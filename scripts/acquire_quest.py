#!/usr/bin/env python3
"""Acquire and freeze a small, independent Shogi Quest benchmark corpus.

Bulk data, caches, account identifiers, and acquisition state stay under a
Git-external runtime.  Only the explicitly selected, anonymized CSA snapshot is
written into the repository by the ``snapshot`` subcommand.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from prepare import REPO, sha256


CONFIG_PATH = REPO / "config/quest-corpus.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
DEFAULT_CORPUS_ROOT = (
    Path.home() / ".local/share/sekirei-weight2" / CONFIG["corpus_id"]
)
USER_AGENT = "sekirei-weight2/0.1 (+https://github.com/phni3j9a/sekirei-weight2)"
GAME_ID_RE = re.compile(r"^[a-z0-9]{8,32}$")
MOVE_RE = re.compile(
    r"^[+-][0-9]{4}(?:FU|KY|KE|GI|KI|KA|HI|OU|TO|NY|NK|NG|UM|RY)$"
)
NAME_RE = re.compile(r"^N[+-]")
STANDARD_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
STANDARD_BOARD = (
    "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY",
    "P2 * -HI *  *  *  *  * -KA * ",
    "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU",
    "P4 *  *  *  *  *  *  *  *  * ",
    "P5 *  *  *  *  *  *  *  *  * ",
    "P6 *  *  *  *  *  *  *  *  * ",
    "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU",
    "P8 * +KA *  *  *  *  * +HI * ",
    "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp")
    mode = "wb" if isinstance(data, bytes) else "w"
    kwargs = {} if mode == "wb" else {"encoding": "utf-8"}
    with staged.open(mode, **kwargs) as stream:
        stream.write(data)
        stream.flush()
    staged.replace(path)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def user_key(user_id):
    return digest_bytes(user_id.casefold().encode())


def looks_like_bot(player):
    player_id = str(player.get("id", ""))
    name = str(player.get("name", ""))
    avatar = str(player.get("avatar", ""))
    return (
        player_id.startswith(":")
        or name.startswith(":")
        or avatar.casefold().startswith("bot_")
    )


def parse_official_attrs(html):
    """Read the public attributes embedded in the server-rendered Nuxt page."""
    match = re.search(r"\.attrs=(\[[^;]*\]);", html)
    if not match:
        raise ValueError("official page has no embedded game attributes")
    try:
        attrs = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise ValueError("official game attributes are not JSON literals") from error
    if not isinstance(attrs, list) or not all(isinstance(item, str) for item in attrs):
        raise ValueError("official game attributes have an unexpected shape")
    return attrs


def parse_history(payload):
    try:
        history = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("history endpoint returned invalid JSON") from error
    games = history.get("games") if isinstance(history, dict) else None
    if not isinstance(games, list) or not all(isinstance(game, dict) for game in games):
        raise ValueError("history response has no valid games list")
    return games


def parse_csa(text):
    """Parse the small CSA subset used by the public download endpoint.

    This is an integrity/normalization pass.  Actual move legality is checked
    separately with the pinned cshogi environment before a game is accepted.
    """
    if not text.strip():
        raise ValueError("CSA is empty")
    if "\x00" in text:
        raise ValueError("CSA contains a NUL byte")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line for line in lines if line]
    board = tuple(line for line in lines if re.match(r"^P[1-9]", line))
    if board != STANDARD_BOARD:
        raise ValueError("CSA is not a standard-even starting position")
    turns = [line for line in lines if line in ("+", "-")]
    if turns != ["+"]:
        raise ValueError("CSA does not start with black to move")
    moves = [line for line in lines if MOVE_RE.fullmatch(line)]
    if not moves:
        raise ValueError("CSA contains no moves")
    malformed_moves = [
        line for line in lines
        if re.match(r"^[+-][0-9]", line) and not MOVE_RE.fullmatch(line)
    ]
    if malformed_moves:
        raise ValueError("CSA contains a malformed move")
    for index, move in enumerate(moves):
        expected = "+" if index % 2 == 0 else "-"
        if move[0] != expected:
            raise ValueError(f"CSA turn order breaks at ply {index + 1}")
    names = [line for line in lines if NAME_RE.match(line)]
    if len(names) != 2 or not names[0].startswith("N+") or not names[1].startswith("N-"):
        raise ValueError("CSA must contain one name for each side")
    canonical = "\n".join((*STANDARD_BOARD, "+", *moves)) + "\n"
    return {
        "moves": moves,
        "plies": len(moves),
        "canonical": canonical,
        "canonical_sha256": digest_bytes(canonical.encode()),
    }


def anonymized_csa(parsed):
    return "\n".join((
        "V2.2",
        "N+black",
        "N-white",
        *STANDARD_BOARD,
        "+",
        *parsed["moves"],
        "",
    ))


def require_cshogi():
    try:
        cshogi = importlib.import_module("cshogi")
        csa = importlib.import_module("cshogi.CSA")
    except ImportError as error:
        raise RuntimeError(
            "CSA legality checking needs the pinned audit environment; run "
            "`python3 scripts/prepare.py audit-deps` and invoke this script "
            "with <runtime>/venv/bin/python"
        ) from error
    expected = None
    for line in (REPO / "config/audit-requirements.txt").read_text().splitlines():
        if line.casefold().startswith("cshogi=="):
            expected = line.split("==", 1)[1]
            break
    actual = importlib.metadata.version("cshogi")
    if expected is None or actual != expected:
        raise RuntimeError(
            f"cshogi version mismatch: expected {expected!r}, found {actual!r}"
        )
    return cshogi, csa


def validate_legal_csa(text, expected_plies, cshogi, csa):
    try:
        games = csa.Parser.parse_str(text)
    except Exception as error:
        raise ValueError(f"cshogi rejected CSA: {error}") from error
    if len(games) != 1:
        raise ValueError(f"cshogi parsed {len(games)} games instead of one")
    game = games[0]
    if game.sfen != STANDARD_SFEN:
        raise ValueError("cshogi did not parse the standard initial position")
    if len(game.moves) != expected_plies:
        raise ValueError("cshogi and text parser disagree about move count")
    board = cshogi.Board(game.sfen)
    for index, move in enumerate(game.moves):
        if not board.is_legal(move):
            raise ValueError(f"illegal move at ply {index + 1}")
        board.push(move)


class FetchError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class PoliteClient:
    def __init__(self, delay_seconds, max_network_requests=None):
        if delay_seconds < 1.0:
            raise ValueError("request delay must be at least 1.0 second")
        self.delay_seconds = delay_seconds
        self.max_network_requests = max_network_requests
        self.last_request = None
        self.network_requests = 0
        self.cache_hits = 0

    def get(self, url, cache_path, params=None, accept="*/*"):
        if cache_path.exists():
            self.cache_hits += 1
            return cache_path.read_bytes()
        if (self.max_network_requests is not None
                and self.network_requests >= self.max_network_requests):
            raise FetchError("configured network request limit reached")
        query = urlencode(params or {})
        target = f"{url}?{query}" if query else url
        for attempt in range(3):
            if self.last_request is not None:
                remaining = self.delay_seconds - (time.monotonic() - self.last_request)
                if remaining > 0:
                    time.sleep(remaining)
            request = Request(target, headers={
                "Accept": accept,
                "User-Agent": USER_AGENT,
            })
            self.last_request = time.monotonic()
            self.network_requests += 1
            try:
                with urlopen(request, timeout=30) as response:
                    data = response.read(2 * 1024 * 1024 + 1)
                    if len(data) > 2 * 1024 * 1024:
                        raise FetchError(f"response is unexpectedly large: {target}")
                atomic_write(cache_path, data)
                return data
            except HTTPError as error:
                if error.code == 404:
                    raise FetchError(f"HTTP 404: {target}", status=404) from error
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise FetchError(f"HTTP {error.code}: {target}", status=error.code) from error
            except (URLError, TimeoutError) as error:
                if attempt == 2:
                    raise FetchError(f"request failed: {target}: {error}") from error
            time.sleep(min(30, 2 ** (attempt + 1)))
        raise AssertionError("unreachable retry loop")


def new_state():
    return {
        "schema_version": 1,
        "corpus_id": CONFIG["corpus_id"],
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config_sha256": sha256(CONFIG_PATH),
        "source_policy": {
            "public_history_and_download_frontend": CONFIG["history_url"],
            "official_game_page": CONFIG["official_game_url"],
            "terms_url": CONFIG["terms_url"],
            "terms_checked_date": CONFIG["terms_checked_date"],
            "user_agent": USER_AGENT,
        },
        "pending_users": list(CONFIG["seed_users"]),
        "known_users": {},
        "history_queries": {},
        "games": {},
        "canonical_games": {},
        "request_stats": {"network_requests": 0, "cache_hits": 0},
    }


def load_state(root):
    path = root / "state.json"
    if not path.exists():
        state = new_state()
        save_state(root, state)
        return state
    state = json.loads(path.read_text())
    if state.get("schema_version") != 1 or state.get("corpus_id") != CONFIG["corpus_id"]:
        raise RuntimeError("acquisition state has an unsupported schema or corpus ID")
    if state.get("config_sha256") != sha256(CONFIG_PATH):
        raise RuntimeError(
            "acquisition config changed; preserve this corpus and start with a new root"
        )
    return state


def state_counts(state):
    dispositions = Counter(game["disposition"] for game in state["games"].values())
    reasons = Counter(
        game.get("reason", "") for game in state["games"].values()
        if game["disposition"] == "rejected"
    )
    return dispositions, reasons


def save_state(root, state):
    state["updated_at"] = utc_now()
    atomic_write(root / "state.json", json_bytes(state))
    dispositions, reasons = state_counts(state)
    accepted = [
        game for game in state["games"].values()
        if game["disposition"] == "accepted"
    ]
    aggregate = digest_bytes(
        "".join(sorted(game["canonical_sha256"] for game in accepted)).encode()
    )
    manifest = {
        "schema_version": 1,
        "corpus_id": state["corpus_id"],
        "config_sha256": state["config_sha256"],
        "created_at": state["created_at"],
        "updated_at": state["updated_at"],
        "accepted_games": dispositions["accepted"],
        "rejected_games": dispositions["rejected"],
        "rejected_by_reason": dict(sorted(reasons.items())),
        "pending_users": len(state["pending_users"]),
        "history_queries": len(state["history_queries"]),
        "corpus_canonical_sha256": aggregate,
        "request_stats": state["request_stats"],
        "source_policy": state["source_policy"],
    }
    atomic_write(root / "manifest.json", json_bytes(manifest))


def remember_user(state, player):
    player_id = player.get("id")
    if not isinstance(player_id, str) or not player_id or len(player_id) > 100:
        return
    key = user_key(player_id)
    if key not in state["known_users"]:
        state["known_users"][key] = {
            "id": player_id,
            "bot_marker": looks_like_bot(player),
        }
        if not looks_like_bot(player):
            state["pending_users"].append(player_id)


def compact_metadata(game, game_type):
    players = []
    for player in game.get("players", []):
        players.append({
            key: player[key]
            for key in ("id", "name", "oldR", "oldD", "avatar")
            if key in player
        })
    return {
        "id": game.get("id"),
        "game_type": game_type,
        "created": game.get("created"),
        "players": players,
        "final_status": game.get("finalStatus"),
        "length": game.get("length"),
        "handicap": game.get("handicap"),
        "move_exists": game.get("moveExists"),
    }


def reject(state, game_id, metadata, reason):
    state["games"][game_id] = {
        "disposition": "rejected",
        "reason": reason,
        "metadata": metadata,
    }


def process_game(root, state, game, game_type, client, cshogi, csa):
    game_id = game.get("id")
    if not isinstance(game_id, str) or not GAME_ID_RE.fullmatch(game_id):
        return False
    players = game.get("players")
    if isinstance(players, list):
        for player in players:
            if isinstance(player, dict):
                remember_user(state, player)
    if game_id in state["games"]:
        return False

    metadata = compact_metadata(game, game_type)
    if not isinstance(players, list) or len(players) != 2:
        reject(state, game_id, metadata, "unexpected_players")
        return False
    if game.get("handicap"):
        reject(state, game_id, metadata, "handicap")
        return False
    if any(looks_like_bot(player) for player in players):
        reject(state, game_id, metadata, "bot_marker")
        return False

    official_url = CONFIG["official_game_url"].format(game_id=game_id)
    try:
        html_bytes = client.get(
            official_url,
            root / "cache/official" / f"{game_id}.html",
            accept="text/html",
        )
    except FetchError as error:
        if error.status == 404:
            reject(state, game_id, metadata, "official_page_missing")
            return False
        raise
    try:
        attrs = parse_official_attrs(html_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        reject(state, game_id, metadata, "official_attributes_invalid")
        return False
    required_attr = CONFIG["required_official_attribute"]
    if required_attr not in attrs:
        reject(state, game_id, metadata, "official_human_marker_missing")
        return False

    try:
        csa_bytes = client.get(
            CONFIG["download_url"],
            root / "cache/csa" / f"{game_id}.csa",
            params={"id": game_id},
            accept="application/octet-stream,text/plain",
        )
    except FetchError as error:
        if error.status == 404:
            reject(state, game_id, metadata, "csa_missing")
            return False
        raise
    if not csa_bytes:
        reject(state, game_id, metadata, "csa_empty")
        return False
    try:
        text = csa_bytes.decode("utf-8")
        parsed = parse_csa(text)
        validate_legal_csa(text, parsed["plies"], cshogi, csa)
    except (UnicodeDecodeError, ValueError):
        reject(state, game_id, metadata, "csa_invalid_or_illegal")
        return False
    reported_length = game.get("length")
    if isinstance(reported_length, int) and reported_length not in (
            parsed["plies"], parsed["plies"] + 1):
        reject(state, game_id, metadata, "move_count_mismatch")
        return False
    if parsed["canonical_sha256"] in state["canonical_games"]:
        reject(state, game_id, metadata, "duplicate_moves")
        return False

    raw_hash = digest_bytes(csa_bytes)
    raw_path = root / "games" / f"{raw_hash}.csa"
    if raw_path.exists() and sha256(raw_path) != raw_hash:
        raise RuntimeError(f"corrupt content-addressed game: {raw_path}")
    if not raw_path.exists():
        atomic_write(raw_path, csa_bytes)
    state["games"][game_id] = {
        "disposition": "accepted",
        "metadata": metadata,
        "official_attrs": attrs,
        "source_url": official_url,
        "raw_path": str(raw_path.relative_to(root)),
        "raw_bytes": len(csa_bytes),
        "raw_sha256": raw_hash,
        "plies": parsed["plies"],
        "canonical_sha256": parsed["canonical_sha256"],
        "accepted_at": utc_now(),
    }
    state["canonical_games"][parsed["canonical_sha256"]] = game_id
    return True


def crawl(root, args):
    cshogi, csa = require_cshogi()
    state = load_state(root)
    client = PoliteClient(args.delay, args.max_network_requests)
    accepted_at_start = state_counts(state)[0]["accepted"]
    target = args.target
    print(f"Corpus: {root}")
    print(f"Accepted: {accepted_at_start}/{target}; delay: {args.delay:.1f}s")

    try:
        while state_counts(state)[0]["accepted"] < target:
            if not state["pending_users"]:
                raise RuntimeError("user crawl queue exhausted before reaching the target")
            current_user = state["pending_users"].pop(0)
            current_key = user_key(current_user)
            if current_key not in state["known_users"]:
                state["known_users"][current_key] = {
                    "id": current_user,
                    "bot_marker": current_user.startswith(":"),
                }
            for game_type in CONFIG["game_types"]:
                query_key = f"{current_key}:{game_type}"
                if query_key in state["history_queries"]:
                    continue
                history_cache = root / "cache/history" / f"{query_key}.json"
                payload = client.get(
                    CONFIG["history_url"],
                    history_cache,
                    params={"userId": current_user, "gtype": game_type},
                    accept="application/json",
                )
                games = parse_history(payload)
                for game in games:
                    if not isinstance(game, dict):
                        continue
                    changed = process_game(
                        root, state, game, game_type, client, cshogi, csa
                    )
                    state["request_stats"]["network_requests"] += client.network_requests
                    state["request_stats"]["cache_hits"] += client.cache_hits
                    client.network_requests = 0
                    client.cache_hits = 0
                    save_state(root, state)
                    accepted = state_counts(state)[0]["accepted"]
                    if changed and accepted % args.progress_every == 0:
                        rejected = state_counts(state)[0]["rejected"]
                        print(
                            f"Accepted {accepted}/{target}; rejected {rejected}; "
                            f"queued users {len(state['pending_users'])}",
                            flush=True,
                        )
                    if accepted >= target:
                        break
                if state_counts(state)[0]["accepted"] >= target:
                    break
                state["history_queries"][query_key] = {
                    "user_key": current_key,
                    "game_type": game_type,
                    "games": len(games),
                    "retrieved_at": utc_now(),
                    "cache_sha256": sha256(history_cache),
                }
                save_state(root, state)
    finally:
        state["request_stats"]["network_requests"] += client.network_requests
        state["request_stats"]["cache_hits"] += client.cache_hits
        save_state(root, state)

    manifest = json.loads((root / "manifest.json").read_text())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"PASS: acquired {manifest['accepted_games']} verified games")


def verify_corpus(root):
    cshogi, csa = require_cshogi()
    state = load_state(root)
    accepted = 0
    canonical = {}
    for game_id, game in state["games"].items():
        if game["disposition"] != "accepted":
            continue
        path = root / game["raw_path"]
        if path.stat().st_size != game["raw_bytes"] or sha256(path) != game["raw_sha256"]:
            raise RuntimeError(f"raw game hash/size mismatch: {game_id}")
        text = path.read_text()
        parsed = parse_csa(text)
        validate_legal_csa(text, parsed["plies"], cshogi, csa)
        if parsed["canonical_sha256"] != game["canonical_sha256"]:
            raise RuntimeError(f"canonical game hash mismatch: {game_id}")
        if parsed["canonical_sha256"] in canonical:
            raise RuntimeError(f"duplicate move sequence: {game_id}")
        canonical[parsed["canonical_sha256"]] = game_id
        if CONFIG["required_official_attribute"] not in game["official_attrs"]:
            raise RuntimeError(f"missing official human marker: {game_id}")
        accepted += 1
    if canonical != state["canonical_games"]:
        raise RuntimeError("canonical game index differs from accepted games")
    print(f"PASS: {accepted} unique, standard, human-marked legal games")
    return state


def game_ratings(game):
    values = []
    for player in game["metadata"]["players"]:
        rating = player.get("oldR")
        if not isinstance(rating, (int, float)):
            return None
        values.append(float(rating))
    return values


def selection_rank(game_id, game):
    seed = CONFIG["selection"]["seed"]
    value = f"{seed}\0{game['canonical_sha256']}\0{game_id}".encode()
    return digest_bytes(value)


def choose_snapshot(state):
    spec = CONFIG["selection"]
    candidates = []
    for game_id, game in state["games"].items():
        if game["disposition"] != "accepted":
            continue
        ratings = game_ratings(game)
        if ratings is None or min(ratings) < spec["min_rating"]:
            continue
        if max(ratings) - min(ratings) > spec["max_rating_gap"]:
            continue
        if not spec["min_plies"] <= game["plies"] <= spec["max_plies"]:
            continue
        candidates.append((selection_rank(game_id, game), game_id, game))
    candidates.sort()

    chosen = []
    used_players = set()
    for bucket in spec["buckets"]:
        bucket_games = []
        for rank, game_id, game in candidates:
            if game["metadata"]["game_type"] != bucket["game_type"]:
                continue
            player_keys = {
                user_key(player["id"])
                for player in game["metadata"]["players"]
            }
            if player_keys & used_players:
                continue
            bucket_games.append((rank, game_id, game, player_keys))
            used_players.update(player_keys)
            if len(bucket_games) == bucket["count"]:
                break
        if len(bucket_games) != bucket["count"]:
            raise RuntimeError(
                f"only {len(bucket_games)} games satisfy selection bucket {bucket}"
            )
        for item in bucket_games:
            chosen.append((bucket["split"], *item[:3]))
    return chosen


def write_snapshot(root, output):
    state = verify_corpus(root)
    accepted = state_counts(state)[0]["accepted"]
    if accepted < CONFIG["target_games"]:
        raise RuntimeError(
            f"snapshot needs {CONFIG['target_games']} accepted games; found {accepted}"
        )
    if output.exists():
        raise RuntimeError(f"snapshot already exists; refusing to replace it: {output}")
    output.mkdir(parents=True)
    selected = choose_snapshot(state)
    records = []
    counters = Counter()
    for split, rank, game_id, game in selected:
        counters[split] += 1
        snapshot_id = f"{split}-{counters[split]:02d}"
        parsed = parse_csa((root / game["raw_path"]).read_text())
        content = anonymized_csa(parsed)
        path = output / split / f"{snapshot_id}.csa"
        atomic_write(path, content)
        records.append({
            "snapshot_id": snapshot_id,
            "split": split,
            "game_type": game["metadata"]["game_type"],
            "created": game["metadata"]["created"],
            "plies": game["plies"],
            "final_status": game["metadata"]["final_status"],
            "source_game_id": game_id,
            "source_url": game["source_url"],
            "source_csa_sha256": game["raw_sha256"],
            "canonical_sha256": game["canonical_sha256"],
            "snapshot_csa_sha256": sha256(path),
            "selection_rank": rank,
        })
    corpus_manifest = json.loads((root / "manifest.json").read_text())
    manifest = {
        "schema_version": 1,
        "benchmark_id": "shogiquest-benchmark-v1",
        "frozen_before_engine_analysis": True,
        "source": "Shogi Quest",
        "source_terms_url": CONFIG["terms_url"],
        "source_terms_checked_date": CONFIG["terms_checked_date"],
        "source_corpus_id": CONFIG["corpus_id"],
        "source_corpus_accepted_games": accepted,
        "source_corpus_canonical_sha256": corpus_manifest["corpus_canonical_sha256"],
        "anonymization": (
            "CSA player names and ratings replaced; source game IDs retained for provenance"
        ),
        "human_game_check": (
            "both list records lack known bot markers and official page contains "
            + CONFIG["required_official_attribute"]
        ),
        "legality_check": "all moves replayed by pinned cshogi",
        "selection": CONFIG["selection"],
        "games": records,
    }
    atomic_write(output / "manifest.json", json_bytes(manifest))
    print(f"PASS: wrote {len(records)} frozen benchmark games to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_CORPUS_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    crawl_parser = subparsers.add_parser("crawl", help="resume public acquisition")
    crawl_parser.add_argument("--target", type=int, default=CONFIG["target_games"])
    crawl_parser.add_argument(
        "--delay", type=float, default=CONFIG["request_delay_seconds"]
    )
    crawl_parser.add_argument("--max-network-requests", type=int)
    crawl_parser.add_argument("--progress-every", type=int, default=10)
    crawl_parser.add_argument(
        "--dry-run", action="store_true",
        help="print the immutable acquisition plan without writing or requesting",
    )
    subparsers.add_parser("verify", help="verify all local accepted games")
    snapshot_parser = subparsers.add_parser(
        "snapshot", help="freeze the deterministic development/final benchmark"
    )
    snapshot_parser.add_argument(
        "--output", type=Path, default=REPO / "benchmarks/shogiquest-v1"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if getattr(args, "target", 1) < 1:
        raise SystemExit("--target must be positive")
    if getattr(args, "progress_every", 1) < 1:
        raise SystemExit("--progress-every must be positive")
    if args.command == "crawl" and args.delay < 1.0:
        raise SystemExit("--delay must be at least 1.0 second")
    if (getattr(args, "max_network_requests", None) is not None
            and args.max_network_requests < 1):
        raise SystemExit("--max-network-requests must be positive")
    root = args.root.expanduser().resolve()
    if args.command == "crawl" and args.dry_run:
        print(json.dumps({
            "corpus_root": str(root),
            "target_games": args.target,
            "game_types": CONFIG["game_types"],
            "seed_users": CONFIG["seed_users"],
            "request_delay_seconds": args.delay,
            "max_network_requests": args.max_network_requests,
            "minimum_game_request_seconds": round(2 * args.target * args.delay, 1),
            "writes": False,
            "network_requests": False,
            "config_sha256": sha256(CONFIG_PATH),
        }, indent=2, ensure_ascii=False))
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".acquire.lock").open("a") as lockfile:
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another acquisition process holds {root}") from error
        if args.command == "crawl":
            crawl(root, args)
        elif args.command == "verify":
            verify_corpus(root)
        elif args.command == "snapshot":
            write_snapshot(root, args.output.resolve())
        else:
            raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except (FetchError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
