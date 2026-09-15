#!/usr/bin/env python3
"""Run and inspect the fixed development benchmark.

The benchmark intentionally has no third-party Python dependency.  CSA is
converted from the standard initial position with a small, strict board
tracker, and every observation is made by a fresh USI process.  Large run
files belong in the external runtime; this module only writes there when a
subcommand is invoked.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import select
import signal
import subprocess
import sys
import threading
import time

from prepare import DEFAULT_RUNTIME, REPO, sha256, verify_weight


CONFIG_PATH = REPO / "config/development-benchmark.json"
BENCHMARK_MANIFEST_PATH = REPO / "benchmarks/shogiquest-v1/manifest.json"
DEVELOPMENT_ROOT = REPO / "benchmarks/shogiquest-v1/development"

STANDARD_BOARD_LINES = (
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
STANDARD_SFEN = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/"
    "LNSGKGSNL b - 1"
)

BASE_PIECES = ("FU", "KY", "KE", "GI", "KI", "KA", "HI", "OU")
PROMOTED_PIECES = {"FU": "TO", "KY": "NY", "KE": "NK", "GI": "NG", "KA": "UM", "HI": "RY"}
UNPROMOTED_PIECES = {value: key for key, value in PROMOTED_PIECES.items()}
GOLD_LIKE = {"KI", "TO", "NY", "NK", "NG"}
PIECE_TO_DROP = {"FU": "P", "KY": "L", "KE": "N", "GI": "S", "KI": "G", "KA": "B", "HI": "R"}
DROP_TO_PIECE = {value: key for key, value in PIECE_TO_DROP.items()}

CSA_MOVE_RE = re.compile(
    r"^(?P<side>[+-])(?P<from>[0-9]{2})(?P<to>[0-9]{2})"
    r"(?P<piece>FU|KY|KE|GI|KI|KA|HI|OU|TO|NY|NK|NG|UM|RY)$"
)
USI_NORMAL_RE = re.compile(r"^[1-9][a-i][1-9][a-i]\+?$")
USI_DROP_RE = re.compile(r"^[PLNSGBR]\*[1-9][a-i]$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class BenchmarkError(RuntimeError):
    """An input, configuration, protocol, or run-integrity error."""


class ProtocolError(BenchmarkError):
    """The engine did not speak the required USI subset."""


class StartupError(BenchmarkError):
    """The engine could not be started or initialized."""


class ConfigurationError(BenchmarkError):
    """Pinned engine or benchmark configuration was not usable."""


class ProcessExitError(BenchmarkError):
    """A USI process exited before a normal lifecycle completed."""


class CleanupError(ProcessExitError):
    """The process group could not be safely proven/cleaned."""


class ObservationTimeout(BenchmarkError):
    """An observation exceeded its phase deadline."""


class NodeBudgetError(BenchmarkError):
    """An observation does not meet the configured node evidence policy."""


TECHNICAL_FAILURE_STATUSES = {
    "timeout",
    "protocol_failure",
    "startup_failure",
    "config_failure",
    "exit_failure",
    "node_budget_failure",
    "cleanup_failure",
}
OBSERVATION_STATUSES = {"exact_cp", "bound_cp", "mate", "no_score"}
OBSERVATION_COMPARE_FIELDS = (
    "status",
    "bestmove",
    "bestmove_kind",
    "pv",
    "pv_head",
    "score_kind",
    "score_cp_stm",
    "score_cp_sente",
    "reported_cp_stm",
    "reported_cp_sente",
    "score_bound_stm",
    "score_bound_sente",
    "score_mate_stm",
    "mate_distance",
    "mate_distance_known",
    "mate_sign",
    "winner",
    "winner_stm",
    "winner_sente",
    "reported_nodes_at_score",
    "last_reported_nodes",
    "engine_time_ms",
    "engine_time",
    "wall_go_to_bestmove_ns",
    "wall_go_to_bestmove_seconds",
    "raw_score_line",
    "raw_bestmove_line",
    "info_count",
    "failure_reason",
)


def canonical_json_bytes(value):
    """Return the one canonical JSON representation used for all hashes."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def json_sha256(value):
    return digest_bytes(canonical_json_bytes(value))


def atomic_write(path, data):
    """Write bytes/text durably, then replace the destination atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    binary = isinstance(data, bytes)
    mode = "wb" if binary else "w"
    kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
    try:
        with staged.open(mode, **kwargs) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            staged.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path, value):
    atomic_write(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def side_name(side):
    return "black" if side == 1 else "white"


def opposite_side(side):
    return -1 if side == 1 else 1


def _square_from_csa(value):
    if not re.fullmatch(r"[1-9][1-9]", value):
        raise ValueError(f"invalid CSA square: {value}")
    return int(value[0]), int(value[1])


def _square_to_usi(square):
    file_, rank = square
    return f"{file_}{chr(ord('a') + rank - 1)}"


def _square_from_usi(value):
    if not re.fullmatch(r"[1-9][a-i]", value):
        raise ValueError(f"invalid USI square: {value}")
    return int(value[0]), ord(value[1]) - ord("a") + 1


def unpromote(piece):
    return UNPROMOTED_PIECES.get(piece, piece)


def promote(piece):
    return PROMOTED_PIECES.get(piece)


def _in_board(square):
    return all(1 <= value <= 9 for value in square)


def _in_promotion_zone(side, rank):
    return rank <= 3 if side == 1 else rank >= 7


class BoardTracker:
    """Strict standard-start-position board state used for CSA and USI."""

    def __init__(self, board=None, hands=None, side=1):
        self.board = dict(board or {})
        self.hands = {
            1: {piece: 0 for piece in BASE_PIECES},
            -1: {piece: 0 for piece in BASE_PIECES},
        }
        for owner, values in (hands or {}).items():
            for piece, count in values.items():
                if piece not in BASE_PIECES or count < 0:
                    raise ValueError("invalid hand state")
                self.hands[int(owner)][piece] = int(count)
        if side not in (1, -1):
            raise ValueError("side must be black (+1) or white (-1)")
        self.side = side

    @classmethod
    def standard(cls):
        board = {}
        first = ("KY", "KE", "GI", "KI", "OU", "KI", "GI", "KE", "KY")
        for file_, piece in enumerate(first, 1):
            board[(file_, 9)] = (1, piece)
            board[(file_, 1)] = (-1, piece)
        board[(2, 8)] = (1, "HI")
        board[(8, 8)] = (1, "KA")
        board[(8, 2)] = (-1, "HI")
        board[(2, 2)] = (-1, "KA")
        for file_ in range(1, 10):
            board[(file_, 7)] = (1, "FU")
            board[(file_, 3)] = (-1, "FU")
        return cls(board=board)

    def clone(self):
        return BoardTracker(
            board=self.board,
            hands=self.hands,
            side=self.side,
        )

    def _path_clear(self, source, target):
        df = target[0] - source[0]
        dr = target[1] - source[1]
        step_f = 0 if df == 0 else (1 if df > 0 else -1)
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        current = (source[0] + step_f, source[1] + step_r)
        while current != target:
            if current in self.board:
                return False
            current = (current[0] + step_f, current[1] + step_r)
        return True

    def _piece_can_move(self, owner, piece, source, target):
        df = target[0] - source[0]
        dr = target[1] - source[1]
        forward = -1 if owner == 1 else 1
        adf, adr = abs(df), abs(dr)
        if piece == "FU":
            return df == 0 and dr == forward
        if piece == "KY":
            return df == 0 and dr * forward > 0 and self._path_clear(source, target)
        if piece == "KE":
            return adf == 1 and dr == 2 * forward
        if piece == "GI":
            return (dr == forward and adf <= 1) or (dr == -forward and adf == 1)
        if piece in GOLD_LIKE or piece == "KI":
            return (dr == forward and adf <= 1) or (dr == 0 and adf == 1) or (dr == -forward and df == 0)
        if piece == "KA":
            return adf == adr and adf > 0 and self._path_clear(source, target)
        if piece == "HI":
            return ((df == 0) != (dr == 0)) and self._path_clear(source, target)
        if piece == "OU":
            return max(adf, adr) == 1
        if piece == "UM":
            diagonal = adf == adr and adf > 0 and self._path_clear(source, target)
            orthogonal = max(adf, adr) == 1 and (df == 0 or dr == 0)
            return diagonal or orthogonal
        if piece == "RY":
            straight = ((df == 0) != (dr == 0)) and self._path_clear(source, target)
            diagonal = adf == 1 and adr == 1
            return straight or diagonal
        return False

    def _validate_drop(self, owner, piece, target):
        if piece not in PIECE_TO_DROP or self.hands[owner][piece] <= 0:
            raise ValueError("drop piece is not in hand")
        if target in self.board:
            raise ValueError("drop target is occupied")
        forward_last = 1 if owner == 1 else 9
        forward_second_last = (1, 2) if owner == 1 else (8, 9)
        if piece in ("FU", "KY") and target[1] == forward_last:
            raise ValueError("piece cannot be dropped on the last rank")
        if piece == "KE" and target[1] in forward_second_last:
            raise ValueError("knight cannot be dropped on the last two ranks")
        if piece == "FU":
            if any(
                owner == board_owner and square[0] == target[0] and board_piece == "FU"
                for square, (board_owner, board_piece) in self.board.items()
            ):
                raise ValueError("two unpromoted pawns on one file")

    def _validate_normal(self, owner, source, target, result_piece):
        source_item = self.board.get(source)
        if source_item is None:
            raise ValueError("move source is empty")
        source_owner, source_piece = source_item
        if source_owner != owner:
            raise ValueError("move source belongs to the other side")
        target_item = self.board.get(target)
        if target_item is not None:
            target_owner, target_piece = target_item
            if target_owner == owner:
                raise ValueError("move target belongs to the moving side")
            if target_piece == "OU":
                raise ValueError("capturing a king is not a legal CSA move")
        if result_piece not in (*BASE_PIECES, *PROMOTED_PIECES.values()):
            raise ValueError("unknown resulting piece")
        if not self._piece_can_move(owner, source_piece, source, target):
            raise ValueError("piece movement is not legal")
        if result_piece == source_piece:
            pass
        elif promote(source_piece) == result_piece:
            if not _in_promotion_zone(owner, source[1]) and not _in_promotion_zone(owner, target[1]):
                raise ValueError("promotion is outside the promotion zone")
        else:
            raise ValueError("CSA resulting piece does not match source piece")
        base = unpromote(source_piece)
        if result_piece == base:
            last_rank = 1 if owner == 1 else 9
            second_last = (1, 2) if owner == 1 else (8, 9)
            if base in ("FU", "KY") and target[1] == last_rank:
                raise ValueError("unpromoted piece cannot move to the last rank")
            if base == "KE" and target[1] in second_last:
                raise ValueError("unpromoted knight cannot move to the last two ranks")

    def _apply(self, owner, source, target, result_piece):
        target_item = self.board.get(target)
        if target_item is not None:
            captured = unpromote(target_item[1])
            if captured != "OU":
                self.hands[owner][captured] += 1
        if source == (0, 0):
            self.hands[owner][result_piece] -= 1
        else:
            del self.board[source]
        self.board[target] = (owner, result_piece)
        self.side = opposite_side(self.side)

    def apply_csa(self, line):
        match = CSA_MOVE_RE.fullmatch(line)
        if not match:
            raise ValueError(f"malformed CSA move: {line}")
        owner = 1 if match.group("side") == "+" else -1
        if owner != self.side:
            raise ValueError("CSA side-to-move transition is invalid")
        source_text, target_text = match.group("from"), match.group("to")
        if target_text == "00" or target_text[0] == "0" or target_text[1] == "0":
            raise ValueError("CSA destination is invalid")
        target = _square_from_csa(target_text)
        result_piece = match.group("piece")
        if source_text == "00":
            if result_piece not in BASE_PIECES or result_piece == "OU":
                raise ValueError("CSA drops must use an unpromoted non-king piece")
            self._validate_drop(owner, result_piece, target)
            usi = f"{PIECE_TO_DROP[result_piece]}*{_square_to_usi(target)}"
            self._apply(owner, (0, 0), target, result_piece)
            return usi
        source = _square_from_csa(source_text)
        self._validate_normal(owner, source, target, result_piece)
        usi = _square_to_usi(source) + _square_to_usi(target)
        if result_piece != self.board[source][1]:
            usi += "+"
        self._apply(owner, source, target, result_piece)
        return usi

    def apply_usi(self, move):
        if USI_DROP_RE.fullmatch(move):
            piece = DROP_TO_PIECE[move[0]]
            target = _square_from_usi(move[2:])
            self._validate_drop(self.side, piece, target)
            self._apply(self.side, (0, 0), target, piece)
            return
        if not USI_NORMAL_RE.fullmatch(move):
            raise ValueError(f"malformed USI move: {move}")
        source = _square_from_usi(move[:2])
        target = _square_from_usi(move[2:4])
        source_piece = self.board.get(source)
        if source_piece is None:
            raise ValueError("USI move source is empty")
        result_piece = source_piece[1]
        if move.endswith("+"):
            promoted = promote(result_piece)
            if promoted is None:
                raise ValueError("USI move asks for an invalid promotion")
            result_piece = promoted
        self._validate_normal(self.side, source, target, result_piece)
        self._apply(self.side, source, target, result_piece)

    def is_legal_usi(self, move):
        try:
            copy = self.clone()
            copy.apply_usi(move)
        except (TypeError, ValueError, KeyError):
            return False
        return True


def _normalise_csa_text(text):
    if not isinstance(text, str):
        raise ValueError("CSA input must be text")
    if "\x00" in text:
        raise ValueError("CSA contains a NUL byte")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_csa_text(text, game_id=None):
    """Parse and replay one strict standard-start-position CSA game."""
    text = _normalise_csa_text(text)
    lines = [line for line in text.split("\n") if line]
    if not lines:
        raise ValueError("CSA is empty")
    board_lines = tuple(line for line in lines if re.fullmatch(r"P[1-9].*", line))
    if board_lines and board_lines != STANDARD_BOARD_LINES:
        raise ValueError("CSA has a non-standard explicit initial position")
    if "PI" not in lines and not board_lines:
        raise ValueError("CSA has no standard initial-position declaration")
    if lines.count("PI") > 1:
        raise ValueError("CSA has duplicate PI declarations")
    turns = [line for line in lines if line in ("+", "-")]
    if turns != ["+"]:
        raise ValueError("CSA must have exactly one black-to-move marker")
    moves = []
    for line in lines:
        if CSA_MOVE_RE.fullmatch(line):
            moves.append(line)
        elif re.match(r"^[+-][0-9]", line):
            raise ValueError(f"CSA contains a malformed move: {line}")
    if not moves:
        raise ValueError("CSA contains no moves")
    board = BoardTracker.standard()
    usi_moves = []
    boards_after = []
    for index, move in enumerate(moves, 1):
        try:
            usi_moves.append(board.apply_csa(move))
        except ValueError as error:
            raise ValueError(f"invalid CSA move at ply {index}: {error}") from error
        boards_after.append(board.clone())
    canonical = "\n".join((*STANDARD_BOARD_LINES, "+", *moves)) + "\n"
    return {
        "game_id": game_id,
        "moves": moves,
        "usi_moves": usi_moves,
        "plies": len(moves),
        "canonical": canonical,
        "canonical_sha256": digest_bytes(canonical.encode("utf-8")),
        "boards_after": boards_after,
    }


def parse_csa_file(path, game_id=None):
    path = Path(path)
    return parse_csa_text(path.read_text(encoding="utf-8"), game_id=game_id)


def development_manifest_entries():
    try:
        manifest = json.loads(BENCHMARK_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot read benchmark manifest: {error}") from error
    entries = [item for item in manifest.get("games", []) if item.get("split") == "development"]
    if len(entries) != 5:
        raise ConfigurationError(f"expected five development manifest entries, got {len(entries)}")
    return {item["snapshot_id"]: item for item in entries}


def _safe_development_path(relative):
    relative_path = Path(relative)
    if relative_path.is_absolute() or "final" in relative_path.parts:
        raise ConfigurationError("the development runner cannot select final data")
    path = (REPO / "benchmarks/shogiquest-v1" / relative_path).resolve()
    try:
        path.relative_to(DEVELOPMENT_ROOT.resolve())
    except ValueError as error:
        raise ConfigurationError("benchmark input is outside the development directory") from error
    return path


def load_config(path=CONFIG_PATH):
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot read benchmark config: {error}") from error
    if config.get("schema_version") != 1:
        raise ConfigurationError("unsupported development benchmark config schema")
    if config.get("split") != "development":
        raise ConfigurationError("benchmark config must select the development split")
    for item in config.get("games", []):
        if "final" in str(item.get("path", "")).split("/"):
            raise ConfigurationError("final data is not selectable by this runner")
    return config


def load_development_games(config=None):
    config = config or load_config()
    entries = development_manifest_entries()
    games = []
    expected_csa = config.get("hashes", {}).get("development_csa_sha256", {})
    for item in config.get("games", []):
        game_id = item.get("id")
        if game_id not in entries:
            raise ConfigurationError(f"config game is not a development manifest entry: {game_id}")
        path = _safe_development_path(item.get("path", ""))
        game = parse_csa_file(path, game_id=game_id)
        manifest_item = entries[game_id]
        if game["plies"] != item.get("plies") or game["plies"] != manifest_item.get("plies"):
            raise ConfigurationError(f"ply count mismatch for {game_id}")
        raw_hash = sha256(path)
        if raw_hash != manifest_item.get("snapshot_csa_sha256"):
            raise ConfigurationError(f"snapshot CSA hash mismatch for {game_id}")
        if item.get("sha256") and raw_hash != item["sha256"]:
            raise ConfigurationError(f"configured CSA hash mismatch for {game_id}")
        if item.get("canonical_sha256") and game["canonical_sha256"] != item["canonical_sha256"]:
            raise ConfigurationError(f"configured canonical hash mismatch for {game_id}")
        expected_csa[game_id] = raw_hash
        game["csa_sha256"] = raw_hash
        game["source_manifest"] = {
            key: manifest_item[key]
            for key in ("canonical_sha256", "snapshot_id", "game_type", "plies")
            if key in manifest_item
        }
        games.append(game)
    if [game["game_id"] for game in games] != [item.get("id") for item in config.get("games", [])]:
        raise ConfigurationError("development game order is not deterministic")
    if len(games) != 5:
        raise ConfigurationError(f"expected five development games, got {len(games)}")
    return games


def development_csa_hash(games):
    records = [
        {
            "game_id": game["game_id"],
            "csa_sha256": game["csa_sha256"],
            "canonical_sha256": game["canonical_sha256"],
            "plies": game["plies"],
        }
        for game in games
    ]
    return json_sha256(records)


def build_universe(config=None):
    config = config or load_config()
    games = load_development_games(config)
    records = []
    for game in games:
        occurrences = []
        for ply, board in enumerate(game["boards_after"], 1):
            history = game["usi_moves"][:ply]
            occurrences.append(
                {
                    "occurrence_id": f"{game['game_id']}:{ply:03d}",
                    "game_id": game["game_id"],
                    "ply": ply,
                    "side_to_move": side_name(board.side),
                    "moves": history,
                    "position": "position startpos moves " + " ".join(history),
                }
            )
        records.append(
            {
                "game_id": game["game_id"],
                "plies": game["plies"],
                "csa_sha256": game["csa_sha256"],
                "canonical_sha256": game["canonical_sha256"],
                "occurrences": occurrences,
            }
        )
    universe = {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "split": "development",
        "games": records,
    }
    return universe, games


def universe_hash(universe):
    return json_sha256(universe)


def plan_hash(plan):
    """Hash the deterministic plan payload, excluding optional local checks."""
    return json_sha256({
        key: value for key, value in plan.items()
        if key not in ("plan_sha256", "cshogi")
    })


def validate_config_and_universe(config=None):
    config = config or load_config()
    universe, games = build_universe(config)
    hashes = config.get("hashes", {})
    expected_manifest_hash = hashes.get("benchmark_manifest_sha256")
    actual_manifest_hash = sha256(BENCHMARK_MANIFEST_PATH)
    if expected_manifest_hash != actual_manifest_hash:
        raise ConfigurationError(
            f"benchmark manifest hash mismatch: expected {expected_manifest_hash}, got {actual_manifest_hash}"
        )
    expected_csa_hash = hashes.get("development_csa_aggregate_sha256")
    actual_csa_hash = development_csa_hash(games)
    if expected_csa_hash != actual_csa_hash:
        raise ConfigurationError(
            f"development CSA aggregate hash mismatch: expected {expected_csa_hash}, got {actual_csa_hash}"
        )
    expected_universe_hash = hashes.get("universe_sha256")
    actual_universe_hash = universe_hash(universe)
    if expected_universe_hash != actual_universe_hash:
        raise ConfigurationError(
            f"development universe hash mismatch: expected {expected_universe_hash}, got {actual_universe_hash}"
        )
    expected_total = config.get("universe", {}).get("total_occurrences")
    actual_total = sum(game["plies"] for game in games)
    if expected_total != actual_total:
        raise ConfigurationError(f"expected {expected_total} occurrences, got {actual_total}")
    expected_plies = config.get("universe", {}).get("plies", {})
    actual_plies = {game["game_id"]: game["plies"] for game in games}
    if expected_plies != actual_plies:
        raise ConfigurationError(f"configured game ply counts differ: {expected_plies} != {actual_plies}")
    return universe, games


def _strict_positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{field} must be a positive integer")
    return value


def validate_run_policy(config, run_type, *, require_formal_gate=False):
    """Validate repetition and absolute node-policy fields for one run."""
    if run_type not in ("pilot", "formal"):
        raise ConfigurationError("run type must be pilot or formal")
    _strict_positive_int(config.get("requested_nodes"), "requested_nodes")
    section = config.get(run_type)
    if not isinstance(section, dict):
        raise ConfigurationError(f"missing {run_type} run policy")
    if "node_anomaly_ceiling" in section:
        raise ConfigurationError(f"{run_type}.node_anomaly_ceiling is obsolete; use max_reported_nodes")
    repetitions = _strict_positive_int(section.get("repetitions"), f"{run_type}.repetitions")
    max_nodes = section.get("max_reported_nodes")
    if run_type == "pilot":
        if max_nodes is not None:
            raise ConfigurationError("pilot.max_reported_nodes must be null; pilot has no node ceiling")
        if repetitions != 3:
            raise ConfigurationError("pilot.repetitions must remain the fixed value 3")
    elif max_nodes is not None:
        _strict_positive_int(max_nodes, "formal.max_reported_nodes")
        if max_nodes < config["requested_nodes"]:
            raise ConfigurationError("formal.max_reported_nodes must be >= requested_nodes")
    if run_type == "formal" and repetitions != 1:
        raise ConfigurationError("formal.repetitions must remain the fixed value 1")
    if require_formal_gate and run_type == "formal" and not isinstance(section.get("pilot_evidence"), dict):
        raise ConfigurationError(
            "formal node ceiling gate is not frozen: formal.pilot_evidence is required before formal launch; "
            "complete the pilot and freeze its evidence"
        )
    return {
        "repetitions": repetitions,
        "max_reported_nodes": max_nodes,
    }


def make_plan(run_type, config=None):
    if run_type not in ("pilot", "formal"):
        raise ValueError("run type must be pilot or formal")
    config = config or load_config()
    policy = validate_run_policy(config, run_type)
    universe, games = validate_config_and_universe(config)
    selected = []
    for game in universe["games"]:
        if run_type == "pilot":
            plies = (1, (game["plies"] + 1) // 2, game["plies"])
        else:
            plies = range(1, game["plies"] + 1)
        by_ply = {item["ply"]: item for item in game["occurrences"]}
        selected.extend(by_ply[ply] for ply in plies)
    repetitions = policy["repetitions"]
    payload = {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "split": "development",
        "run_type": run_type,
        "repetitions": repetitions,
        "requested_nodes": config["requested_nodes"],
        "max_reported_nodes": policy["max_reported_nodes"],
        "positions": selected,
        "hashes": {
            "benchmark_manifest_sha256": sha256(BENCHMARK_MANIFEST_PATH),
            "development_csa_aggregate_sha256": development_csa_hash(games),
            "universe_sha256": universe_hash(universe),
        },
    }
    payload["plan_sha256"] = plan_hash(payload)
    return payload, games


def _extract_option_names(lines):
    names = set()
    for line in lines:
        match = re.match(r"^option name (.+?) type (?:spin|check|combo|string|button|filename)\b", line)
        if match:
            names.add(match.group(1))
    return names


def _is_normal_bestmove(move):
    return bool(USI_NORMAL_RE.fullmatch(move) or USI_DROP_RE.fullmatch(move))


def _parse_int_token(tokens, name):
    if name not in tokens:
        return None
    index = tokens.index(name)
    if index + 1 >= len(tokens):
        return None
    try:
        return int(tokens[index + 1])
    except ValueError:
        return None


_INFO_INTEGER_FIELDS = {
    "depth",
    "seldepth",
    "time",
    "nodes",
    "nps",
    "hashfull",
    "tbhits",
    "currmovenumber",
    "multipv",
}
_INFO_SINGLE_VALUE_FIELDS = {"currmove"}


def _parse_usi_info_line(line):
    """Parse the structured USI ``info`` subcommands used by this runner.

    In particular, ``info string`` is deliberately not parsed as an engine
    observation.  A string payload is free text and may contain words such as
    ``score`` or ``pv`` without having USI score semantics.
    """
    tokens = line.split()
    if len(tokens) < 2 or tokens[0] != "info" or tokens[1] == "string":
        return None
    if "string" in tokens[2:]:
        # Truncate before parsing any subcommand after the free-text marker.
        # This also covers a malformed ``string`` token placed after a PV,
        # whose move list otherwise consumes the rest of the line.
        tokens = tokens[:tokens.index("string", 2)]
    values = {"tokens": tokens, "line": line}
    index = 1
    while index < len(tokens):
        key = tokens[index]
        if key == "score":
            if index + 2 >= len(tokens):
                return None
            kind, raw = tokens[index + 1:index + 3]
            if kind not in ("cp", "mate"):
                return None
            values["score_kind"] = kind
            values["score_raw"] = raw
            index += 3
            bound = "exact"
            if index < len(tokens) and tokens[index] in ("lowerbound", "upperbound"):
                bound = tokens[index]
                index += 1
            values["bound_stm"] = bound
            continue
        if key == "pv":
            values["pv"] = tokens[index + 1:]
            break
        if key == "string":
            # USI ``string`` consumes the rest of this info line.  A later
            # score/nodes/pv token is diagnostic free text, not another USI
            # subcommand.  Fields parsed before this token remain available.
            break
        if key in _INFO_INTEGER_FIELDS:
            if index + 1 >= len(tokens):
                return None
            try:
                values[key] = int(tokens[index + 1])
            except ValueError:
                return None
            index += 2
            continue
        if key in _INFO_SINGLE_VALUE_FIELDS:
            if index + 1 >= len(tokens):
                return None
            index += 2
            continue
        # These fields contain a variable-length move list.  They are valid
        # USI, but cannot be followed by a score in the grammar we need here.
        if key in ("refutation", "currline"):
            break
        # Unknown tokens are not treated as a reason to search the remainder
        # of the line for a substring that happens to look like a score.
        return None
    return values


def _observation_window(lines):
    """Return raw events after this attempt's go and through its bestmove."""
    values = list(lines)
    has_events = any(isinstance(item, dict) for item in values)
    if not has_events:
        window = []
        for item in values:
            window.append(item)
            if str(item).startswith("bestmove "):
                break
        return window
    go_indexes = [
        index for index, item in enumerate(values)
        if isinstance(item, dict)
        and item.get("direction") == "send"
        and re.fullmatch(r"go nodes [0-9]+", str(item.get("line", "")))
    ]
    if not go_indexes:
        return []
    window = []
    for item in values[go_indexes[-1] + 1:]:
        window.append(item)
        if (
            isinstance(item, dict)
            and item.get("direction") == "receive"
            and str(item.get("line", "")).startswith("bestmove ")
        ):
            break
    return window


def _primary_scored_infos(lines):
    infos = []
    for item in _observation_window(lines):
        if isinstance(item, dict):
            if item.get("direction") != "receive":
                continue
            line = item.get("line", "")
        else:
            line = str(item)
        if line.startswith("bestmove "):
            continue
        parsed = _parse_usi_info_line(line)
        if parsed is None or "score_kind" not in parsed or not parsed.get("pv"):
            continue
        tokens = parsed["tokens"]
        if "multipv" in tokens:
            index = tokens.index("multipv")
            if index + 1 >= len(tokens) or tokens[index + 1] != "1":
                continue
        infos.append({
            "line": line,
            "tokens": tokens,
            "kind": parsed["score_kind"],
            "raw": parsed["score_raw"],
            "bound_stm": parsed.get("bound_stm", "exact"),
            "pv": parsed["pv"],
            "nodes": parsed.get("nodes"),
            "time_ms": parsed.get("time"),
        })
    return infos


def _last_info_measurements(lines):
    last_nodes = None
    last_time = None
    for item in _observation_window(lines):
        if isinstance(item, dict):
            if item.get("direction") != "receive":
                continue
            line = item.get("line", "")
        else:
            line = str(item)
        parsed = _parse_usi_info_line(line)
        if parsed is None:
            continue
        nodes = parsed.get("nodes")
        engine_time = parsed.get("time")
        if nodes is not None:
            last_nodes = nodes
        if engine_time is not None:
            last_time = engine_time
    return last_nodes, last_time


def _bestmove_line(lines):
    for item in _observation_window(lines):
        if isinstance(item, dict):
            if item.get("direction") != "receive":
                continue
            line = item.get("line", "")
        else:
            line = str(item)
        if line.startswith("bestmove "):
            return line
    return None


def _event_offset(lines, predicate):
    for item in _observation_window(lines):
        if not isinstance(item, dict) or item.get("direction") != "receive":
            continue
        if predicate(item.get("line", "")):
            return item.get("offset_ns")
    return None


def _winner_for_mate(raw, side):
    # The sign is deliberately lexical: -0 is not collapsed into 0.
    if raw.startswith("+"):
        return side_name(side)
    if raw.startswith("-"):
        return side_name(opposite_side(side))
    if raw == "0":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value > 0:
        return side_name(side)
    if value < 0:
        return side_name(opposite_side(side))
    return None


def _winner_stm_for_mate(raw):
    if raw.startswith("+"):
        return "side_to_move"
    if raw.startswith("-"):
        return "opponent"
    try:
        value = int(raw)
    except ValueError:
        return None
    if value > 0:
        return "side_to_move"
    if value < 0:
        return "opponent"
    return None


def parse_usi_observation(
    lines,
    side_to_move,
    *,
    board=None,
    requested_nodes=None,
    max_reported_nodes=None,
    anomaly_ceiling=None,
    wall_go_to_bestmove_ns=None,
):
    """Parse the final primary USI score without converting mate to cp.

    ``lines`` may be plain strings (convenient for unit tests) or raw event
    dictionaries emitted by :class:`USIProcess`.  A bound is never replaced
    by an earlier exact score.
    """
    if side_to_move in ("black", 1, "+"):
        side = 1
    elif side_to_move in ("white", -1, "-"):
        side = -1
    else:
        raise ValueError(f"invalid side to move: {side_to_move!r}")
    best_line = _bestmove_line(lines)
    last_nodes, last_time = _last_info_measurements(lines)
    infos = _primary_scored_infos(lines)
    result = {
        "schema_version": 1,
        "status": "protocol_failure" if best_line is None else "no_score",
        "bestmove": None,
        "bestmove_kind": None,
        "pv": None,
        "pv_head": None,
        "score_kind": None,
        "score_cp_stm": None,
        "score_cp_sente": None,
        "reported_cp_stm": None,
        "reported_cp_sente": None,
        "score_bound_stm": None,
        "score_bound_sente": None,
        "score_mate_stm": None,
        "mate_distance": None,
        "mate_distance_known": None,
        "mate_sign": None,
        "winner": None,
        "winner_stm": None,
        "winner_sente": None,
        "reported_nodes_at_score": None,
        "last_reported_nodes": last_nodes,
        "engine_time_ms": last_time,
        "engine_time": last_time,
        "wall_go_to_bestmove_ns": wall_go_to_bestmove_ns,
        "wall_go_to_bestmove_seconds": None if wall_go_to_bestmove_ns is None else wall_go_to_bestmove_ns / 1_000_000_000,
        "raw_score_line": None,
        "raw_bestmove_line": best_line,
        "info_count": len(infos),
        "failure_reason": None,
    }
    if best_line is None:
        result["failure_reason"] = "missing_bestmove"
        return result
    tokens = best_line.split()
    if len(tokens) < 2 or not tokens[1]:
        result["status"] = "protocol_failure"
        result["failure_reason"] = "empty_bestmove"
        return result
    bestmove = tokens[1]
    result["bestmove"] = bestmove
    if bestmove == "resign":
        result["bestmove_kind"] = "resign"
    elif bestmove == "win":
        result["bestmove_kind"] = "win"
    elif _is_normal_bestmove(bestmove):
        result["bestmove_kind"] = "normal"
    else:
        result["bestmove_kind"] = "other_special"
    if not infos:
        result["failure_reason"] = "no_primary_scored_pv"
        return result
    info = infos[-1]
    result.update(
        pv=info["pv"],
        pv_head=info["pv"][0],
        score_kind=info["kind"],
        reported_nodes_at_score=info["nodes"],
        raw_score_line=info["line"],
        score_bound_stm=info["bound_stm"],
    )
    bound = info["bound_stm"]
    if bound == "exact":
        result["score_bound_sente"] = "exact"
    elif side == 1:
        result["score_bound_sente"] = bound
    else:
        result["score_bound_sente"] = "lowerbound" if bound == "upperbound" else "upperbound"
    if info["kind"] == "cp":
        try:
            value = int(info["raw"])
        except ValueError:
            result["status"] = "protocol_failure"
            result["failure_reason"] = "invalid_cp_token"
            return result
        result["reported_cp_stm"] = value
        result["reported_cp_sente"] = value if side == 1 else -value
        if bound == "exact":
            result["score_cp_stm"] = value
            result["score_cp_sente"] = value if side == 1 else -value
            result["status"] = "exact_cp"
        else:
            result["status"] = "bound_cp"
    else:
        raw = info["raw"]
        if not re.fullmatch(r"(?:[+-]?[0-9]+|[+-])", raw):
            result["status"] = "protocol_failure"
            result["failure_reason"] = "invalid_mate_token"
            return result
        result["score_mate_stm"] = raw
        if raw in ("+", "-"):
            result["mate_distance"] = None
            result["mate_distance_known"] = False
        else:
            result["mate_distance"] = abs(int(raw))
            result["mate_distance_known"] = True
        result["mate_sign"] = "+" if raw.startswith("+") else "-" if raw.startswith("-") else ""
        result["winner"] = _winner_for_mate(raw, side)
        result["winner_stm"] = _winner_stm_for_mate(raw)
        result["winner_sente"] = result["winner"]
        result["status"] = "mate"
    if result["bestmove_kind"] == "normal":
        if result["pv_head"] != bestmove:
            result["status"] = "protocol_failure"
            result["failure_reason"] = "pv_bestmove_mismatch"
        elif board is not None and not board.is_legal_usi(bestmove):
            result["status"] = "protocol_failure"
            result["failure_reason"] = "illegal_bestmove"
    elif result["bestmove_kind"] is not None:
        # A special bestmove is retained as a special outcome, not a numeric
        # success, even if an engine printed a stale score before it.
        result["status"] = "no_score"
        result["failure_reason"] = "special_bestmove"
    if (
        result["status"] in ("exact_cp", "bound_cp", "mate")
        and requested_nodes is not None
    ):
        if last_nodes is None or last_nodes <= 0:
            result["status"] = "node_budget_failure"
            result["failure_reason"] = "missing_or_nonpositive_nodes"
        else:
            # ``anomaly_ceiling`` is retained only for callers of the old
            # helper API.  Benchmark plans use the unambiguous absolute
            # ``max_reported_nodes`` field.
            if max_reported_nodes is None and anomaly_ceiling is not None:
                max_reported_nodes = math.ceil(requested_nodes * (1 + anomaly_ceiling))
            observed_nodes = max(
                last_nodes,
                result["reported_nodes_at_score"] or 0,
            )
            if max_reported_nodes is not None and observed_nodes > max_reported_nodes:
                result["status"] = "node_budget_failure"
                result["failure_reason"] = "node_count_above_ceiling"
    return result


def parse_result(lines, side, *, allow_bounds=False):
    """Compatibility parser used by the environment smoke tests."""
    result = parse_usi_observation(lines, side)
    if result["status"] == "protocol_failure":
        raise RuntimeError(result["failure_reason"] or "USI protocol failure")
    if result["status"] == "bound_cp" and not allow_bounds:
        raise RuntimeError("final scored PV is a bound")
    if result["status"] not in ("exact_cp", "bound_cp"):
        raise RuntimeError("expected finite cp score for smoke position")
    if result["bestmove_kind"] != "normal":
        raise RuntimeError("unexpected smoke bestmove")
    return {
        "nodes": result["last_reported_nodes"],
        "depth": 0,
        "bestmove": result["bestmove"],
        "reported_cp_stm": result["reported_cp_stm"],
        "score_bound_stm": result["score_bound_stm"],
        "score_bound_sente": result["score_bound_sente"],
        "score_cp_stm": result["score_cp_stm"],
        "score_cp_sente": result["score_cp_sente"],
    }


_SUPERVISOR_FLAG = "--_usi-supervisor"
_PR_SET_CHILD_SUBREAPER = 36
_SUPERVISOR_STATUS_SCHEMA = 1
_SUPERVISOR_CLEANUP_FAILURE_EXIT = 125
_SUPERVISOR_CLEANUP_DEADLINE_SECONDS = 1.0
_SUPERVISOR_TERM_GRACE_SECONDS = 0.20
_SUPERVISOR_QUIESCENCE_SECONDS = 0.05
_SUPERVISOR_POLL_SECONDS = 0.005
SUPERVISOR_RECORDED_STATUSES = {
    "ok",
    "startup_failure",
    "cleanup_failure",
    "not_started",
    "missing",
    "invalid",
    "pending",
}


class _SupervisorCleanupProofError(RuntimeError):
    """The supervisor lost proof needed for safe descendant cleanup."""


def _close_fd(fd):
    try:
        os.close(fd)
    except OSError:
        pass


def _supervisor_status_payload(status, failure=None, engine_returncode=None):
    return {
        "schema_version": _SUPERVISOR_STATUS_SCHEMA,
        "kind": "supervisor_outcome",
        "supervisor_status": status,
        "supervisor_failure": failure,
        "engine_returncode": engine_returncode,
    }


def _write_supervisor_status(fd, payload):
    if not isinstance(payload, dict):
        payload = _supervisor_status_payload("invalid", "supervisor status was not an object")
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8", "replace")
    try:
        while data:
            written = os.write(fd, data)
            data = data[written:]
    except OSError:
        pass


def _enable_child_subreaper():
    """Enable Linux subreaping for the runner-owned supervisor only."""
    if platform.system() != "Linux":
        return False, "the USI supervisor requires Linux child-subreaper support"
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        prctl.restype = ctypes.c_int
        if prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            error_number = ctypes.get_errno()
            return False, os.strerror(error_number)
    except (AttributeError, OSError) as error:
        return False, str(error)
    return True, None


def _forward_supervisor_fd(source_fd, sink_fd, done, shutdown, relay_stop=None):
    relay_stop = shutdown if relay_stop is None else relay_stop
    try:
        while True:
            if source_fd == 0:
                # stdin belongs to the runner and may stay open after the
                # engine has exited.  Polling lets the supervisor close this
                # relay as part of the bounded cleanup phase instead of
                # waiting for the runner's later pipe close.
                try:
                    readable, _, _ = select.select((source_fd,), (), (), 0.05)
                except (OSError, ValueError):
                    break
                if not readable:
                    if relay_stop.is_set():
                        break
                    continue
            try:
                data = os.read(source_fd, 65536)
            except InterruptedError:
                continue
            except OSError:
                if source_fd == 0:
                    shutdown.set()
                break
            if not data:
                if source_fd == 0:
                    relay_stop.set()
                break
            view = memoryview(data)
            while view:
                try:
                    written = os.write(sink_fd, view)
                except InterruptedError:
                    continue
                except (BrokenPipeError, OSError):
                    shutdown.set()
                    view = view[:0]
                    break
                view = view[written:]
    finally:
        done.set()
        _close_fd(source_fd)
        _close_fd(sink_fd)


def _supervisor_wait_status_code(status):
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1


def _supervisor_proc_stat(pid):
    """Read the identity fields needed before opening a child pidfd."""
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii")
    except (OSError, ValueError):
        return None
    closing = text.rfind(") ")
    if closing < 0:
        return None
    fields = text[closing + 2:].split()
    if len(fields) < 20:
        return None
    try:
        return {
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
            "starttime": int(fields[19]),
        }
    except ValueError:
        return None


def _supervisor_direct_children(supervisor_pid):
    """Enumerate direct waitable children, failing closed on bad evidence."""
    path = Path(f"/proc/{int(supervisor_pid)}/task/{int(supervisor_pid)}/children")
    try:
        text = path.read_text(encoding="ascii")
    except OSError as error:
        raise _SupervisorCleanupProofError(f"cannot enumerate supervisor children: {error}") from error
    children = set()
    for token in text.split():
        if not token.isdigit():
            raise _SupervisorCleanupProofError(
                f"supervisor children enumeration contained a non-PID token: {token!r}"
            )
        pid = int(token)
        if pid <= 0 or pid == int(supervisor_pid):
            raise _SupervisorCleanupProofError(
                f"supervisor children enumeration contained an invalid PID: {token!r}"
            )
        children.add(pid)
    return children


def _supervisor_pidfd_capable():
    return (
        platform.system() == "Linux"
        and callable(getattr(os, "pidfd_open", None))
        and callable(getattr(signal, "pidfd_send_signal", None))
        and callable(getattr(os, "waitid", None))
        and all(hasattr(os, name) for name in ("P_PID", "WEXITED", "WNOWAIT", "WNOHANG"))
    )


def _supervisor_read_exec_failure(fd):
    if fd is None:
        return None
    chunks = []
    try:
        os.set_blocking(fd, False)
        while True:
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                break
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        _close_fd(fd)
    if not chunks:
        return None
    return b"".join(chunks).decode("utf-8", "replace").strip() or "unknown exec error"


def _supervisor_signal_pidfd(record, signum):
    """Signal a still-owned identity only through its already-open pidfd."""
    current = _supervisor_proc_stat(record["pid"])
    if current is None or current["state"] == "Z":
        return
    if current["starttime"] != record["starttime"]:
        raise _SupervisorCleanupProofError(
            f"supervisor child identity changed before signal: {record['pid']}"
        )
    try:
        signal.pidfd_send_signal(record["pidfd"], signum, None, 0)
    except ProcessLookupError:
        # The pidfd is still the identity proof; the child may have exited in
        # the small interval between /proc inspection and the signal call.
        return
    except InterruptedError:
        try:
            signal.pidfd_send_signal(record["pidfd"], signum, None, 0)
        except ProcessLookupError:
            return
        except OSError as error:
            raise _SupervisorCleanupProofError(
                f"cannot signal child {record['pid']} through pidfd: {error}"
            ) from error
        except (AttributeError, TypeError) as error:
            raise _SupervisorCleanupProofError(
                f"cannot signal child {record['pid']} through pidfd: {error}"
            ) from error
    except (AttributeError, TypeError) as error:
        raise _SupervisorCleanupProofError(
            f"cannot signal child {record['pid']} through pidfd: {error}"
        ) from error
    except OSError as error:
        if error.errno == errno.ESRCH:  # Already exited.
            return
        raise _SupervisorCleanupProofError(
            f"cannot signal child {record['pid']} through pidfd: {error}"
        ) from error


def _supervisor_decode_waitid_status(wait_result):
    if wait_result.si_code == getattr(os, "CLD_EXITED", 1):
        return int(wait_result.si_status)
    return 128 + int(wait_result.si_status)


def _usi_supervisor_main(arguments):
    """Keep an isolated anchor while directly supervising and reaping the engine."""
    if len(arguments) != 2:
        return 64
    binary, raw_status_fd = arguments
    try:
        status_fd = int(raw_status_fd)
    except ValueError:
        return 64

    def finish(status, failure=None, engine_returncode=None, returncode=None):
        _write_supervisor_status(
            status_fd,
            _supervisor_status_payload(status, failure, engine_returncode),
        )
        _close_fd(status_fd)
        if returncode is not None:
            return returncode
        if status == "cleanup_failure":
            return _SUPERVISOR_CLEANUP_FAILURE_EXIT
        if status == "startup_failure":
            return 126
        return _supervisor_wait_status_code(engine_returncode if engine_returncode is not None else 1)

    try:
        os.set_inheritable(status_fd, True)
    except OSError as error:
        return finish("startup_failure", f"cannot prepare status channel: {error}")

    ready, reason = _enable_child_subreaper()
    if not ready:
        return finish(
            "cleanup_failure",
            f"cannot enable child subreaper: {reason}",
            returncode=_SUPERVISOR_CLEANUP_FAILURE_EXIT,
        )
    try:
        # The supervisor is the session leader and process-group leader.  It
        # is the stable ownership anchor; the runner never needs to discover
        # children by racing a /proc poll after the engine has exited.
        if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
            return finish(
                "cleanup_failure",
                "supervisor is not an isolated session leader",
                returncode=_SUPERVISOR_CLEANUP_FAILURE_EXIT,
            )
    except OSError as error:
        return finish(
            "cleanup_failure",
            f"cannot verify supervisor session: {error}",
            returncode=_SUPERVISOR_CLEANUP_FAILURE_EXIT,
        )

    if not _supervisor_pidfd_capable():
        return finish(
            "cleanup_failure",
            "required pidfd and waitid cleanup capabilities are unavailable",
            returncode=_SUPERVISOR_CLEANUP_FAILURE_EXIT,
        )
    try:
        _supervisor_direct_children(os.getpid())
    except _SupervisorCleanupProofError as error:
        return finish("cleanup_failure", str(error), returncode=_SUPERVISOR_CLEANUP_FAILURE_EXIT)

    try:
        engine_stdin_read, engine_stdin_write = os.pipe()
        engine_stdout_read, engine_stdout_write = os.pipe()
        exec_status_read, exec_status_write = os.pipe()
    except OSError as error:
        for fd in locals().get("engine_stdin_read", None), locals().get("engine_stdin_write", None), locals().get("engine_stdout_read", None), locals().get("engine_stdout_write", None):
            if fd is not None:
                _close_fd(fd)
        return finish("startup_failure", f"cannot create USI pipes: {error}")

    try:
        engine_pid = os.fork()
    except OSError as error:
        _close_fd(engine_stdin_read)
        _close_fd(engine_stdin_write)
        _close_fd(engine_stdout_read)
        _close_fd(engine_stdout_write)
        _close_fd(exec_status_read)
        _close_fd(exec_status_write)
        return finish("startup_failure", f"cannot fork engine: {error}")

    if engine_pid == 0:
        # No shell, no altered argv: the actual engine is a direct exec with
        # the one argument vector used by the benchmark.
        try:
            _close_fd(exec_status_read)
            os.dup2(engine_stdin_read, 0)
            os.dup2(engine_stdout_write, 1)
            os.dup2(engine_stdout_write, 2)
            os.set_inheritable(status_fd, False)
            for fd in (engine_stdin_read, engine_stdin_write, engine_stdout_read, engine_stdout_write, status_fd):
                if fd not in (0, 1, 2):
                    _close_fd(fd)
            os.execv(binary, [binary])
        except OSError as error:
            try:
                message = f"cannot exec {binary}: {error}".encode("utf-8", "replace")
                os.write(exec_status_write, message)
            except OSError:
                pass
            _close_fd(exec_status_write)
            os._exit(127)

    _close_fd(engine_stdin_read)
    _close_fd(engine_stdout_write)
    _close_fd(exec_status_write)
    shutdown = threading.Event()
    input_stop = threading.Event()
    input_done = threading.Event()
    output_done = threading.Event()

    def request_shutdown(_signum, _frame):
        shutdown.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGHUP, request_shutdown)
    input_thread = threading.Thread(
        target=_forward_supervisor_fd,
        args=(0, engine_stdin_write, input_done, shutdown, input_stop),
        daemon=True,
    )
    output_thread = threading.Thread(
        target=_forward_supervisor_fd,
        args=(engine_stdout_read, 1, output_done, shutdown),
        daemon=True,
    )
    input_thread.start()
    output_thread.start()

    engine_status = None
    engine_reaped = False
    engine_exec_failure = None
    tracked = {}
    cleanup_failure = None
    cleanup_started = None
    cleanup_deadline = None
    term_stage = False
    kill_stage = False
    quiescent_since = None

    def note_cleanup_failure(error):
        nonlocal cleanup_failure
        if cleanup_failure is None:
            cleanup_failure = str(error)
        shutdown.set()

    def track_direct_children():
        """Open pidfds before any wait can reap the enumerated children."""
        child_pids = _supervisor_direct_children(os.getpid())
        for pid in sorted(child_pids):
            record = tracked.get(pid)
            if record is not None:
                current = _supervisor_proc_stat(pid)
                if current is not None and current["starttime"] != record["starttime"]:
                    raise _SupervisorCleanupProofError(
                        f"supervisor child identity changed while tracked: {pid}"
                    )
                continue
            snapshot = _supervisor_proc_stat(pid)
            if snapshot is None or snapshot["ppid"] != os.getpid():
                raise _SupervisorCleanupProofError(
                    f"cannot prove direct child identity before pidfd open: {pid}"
                )
            try:
                pidfd = os.pidfd_open(pid, 0)
            except OSError as error:
                raise _SupervisorCleanupProofError(
                    f"cannot open pidfd for direct child {pid}: {error}"
                ) from error
            after = _supervisor_proc_stat(pid)
            if after is None or after["starttime"] != snapshot["starttime"]:
                _close_fd(pidfd)
                raise _SupervisorCleanupProofError(
                    f"direct child identity changed during pidfd acquisition: {pid}"
                )
            tracked[pid] = {
                "pid": pid,
                "pidfd": pidfd,
                "starttime": snapshot["starttime"],
                "term_sent": False,
                "kill_sent": False,
            }
        return child_pids

    def send_pending(signum, marker):
        for record in list(tracked.values()):
            if record[marker]:
                continue
            _supervisor_signal_pidfd(record, signum)
            record[marker] = True

    def observe_engine():
        nonlocal engine_status
        if engine_status is not None:
            return
        try:
            wait_result = os.waitid(
                os.P_PID,
                engine_pid,
                os.WEXITED | os.WNOWAIT | os.WNOHANG,
            )
        except ChildProcessError:
            # This is only valid after our own reaper has recorded the engine.
            if not engine_reaped:
                raise _SupervisorCleanupProofError("engine disappeared before supervisor could observe it")
            return
        except OSError as error:
            raise _SupervisorCleanupProofError(f"cannot observe engine exit with waitid: {error}") from error
        if wait_result is not None and getattr(wait_result, "si_pid", 0) == engine_pid:
            engine_status = _supervisor_decode_waitid_status(wait_result)

    def reap_available():
        nonlocal engine_status, engine_reaped
        got_echild = False
        while True:
            # Re-enumerate for every waitpid call, not merely once per outer
            # loop.  Reaping the engine can reparent a descendant between two
            # successive wait results; that descendant must have a pidfd
            # before the next wait is allowed to reap it.
            track_direct_children()
            try:
                child_pid, child_status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                got_echild = True
                break
            except InterruptedError:
                continue
            if child_pid == 0:
                break
            record = tracked.pop(child_pid, None)
            if record is None:
                raise _SupervisorCleanupProofError(
                    f"supervisor reaped an untracked child without pidfd proof: {child_pid}"
                )
            _close_fd(record["pidfd"])
            if child_pid == engine_pid:
                engine_status = _supervisor_wait_status_code(child_status)
                engine_reaped = True
        return got_echild

    try:
        while True:
            try:
                observe_engine()
                child_pids = track_direct_children()
            except _SupervisorCleanupProofError as error:
                note_cleanup_failure(error)
                child_pids = None

            if engine_status is not None:
                # Once the engine itself is known to have exited, no further
                # runner commands can be useful.  Stop only the input relay;
                # surviving descendants remain owned and visible until the
                # runner explicitly requests cleanup.
                input_stop.set()
            if shutdown.is_set() and cleanup_started is None:
                cleanup_started = time.monotonic()
                cleanup_deadline = cleanup_started + _SUPERVISOR_CLEANUP_DEADLINE_SECONDS
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
                _close_fd(engine_stdin_write)

            now = time.monotonic()
            if shutdown.is_set() and not term_stage:
                try:
                    send_pending(signal.SIGTERM, "term_sent")
                    term_stage = True
                except _SupervisorCleanupProofError as error:
                    note_cleanup_failure(error)
            if (
                shutdown.is_set()
                and term_stage
                and not kill_stage
                and cleanup_started is not None
                and now >= cleanup_started + _SUPERVISOR_TERM_GRACE_SECONDS
            ):
                try:
                    send_pending(signal.SIGKILL, "kill_sent")
                    kill_stage = True
                except _SupervisorCleanupProofError as error:
                    note_cleanup_failure(error)

            try:
                # A fresh enumeration is required immediately before every
                # reap batch.  A child that was reparented after engine exit
                # therefore has a pidfd before it can be waited on.
                child_pids = track_direct_children()
                got_echild = reap_available()
                child_pids = track_direct_children()
            except _SupervisorCleanupProofError as error:
                note_cleanup_failure(error)
                got_echild = False
                child_pids = None

            if engine_status is not None and engine_reaped and not tracked and child_pids == set() and got_echild:
                if quiescent_since is None:
                    quiescent_since = time.monotonic()
                elif time.monotonic() - quiescent_since >= _SUPERVISOR_QUIESCENCE_SECONDS:
                    if input_done.is_set() and output_done.is_set() and cleanup_failure is None:
                        break
            else:
                quiescent_since = None

            if cleanup_deadline is not None and time.monotonic() >= cleanup_deadline:
                if not kill_stage:
                    try:
                        send_pending(signal.SIGKILL, "kill_sent")
                        kill_stage = True
                    except _SupervisorCleanupProofError as error:
                        note_cleanup_failure(error)
                if (
                    engine_status is None
                    or not engine_reaped
                    or tracked
                    or child_pids not in (None, set())
                    or not input_done.is_set()
                    or not output_done.is_set()
                ):
                    note_cleanup_failure(
                        "supervisor cleanup deadline exceeded before all children and relays were proven closed"
                    )
                break
            time.sleep(_SUPERVISOR_POLL_SECONDS)
    finally:
        _close_fd(engine_stdin_write)
        _close_fd(engine_stdout_read)
        input_thread.join(timeout=0.2)
        output_thread.join(timeout=1)
        engine_exec_failure = _supervisor_read_exec_failure(exec_status_read)
        for record in tracked.values():
            _close_fd(record["pidfd"])

    if cleanup_failure is not None:
        supervisor_status = "cleanup_failure"
        supervisor_failure = cleanup_failure
    elif engine_exec_failure:
        supervisor_status = "startup_failure"
        supervisor_failure = engine_exec_failure
    else:
        supervisor_status = "ok"
        supervisor_failure = None
    return finish(supervisor_status, supervisor_failure, engine_status)


class USIProcess:
    """One fresh engine, anchored by a runner-owned supervisor."""

    def __init__(self, binary, cwd, timeout):
        self.binary = Path(binary).expanduser().resolve()
        self.cwd = Path(cwd)
        self.timeout = float(timeout)
        self.started_ns = time.monotonic_ns()
        self.events = []
        self._events_lock = threading.Lock()
        self.lines = queue.Queue()
        self._supervisor_status = None
        self._force_killed = False
        status_read, status_write = os.pipe()
        self._supervisor_status_read = status_read
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    _SUPERVISOR_FLAG,
                    str(self.binary),
                    str(status_write),
                ],
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
                env=dict(os.environ, RAYON_NUM_THREADS="1"),
                pass_fds=(status_write,),
            )
        except OSError as error:
            _close_fd(status_read)
            raise StartupError(f"cannot start {self.binary}: {error}") from error
        finally:
            _close_fd(status_write)
        self.leader_pid = self.process.pid
        leader = self._read_proc_stat(self.leader_pid)
        try:
            self.pgid = os.getpgid(self.leader_pid)
        except ProcessLookupError as error:
            try:
                self.process.kill()
                self.process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
            _close_fd(self._supervisor_status_read)
            self._supervisor_status_read = None
            raise StartupError(f"USI session leader disappeared during startup: {self.binary}") from error
        self._proc_identity_available = leader is not None
        if leader is not None:
            self.leader_starttime = leader["starttime"]
            self.session_id = leader["session"]
            if leader["pgrp"] != self.pgid or leader["session"] != self.session_id:
                try:
                    self.process.kill()
                    self.process.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                _close_fd(self._supervisor_status_read)
                self._supervisor_status_read = None
                raise StartupError("spawned USI process is not an isolated session")
        else:
            # Without a starttime there is no safe PID-reuse proof.  The
            # process may still be usable for parsing, but cleanup will fail
            # closed instead of sending to an unowned numeric PGID.
            self.leader_starttime = None
            self.session_id = self.pgid
        self._known_group_identities = {}
        self._identity_lock = threading.Lock()
        self._ownership_ambiguous = not self._proc_identity_available
        self._refresh_group_members()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    @staticmethod
    def _read_proc_stat(pid):
        try:
            text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii")
        except (OSError, ValueError):
            return None
        closing = text.rfind(") ")
        if closing < 0:
            return None
        fields = text[closing + 2:].split()
        if len(fields) < 20:
            return None
        try:
            return {
                "state": fields[0],
                "ppid": int(fields[1]),
                "pgrp": int(fields[2]),
                "session": int(fields[3]),
                # /proc field 22 is index 19 after the state field.
                "starttime": int(fields[19]),
            }
        except ValueError:
            return None

    def _refresh_group_members(self, *, remember=True):
        proc_root = Path("/proc")
        if not proc_root.is_dir():
            return None
        members = []
        try:
            entries = list(proc_root.iterdir())
        except OSError:
            return None
        for entry in entries:
            if not entry.name.isdigit():
                continue
            stat = self._read_proc_stat(int(entry.name))
            if stat is None or stat["pgrp"] != self.pgid or stat["session"] != self.session_id:
                continue
            identity = (int(entry.name), stat["starttime"])
            if remember:
                with self._identity_lock:
                    self._known_group_identities.setdefault(identity[0], identity[1])
            members.append((identity[0], stat))
        return members

    def _leader_is_owned_alive(self):
        leader = self._read_proc_stat(self.leader_pid)
        return bool(
            leader is not None
            and leader["state"] != "Z"
            and leader["starttime"] == self.leader_starttime
            and leader["pgrp"] == self.pgid
            and leader["session"] == self.session_id
        )

    def _owned_group_members(self):
        """Return our session members, or None when PID reuse is ambiguous."""
        if self._ownership_ambiguous or not self._proc_identity_available:
            return None
        # Do not remember newly observed identities after the original leader
        # is gone.  The supervisor remains the session anchor through child
        # cleanup, so a leader loss is an ownership failure rather than a
        # reason to adopt an unknown same-PGID process.
        members = self._refresh_group_members(remember=False)
        if members is None:
            return None
        leader = self._read_proc_stat(self.leader_pid)
        if (
            leader is not None
            and leader["state"] != "Z"
            and leader["starttime"] == self.leader_starttime
        ):
            if leader["pgrp"] != self.pgid or leader["session"] != self.session_id:
                self._ownership_ambiguous = True
                return None
            for pid, stat in members:
                with self._identity_lock:
                    self._known_group_identities.setdefault(pid, stat["starttime"])
            return members
        with self._identity_lock:
            known = dict(self._known_group_identities)
        # The original leader is gone or its PID was reused.  A complete
        # snapshot is safe only when every live member is an identity observed
        # before ownership was lost.  This permits a registered surviving
        # child, but never adopts an unknown same-PGID process.
        owned = []
        for item in members:
            pid, stat = item
            if stat["state"] == "Z":
                if known.get(pid) == stat["starttime"]:
                    owned.append(item)
                continue
            if known.get(pid) != stat["starttime"]:
                return None
            owned.append(item)
        return owned

    @staticmethod
    def _is_executable(stat, pid):
        if stat["state"] == "Z":
            return False
        try:
            exe = Path(f"/proc/{pid}/exe")
            return exe.exists() or exe.is_symlink()
        except OSError:
            return False

    def _executable_group_members(self):
        members = self._owned_group_members()
        if members is None:
            raise CleanupError("cannot verify ownership of the USI process group")
        return [item for item in members if self._is_executable(item[1], item[0])]

    def _signal_owned_members(self, members, signum):
        """Signal exact registered PID/starttime identities after leader exit."""
        for pid, snapshot in members:
            current = self._read_proc_stat(pid)
            if current is None or current["state"] == "Z":
                continue
            if (
                current["starttime"] != snapshot["starttime"]
                or current["pgrp"] != self.pgid
                or current["session"] != self.session_id
            ):
                raise CleanupError("owned USI child identity changed during cleanup")
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                continue

    def _wait_group_clear(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if not self._executable_group_members():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)

    def _record(self, direction, line):
        with self._events_lock:
            event = {
                "offset_ns": max(0, time.monotonic_ns() - self.started_ns),
                "direction": direction,
                "line": line,
            }
            self.events.append(event)
        return event

    def _read(self):
        try:
            for raw in self.process.stdout:
                line = raw[:-2] if raw.endswith("\r\n") else raw[:-1] if raw.endswith(("\n", "\r")) else raw
                event = self._record("receive", line)
                self.lines.put(event)
        finally:
            self.lines.put(None)

    def _read_supervisor_status(self, *, block):
        fd = self._supervisor_status_read
        if fd is None:
            return
        try:
            os.set_blocking(fd, block)
            chunks = []
            while True:
                try:
                    chunk = os.read(fd, 4096)
                except BlockingIOError:
                    break
                except OSError:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks).decode("utf-8", "replace")
            lines = [line for line in raw.splitlines() if line.strip()]
            if len(lines) != 1:
                self._supervisor_status = {
                    "schema_version": _SUPERVISOR_STATUS_SCHEMA,
                    "kind": "supervisor_outcome",
                    "supervisor_status": "missing" if not lines else "invalid",
                    "supervisor_failure": "supervisor did not provide exactly one terminal status",
                    "engine_returncode": None,
                }
            else:
                try:
                    status = json.loads(lines[0])
                except json.JSONDecodeError:
                    status = None
                if (
                    not isinstance(status, dict)
                    or status.get("schema_version") != _SUPERVISOR_STATUS_SCHEMA
                    or status.get("kind") != "supervisor_outcome"
                    or status.get("supervisor_status") not in {"ok", "startup_failure", "cleanup_failure"}
                    or (
                        status.get("engine_returncode") is not None
                        and (
                            isinstance(status.get("engine_returncode"), bool)
                            or not isinstance(status.get("engine_returncode"), int)
                        )
                    )
                    or (
                        status.get("supervisor_failure") is not None
                        and not isinstance(status.get("supervisor_failure"), str)
                    )
                    or set(status) != {
                        "schema_version",
                        "kind",
                        "supervisor_status",
                        "supervisor_failure",
                        "engine_returncode",
                    }
                ):
                    self._supervisor_status = {
                        "schema_version": _SUPERVISOR_STATUS_SCHEMA,
                        "kind": "supervisor_outcome",
                        "supervisor_status": "invalid",
                        "supervisor_failure": "supervisor terminal status was invalid",
                        "engine_returncode": None,
                    }
                else:
                    self._supervisor_status = status
        finally:
            _close_fd(fd)
            self._supervisor_status_read = None

    def send(self, line):
        self._record("send", line)
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise ProcessExitError(f"USI stdin closed while sending {line!r}") from error

    def until(self, predicate, timeout=None):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ObservationTimeout("USI deadline exceeded")
            try:
                event = self.lines.get(timeout=remaining)
            except queue.Empty as error:
                raise ObservationTimeout("USI deadline exceeded") from error
            if event is None:
                raise ProcessExitError("USI engine exited before completing the command")
            line = event["line"]
            if predicate(line):
                return event

    def terminate_group(self):
        # Do not use process.poll() as a proxy for the group: the session
        # leader may have exited while a child still owns stdout.
        members = self._executable_group_members()
        if not members:
            try:
                self.process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
            return
        if self._leader_is_owned_alive():
            try:
                os.killpg(self.pgid, signal.SIGTERM)
            except ProcessLookupError:
                return
        else:
            self._signal_owned_members(members, signal.SIGTERM)
        try:
            self.process.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            pass
        if self._wait_group_clear(0.75):
            return
        members = self._executable_group_members()
        if not members:
            return
        if self._leader_is_owned_alive():
            try:
                self._force_killed = True
                os.killpg(self.pgid, signal.SIGKILL)
            except ProcessLookupError:
                return
        else:
            self._signal_owned_members(members, signal.SIGKILL)
        try:
            self.process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        if not self._wait_group_clear(3):
            raise CleanupError("could not kill USI process group")
        if self._force_killed:
            raise CleanupError("USI supervisor was force-killed before cleanup status was proven")

    def close(self, *, normal=False):
        error = None
        try:
            if normal and self.process.poll() is None:
                try:
                    self.send("quit")
                    self.process.wait(timeout=5)
                except ProcessExitError as caught:
                    error = caught
                except subprocess.TimeoutExpired:
                    error = CleanupError("USI session leader did not exit after quit")
            try:
                # Always inspect and clean the full group, including after the
                # parent has already exited.  This is also what closes inherited
                # stdout handles held by a surviving child.
                self.terminate_group()
            except ProcessExitError as caught:
                # An ownership/cleanup ambiguity must never be downgraded to
                # an earlier quit/write failure: callers use CleanupError to
                # abort subsequent benchmark attempts safely.
                if error is None or isinstance(caught, CleanupError):
                    error = caught
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired as caught:
                if error is None or not isinstance(error, CleanupError):
                    error = CleanupError("USI session leader did not exit")
            try:
                self._read_supervisor_status(block=self.process.poll() is not None)
            except OSError:
                self._supervisor_status = {
                    "schema_version": _SUPERVISOR_STATUS_SCHEMA,
                    "kind": "supervisor_outcome",
                    "supervisor_status": "invalid",
                    "supervisor_failure": "could not read supervisor terminal status",
                    "engine_returncode": None,
                }
            if not isinstance(self._supervisor_status, dict):
                error = CleanupError("USI supervisor produced no terminal cleanup status")
            elif self._supervisor_status.get("supervisor_status") == "cleanup_failure":
                error = CleanupError(
                    self._supervisor_status.get("supervisor_failure")
                    or "USI supervisor reported cleanup failure"
                )
            elif self._supervisor_status.get("supervisor_status") not in ("ok", "startup_failure"):
                error = CleanupError(
                    self._supervisor_status.get("supervisor_failure")
                    or "USI supervisor terminal status was not proven"
                )
            if self._force_killed:
                error = CleanupError("USI supervisor was force-killed before cleanup status was proven")
        finally:
            self.reader.join(timeout=3)
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
            if self.process.stdout is not None:
                try:
                    self.process.stdout.close()
                except OSError:
                    pass
        if error is not None:
            raise type(error)(str(error)) from error
        if (
            self.process.returncode not in (0, None)
            and isinstance(self._supervisor_status, dict)
            and self._supervisor_status.get("supervisor_status") != "startup_failure"
        ):
            raise ProcessExitError(f"USI engine exited with {self.process.returncode}")


def _engine_error(line):
    lowered = line.casefold()
    if "error!" in lowered:
        return True
    if any(marker in lowered for marker in ("unknown option", "no such option", "file not found", "unknown command")):
        return True
    return bool(re.search(r"(?:failed|cannot|could not|error!).*(?:load|open|weight|evaldir)", lowered))


def _engine_error_is_configuration(line):
    lowered = line.casefold()
    return any(marker in lowered for marker in ("unknown option", "no such option", "file not found", "unknown command", "evaldir", "weight"))


def _wait_for(process, predicate, phase):
    start_index = len(process.events)
    while True:
        event = process.until(predicate)
        for seen in process.events[start_index:]:
            if seen["direction"] == "receive" and _engine_error(seen["line"]):
                if _engine_error_is_configuration(seen["line"]):
                    raise ConfigurationError(f"engine reported configuration error during {phase}: {seen['line']}")
                if phase in ("usi", "isready"):
                    raise StartupError(f"engine reported startup error during {phase}: {seen['line']}")
                raise ProtocolError(f"engine reported protocol error during {phase}: {seen['line']}")
            if (
                seen["direction"] == "receive"
                and seen["line"].startswith("bestmove ")
                and not predicate(seen["line"])
            ):
                raise ProtocolError(f"unexpected bestmove during {phase}: {seen['line']}")
        return event


def _write_raw_log(path, events):
    data = "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events)
    atomic_write(path, data)


OUTCOME_FIELDS = (
    "status",
    "failure_reason",
    "outcome_phase",
    "completed_before_deadline",
    "cleanup_status",
    "cleanup_failure",
    "original_status",
    "original_failure_reason",
    "original_outcome_phase",
    "engine_returncode",
    "supervisor_status",
    "supervisor_failure",
)


def _outcome_payload(result):
    return {field: result.get(field) for field in OUTCOME_FIELDS}


def _runner_event(result, events=()):
    """Return the one authoritative non-engine runner outcome event."""
    last_offset = max(
        (event.get("offset_ns", -1) for event in events if isinstance(event, dict)),
        default=-1,
    )
    outcome = _outcome_payload(result)
    event = {
        "event": "runner",
        "kind": "outcome",
        "direction": "runner",
        "line": "",
        "offset_ns": last_offset + 1,
    }
    event.update(outcome)
    return event


def _mark_cleanup_failure(result, error):
    """Retain the original result while making unsafe cleanup terminal."""
    original = dict(result or {})
    marked = dict(original)
    marked.update(
        status="cleanup_failure",
        failure_reason=f"cleanup failed: {error}",
        outcome_phase="cleanup",
        cleanup_status="failure",
        cleanup_failure=str(error),
        original_status=original.get("status"),
        original_failure_reason=original.get("failure_reason"),
        original_outcome_phase=original.get("outcome_phase"),
    )
    return marked


def run_engine_attempt(
    binary,
    position,
    side_to_move,
    options,
    *,
    cwd,
    requested_nodes,
    timeout_seconds,
    anomaly_ceiling=None,
    max_reported_nodes=None,
    board=None,
    raw_log_path=None,
):
    """Run one engine lifecycle and return an observation plus raw events."""
    process = None
    result = None
    phase = "popen"
    completed_before_deadline = False
    normal_close = False
    cleanup_status = "not_required"
    cleanup_failure = None
    engine_returncode = None
    supervisor_status = "not_started"
    supervisor_failure = None
    try:
        process = USIProcess(binary, cwd, timeout_seconds)
        cleanup_status = "pending"
        phase = "usi"
        process.send("usi")
        _wait_for(process, lambda line: line == "usiok", "usi")
        advertised = _extract_option_names(
            [event["line"] for event in process.events if event["direction"] == "receive"]
        )
        missing = sorted(set(options) - advertised)
        if missing:
            raise ConfigurationError(f"engine does not advertise configured options: {missing}")
        phase = "setoption"
        for key, value in options.items():
            process.send(f"setoption name {key} value {value}")
        phase = "isready"
        process.send("isready")
        _wait_for(process, lambda line: line == "readyok", "isready")
        phase = "usinewgame"
        process.send("usinewgame")
        phase = "position"
        process.send(position)
        phase = "go"
        process.send(f"go nodes {requested_nodes}")
        _wait_for(process, lambda line: line.startswith("bestmove "), "go")
        completed_before_deadline = True
        go_events = [
            event for event in process.events
            if event["direction"] == "send" and event["line"].startswith("go nodes ")
        ]
        best_offset = _event_offset(process.events, lambda line: line.startswith("bestmove "))
        go_offset = go_events[-1]["offset_ns"] if go_events else None
        elapsed_ns = None if best_offset is None or go_offset is None else max(0, best_offset - go_offset)
        result = parse_usi_observation(
            process.events,
            side_to_move,
            board=board,
            requested_nodes=requested_nodes,
            max_reported_nodes=max_reported_nodes,
            anomaly_ceiling=anomaly_ceiling,
            wall_go_to_bestmove_ns=elapsed_ns,
        )
        normal_close = True
        phase = "complete"
    except ObservationTimeout as error:
        result = {
            "schema_version": 1,
            "status": "timeout",
            "failure_reason": str(error),
            "wall_go_to_bestmove_ns": None,
        }
    except ConfigurationError as error:
        result = {
            "schema_version": 1,
            "status": "config_failure",
            "failure_reason": str(error),
        }
    except StartupError as error:
        result = {"schema_version": 1, "status": "startup_failure", "failure_reason": str(error)}
        supervisor_failure = str(error)
    except ProcessExitError as error:
        result = {"schema_version": 1, "status": "exit_failure", "failure_reason": str(error)}
    except ProtocolError as error:
        result = {"schema_version": 1, "status": "protocol_failure", "failure_reason": str(error)}
    except BenchmarkError as error:
        result = {"schema_version": 1, "status": "protocol_failure", "failure_reason": str(error)}
    finally:
        if process is not None:
            try:
                process.close(normal=normal_close)
                cleanup_status = "ok"
            except CleanupError as error:
                cleanup_status = "failure"
                cleanup_failure = str(error)
            except ProcessExitError as error:
                cleanup_status = "ok"
                if result is None or result.get("status") in OBSERVATION_STATUSES:
                    result = {
                        "schema_version": 1,
                        "status": "exit_failure",
                        "failure_reason": str(error),
                    }
            supervisor_status = process._supervisor_status
            if isinstance(supervisor_status, dict):
                engine_returncode = supervisor_status.get("engine_returncode")
                supervisor_failure = supervisor_status.get("supervisor_failure")
                terminal_status = supervisor_status.get("supervisor_status")
                if terminal_status == "cleanup_failure" and cleanup_failure is None:
                    cleanup_status = "failure"
                    cleanup_failure = supervisor_failure or "supervisor reported cleanup failure"
                elif terminal_status in ("missing", "invalid", "pending") and cleanup_failure is None:
                    cleanup_status = "failure"
                    cleanup_failure = supervisor_failure or "supervisor cleanup status was not proven"
                supervisor_status = terminal_status
            else:
                supervisor_status = "missing"
                if cleanup_failure is None:
                    cleanup_status = "failure"
                    cleanup_failure = "supervisor did not provide a terminal status"
            if (
                cleanup_failure is None
                and supervisor_status == "startup_failure"
                and (result is None or result.get("status") == "exit_failure")
            ):
                result = {
                    "schema_version": 1,
                    "status": "startup_failure",
                    "failure_reason": supervisor_failure or "USI supervisor startup failed",
                }
        if result is None:
            result = {"schema_version": 1, "status": "exit_failure", "failure_reason": "no result"}
        result.setdefault("schema_version", 1)
        result.setdefault("outcome_phase", phase)
        result.setdefault("completed_before_deadline", completed_before_deadline)
        result.setdefault("cleanup_status", cleanup_status)
        result.setdefault("cleanup_failure", cleanup_failure)
        result.setdefault("original_status", None)
        result.setdefault("original_failure_reason", None)
        result.setdefault("original_outcome_phase", None)
        result.setdefault("engine_returncode", engine_returncode)
        result.setdefault("supervisor_status", supervisor_status)
        result.setdefault("supervisor_failure", supervisor_failure)
        if cleanup_failure is not None:
            result = _mark_cleanup_failure(result, cleanup_failure)
        elif result.get("cleanup_status") == "pending":
            result["cleanup_status"] = cleanup_status
        result["binary_identity"] = binary_identity(binary)
        result["requested_nodes"] = requested_nodes
        result["options"] = dict(options)
        result["binary"] = str(Path(binary).expanduser().resolve())
        if raw_log_path is not None:
            events = [] if process is None else list(process.events)
            events.append(_runner_event(result, events))
            _write_raw_log(raw_log_path, events)
    result["raw_log_sha256"] = sha256(raw_log_path) if raw_log_path is not None and Path(raw_log_path).exists() else None
    return result


@contextmanager
def nonblocking_lock(path, exclusive):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(stream, mode | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BenchmarkError(f"lock is busy: {path}") from error
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _git_capture(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise BenchmarkError(f"cannot inspect repository: {error}") from error


def repo_identity():
    return {
        "commit": _git_capture("rev-parse", "HEAD"),
        "dirty": bool(_git_capture("status", "--porcelain=v1", "--untracked-files=all")),
        "status_porcelain": _git_capture("status", "--porcelain=v1", "--untracked-files=all"),
    }


def _model_identity(config):
    model = config.get("candidate_model", {"kind": "material_fallback", "path": None})
    kind = model.get("kind")
    if kind == "material_fallback":
        return {"kind": kind, "path": None, "sha256": None, "bytes": None}
    path_value = model.get("path")
    if not path_value:
        raise ConfigurationError("candidate model path is required for a non-fallback model")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"candidate model is missing: {path}")
    return {"kind": kind, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def runtime_identity(runtime, config):
    runtime = Path(runtime)
    build_path = runtime / "build-manifest.json"
    build = json.loads(build_path.read_text()) if build_path.exists() else {"missing": str(build_path)}
    binaries = {}
    for engine in config.get("engines", {}).values():
        name = engine["binary"]
        path = runtime / "bin" / name
        binaries[name] = binary_identity(path)
    teacher = config["engines"]["teacher"]
    weight = None
    if teacher.get("weight_required", True):
        try:
            weight_path = verify_weight(runtime)
            weight = {"path": str(weight_path), "sha256": sha256(weight_path), "bytes": weight_path.stat().st_size}
        except (OSError, RuntimeError) as error:
            raise ConfigurationError(f"primary teacher weight is not verified: {error}") from error
    return {
        "build_manifest": build,
        "binaries": binaries,
        "teacher_weight": weight,
        "candidate_model": _model_identity(config),
    }


def binary_identity(path):
    """Return the stable identity recorded for one executable."""
    path = Path(path).expanduser().resolve()
    try:
        stat = path.stat()
        exists = path.is_file()
    except OSError:
        stat = None
        exists = False
    return {
        "path": str(path),
        "exists": exists,
        "sha256": sha256(path) if exists else None,
        "bytes": stat.st_size if exists and stat is not None else None,
    }


def validate_runtime(runtime, config):
    """Require the prepared runtime to match the pinned toolchain exactly."""
    runtime = Path(runtime)
    manifest_path = runtime / "build-manifest.json"
    if not manifest_path.is_file():
        raise ConfigurationError(f"build manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"build manifest is invalid: {manifest_path}") from error
    expected_lock = sha256(REPO / "config/toolchain.lock.json")
    if manifest.get("lock_sha256") != expected_lock:
        raise ConfigurationError("build manifest does not match the current toolchain lock")
    for engine in config.get("engines", {}).values():
        name = engine["binary"]
        path = runtime / "bin" / name
        expected = manifest.get("binaries", {}).get(name, {}).get("sha256")
        if not path.is_file() or not expected or sha256(path) != expected:
            raise ConfigurationError(f"prepared binary is missing or hash-mismatched: {name}")
    runtime_identity(runtime, config)
    return manifest


def host_identity():
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_model or platform.processor(),
        "python": sys.version,
        "rayon_num_threads": "1",
    }


def execution_identity(config, plan, runtime):
    """Return the pilot/formal identity shared by one observation.

    Run type, sampled/full position set, repetitions, and the post-pilot
    absolute ceiling intentionally do not belong here.  Everything that can
    change a USI observation or the interpretation of its raw evidence does.
    In particular, this identity deliberately excludes the mutable config
    digest and repository dirty/commit state so freezing pilot evidence and
    documenting that freeze do not invalidate the evidence.
    """
    runtime_info = runtime_identity(runtime, config)
    resolved_options = _runtime_options(config, runtime)
    resolved_option_order = {
        engine_id: list(options)
        for engine_id, options in resolved_options.items()
    }
    runner_hash = sha256(Path(__file__))
    return {
        "schema_version": 1,
        "runner": {
            "path": "scripts/benchmark.py",
            "sha256": runner_hash,
        },
        "parser": {
            "id": "usi-observation-v1",
            "sha256": runner_hash,
        },
        "benchmark": {
            "benchmark_id": config["benchmark_id"],
            "split": "development",
            "manifest_sha256": plan["hashes"]["benchmark_manifest_sha256"],
            "development_csa_aggregate_sha256": plan["hashes"]["development_csa_aggregate_sha256"],
            "universe_sha256": plan["hashes"]["universe_sha256"],
        },
        "requested_nodes": config["requested_nodes"],
        "timeout_seconds": config["timeout_seconds"],
        "environment": {
            "jobs": 1,
            "threads": 1,
            "rayon_num_threads": "1",
        },
        "engines": config["engines"],
        "resolved_options": resolved_options,
        "resolved_option_order": resolved_option_order,
        "runtime": runtime_info,
        "toolchain_lock_sha256": sha256(REPO / "config/toolchain.lock.json"),
        "teacher_weight": runtime_info.get("teacher_weight"),
        "candidate_model": runtime_info.get("candidate_model"),
        "host": host_identity(),
    }


def fingerprint_payload(config, plan, runtime, run_type):
    runtime_info = runtime_identity(runtime, config)
    resolved_options = _runtime_options(config, runtime)
    return {
        "schema_version": 1,
        "runner": {
            "path": "scripts/benchmark.py",
            "sha256": sha256(Path(__file__)),
        },
        "benchmark": {
            "manifest_sha256": sha256(BENCHMARK_MANIFEST_PATH),
            "development_csa_aggregate_sha256": plan["hashes"]["development_csa_aggregate_sha256"],
            "universe_sha256": plan["hashes"]["universe_sha256"],
            "plan_sha256": plan["plan_sha256"],
        },
        "toolchain_lock_sha256": sha256(REPO / "config/toolchain.lock.json"),
        "execution_identity": execution_identity(config, plan, runtime),
        "runtime": runtime_info,
        "engines": config["engines"],
        "resolved_options": resolved_options,
        "resolved_option_order": {
            engine_id: list(options)
            for engine_id, options in resolved_options.items()
        },
        "limits": {
            "requested_nodes": config["requested_nodes"],
            "timeout_seconds": config["timeout_seconds"],
            "jobs": 1,
            "repetitions": plan["repetitions"],
            "run_type": run_type,
            "max_reported_nodes": plan.get("max_reported_nodes"),
        },
        "host": host_identity(),
    }


def _run_id(run_type):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{run_type}-{stamp}-{os.getpid()}"


def _validate_run_id(value):
    if not isinstance(value, str) or not RUN_ID_RE.fullmatch(value) or value in (".", ".."):
        raise ValueError("run id must contain only letters, numbers, dot, underscore, and hyphen")


def _attempt_id(engine_id, occurrence, repetition):
    return f"{engine_id}--{occurrence['game_id']}--p{occurrence['ply']:03d}--r{repetition:02d}"


def _attempt_path(run_dir, attempt_id):
    return run_dir / "attempts" / f"{attempt_id}.json"


def _log_path(run_dir, attempt_id):
    return run_dir / "logs" / f"{attempt_id}.jsonl"


def _read_raw_event_log(path):
    """Read and minimally validate one immutable raw event log."""
    events = []
    previous_offset = -1
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BenchmarkError(f"cannot read raw event log: {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise BenchmarkError(f"invalid raw event JSON at {path}:{line_number}") from error
        if not isinstance(event, dict):
            raise BenchmarkError(f"raw event is not an object: {path}:{line_number}")
        offset = event.get("offset_ns")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < previous_offset:
            raise BenchmarkError(f"raw event offsets are not monotonic: {path}:{line_number}")
        previous_offset = offset
        if event.get("event") == "runner":
            if (
                event.get("kind") != "outcome"
                or event.get("direction") != "runner"
                or event.get("line") != ""
                or not isinstance(event.get("status"), str)
                or not isinstance(event.get("outcome_phase"), str)
                or not isinstance(event.get("completed_before_deadline"), bool)
                or event.get("cleanup_status") not in ("not_required", "pending", "ok", "failure")
                or event.get("supervisor_status") not in SUPERVISOR_RECORDED_STATUSES
                or (
                    event.get("engine_returncode") is not None
                    and (
                        isinstance(event.get("engine_returncode"), bool)
                        or not isinstance(event.get("engine_returncode"), int)
                    )
                )
                or (
                    event.get("supervisor_failure") is not None
                    and not isinstance(event.get("supervisor_failure"), str)
                )
                or set(event) != {
                    "event",
                    "kind",
                    "direction",
                    "line",
                    "offset_ns",
                    *OUTCOME_FIELDS,
                }
            ):
                raise BenchmarkError(f"invalid runner event: {path}:{line_number}")
            events.append(event)
            continue
        if (
            event.get("direction") not in ("send", "receive")
            or not isinstance(event.get("line"), str)
            or set(event) != {"direction", "line", "offset_ns"}
        ):
            raise BenchmarkError(f"invalid engine event: {path}:{line_number}")
        events.append(event)
    runners = [event for event in events if event.get("event") == "runner"]
    if len(runners) != 1 or not events or events[-1].get("event") != "runner":
        raise BenchmarkError(f"raw event log must end with exactly one runner outcome: {path}")
    return events


def _runner_outcome(events):
    outcomes = [event for event in events if event.get("event") == "runner"]
    if len(outcomes) != 1:
        raise BenchmarkError("raw event log must contain exactly one runner outcome")
    return outcomes[0]


def _expected_options_from_manifest(manifest, config):
    payload = manifest.get("fingerprint_payload")
    resolved = payload.get("resolved_options") if isinstance(payload, dict) else None
    if not isinstance(resolved, dict):
        raise BenchmarkError("run fingerprint lacks resolved options")
    if not all(isinstance(resolved.get(engine), dict) for engine in config["engines"]):
        raise BenchmarkError("run fingerprint lacks resolved options for every engine")
    orders = payload.get("resolved_option_order", {}) if isinstance(payload, dict) else {}
    if not isinstance(orders, dict) or not all(isinstance(orders.get(engine), list) for engine in config["engines"]):
        raise BenchmarkError("run fingerprint lacks resolved option order for every engine")
    if any(set(orders[engine]) != set(resolved[engine]) for engine in config["engines"]):
        raise BenchmarkError("run fingerprint resolved option order does not match its options")
    return {
        engine: {name: resolved[engine][name] for name in orders[engine]}
        for engine in config["engines"]
    }


def _expected_send_lines(expected, expected_options, requested_nodes):
    lines = ["usi"]
    lines.extend(
        f"setoption name {key} value {value}"
        for key, value in expected_options.items()
    )
    lines.extend(
        (
            "isready",
            "usinewgame",
            expected["position"],
            f"go nodes {requested_nodes}",
        )
    )
    return lines


def _validate_engine_lifecycle(
    events,
    expected,
    expected_options,
    requested_nodes,
    status,
    completed_before_deadline,
):
    """Validate command ordering and the matching USI handshake evidence."""
    engine_events = [event for event in events if event.get("event") != "runner"]
    send_lines = [event["line"] for event in engine_events if event.get("direction") == "send"]
    base = _expected_send_lines(expected, expected_options, requested_nodes)
    full_statuses = OBSERVATION_STATUSES | {"protocol_failure", "node_budget_failure"}
    completed = bool(completed_before_deadline)
    full = completed and status in full_statuses
    allowed_sequences = (base + ["quit"], base)
    if full:
        if send_lines != base + ["quit"]:
            raise BenchmarkError(f"raw USI send lifecycle mismatch for {expected['attempt_id']}")
    elif not any(send_lines == candidate[:len(send_lines)] for candidate in allowed_sequences):
        raise BenchmarkError(f"raw USI send prefix mismatch for {expected['attempt_id']}")

    if not full:
        return

    usiok = [
        index for index, event in enumerate(engine_events)
        if event.get("direction") == "receive" and event["line"] == "usiok"
    ]
    if len(usiok) != 1:
        raise BenchmarkError(f"raw USI lifecycle must contain one usiok: {expected['attempt_id']}")
    advertised = _extract_option_names(
        [
            event["line"] for event in engine_events[:usiok[0]]
            if event.get("direction") == "receive"
        ]
    )
    missing = sorted(set(expected_options) - advertised)
    if missing:
        raise BenchmarkError(
            f"raw USI option advertisement is missing {missing}: {expected['attempt_id']}"
        )
    readyok = [
        index for index, event in enumerate(engine_events)
        if event.get("direction") == "receive" and event["line"] == "readyok"
    ]
    if len(readyok) != 1:
        raise BenchmarkError(f"raw USI lifecycle must contain one readyok: {expected['attempt_id']}")
    usi_send_index = next(
        index for index, event in enumerate(engine_events)
        if event.get("direction") == "send" and event["line"] == "usi"
    )
    setoption_indices = [
        index for index, event in enumerate(engine_events)
        if event.get("direction") == "send" and event["line"].startswith("setoption name ")
    ]
    isready_index = next(
        index for index, event in enumerate(engine_events)
        if event.get("direction") == "send" and event["line"] == "isready"
    )
    usinewgame_index = next(
        index for index, event in enumerate(engine_events)
        if event.get("direction") == "send" and event["line"] == "usinewgame"
    )
    position_index = next(
        index for index, event in enumerate(engine_events)
        if event.get("direction") == "send" and event["line"] == expected["position"]
    )
    go_index = next(
        index for index, event in enumerate(engine_events)
        if event.get("direction") == "send" and event["line"] == f"go nodes {requested_nodes}"
    )
    if usi_send_index >= usiok[0]:
        raise BenchmarkError(f"raw usiok is not after usi: {expected['attempt_id']}")
    if not all(usiok[0] < index < isready_index for index in setoption_indices):
        raise BenchmarkError(
            f"raw setoption is not after usiok and before isready: {expected['attempt_id']}"
        )
    if not (usiok[0] < isready_index < readyok[0] < usinewgame_index < position_index < go_index):
        raise BenchmarkError(f"raw readyok is before isready: {expected['attempt_id']}")
    bestmoves = [
        index for index, event in enumerate(engine_events)
        if event.get("direction") == "receive" and event["line"].startswith("bestmove ")
    ]
    if len(bestmoves) != 1:
        raise BenchmarkError(f"raw USI lifecycle must contain one bestmove: {expected['attempt_id']}")
    quit_index = next(
        index for index, event in enumerate(engine_events)
        if event.get("direction") == "send" and event["line"] == "quit"
    )
    if bestmoves[0] <= go_index or bestmoves[0] >= quit_index:
        raise BenchmarkError(f"raw bestmove appeared before the observation: {expected['attempt_id']}")


def _compare_raw_observation(record, observed, runner_status):
    saved = record["result"]
    saved_status = record["status"]
    observed_status = observed.get("status")
    parser_statuses = OBSERVATION_STATUSES | {"protocol_failure", "node_budget_failure"}
    if saved_status in parser_statuses and observed_status != saved_status:
        raise BenchmarkError(
            f"raw observation status mismatch for {record['attempt_id']}: "
            f"saved {saved_status}, reparsed {observed_status}"
        )
    if runner_status != saved_status:
        raise BenchmarkError(f"runner status mismatch for {record['attempt_id']}")
    for field in OBSERVATION_COMPARE_FIELDS:
        if field in ("status", "failure_reason"):
            continue
        # Runner-generated technical failures intentionally contain only the
        # fields available before the process failed.  There is no semantic
        # value to compare for a field they did not save.
        if field not in saved:
            continue
        if saved.get(field) != observed.get(field):
            raise BenchmarkError(f"raw observation {field} mismatch for {record['attempt_id']}")
    if saved_status in ("protocol_failure", "node_budget_failure"):
        if saved.get("failure_reason") != observed.get("failure_reason"):
            raise BenchmarkError(f"raw observation failure reason mismatch for {record['attempt_id']}")


def _go_to_bestmove_ns(events):
    go_offsets = [
        event["offset_ns"] for event in events
        if event.get("direction") == "send"
        and re.fullmatch(r"go nodes [0-9]+", event.get("line", ""))
    ]
    best_offset = _event_offset(events, lambda line: line.startswith("bestmove "))
    if not go_offsets or best_offset is None:
        return None
    return max(0, best_offset - go_offsets[-1])


def _validate_recorded_attempt(
    path,
    run_fingerprint,
    expected=None,
    *,
    expected_options=None,
    expected_binary_identity=None,
    requested_nodes=None,
    max_reported_nodes=None,
    board=None,
):
    path = Path(path)
    if expected is not None and path.name != f"{expected['attempt_id']}.json":
        raise BenchmarkError(f"attempt filename does not match attempt_id: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"corrupt attempt result: {path}: {error}") from error
    if record.get("schema_version") != 1 or record.get("run_fingerprint") != run_fingerprint:
        raise BenchmarkError(f"attempt fingerprint/schema mismatch: {path}")
    if not isinstance(record.get("result"), dict):
        raise BenchmarkError(f"attempt result is not an object: {path}")
    if record.get("status") != record["result"].get("status"):
        raise BenchmarkError(f"attempt top-level/nested status mismatch: {path}")
    if record.get("status") not in (*OBSERVATION_STATUSES, *TECHNICAL_FAILURE_STATUSES):
        raise BenchmarkError(f"attempt has an unknown status: {path}")
    position = record.get("position")
    if not isinstance(position, str) or record.get("position_sha256") != digest_bytes(position.encode("utf-8")):
        raise BenchmarkError(f"attempt position hash mismatch: {path}")
    if expected is not None:
        for field in ("ply", "repetition"):
            if isinstance(record.get(field), bool) or not isinstance(record.get(field), int):
                raise BenchmarkError(f"attempt {field} has an invalid type: {path}")
        for key in ("attempt_id", "engine_id", "game_id", "ply", "repetition", "position", "side_to_move"):
            if record.get(key) != expected.get(key):
                raise BenchmarkError(f"attempt identity/position mismatch: {path}")
        if record.get("attempt_id") != path.stem:
            raise BenchmarkError(f"attempt filename does not match attempt_id: {path}")
    result = record["result"]
    if not isinstance(record.get("outcome"), dict) or set(record["outcome"]) != set(OUTCOME_FIELDS):
        raise BenchmarkError(f"attempt outcome is missing: {path}")
    if record["outcome"].get("status") != record.get("status"):
        raise BenchmarkError(f"attempt top-level/outcome status mismatch: {path}")
    if requested_nodes is not None and (
        isinstance(result.get("requested_nodes"), bool)
        or not isinstance(result.get("requested_nodes"), int)
        or result.get("requested_nodes") != requested_nodes
    ):
        raise BenchmarkError(f"attempt requested_nodes mismatch: {path}")
    if expected_options is not None and result.get("options") != expected_options:
        raise BenchmarkError(f"attempt configured options mismatch: {path}")
    if expected_binary_identity is not None:
        if result.get("binary_identity") != expected_binary_identity:
            raise BenchmarkError(f"attempt binary identity mismatch: {path}")
        if result.get("binary") != expected_binary_identity.get("path"):
            raise BenchmarkError(f"attempt binary path mismatch: {path}")
    raw_log = record.get("raw_log")
    expected_log = f"{record.get('attempt_id')}.jsonl"
    if (
        not isinstance(raw_log, str)
        or Path(raw_log).name != raw_log
        or raw_log in ("", ".", "..")
        or raw_log != expected_log
    ):
        raise BenchmarkError(f"attempt raw log path is unsafe: {path}")
    log = path.parent.parent / "logs" / raw_log
    if not log.is_file() or record.get("raw_log_sha256") != sha256(log):
        raise BenchmarkError(f"attempt raw log hash mismatch: {path}")
    if result.get("raw_log_sha256") != record.get("raw_log_sha256"):
        raise BenchmarkError(f"attempt nested raw log hash mismatch: {path}")
    events = _read_raw_event_log(log)
    runner = _runner_outcome(events)
    for field in OUTCOME_FIELDS:
        if runner.get(field) != result.get(field) or runner.get(field) != record["outcome"].get(field):
            raise BenchmarkError(f"attempt outcome does not match runner event ({field}): {path}")
    if expected is not None:
        if result.get("completed_before_deadline") is not True and record.get("status") in OBSERVATION_STATUSES:
            raise BenchmarkError(f"successful attempt was not completed before deadline: {path}")
        _validate_engine_lifecycle(
            events,
            expected,
            expected_options or {},
            requested_nodes,
            record["status"],
            runner["completed_before_deadline"],
        )
        should_parse = record["status"] in OBSERVATION_STATUSES or (
            runner["completed_before_deadline"]
            and record["status"] in ("protocol_failure", "node_budget_failure")
        )
        if should_parse:
            observed = parse_usi_observation(
                events,
                expected["side_to_move"],
                board=board,
                requested_nodes=requested_nodes,
                max_reported_nodes=max_reported_nodes,
                wall_go_to_bestmove_ns=_go_to_bestmove_ns(events),
            )
            _compare_raw_observation(record, observed, runner["status"])
        elif record["status"] in TECHNICAL_FAILURE_STATUSES and not runner["completed_before_deadline"]:
            semantic_fields = (
                "score_kind",
                "score_cp_stm",
                "score_cp_sente",
                "reported_cp_stm",
                "reported_cp_sente",
                "score_bound_stm",
                "score_bound_sente",
                "score_mate_stm",
                "mate_distance",
                "winner",
                "winner_stm",
                "winner_sente",
                "reported_nodes_at_score",
                "raw_score_line",
            )
            if any(result.get(field) is not None for field in semantic_fields):
                raise BenchmarkError(f"raw observation technical outcome promoted a late observation: {path}")
    return record


def expected_attempt_matrix(plan, config):
    """Build the exact attempt identity set for a plan."""
    expected = {}
    positions = plan.get("positions")
    if not isinstance(positions, list):
        raise BenchmarkError("run plan positions must be a list")
    for occurrence in positions:
        if not isinstance(occurrence, dict):
            raise BenchmarkError("run plan contains a non-object occurrence")
        for field in ("game_id", "ply", "position", "side_to_move"):
            if field not in occurrence:
                raise BenchmarkError(f"run plan occurrence lacks {field}")
        for repetition in range(1, plan["repetitions"] + 1):
            for engine_id in config["engines"]:
                attempt_id = _attempt_id(engine_id, occurrence, repetition)
                if attempt_id in expected:
                    raise BenchmarkError(f"run plan has duplicate attempt identity: {attempt_id}")
                expected[attempt_id] = {
                    "attempt_id": attempt_id,
                    "engine_id": engine_id,
                    "game_id": occurrence["game_id"],
                    "ply": occurrence["ply"],
                    "repetition": repetition,
                    "position": occurrence["position"],
                    "side_to_move": occurrence["side_to_move"],
                }
    return expected


def _board_map_from_games(games):
    return {
        (game["game_id"], index): board
        for game in games
        for index, board in enumerate(game["boards_after"], 1)
    }


def validate_run_artifacts(
    run_dir,
    config=None,
    *,
    require_complete=False,
    runtime_options=None,
    expected_run_type=None,
):
    """Validate the fixed development run and every recorded attempt.

    This is the single strict path shared by resume, local report, and public
    export.  ``require_complete=False`` is only for resuming a running run.
    """
    config = config or load_config()
    run_dir = Path(run_dir).expanduser().resolve()
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"cannot read run manifest/plan: {run_dir}: {error}") from error
    if manifest.get("schema_version") != 1:
        raise BenchmarkError("run manifest has an unsupported schema")
    if plan.get("schema_version") != 1:
        raise BenchmarkError("run plan has an unsupported schema")
    run_type = manifest.get("run_type")
    if expected_run_type is not None and run_type != expected_run_type:
        raise BenchmarkError("run type does not match the expected artifact")
    if run_type not in ("pilot", "formal"):
        raise BenchmarkError("run manifest has an invalid run type")
    try:
        _validate_run_id(manifest.get("run_id", ""))
    except ValueError as error:
        raise BenchmarkError("run manifest run_id is unsafe") from error
    if manifest.get("run_id") != run_dir.name or "final" in manifest["run_id"].casefold():
        raise BenchmarkError("run manifest identifies a non-development or mismatched run")
    policy = validate_run_policy(config, run_type)
    if manifest.get("benchmark_id") != config.get("benchmark_id") or manifest.get("split") != "development":
        raise BenchmarkError("run manifest is not the fixed development benchmark")
    if plan.get("benchmark_id") != config.get("benchmark_id") or plan.get("split") != "development":
        raise BenchmarkError("run plan is not the fixed development benchmark")
    expected_plan, games = make_plan(run_type, config)
    if plan_hash(plan) != plan.get("plan_sha256"):
        raise BenchmarkError("run plan hash mismatch")
    expected_semantics = {
        key: value for key, value in expected_plan.items()
        if key not in ("plan_sha256", "cshogi")
    }
    actual_semantics = {
        key: value for key, value in plan.items()
        if key not in ("plan_sha256", "cshogi")
    }
    if actual_semantics != expected_semantics:
        raise BenchmarkError("run plan does not match the current fixed development plan")
    if manifest.get("plan_sha256") != plan.get("plan_sha256"):
        raise BenchmarkError("run manifest plan hash mismatch")
    attempt_matrix = manifest.get("attempt_matrix")
    expected_matrix = {
        "engine_ids": list(config["engines"]),
        "position_count": len(plan.get("positions", [])),
        "repetitions": plan.get("repetitions"),
        "expected_attempts": len(plan.get("positions", [])) * plan.get("repetitions", 0) * len(config["engines"]),
    }
    if attempt_matrix != expected_matrix:
        raise BenchmarkError("run manifest attempt matrix does not match the current development plan")
    if not isinstance(manifest.get("fingerprint"), str) or not manifest.get("fingerprint"):
        raise BenchmarkError("run manifest fingerprint is missing")
    fingerprint_payload = manifest.get("fingerprint_payload")
    if not isinstance(fingerprint_payload, dict):
        raise BenchmarkError("run manifest fingerprint payload is missing")
    if json_sha256(fingerprint_payload) != manifest.get("fingerprint"):
        raise BenchmarkError("run manifest fingerprint payload hash mismatch")
    runner_identity = fingerprint_payload.get("runner")
    if not isinstance(runner_identity, dict) or runner_identity.get("path") != "scripts/benchmark.py":
        raise BenchmarkError("run manifest runner identity is invalid")
    if runner_identity.get("sha256") != sha256(Path(__file__)):
        raise BenchmarkError("run manifest was produced by a different benchmark runner")
    execution_payload = fingerprint_payload.get("execution_identity")
    if not isinstance(execution_payload, dict):
        raise BenchmarkError("run manifest execution identity is missing")
    parser_identity = execution_payload.get("parser")
    if (
        not isinstance(parser_identity, dict)
        or parser_identity.get("id") != "usi-observation-v1"
        or parser_identity.get("sha256") != sha256(Path(__file__))
    ):
        raise BenchmarkError("run manifest parser identity is invalid")
    if manifest.get("runtime_identity") != execution_payload.get("runtime"):
        raise BenchmarkError("run manifest runtime identity does not match its fingerprint")
    if (
        execution_payload.get("runner") != runner_identity
        or execution_payload.get("engines") != fingerprint_payload.get("engines")
        or execution_payload.get("resolved_options") != fingerprint_payload.get("resolved_options")
        or execution_payload.get("resolved_option_order") != fingerprint_payload.get("resolved_option_order")
        or execution_payload.get("toolchain_lock_sha256") != fingerprint_payload.get("toolchain_lock_sha256")
    ):
        raise BenchmarkError("run manifest execution identity is inconsistent with its fingerprint payload")
    # The formal pilot gate is deliberately frozen by editing this config
    # after the pilot.  Plan/hash comparisons below bind all scoring inputs;
    # mutable config/repository provenance is kept outside the fingerprint so
    # the freeze and its documentation do not invalidate the pilot evidence.
    if fingerprint_payload.get("toolchain_lock_sha256") != sha256(REPO / "config/toolchain.lock.json"):
        raise BenchmarkError("run manifest toolchain lock does not match the current repository")
    fingerprint_benchmark = fingerprint_payload.get("benchmark")
    expected_hashes = expected_plan["hashes"]
    benchmark_hash_pairs = {
        "manifest_sha256": expected_hashes["benchmark_manifest_sha256"],
        "development_csa_aggregate_sha256": expected_hashes["development_csa_aggregate_sha256"],
        "universe_sha256": expected_hashes["universe_sha256"],
        "plan_sha256": expected_hashes.get("plan_sha256", plan["plan_sha256"]),
    }
    if not isinstance(fingerprint_benchmark, dict) or any(
        fingerprint_benchmark.get(key) != value
        for key, value in benchmark_hash_pairs.items()
    ):
        raise BenchmarkError("run manifest benchmark hashes do not match the current development benchmark")
    if execution_payload.get("benchmark") != {
        "benchmark_id": config["benchmark_id"],
        "split": "development",
        "manifest_sha256": expected_hashes["benchmark_manifest_sha256"],
        "development_csa_aggregate_sha256": expected_hashes["development_csa_aggregate_sha256"],
        "universe_sha256": expected_hashes["universe_sha256"],
    }:
        raise BenchmarkError("run manifest execution identity benchmark is not the fixed development benchmark")
    limits = fingerprint_payload.get("limits")
    if not isinstance(limits, dict) or any(
        limits.get(key) != value
        for key, value in {
            "requested_nodes": expected_plan["requested_nodes"],
            "repetitions": expected_plan["repetitions"],
            "run_type": run_type,
            "max_reported_nodes": expected_plan.get("max_reported_nodes"),
        }.items()
    ):
        raise BenchmarkError("run manifest limits do not match the current development plan")
    if (
        execution_payload.get("requested_nodes") != expected_plan["requested_nodes"]
        or execution_payload.get("timeout_seconds") != config["timeout_seconds"]
        or execution_payload.get("environment") != {"jobs": 1, "threads": 1, "rayon_num_threads": "1"}
    ):
        raise BenchmarkError("run manifest execution limits/environment are inconsistent")
    universe_path = run_dir / "universe.json"
    if not universe_path.is_file():
        raise BenchmarkError("run canonical universe is missing")
    try:
        universe = json.loads(universe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError("run canonical universe is invalid") from error
    expected_universe, _ = build_universe(config)
    if universe != expected_universe or universe_hash(universe) != expected_plan["hashes"]["universe_sha256"]:
        raise BenchmarkError("run universe does not match the current fixed development universe")
    expected = expected_attempt_matrix(plan, config)
    attempts_root = run_dir / "attempts"
    logs_root = run_dir / "logs"
    if not attempts_root.is_dir() or not logs_root.is_dir():
        raise BenchmarkError("run attempts/logs directory is missing")
    actual_paths = sorted(attempts_root.iterdir(), key=lambda path: path.name)
    actual_names = {path.name for path in actual_paths}
    expected_names = {f"{attempt_id}.json" for attempt_id in expected}
    unexpected_names = sorted(actual_names - expected_names)
    if unexpected_names:
        raise BenchmarkError(f"run has extra or renamed attempt files: {unexpected_names}")
    log_paths = sorted(logs_root.iterdir(), key=lambda path: path.name)
    expected_log_names = {f"{path.stem}.jsonl" for path in actual_paths if path.is_file()}
    unexpected_logs = sorted(
        path.name for path in log_paths
        if not path.is_file() or path.name not in expected_log_names
    )
    if unexpected_logs:
        raise BenchmarkError(f"run has extra or renamed raw logs: {unexpected_logs}")
    if require_complete and manifest.get("status") != "complete":
        raise BenchmarkError("refusing to report an incomplete run")
    if manifest.get("status") not in ("running", "incomplete", "complete"):
        raise BenchmarkError("run manifest has an invalid status")
    if manifest.get("status") == "complete" and actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise BenchmarkError(f"complete run attempt set mismatch: missing={missing}, extra={extra}")
    manifest_options = _expected_options_from_manifest(manifest, config)
    expected_options = runtime_options if runtime_options is not None else manifest_options
    runtime_binaries = manifest.get("runtime_identity", {}).get("binaries", {})
    board_map = _board_map_from_games(games)
    records = []
    for path in actual_paths:
        if not path.is_file() or path.name not in expected_names:
            raise BenchmarkError(f"invalid attempt file: {path}")
        attempt_id = path.stem
        occurrence = expected[attempt_id]
        board = board_map.get((occurrence["game_id"], occurrence["ply"]))
        fingerprint_engines = fingerprint_payload.get("engines", {})
        binary_name = (
            fingerprint_engines.get(occurrence["engine_id"], {}).get("binary")
            if isinstance(fingerprint_engines, dict)
            else None
        )
        if not isinstance(binary_name, str):
            binary_name = config["engines"][occurrence["engine_id"]]["binary"]
        expected_binary = runtime_binaries.get(binary_name)
        if not isinstance(expected_binary, dict):
            raise BenchmarkError(f"run manifest lacks binary identity for {binary_name}")
        records.append(
            _validate_recorded_attempt(
                path,
                manifest["fingerprint"],
                occurrence,
                expected_options=expected_options.get(occurrence["engine_id"]),
                expected_binary_identity=expected_binary,
                requested_nodes=plan["requested_nodes"],
                max_reported_nodes=policy["max_reported_nodes"],
                board=board.clone() if board is not None else None,
            )
        )
    if manifest.get("status") == "complete":
        total = len(expected)
        if manifest.get("completed_attempts") != total or manifest.get("total_attempts") != total:
            raise BenchmarkError("complete run counters do not match the exact attempt matrix")
    elif "completed_attempts" in manifest and manifest.get("completed_attempts") != len(records):
        raise BenchmarkError("running run completed_attempts counter does not match evidence")
    return manifest, plan, universe, records, expected, games


def observed_max_reported_nodes(records):
    """Return the maximum of both node-evidence fields across all attempts."""
    values = []
    for record in records:
        result = record.get("result", {})
        for key in ("reported_nodes_at_score", "last_reported_nodes"):
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                values.append(value)
    return max(values) if values else None


def validate_formal_gate(runtime, config):
    """Validate the pilot artifact and frozen absolute formal node gate."""
    policy = validate_run_policy(config, "formal", require_formal_gate=True)
    gate = config["formal"].get("pilot_evidence")
    required = ("pilot_run_id", "pilot_fingerprint", "observed_max_reported_nodes", "max_reported_nodes")
    if any(key not in gate for key in required):
        raise ConfigurationError(f"formal node ceiling gate: formal.pilot_evidence must contain {required}")
    pilot_run_id = gate["pilot_run_id"]
    if not isinstance(pilot_run_id, str):
        raise ConfigurationError("formal.pilot_evidence.pilot_run_id must be a string")
    try:
        _validate_run_id(pilot_run_id)
    except ValueError as error:
        raise ConfigurationError("formal.pilot_evidence.pilot_run_id is unsafe") from error
    if not isinstance(gate["pilot_fingerprint"], str) or not gate["pilot_fingerprint"]:
        raise ConfigurationError("formal.pilot_evidence.pilot_fingerprint must be a non-empty string")
    observed_claim = gate["observed_max_reported_nodes"]
    chosen_claim = gate["max_reported_nodes"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (observed_claim, chosen_claim)):
        raise ConfigurationError("formal.pilot_evidence node values must be positive integers")
    if policy["max_reported_nodes"] != chosen_claim:
        raise ConfigurationError("formal.max_reported_nodes does not match frozen pilot evidence")
    if chosen_claim < observed_claim:
        raise ConfigurationError(
            "formal node gate must satisfy max_reported_nodes >= observed_max_reported_nodes; "
            "observed evidence may be below requested_nodes because go nodes is an upper limit"
        )
    pilot_dir = Path(runtime).expanduser().resolve() / "runs" / pilot_run_id
    if not pilot_dir.is_dir():
        raise ConfigurationError(f"frozen pilot artifact is missing from this runtime: {pilot_dir}")
    manifest, plan, _universe, records, _expected, _games = validate_run_artifacts(
        pilot_dir,
        config,
        require_complete=True,
        expected_run_type="pilot",
    )
    expected_total = len(plan["positions"]) * plan["repetitions"] * len(config["engines"])
    if expected_total != 90 or len(records) != 90:
        raise ConfigurationError("formal gate pilot artifact must contain exactly 90 attempts")
    if manifest.get("fingerprint") != gate["pilot_fingerprint"]:
        raise ConfigurationError("formal gate pilot fingerprint does not match the artifact")
    formal_plan, _ = make_plan("formal", config)
    pilot_execution_identity = manifest.get("fingerprint_payload", {}).get("execution_identity")
    current_execution_identity = execution_identity(config, formal_plan, runtime)
    if pilot_execution_identity != current_execution_identity:
        raise ConfigurationError("formal gate execution identity does not match the current formal environment")
    observed_actual = observed_max_reported_nodes(records)
    if observed_actual is None:
        raise ConfigurationError(
            "formal gate requires node evidence in at least one valid pilot result; "
            "all-valid-but-no-node-evidence is rejected closed"
        )
    if observed_actual != observed_claim:
        raise ConfigurationError(
            f"formal gate observed maximum mismatch: frozen {observed_claim}, actual {observed_actual}"
        )
    return gate


def _runtime_options(config, runtime):
    options = {}
    for engine_id, engine in config["engines"].items():
        options[engine_id] = dict(engine.get("options", {}))
    teacher = options["teacher"]
    if config["engines"]["teacher"].get("weight_required", True):
        weight = verify_weight(runtime)
        teacher["EvalDir"] = str(weight.parent)
    return options


def _board_map(games):
    return {
        (game["game_id"], index): board
        for game in games
        for index, board in enumerate(game["boards_after"], 1)
    }


def _execute_run_locked(runtime, config, run_type, *, run_id=None, resume=False):
    runtime = Path(runtime).expanduser().resolve()
    validate_run_policy(config, run_type)
    if run_type == "formal":
        validate_formal_gate(runtime, config)
    if resume:
        if not run_id:
            raise ValueError("resume requires --run-id")
        _validate_run_id(run_id)
        run_dir = runtime / "runs" / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise BenchmarkError(f"run manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_type") != run_type:
            raise BenchmarkError("resume run type does not match manifest")
        plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        validate_runtime(runtime, config)
        current_payload = fingerprint_payload(config, plan, runtime, run_type)
        current_fingerprint = json_sha256(current_payload)
        if manifest.get("fingerprint") != current_fingerprint:
            raise BenchmarkError("run fingerprint mismatch; refusing to overwrite or resume")
        options = _runtime_options(config, runtime)
        manifest, plan, _universe, _records, _expected, games = validate_run_artifacts(
            run_dir,
            config,
            require_complete=False,
            runtime_options=options,
            expected_run_type=run_type,
        )
        if any(record.get("status") == "cleanup_failure" for record in _records):
            raise BenchmarkError("previous cleanup failure makes this run unsafe to resume")
    else:
        if run_id is None:
            run_id = _run_id(run_type)
        _validate_run_id(run_id)
        run_dir = runtime / "runs" / run_id
        if any((run_dir / name).exists() for name in ("manifest.json", "plan.json", "attempts", "logs")):
            raise BenchmarkError(f"run already exists; use --resume: {run_dir}")
        plan, games = make_plan(run_type, config)
        plan["cshogi"] = compare_cshogi(games)
        validate_runtime(runtime, config)
        runtime_info = runtime_identity(runtime, config)
        payload = fingerprint_payload(config, plan, runtime, run_type)
        current_fingerprint = json_sha256(payload)
        expected_total = len(plan["positions"]) * plan["repetitions"] * len(config["engines"])
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "run_type": run_type,
            "benchmark_id": config["benchmark_id"],
            "split": "development",
            "created_at": utc_now(),
            "provenance": {
                "config_sha256": sha256(CONFIG_PATH),
                "repo": repo_identity(),
            },
            "fingerprint": current_fingerprint,
            "fingerprint_payload": payload,
            "plan_sha256": plan["plan_sha256"],
            "runtime_identity": runtime_info,
            "attempt_matrix": {
                "engine_ids": list(config["engines"]),
                "position_count": len(plan["positions"]),
                "repetitions": plan["repetitions"],
                "expected_attempts": expected_total,
            },
            "status": "running",
        }
        (run_dir / "attempts").mkdir(parents=True, exist_ok=False)
        (run_dir / "logs").mkdir(parents=True, exist_ok=False)
        universe, _ = build_universe(config)
        atomic_write_json(run_dir / "universe.json", universe)
        atomic_write_json(run_dir / "plan.json", plan)
        atomic_write_json(run_dir / "manifest.json", manifest)
    if not resume:
        options = _runtime_options(config, runtime)
    board_map = _board_map(games)
    completed = 0
    total = len(plan["positions"]) * plan["repetitions"] * len(config["engines"])
    for occurrence in plan["positions"]:
        for repetition in range(1, plan["repetitions"] + 1):
            for engine_id, engine in config["engines"].items():
                attempt_id = _attempt_id(engine_id, occurrence, repetition)
                attempt_path = _attempt_path(run_dir, attempt_id)
                expected_attempt = {
                    "attempt_id": attempt_id,
                    "engine_id": engine_id,
                    "game_id": occurrence["game_id"],
                    "ply": occurrence["ply"],
                    "repetition": repetition,
                    "position": occurrence["position"],
                    "side_to_move": occurrence["side_to_move"],
                }
                expected_binary = manifest["runtime_identity"]["binaries"][engine["binary"]]
                if attempt_path.exists():
                    _validate_recorded_attempt(
                        attempt_path,
                        manifest["fingerprint"],
                        expected_attempt,
                        expected_options=options[engine_id],
                        expected_binary_identity=expected_binary,
                        requested_nodes=plan["requested_nodes"],
                        max_reported_nodes=plan.get("max_reported_nodes"),
                        board=board_map[(occurrence["game_id"], occurrence["ply"])].clone(),
                    )
                    if json.loads(attempt_path.read_text(encoding="utf-8")).get("status") == "cleanup_failure":
                        raise BenchmarkError("previous cleanup failure makes this run unsafe to resume")
                    completed += 1
                    continue
                board = board_map[(occurrence["game_id"], occurrence["ply"])].clone()
                result = run_engine_attempt(
                    runtime / "bin" / engine["binary"],
                    occurrence["position"],
                    occurrence["side_to_move"],
                    options[engine_id],
                    cwd=run_dir,
                    requested_nodes=plan["requested_nodes"],
                    timeout_seconds=config["timeout_seconds"],
                    max_reported_nodes=plan.get("max_reported_nodes"),
                    board=board,
                    raw_log_path=_log_path(run_dir, attempt_id),
                )
                record = {
                    "schema_version": 1,
                    "run_fingerprint": manifest["fingerprint"],
                    "attempt_id": attempt_id,
                    "engine_id": engine_id,
                    "game_id": occurrence["game_id"],
                    "ply": occurrence["ply"],
                    "repetition": repetition,
                    "position": occurrence["position"],
                    "side_to_move": occurrence["side_to_move"],
                    "position_sha256": digest_bytes(occurrence["position"].encode("utf-8")),
                    "status": result["status"],
                    "outcome": _outcome_payload(result),
                    "result": result,
                    "raw_log": f"{attempt_id}.jsonl",
                    "raw_log_sha256": result.get("raw_log_sha256"),
                }
                atomic_write_json(attempt_path, record)
                completed += 1
                print(f"[{completed}/{total}] {attempt_id}: {result['status']}", flush=True)
                if result["status"] == "cleanup_failure":
                    raise BenchmarkError(
                        f"cleanup failure aborted benchmark after {attempt_id}: "
                        f"{result.get('failure_reason')}"
                    )
    manifest["status"] = "complete"
    manifest["completed_attempts"] = completed
    manifest["total_attempts"] = total
    atomic_write_json(run_dir / "manifest.json", manifest)
    validate_run_artifacts(
        run_dir,
        config,
        require_complete=True,
        runtime_options=options,
        expected_run_type=run_type,
    )
    return manifest


def execute_run(runtime, config, run_type, *, run_id=None, resume=False):
    validate_run_policy(config, run_type)
    runtime = Path(runtime).expanduser().resolve()
    if run_type == "formal":
        # Check before creating a run directory so a refused formal launch is
        # side-effect free.  _execute_run_locked repeats this under the locks.
        validate_formal_gate(runtime, config)
    runtime.mkdir(parents=True, exist_ok=True)
    if not resume and run_id is None:
        run_id = _run_id(run_type)
    # The prepare shared lock prevents a build/import from changing binaries or
    # weights while the benchmark's exclusive run lock prevents another run.
    with nonblocking_lock(runtime / ".prepare.lock", exclusive=False):
        with nonblocking_lock(runtime / ".benchmark.lock", exclusive=True):
            if resume:
                if not run_id:
                    raise ValueError("resume requires --run-id")
                run_dir = runtime / "runs" / run_id
                if not (run_dir.is_dir() and (run_dir / "manifest.json").is_file()):
                    raise BenchmarkError(f"cannot resume an incomplete or missing run: {run_dir}")
                run_lock_path = runtime / "runs" / run_id / ".run.lock"
            else:
                run_dir = runtime / "runs" / run_id
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                if run_dir.exists():
                    raise BenchmarkError(f"run already exists; use --resume: {run_dir}")
                run_dir.mkdir()
                run_lock_path = runtime / "runs" / run_id / ".run.lock"
            with nonblocking_lock(run_lock_path, exclusive=True):
                return _execute_run_locked(runtime, config, run_type, run_id=run_id, resume=resume)


def compare_cshogi(games):
    """Compare every converted move with pinned cshogi when importable."""
    try:
        import importlib
        import importlib.metadata
        cshogi = importlib.import_module("cshogi")
        version = importlib.metadata.version("cshogi")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return {"available": False, "reason": "cshogi is not installed", "checked_moves": 0}
    if version != "1.0.4":
        return {"available": False, "reason": f"cshogi version is {version}, expected 1.0.4", "checked_moves": 0}
    checked = 0
    for game in games:
        board = cshogi.Board(STANDARD_SFEN)
        for move in game["usi_moves"]:
            try:
                move_id = board.move_from_usi(move)
                if not board.is_legal(move_id):
                    raise ValueError("cshogi says converted move is illegal")
                board.push(move_id)
            except Exception as error:
                raise BenchmarkError(f"cshogi conversion mismatch at {game['game_id']} ply {checked + 1}: {error}") from error
            checked += 1
    return {"available": True, "version": version, "checked_moves": checked, "matched": True}


def status(runtime):
    root = Path(runtime).expanduser().resolve() / "runs"
    rows = []
    if not root.is_dir():
        return rows
    for path in sorted(root.iterdir()):
        if not path.is_dir() or "final" in path.parts:
            continue
        manifest = path / "manifest.json"
        if manifest.is_file():
            try:
                value = json.loads(manifest.read_text())
            except json.JSONDecodeError:
                value = {"run_id": path.name, "status": "corrupt"}
            rows.append({key: value.get(key) for key in ("run_id", "run_type", "status", "completed_attempts", "total_attempts", "fingerprint")})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "pilot", "formal", "resume", "status"))
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--run-id")
    parser.add_argument("--run-type", choices=("pilot", "formal"))
    parser.add_argument("--type", dest="run_type_alias", choices=("pilot", "formal"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cshogi", action="store_true", help="compare all converted moves with pinned cshogi")
    args = parser.parse_args(argv)
    config = load_config()
    runtime = args.runtime.expanduser().resolve()
    if args.action == "status":
        print(json.dumps(status(runtime), indent=2, ensure_ascii=False))
        return 0
    if args.action == "plan":
        run_type = args.run_type or args.run_type_alias or "formal"
        if args.run_id:
            parser.error("plan uses --run-type; --run-id is reserved for engine runs")
        plan, games = make_plan(run_type, config)
        # The check is opportunistic: CI's system Python remains dependency
        # free, while the pinned audit Python records all 570 matches.
        plan["cshogi"] = compare_cshogi(games)
        output = args.output or (runtime / "plans" / f"development-{run_type}.json")
        universe, _ = build_universe(config)
        universe_output = output.parent / "development-universe.json"
        atomic_write_json(universe_output, universe)
        atomic_write_json(output, plan)
        print(json.dumps({
            "run_type": run_type,
            "positions": len(plan["positions"]),
            "repetitions": plan["repetitions"],
            "attempts": len(plan["positions"]) * plan["repetitions"] * 2,
            "requested_nodes": plan["requested_nodes"],
            "max_reported_nodes": plan.get("max_reported_nodes"),
            "universe_sha256": plan["hashes"]["universe_sha256"],
            "development_csa_aggregate_sha256": plan["hashes"]["development_csa_aggregate_sha256"],
            "universe_output": str(universe_output),
            "output": str(output),
        }, indent=2, ensure_ascii=False))
        return 0
    if args.action in ("pilot", "formal"):
        manifest = execute_run(runtime, config, args.action, run_id=args.run_id)
        print(json.dumps({key: manifest.get(key) for key in ("run_id", "run_type", "status", "fingerprint")}, indent=2))
        return 0
    if args.action == "resume":
        if not args.run_id:
            parser.error("resume requires --run-id")
        run_dir = runtime / "runs" / args.run_id
        manifest = json.loads((run_dir / "manifest.json").read_text())
        run_type = manifest.get("run_type")
        if run_type not in ("pilot", "formal"):
            raise ConfigurationError("resume manifest has an invalid run type")
        manifest = execute_run(runtime, config, run_type, run_id=args.run_id, resume=True)
        print(json.dumps({key: manifest.get(key) for key in ("run_id", "run_type", "status", "fingerprint")}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == _SUPERVISOR_FLAG:
        raise SystemExit(_usi_supervisor_main(sys.argv[2:]))
    try:
        raise SystemExit(main())
    except (BenchmarkError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
