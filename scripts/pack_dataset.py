#!/usr/bin/env python3
"""Prepare a bounded, game-split sample from the pinned GenSfen corpus."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

from audit_pack import corpus_files, iter_pack_positions, require_decoder
from benchmark import atomic_write, atomic_write_json, nonblocking_lock
from prepare import sha256
from acquire_quest import parse_csa


def position_key(sfen):
    """Ignore ply counter for duplicate/leakage detection, retain side and hand."""
    return " ".join(sfen.split()[:3])


def game_identity(rows):
    value = position_key(rows[0]["sfen"]) + "\n" + " ".join(r["selected_move"] for r in rows)
    return hashlib.sha256(value.encode()).hexdigest()


def split_for_game(game_id, seed):
    value = hashlib.sha256(f"{seed}:{game_id}".encode()).hexdigest()
    return "holdout" if int(value[:16], 16) % 10 == 0 else "train"


def independent_position_exclusions(root, cshogi):
    """Exclude the whole acquired pool, without opening/selecting final files."""
    import cshogi.CSA
    files = sorted((root / "games").glob("*.csa"))
    manifest = json.loads((root / "manifest.json").read_text())
    if len(files) != 1000 or manifest["accepted_games"] != 1000:
        raise RuntimeError("expected the complete fixed independent 1000-game pool")
    keys = set()
    canonical_hashes = []
    for path in files:
        if sha256(path) != path.stem:
            raise RuntimeError("independent corpus file hash mismatch")
        source_text = path.read_text()
        canonical_hashes.append(parse_csa(source_text)["canonical_sha256"])
        games = cshogi.CSA.Parser.parse_str(source_text)
        if len(games) != 1:
            raise RuntimeError("invalid independent corpus game")
        game = games[0]
        board = cshogi.Board(game.sfen)
        keys.add(position_key(board.sfen()))
        for move in game.moves:
            if not board.is_legal(move):
                raise RuntimeError("illegal independent corpus move")
            board.push(move)
            keys.add(position_key(board.sfen()))
    aggregate = hashlib.sha256("".join(sorted(canonical_hashes)).encode()).hexdigest()
    if aggregate != manifest.get("corpus_canonical_sha256"):
        raise RuntimeError("independent pool canonical aggregate mismatch")
    return keys, {"games": len(files), "unique_positions": len(keys),
                  "corpus_canonical_sha256": aggregate,
                  "source_manifest_sha256": sha256(root / "manifest.json"),
                  "policy": "exclude entire acquired pool; no final split files opened"}


def remove_shared_positions(groups):
    owners = defaultdict(set)
    for split, rows in groups.items():
        for row in rows:
            owners[position_key(row["sfen"])].add(split)
    shared = {key for key, splits in owners.items() if len(splits) > 1}
    result = {}
    for split, rows in groups.items():
        seen = set()
        kept = []
        for row in rows:
            key = position_key(row["sfen"])
            if key not in shared and key not in seen:
                seen.add(key)
                kept.append(row)
        result[split] = kept
    return result, len(shared)


def jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def prepare(args):
    if not 1 <= args.games_per_pack <= 1000:
        raise ValueError("games-per-pack must be 1..1000 for bounded preparation")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError("dataset output already exists; use a new run directory")
    if shutil.disk_usage(output.parent).free < 2 * 2**30:
        raise RuntimeError("dataset preparation needs 2 GiB free")
    cshogi, numpy, dependencies = require_decoder()
    started = time.monotonic()
    with nonblocking_lock(args.corpus_runtime / ".prepare.lock", exclusive=False):
        manifest_path, manifest, files = corpus_files(args.corpus_runtime, "suisho11beta-1m")
        excluded, exclusion_meta = independent_position_exclusions(args.quest_runtime, cshogi)
        groups = {"train": [], "holdout": []}
        game_ids = set()
        selected_games = []
        counts = Counter()
        for path in files:
            positions = iter_pack_positions(path, cshogi, numpy)
            for index, items in itertools.groupby(positions, key=lambda r: r["game_index"]):
                if index >= args.games_per_pack:
                    break
                rows = list(items)
                if len(rows) > 2048:
                    raise RuntimeError("unexpected game length")
                gid = game_identity(rows)
                if gid in game_ids:
                    counts["duplicate_games"] += 1
                    continue
                game_ids.add(gid)
                split = split_for_game(gid, args.seed)
                selected = []
                for row in rows:
                    counts["decoded_positions"] += 1
                    if row["game_ply"] < 16 or row["ply_in_game"] % 4:
                        continue
                    if abs(row["teacher_eval_cp_stm"]) >= 30000:
                        counts["mate_scale_excluded"] += 1
                        continue
                    if position_key(row["sfen"]) in excluded:
                        counts["independent_overlap_excluded"] += 1
                        continue
                    selected.append(dict(row, game_id=gid))
                    if len(selected) == 32:
                        break
                groups[split].extend(selected)
                selected_games.append({"game_id": gid, "split": split,
                                       "pack_sha256": path.stem, "game_index": index})
            print(f"sampled pack {len(selected_games)} cumulative games", flush=True)
        before = sum(map(len, groups.values()))
        groups, shared = remove_shared_positions(groups)
        if not groups["train"] or not groups["holdout"]:
            raise RuntimeError("bounded sample must contain both game-level splits")
        output.mkdir(parents=True)
        corpus_hash = sha256(manifest_path)
        teacher_id = f"external:suisho11beta-1m-pack:{corpus_hash}"
        for split, rows in groups.items():
            positions = [{"schema_version": 1, "sfen": r["sfen"],
                          "source": {"kind": "gensfen-pack", "path": r["game_id"],
                                     "ply": r["game_ply"]},
                          "tags": {"side_to_move": r["side_to_move"], "phase": "middlegame"}}
                         for r in rows]
            labels = [{"sfen": r["sfen"], "score_cp": r["teacher_eval_cp_stm"],
                       "teacher_identity": teacher_id, "label_depth": 0} for r in rows]
            jsonl(output / f"{split}.positions.jsonl", positions)
            jsonl(output / f"{split}.labels.jsonl", labels)
        meta = {"schema_version": 1, "teacher_identity": teacher_id,
                "source_corpus_manifest_sha256": corpus_hash,
                "dependencies": dependencies, "games_per_pack": args.games_per_pack,
                "sampling": "first bounded games of each hash-ordered pack; ply>=16/every4/max32",
                "seed": args.seed, "split": "SHA256(seed:complete-game-identity) modulo 10; holdout=0",
                "games": selected_games, "counts": dict(counts),
                "shared_positions_removed_from_both_splits": shared,
                "duplicate_or_shared_positions_removed": before - sum(map(len, groups.values())),
                "positions": {s: len(r) for s, r in groups.items()},
                "independent_exclusions": exclusion_meta,
                "limitations": ["pack labels lack exact/bound flags and original engine identity",
                                "bounded prefix sample; not a uniform sample of the full corpus",
                                "unrecorded game-family relationships cannot be proven"],
                "label_depth": "0 is an external-cache sentinel, not a search depth claim",
                "wall_seconds": time.monotonic() - started,
                "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                          for p in sorted(output.glob("*.jsonl"))}}
        atomic_write_json(output / "manifest.json", meta)
        print(json.dumps({"output": str(output), "positions": meta["positions"],
                          "wall_seconds": meta["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-runtime", type=Path, required=True)
    parser.add_argument("--quest-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-pack", type=int, default=20)
    parser.add_argument("--seed", default="first-weight-v1")
    prepare(parser.parse_args())
