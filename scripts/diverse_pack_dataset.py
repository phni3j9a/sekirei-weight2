#!/usr/bin/env python3
"""Build a fixed-budget, hash-ranked whole-pack sample with frozen holdout bytes.

Only game boundaries are scanned across each complete pack. The selected byte
spans are decoded by the existing pinned legal-replay decoder. This script does
not read development/final split files, create teacher labels, or run training.
"""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import hashlib
import heapq
import itertools
import json
import mmap
from pathlib import Path
import shutil
import struct
import tempfile
import time

from audit_pack import corpus_files, iter_pack_positions, require_decoder
from benchmark import atomic_write, atomic_write_json, nonblocking_lock
from freeze_holdout_dataset import (
    FILES, board_key, checked_sha, load_dataset, require_private_new_output, unchanged,
)
from pack_dataset import game_identity, independent_position_exclusions, jsonl, split_for_game
from prepare import sha256


SAMPLING = "whole-pack SHA256(seed:pack-sha:index) rank; ply>=16/every4/max32"
ORDERING = "hash-ordered packs round-robin by selected game rank; truncate final game rows"
DEFAULT_SEED = "issue19-diverse-v1"
DEFAULT_TRAIN_COUNT = 112681
PROFILE_KIND = "whole-pack-hash-ranked-frozen-holdout-profile-v1"
ROW_FILTER = {"min_game_ply": 16, "ply_stride": 4, "per_game_cap": 32,
              "max_abs_cp_exclusive": 30000, "max_decoded_game_length": 2048}


def producer_sources():
    """Conservative closure of local imports, including imports in functions.

    Imported third-party packages are fixed by require_decoder; local Python
    sources and configuration inputs are bound by their actual bytes.
    """
    scripts = Path(__file__).resolve().parent
    pending, visited = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                source = scripts / (module.split(".")[0] + ".py")
                if source.is_file() and source not in visited:
                    pending.append(source)
    visited.update({scripts.parent / "config" / "toolchain.lock.json",
                    scripts.parent / "config" / "audit-requirements.txt",
                    scripts.parent / "config" / "quest-corpus.json"})
    return {str(path.relative_to(scripts.parent)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(visited)}


def selection_profile(args, frozen, corpus_manifest, corpus_hash):
    """Expected preregistration body; source refs carry no final split identity."""
    return {"schema_version": 1, "kind": PROFILE_KIND,
            "sampling": SAMPLING, "ordering": ORDERING, "row_filter": ROW_FILTER,
            "selection_seed": args.selection_seed, "games_per_pack": args.games_per_pack,
            "train_budget": args.train_count, "split_seed": frozen["manifest"]["seed"],
            "split": frozen["manifest"]["split"],
            "corpus_manifest_sha256": corpus_hash,
            "packs": sorted(({"sha256": p["sha256"], "bytes": p["bytes"]}
                             for p in corpus_manifest["unique_files"]), key=lambda p: p["sha256"]),
            "frozen_dataset_manifest_sha256": args.expected_frozen_manifest_sha256,
            "frozen_holdout_files": {name: frozen["manifest"]["files"][name]
                                     for name in sorted(FILES) if name.startswith("holdout.")},
            "independent_exclusions": frozen["manifest"]["independent_exclusions"],
            "dependencies": frozen["manifest"]["dependencies"],
            "producer_sources": producer_sources()}


def strict_json_bytes(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def typed_equal(actual, expected):
    """JSON booleans must not pass as integers in declaration contracts."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(typed_equal(actual[k], expected[k]) for k in expected))
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def verify_profile(path, digest, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("selection profile differs from the externally pinned SHA")
    profile = strict_json_bytes(data)
    if not typed_equal(profile, expected):
        raise ValueError("selection profile/body does not match the fixed source and arguments")
    return profile


def iter_game_spans(data):
    """Index the format only; legal replay of unselected games is not claimed."""
    offset, index, size = 0, 0, len(data)
    while offset < size:
        start = offset
        flag = data[offset]
        offset += 1
        if flag == 0:
            offset += 34  # HCP32 + game ply u16.
        elif flag != 1:
            raise ValueError(f"unsupported pack start flag at byte {start}")
        if offset > size:
            raise ValueError("truncated pack game header")
        count = 0
        while True:
            if offset + 2 > size:
                raise ValueError("truncated pack move/result")
            move = struct.unpack_from("<H", data, offset)[0]
            to_square, from_square = move & 127, (move >> 7) & 127
            if to_square == from_square:
                if to_square not in (0, 1, 2) or offset + 3 > size:
                    raise ValueError("invalid or truncated pack end record")
                offset += 3  # Result move16 + end-reason byte.
                break
            if offset + 4 > size:
                raise ValueError("truncated pack move evaluation")
            offset += 4
            count += 1
        yield {"game_index": index, "start": start, "end": offset, "positions": count}
        index += 1


def rank_key(seed, pack_hash, game_index):
    return hashlib.sha256(f"{seed}:{pack_hash}:{game_index}".encode()).hexdigest()


def select_spans(data, pack_hash, seed, count):
    """Fixed size, without replacement; content labels do not affect selection."""
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("games-per-pack must be an integer in 1..1000")
    census = Counter()
    def ranked():
        for span in iter_game_spans(data):
            census["games"] += 1
            census["positions"] += span["positions"]
            census["prefix_400_positions" if span["game_index"] < 400
                   else "after_prefix_positions"] += span["positions"]
            yield (rank_key(seed, pack_hash, span["game_index"]), span["game_index"], span)
    chosen = heapq.nsmallest(count, ranked())
    if len(chosen) != count:
        raise ValueError("pack has fewer games than the fixed selection count")
    selected = [dict(span, rank=rank) for rank, _, span in chosen]
    census["selected_prefix_400_games"] = sum(s["game_index"] < 400 for s in selected)
    return selected, dict(census)


def decode_selected(path, pack_hash, spans, cshogi, numpy, temp_parent):
    """Reuse the unchanged decoder; retain full-pack and slice identities apart."""
    if any(span["positions"] > ROW_FILTER["max_decoded_game_length"] for span in spans):
        raise ValueError("selected game exceeds the fixed decoded length cap")
    if sha256(path) != pack_hash:
        raise ValueError("pack changed before selected-game decoding")
    with tempfile.TemporaryDirectory(prefix=".diverse-pack-", dir=temp_parent) as directory:
        selected_path = Path(directory) / "selected.pack"
        # Byte spans are copied in rank order. Decoder-local game indices are
        # mapped back to the verified original pack locations immediately.
        with path.open("rb") as source, selected_path.open("wb") as target:
            for span in spans:
                source.seek(span["start"])
                data = source.read(span["end"] - span["start"])
                if len(data) != span["end"] - span["start"]:
                    raise ValueError("pack span changed/truncated during copy")
                target.write(data)
        subset_hash = sha256(selected_path)
        decoded = {}
        for local_index, items in itertools.groupby(
                iter_pack_positions(selected_path, cshogi, numpy), key=lambda r: r["game_index"]):
            if local_index not in range(len(spans)):
                raise ValueError("selected decoder game index mismatch")
            rows = list(items)
            span = spans[local_index]
            if len(rows) != span["positions"] or len(rows) > 2048:
                raise ValueError("selected game length differs from the byte index")
            decoded[local_index] = [dict(r, pack_sha256=pack_hash,
                                         game_index=span["game_index"], game_offset=span["start"])
                                    for r in rows]
        if any(span["positions"] and i not in decoded for i, span in enumerate(spans)):
            raise ValueError("selected game is missing from legal replay")
        if sha256(selected_path) != subset_hash or sha256(path) != pack_hash:
            raise ValueError("pack or temporary selected bytes changed during decoding")
        return [(span, decoded.get(i, [])) for i, span in enumerate(spans)], subset_hash


def eligible_rows(rows, excluded):
    selected = []
    for row in rows:
        if (row["game_ply"] >= 16 and row["ply_in_game"] % 4 == 0
                and abs(row["teacher_eval_cp_stm"]) < 30000
                and board_key(row["sfen"]) not in excluded):
            selected.append(row)
            if len(selected) == 32:
                break
    return selected


def select_train(games_by_pack, frozen, excluded, train_count):
    """Deduplicate complete games/boards before imposing the fixed row budget."""
    if type(train_count) is not int or not 1 <= train_count <= DEFAULT_TRAIN_COUNT:
        raise ValueError("train-count must be an integer in 1..112681")
    seed = frozen["manifest"]["seed"]
    old_holdout_games = {gid for gid, game in frozen["games"].items()
                         if game["split"] == "holdout"}
    protected = set(frozen["splits"]["holdout"]["keys"])
    seen_games, inventory, candidate_packs, counts = set(), [], [], Counter()
    for pack_games in games_by_pack:
        candidates = []
        for span, rows in pack_games:
            if not rows:
                counts["empty_selected_games"] += 1
                continue
            gid = game_identity(rows)
            if gid in seen_games:
                counts["duplicate_selected_games"] += 1
                continue
            seen_games.add(gid)
            split = split_for_game(gid, seed)
            game = {"game_id": gid, "split": split, "pack_sha256": rows[0]["pack_sha256"],
                    "game_index": span["game_index"], "selection_rank": span["rank"]}
            inventory.append(game)
            sampled = [dict(row, game_id=gid) for row in eligible_rows(rows, excluded)]
            if gid in old_holdout_games or split == "holdout":
                protected.update(board_key(row["sfen"]) for row in sampled)
                counts["reserved_selected_games"] += 1
            else:
                candidates.append(sampled)
        candidate_packs.append(candidates)
    kept, seen_boards = [], set()
    for round_games in itertools.zip_longest(*candidate_packs):
        for rows in round_games:
            for row in rows or ():
                key = board_key(row["sfen"])
                if key in protected:
                    counts["reserved_board_rows"] += 1
                elif key in seen_boards:
                    counts["duplicate_training_board_rows"] += 1
                else:
                    seen_boards.add(key)
                    kept.append(row)
                if len(kept) == train_count:
                    counts["selected_training_games"] = len({r["game_id"] for r in kept})
                    if seen_boards & (protected | excluded):
                        raise ValueError("training/holdout/independent board overlap")
                    return kept, inventory, dict(counts)
    raise ValueError(f"fixed candidate selection has only {len(kept)} usable rows; need {train_count}")


def validate_frozen(frozen, corpus_hash, pack_hashes, dependencies, exclusion_meta):
    manifest = frozen["manifest"]
    for field, expected in (("source_corpus_manifest_sha256", corpus_hash),
                            ("dependencies", dependencies), ("independent_exclusions", exclusion_meta)):
        if manifest.get(field) != expected:
            raise ValueError(f"frozen dataset/source identity mismatch: {field}")
    if {g["pack_sha256"] for g in frozen["games"].values()} != pack_hashes:
        raise ValueError("frozen dataset pack membership mismatch")
    train, holdout = frozen["splits"]["train"], frozen["splits"]["holdout"]
    if train["keys"] & holdout["keys"] or train["game_ids"] & holdout["game_ids"]:
        raise ValueError("frozen dataset train/holdout overlap")
    for split in ("train", "holdout"):
        if any(frozen["games"][gid]["split"] != split for gid in frozen["splits"][split]["game_ids"]):
            raise ValueError("frozen source game is assigned to the wrong split")


def histogram(rows, labels=None):
    result = Counter()
    for row in rows:
        ply = row.get("game_ply", row.get("source", {}).get("ply"))
        cp = abs(row["teacher_eval_cp_stm"] if labels is None else labels[row["sfen"]]["score_cp"])
        result["ply16-63" if ply < 64 else "ply64-127" if ply < 128 else "ply128+"] += 1
        result["abs-cp<100" if cp < 100 else "abs-cp100-999" if cp < 1000 else
               "abs-cp1000-4999" if cp < 5000 else "abs-cp5000+"] += 1
    return dict(result)


def verify_inputs(hashes, quest_root, pool_paths):
    unchanged(hashes)
    if [quest_root / "manifest.json", *sorted((quest_root / "games").glob("*.csa"))] != pool_paths:
        raise ValueError("independent raw pool membership changed during preparation")


def prepare(args):
    for value in (args.expected_frozen_manifest_sha256, args.expected_corpus_manifest_sha256,
                  args.expected_profile_sha256):
        if not checked_sha(value):
            raise ValueError("source manifest and profile hashes must be pinned externally")
    if not isinstance(args.selection_seed, str) or not args.selection_seed:
        raise ValueError("selection seed must be a nonempty string")
    if type(args.games_per_pack) is not int or not 1 <= args.games_per_pack <= 1000:
        raise ValueError("games-per-pack must be an integer in 1..1000")
    if type(args.train_count) is not int or not 1 <= args.train_count <= DEFAULT_TRAIN_COUNT:
        raise ValueError("train-count must be an integer in 1..112681")
    frozen_root = args.frozen_holdout_dataset.expanduser().resolve(strict=True)
    corpus_runtime = args.corpus_runtime.expanduser().resolve(strict=True)
    quest_root = args.quest_runtime.expanduser().resolve(strict=True)
    profile_path = args.profile.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    require_private_new_output(output, frozen_root, corpus_runtime)
    if output.is_relative_to(quest_root):
        raise ValueError("output must be outside the independent corpus")
    if shutil.disk_usage(output.parent).free < 2 * 2**30:
        raise ValueError("dataset preparation requires at least 2 GiB SSD free")
    started = time.monotonic()
    with ExitStack() as locks:
        locks.enter_context(nonblocking_lock(output.parent / ".diverse-dataset.lock", exclusive=True))
        locks.enter_context(nonblocking_lock(corpus_runtime / ".prepare.lock", exclusive=False))
        locks.enter_context(nonblocking_lock(quest_root / ".acquire.lock", exclusive=False))
        cshogi, numpy, dependencies = require_decoder()
        manifest_path, corpus_manifest, files = corpus_files(corpus_runtime, "suisho11beta-1m")
        corpus_hash = sha256(manifest_path)
        if corpus_hash != args.expected_corpus_manifest_sha256:
            raise ValueError("corpus manifest differs from the pinned SHA")
        if sha256(frozen_root / "manifest.json") != args.expected_frozen_manifest_sha256:
            raise ValueError("frozen manifest differs from the pinned SHA")
        frozen = load_dataset(frozen_root)
        profile = verify_profile(profile_path, args.expected_profile_sha256,
                                 selection_profile(args, frozen, corpus_manifest, corpus_hash))
        files = sorted(files, key=lambda p: p.stem)
        pack_hashes = {item["sha256"] for item in corpus_manifest["unique_files"]}
        if len(pack_hashes) != len(files) or any(p.stem not in pack_hashes for p in files):
            raise ValueError("duplicate/renamed pack membership")
        pool_paths = [quest_root / "manifest.json", *sorted((quest_root / "games").glob("*.csa"))]
        source_hashes = {**frozen["hashes"], str(manifest_path): corpus_hash,
                         str(profile_path): args.expected_profile_sha256,
                         **{str(Path(__file__).resolve().parent.parent / name): info["sha256"]
                            for name, info in profile["producer_sources"].items()},
                         **{str(p): sha256(p) for p in [*files, *pool_paths]}}
        excluded, exclusion_meta = independent_position_exclusions(quest_root, cshogi)
        validate_frozen(frozen, corpus_hash, pack_hashes, dependencies, exclusion_meta)
        games_by_pack, indexes = [], []
        for path in files:
            with path.open("rb") as stream:
                if path.stat().st_size == 0:
                    raise ValueError("empty pack")
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    spans, census = select_spans(data, path.stem, args.selection_seed, args.games_per_pack)
            decoded, subset_hash = decode_selected(path, path.stem, spans, cshogi, numpy, output.parent)
            # Complete SFEN/move sequences still define each game ID. Histories
            # used only by the decoder need not occupy RAM across all 13 packs.
            for _, rows in decoded:
                for row in rows:
                    row.pop("position_command", None)
            games_by_pack.append(decoded)
            indexes.append({"pack_sha256": path.stem, "census": census, "selected_spans": spans,
                            "temporary_selected_bytes_sha256": subset_hash})
            print(f"indexed and decoded selected games from pack {len(indexes)}/{len(files)}", flush=True)
        kept, inventory, counts = select_train(games_by_pack, frozen, excluded, args.train_count)
        verify_inputs(source_hashes, quest_root, pool_paths)
        positions = [{"schema_version": 1, "sfen": r["sfen"],
                      "source": {"kind": "gensfen-pack", "path": r["game_id"], "ply": r["game_ply"]},
                      "tags": {"side_to_move": r["side_to_move"], "phase": "middlegame"}} for r in kept]
        labels = [{"sfen": r["sfen"], "score_cp": r["teacher_eval_cp_stm"],
                   "teacher_identity": frozen["manifest"]["teacher_identity"], "label_depth": 0} for r in kept]
        output.mkdir(mode=0o700)
        jsonl(output / "train.positions.jsonl", positions)
        jsonl(output / "train.labels.jsonl", labels)
        for name in sorted(FILES):
            if name.startswith("holdout."):
                atomic_write(output / name, (frozen_root / name).read_bytes())
                if sha256(output / name) != frozen["manifest"]["files"][name]["sha256"]:
                    raise ValueError("frozen holdout copy is not byte-identical")
            (output / name).chmod(0o600)
        verify_inputs(source_hashes, quest_root, pool_paths)
        original = frozen["manifest"]
        games = {g["game_id"]: g for g in inventory}
        # Holdout provenance is preserved independently of the selected location
        # of a duplicate game in another pack.
        games.update({gid: g for gid, g in frozen["games"].items() if g["split"] == "holdout"})
        manifest = {"schema_version": 1, "teacher_identity": original["teacher_identity"],
                    "source_corpus_manifest_sha256": corpus_hash, "dependencies": dependencies,
                    "sampling": SAMPLING, "selection_seed": args.selection_seed,
                    "games_per_pack": args.games_per_pack, "train_budget": args.train_count,
                    "seed": original["seed"], "split": original["split"],
                    "games": [games[gid] for gid in sorted(games)], "counts": counts,
                    "positions": {"train": len(kept), "holdout": original["positions"]["holdout"]},
                    "independent_exclusions": exclusion_meta,
                    "label_depth": "0 is an external-cache sentinel, not a search depth claim",
                    "files": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                              for name in sorted(FILES)},
                    "derivation": {"kind": "whole-pack-hash-ranked-frozen-holdout-v1",
                                   "profile": profile, "profile_sha256": args.expected_profile_sha256,
                                   "producer_sources": profile["producer_sources"],
                                   "input_sha256": source_hashes, "script_sha256": sha256(Path(__file__)),
                                   "ordering": ORDERING, "indexes": indexes,
                                   "frozen_dataset_manifest_sha256": args.expected_frozen_manifest_sha256,
                                   "train_histogram": histogram(kept),
                                   "old_train_histogram": histogram(frozen["splits"]["train"]["positions"],
                                                                    frozen["splits"]["train"]["cache"]),
                                   "old_train_board_overlap": len({board_key(r["sfen"]) for r in kept}
                                                                  & frozen["splits"]["train"]["keys"]),
                                   "input_unchanged": True, "wall_seconds": time.monotonic() - started},
                    "limitations": ["pack labels lack exact/bound flags and original engine identity",
                                    "hash ranking is over pack locations; duplicate games are removed after selection",
                                    "selected games only are legally replayed; full-pack census validates byte boundaries",
                                    "unrecorded game-family relationships cannot be proven",
                                    "fixed holdout has already been used for model selection"]}
        # Recheck after the final self/helper hashes and body construction, then
        # again after publication. A failed post-write check leaves the original
        # failed body available but no successful manifest at the reader's path.
        verify_inputs(source_hashes, quest_root, pool_paths)
        manifest_output = output / "manifest.json"
        atomic_write_json(manifest_output, manifest)
        manifest_output.chmod(0o600)
        try:
            verify_inputs(source_hashes, quest_root, pool_paths)
        except Exception:
            manifest_output.rename(output / "manifest.failed.json")
            raise
        print(json.dumps({"positions": manifest["positions"],
                          "manifest_sha256": sha256(output / "manifest.json"),
                          "wall_seconds": manifest["derivation"]["wall_seconds"]}, indent=2))
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-runtime", type=Path, required=True)
    parser.add_argument("--quest-runtime", type=Path, required=True)
    parser.add_argument("--frozen-holdout-dataset", type=Path, required=True)
    parser.add_argument("--expected-frozen-manifest-sha256", required=True)
    parser.add_argument("--expected-corpus-manifest-sha256", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", default=DEFAULT_SEED)
    parser.add_argument("--games-per-pack", type=int, default=500)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    prepare(parser.parse_args())
