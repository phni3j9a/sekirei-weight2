#!/usr/bin/env python3
"""Build a private expanded training dataset while preserving a frozen holdout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from benchmark import atomic_write_json
from pack_dataset import jsonl, position_key, split_for_game
from prepare import sha256


FILES = {f"{split}.{kind}.jsonl" for split in ("train", "holdout")
         for kind in ("positions", "labels")}
SPLIT_RECIPE = "SHA256(seed:complete-game-identity) modulo 10; holdout=0"
SAMPLING = "first bounded games of each hash-ordered pack; ply>=16/every4/max32"
EXCLUSION_POLICY = "exclude entire acquired pool; no final split files opened"


def checked_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def board_key(sfen):
    """Only check the producer's SFEN fields; do not implement a new shogi parser."""
    if not isinstance(sfen, str):
        raise ValueError("position SFEN must be a string")
    fields = sfen.split()
    if (len(fields) != 4 or fields[1] not in ("b", "w")
            or not fields[3].isascii() or not fields[3].isdecimal() or int(fields[3]) < 1):
        raise ValueError("invalid SFEN fields, side, or ply counter")
    return position_key(sfen)


def load_dataset(root):
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != FILES:
        raise ValueError("expected a version-1 pack dataset with exactly four files")
    source = manifest.get("source_corpus_manifest_sha256")
    if not checked_sha(source) or manifest.get("teacher_identity") != f"external:suisho11beta-1m-pack:{source}":
        raise ValueError("external teacher and source pack identity mismatch")
    if (not isinstance(manifest.get("seed"), str) or not manifest["seed"]
            or manifest.get("split") != SPLIT_RECIPE or manifest.get("sampling") != SAMPLING):
        raise ValueError("unrecognized game split seed or sampling recipe")
    cap = manifest.get("games_per_pack")
    if type(cap) is not int or not 1 <= cap <= 1000:
        raise ValueError("invalid source game cap")
    exclusion = manifest.get("independent_exclusions", {})
    if (exclusion.get("games") != 1000 or type(exclusion.get("unique_positions")) is not int
            or exclusion["unique_positions"] < 1 or exclusion.get("policy") != EXCLUSION_POLICY
            or not checked_sha(exclusion.get("corpus_canonical_sha256"))
            or not checked_sha(exclusion.get("source_manifest_sha256"))):
        raise ValueError("missing verified whole-pool independent exclusions")
    games, locations = {}, set()
    for game in manifest.get("games", []):
        gid = game.get("game_id")
        if (not checked_sha(gid) or gid in games or not checked_sha(game.get("pack_sha256"))
                or type(game.get("game_index")) is not int or not 0 <= game["game_index"] < cap
                or game.get("split") != split_for_game(gid, manifest["seed"])):
            raise ValueError("invalid, duplicated, or misassigned source game identity")
        location = (game["pack_sha256"], game["game_index"])
        if location in locations:
            raise ValueError("duplicate source pack game location")
        locations.add(location)
        games[gid] = game
    if not games:
        raise ValueError("source game inventory is empty")
    hashes = {str(manifest_path): hashlib.sha256(manifest_bytes).hexdigest()}
    rows = {}
    for name in sorted(FILES):
        path = root / name
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        info = manifest["files"][name]
        if digest != info.get("sha256") or len(data) != info.get("bytes"):
            raise ValueError(f"dataset content hash or size mismatch: {name}")
        hashes[str(path)] = digest
        parsed = [json.loads(line) for line in data.decode().splitlines()]
        if any(not isinstance(row, dict) for row in parsed):
            raise ValueError("JSONL rows must be objects")
        rows[name] = parsed
    splits = {}
    for split in ("train", "holdout"):
        positions, labels = rows[f"{split}.positions.jsonl"], rows[f"{split}.labels.jsonl"]
        if not positions or len(positions) != manifest.get("positions", {}).get(split):
            raise ValueError("dataset position count mismatch")
        sfens, keys = set(), set()
        for row in positions:
            sfen = row.get("sfen")
            key = board_key(sfen)
            gid = row.get("source", {}).get("path")
            if (sfen in sfens or key in keys or gid not in games
                    or row.get("schema_version") != 1 or row.get("source", {}).get("kind") != "gensfen-pack"):
                raise ValueError("duplicate board or unknown source game in positions")
            sfens.add(sfen); keys.add(key)
        cache = {}
        for label in labels:
            sfen, cp = label.get("sfen"), label.get("score_cp")
            if (not isinstance(sfen, str) or sfen in cache or type(cp) is not int or abs(cp) >= 30000
                    or type(label.get("label_depth")) is not int or label["label_depth"] != 0
                    or label.get("teacher_identity") != manifest["teacher_identity"]):
                raise ValueError("invalid or duplicate external label")
            cache[sfen] = label
        if set(cache) != sfens:
            raise ValueError("label set must exactly equal the position SFEN set")
        splits[split] = {"positions": positions, "cache": cache, "keys": keys,
                         "game_ids": {p["source"]["path"] for p in positions}}
    return {"root": root, "manifest": manifest, "hashes": hashes, "games": games, "splits": splits}


def verify_sources(expanded, frozen):
    em, fm = expanded["manifest"], frozen["manifest"]
    for field in ("teacher_identity", "source_corpus_manifest_sha256", "seed", "split", "sampling",
                  "dependencies", "independent_exclusions"):
        if field not in em or field not in fm or em[field] != fm[field]:
            raise ValueError(f"input dataset identity mismatch: {field}")
    if em["games_per_pack"] < fm["games_per_pack"]:
        raise ValueError("expanded dataset has a smaller source game cap")
    if ({g["pack_sha256"] for g in expanded["games"].values()}
            != {g["pack_sha256"] for g in frozen["games"].values()}):
        raise ValueError("input pack inventories differ")
    for gid, game in frozen["games"].items():
        if expanded["games"].get(gid) != game:
            raise ValueError("expanded dataset does not preserve original source game identities")
    for split in ("train", "holdout"):
        if any(frozen["games"][gid]["split"] != split for gid in frozen["splits"][split]["game_ids"]):
            raise ValueError("frozen dataset has a game assigned to the wrong split")
    ft, fh = frozen["splits"]["train"], frozen["splits"]["holdout"]
    if ft["keys"] & fh["keys"] or ft["game_ids"] & fh["game_ids"]:
        raise ValueError("frozen source train and holdout overlap")
    if any(expanded["games"][gid]["split"] != "holdout"
           for gid in expanded["splits"]["holdout"]["game_ids"]):
        raise ValueError("expanded holdout contains a training game")


def require_private_new_output(output, expanded, frozen):
    if output.exists() or output.is_relative_to(expanded) or output.is_relative_to(frozen):
        raise ValueError("output must be new and outside both input datasets")
    if not output.parent.is_dir():
        raise ValueError("output parent must be an existing private runtime directory")
    check = subprocess.run(["git", "-C", str(output.parent), "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True)
    if check.returncode == 0 and check.stdout.strip() == "true":
        raise ValueError("dataset output must be private and outside every Git worktree")


def unchanged(hashes):
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("input dataset changed during preparation")


def prepare(args):
    expanded_root = args.expanded_dataset.expanduser().resolve(strict=True)
    frozen_root = args.frozen_holdout_dataset.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    require_private_new_output(output, expanded_root, frozen_root)
    started = time.monotonic()
    expanded, frozen = load_dataset(expanded_root), load_dataset(frozen_root)
    verify_sources(expanded, frozen)
    old_games = {gid for gid, game in frozen["games"].items() if game["split"] == "holdout"}
    reserved_games = {gid for gid, game in expanded["games"].items() if game["split"] == "holdout"}
    old_keys = frozen["splits"]["holdout"]["keys"]
    new_holdout_keys = expanded["splits"]["holdout"]["keys"]
    kept, excluded = [], Counter()
    for position in expanded["splits"]["train"]["positions"]:
        gid, key = position["source"]["path"], board_key(position["sfen"])
        if gid in old_games:
            excluded["frozen_holdout_game_rows"] += 1
        elif gid in reserved_games:
            excluded["new_reserved_holdout_game_rows"] += 1
        elif key in old_keys:
            excluded["frozen_holdout_board_rows"] += 1
        elif key in new_holdout_keys:
            excluded["expanded_reserved_holdout_board_rows"] += 1
        else:
            kept.append(position)
    if not kept:
        raise ValueError("no training positions remain after holdout protection")
    train_games = {row["source"]["path"] for row in kept}
    train_keys = {board_key(row["sfen"]) for row in kept}
    if train_games & reserved_games or train_keys & (old_keys | new_holdout_keys):
        raise ValueError("train and reserved holdout are not disjoint")
    labels = [expanded["splits"]["train"]["cache"][row["sfen"]] for row in kept]
    hashes = {**expanded["hashes"], **frozen["hashes"]}
    unchanged(hashes)
    holdout_bytes = {name: (frozen_root / name).read_bytes()
                     for name in FILES if name.startswith("holdout.")}
    for name, data in holdout_bytes.items():
        if hashlib.sha256(data).hexdigest() != frozen["manifest"]["files"][name]["sha256"]:
            raise ValueError("frozen holdout changed before copying")
    output.mkdir()
    jsonl(output / "train.positions.jsonl", kept)
    jsonl(output / "train.labels.jsonl", labels)
    for name, data in holdout_bytes.items():
        (output / name).write_bytes(data)
    for name in holdout_bytes:
        if sha256(output / name) != frozen["manifest"]["files"][name]["sha256"]:
            raise ValueError("holdout copy is not byte-identical")
    unchanged(hashes)
    manifest = {key: expanded["manifest"][key] for key in (
        "schema_version", "teacher_identity", "source_corpus_manifest_sha256", "dependencies",
        "games_per_pack", "sampling", "seed", "split", "games", "independent_exclusions")}
    manifest.update(
        positions={"train": len(kept), "holdout": len(frozen["splits"]["holdout"]["positions"])},
        files={name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
               for name in sorted(FILES)},
        label_depth="0 is an external-cache sentinel, not a search depth claim",
        limitations=expanded["manifest"].get("limitations", []),
        derivation={"kind": "expanded-train-frozen-holdout-v1", "input_sha256": hashes,
                    "expanded_dataset": str(expanded_root), "frozen_holdout_dataset": str(frozen_root),
                    "script_sha256": sha256(Path(__file__)), "excluded_training_rows": dict(excluded),
                    "expanded_train_count": len(expanded["splits"]["train"]["positions"]),
                    "frozen_holdout_game_count": len(old_games),
                    "reserved_new_holdout_game_count": len(reserved_games - old_games),
                    "holdout_policy": "original bytes frozen; added holdout games remain unused; no final evaluation split created",
                    "input_unchanged": True, "wall_seconds": time.monotonic() - started})
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({"positions": manifest["positions"], "excluded_training_rows": dict(excluded),
                      "frozen_holdout_game_count": len(old_games),
                      "reserved_new_holdout_game_count": len(reserved_games - old_games),
                      "manifest_sha256": sha256(output / "manifest.json")}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--frozen-holdout-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
