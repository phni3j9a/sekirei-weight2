#!/usr/bin/env python3
"""Sample GenSfen .pack labels and reanalyze them with the pinned teacher."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
from pathlib import Path
import re
import struct
import tempfile
import time

from prepare import DEFAULT_RUNTIME, LOCK, REPO, sha256, verify_weight
from smoke import Engine, parse_result


def require_decoder():
    try:
        import cshogi
        import numpy
    except ImportError as error:
        raise RuntimeError(
            "pack audit dependencies are missing; run "
            "`python3 scripts/prepare.py audit-deps`, then invoke this script "
            "with <runtime>/venv/bin/python"
        ) from error
    expected = {}
    for line in (REPO / "config/audit-requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            expected[name.lower()] = version
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"pack audit dependency mismatch: expected {expected}, got {actual}")
    return cshogi, numpy, actual


def read_exact(stream, size, context):
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"truncated {context}: expected {size} bytes, got {len(data)}")
    return data


def iter_pack_positions(path, cshogi, numpy):
    """Yield positions without loading a whole .pack file into memory."""
    pack_hash = sha256(path)
    with path.open("rb") as stream:
        game_index = 0
        while True:
            flag_bytes = stream.read(1)
            if not flag_bytes:
                return
            game_offset = stream.tell() - 1
            start_flag = flag_bytes[0]
            board = cshogi.Board()
            if start_flag == 1:
                base_position = "startpos"
            elif start_flag == 0:
                hcp = read_exact(stream, 32, f"HCP at game {game_index}")
                game_ply = struct.unpack("<H", read_exact(
                    stream, 2, f"game ply at game {game_index}"))[0]
                board.set_hcp(numpy.frombuffer(hcp, dtype=cshogi.HuffmanCodedPos))
                board.move_number = game_ply
                if not board.is_ok():
                    raise ValueError(f"invalid HCP at game {game_index}")
                base_position = "sfen " + board.sfen()
            else:
                raise ValueError(
                    f"unsupported start flag {start_flag} at byte {game_offset}")

            moves = []
            ply_in_game = 0
            while True:
                move_offset = stream.tell()
                move16 = struct.unpack("<H", read_exact(
                    stream, 2, f"move at game {game_index}, ply {ply_in_game}"))[0]
                to_square = move16 & 0x7f
                from_square = (move16 >> 7) & 0x7f
                if to_square == from_square:
                    reason = read_exact(
                        stream, 1, f"end reason at game {game_index}")[0]
                    if to_square not in (0, 1, 2):
                        raise ValueError(
                            f"invalid game result {to_square} at byte {move_offset}")
                    del reason
                    break

                teacher_eval = struct.unpack("<h", read_exact(
                    stream, 2, f"eval at game {game_index}, ply {ply_in_game}"))[0]
                move = board.move_from_move16(move16)
                if not move or not board.is_legal(move):
                    raise ValueError(
                        f"illegal move16 {move16:#06x} at game {game_index}, "
                        f"ply {ply_in_game}")
                selected_move = cshogi.move_to_usi(move)
                position = base_position
                if moves:
                    position += " moves " + " ".join(moves)
                yield {
                    "pack_sha256": pack_hash,
                    "game_index": game_index,
                    "game_offset": game_offset,
                    "ply_in_game": ply_in_game,
                    "game_ply": int(board.move_number),
                    "side_to_move": "black" if board.turn == cshogi.BLACK else "white",
                    "sfen": board.sfen(),
                    "position_command": position,
                    "selected_move": selected_move,
                    "teacher_eval_cp_stm": teacher_eval,
                }
                board.push(move)
                moves.append(selected_move)
                ply_in_game += 1
            game_index += 1


def corpus_files(runtime, corpus_id):
    root = runtime / "data" / "teachers" / corpus_id
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    spec = LOCK["teacher_corpora"][corpus_id]
    if manifest["teacher_id_claim"] != spec["teacher_id"]:
        raise RuntimeError("corpus teacher claim differs from the lock")
    if manifest["requested_nodes_claim"] != spec["requested_nodes"]:
        raise RuntimeError("corpus node-budget claim differs from the lock")
    files = []
    for item in manifest["unique_files"]:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"corpus pack does not match its manifest: {path}")
        files.append(path)
    if not files:
        raise RuntimeError("corpus manifest has no unique pack files")
    return manifest_path, manifest, files


def select_samples(files, count, min_game_ply, cshogi, numpy):
    if count > len(files):
        raise RuntimeError(
            f"requested {count} samples but one-per-file sampling has only "
            f"{len(files)} unique files")
    selected = []
    for path in files:
        for position in iter_pack_positions(path, cshogi, numpy):
            if (position["game_ply"] >= min_game_ply
                    and abs(position["teacher_eval_cp_stm"]) < 32000):
                position["pack_path"] = str(path)
                selected.append(position)
                break
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"found only {len(selected)} usable sample positions")
    return selected


def probe_teacher(runtime, config, weight, sample, output, index):
    options = dict(config["teacher_options"], EvalDir=str(weight.parent))
    with (output / f"sample-{index:02d}.usi.log").open("w") as log:
        engine = Engine(
            runtime / "bin/yaneuraou", output, config["timeout_seconds"], log)
        try:
            engine.send("usi")
            handshake = engine.until(lambda line: line == "usiok")
            advertised = {match[1] for line in handshake
                          if (match := re.match(r"option name (.+?) type ", line))}
            if missing := options.keys() - advertised:
                raise RuntimeError(f"teacher: unsupported options {sorted(missing)}")
            for key, value in options.items():
                engine.send(f"setoption name {key} value {value}")
            engine.send("isready")
            startup = engine.until(lambda line: line == "readyok")
            engine.send("usinewgame")
            engine.send("position " + sample["position_command"])
            started = time.monotonic()
            engine.send(f"go nodes {config['requested_nodes']}")
            lines = engine.until(lambda line: line.startswith("bestmove "))
            result = parse_result(lines, sample["side_to_move"], allow_bounds=True)
            if not 0 < result["nodes"] <= config["requested_nodes"] * 1.02:
                raise RuntimeError(f"unexpected node count: {result['nodes']}")
            result["wall_seconds"] = round(time.monotonic() - started, 3)
            result["node_limit_fraction"] = round(
                result["nodes"] / config["requested_nodes"], 6)
            result["engine_id"] = [line for line in handshake if line.startswith("id ")]
            result["startup_warnings"] = [
                line for line in startup if "warning" in line.lower()]
            result["selected_move_matches"] = result["bestmove"] == sample["selected_move"]
            result["label_delta_cp_stm"] = (
                result["reported_cp_stm"] - sample["teacher_eval_cp_stm"]
                if result["score_bound_stm"] == "exact" else None)
            return result
        finally:
            engine.close()


def summarize_reanalysis(samples):
    results = [sample["reanalysis"] for sample in samples]
    deltas = [result["label_delta_cp_stm"] for result in results
              if result["label_delta_cp_stm"] is not None]
    return {
        "samples": len(results),
        "exact_scores": len(deltas),
        "bounded_scores": len(results) - len(deltas),
        "selected_move_matches": sum(
            result["selected_move_matches"] for result in results),
        "exact_score_mae_cp": round(
            sum(abs(delta) for delta in deltas) / len(deltas), 3) if deltas else None,
        "exact_score_max_abs_error_cp": max(
            (abs(delta) for delta in deltas), default=None),
        "total_wall_seconds": round(sum(
            result["wall_seconds"] for result in results), 3),
        "node_limit_fraction_min": min(
            result["node_limit_fraction"] for result in results),
        "node_limit_fraction_max": max(
            result["node_limit_fraction"] for result in results),
    }


def run_audit(args, runtime):
    cshogi, numpy, dependencies = require_decoder()
    manifest_path, corpus_manifest, files = corpus_files(runtime, args.corpus)
    samples = select_samples(
        files, args.samples, args.min_game_ply, cshogi, numpy)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (runtime / "runs").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"pack-audit-{timestamp}-", dir=runtime / "runs"))
    print("Logs:", output, flush=True)

    summary = {
        "purpose": "teacher_corpus_compatibility_audit_not_provenance_proof",
        "teacher_id": LOCK["primary_teacher"]["id"],
        "corpus_id": args.corpus,
        "corpus_manifest": str(manifest_path),
        "corpus_manifest_sha256": sha256(manifest_path),
        "unique_pack_files": len(corpus_manifest["unique_files"]),
        "requested_nodes": LOCK["teacher_corpora"][args.corpus]["requested_nodes"],
        "sampling": "first finite label at or after min_game_ply, one per unique pack",
        "min_game_ply": args.min_game_ply,
        "dependencies": dependencies,
        "reanalysis_state": "fresh_engine_process_and_empty_hash_per_sample",
        "known_limitation": (
            "The pack format records neither engine revision/options nor score bounds; "
            "the original generator also reused per-side processes within each game."),
        "samples": samples,
    }
    if not args.no_reanalyze:
        weight = verify_weight(runtime)
        build_manifest = json.loads((runtime / "build-manifest.json").read_text())
        if build_manifest["lock_sha256"] != sha256(REPO / "config/toolchain.lock.json"):
            raise RuntimeError("build manifest does not match current toolchain lock")
        expected_binary = build_manifest["binaries"]["yaneuraou"]["sha256"]
        if sha256(runtime / "bin/yaneuraou") != expected_binary:
            raise RuntimeError("teacher binary does not match the build manifest")
        config = json.loads((REPO / "config/smoke.json").read_text())
        if config["requested_nodes"] != summary["requested_nodes"]:
            raise RuntimeError("smoke and corpus node budgets differ")
        for index, sample in enumerate(samples, 1):
            sample["reanalysis"] = probe_teacher(
                runtime, config, weight, sample, output, index)
        summary["result"] = summarize_reanalysis(samples)
        summary["build_manifest_sha256"] = sha256(runtime / "build-manifest.json")
        summary["weight_sha256"] = sha256(weight)

    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["samples"], indent=2, ensure_ascii=False))
    print("PASS:", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--corpus", default="suisho11beta-1m",
                        choices=sorted(LOCK["teacher_corpora"]))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--min-game-ply", type=int, default=20)
    parser.add_argument("--no-reanalyze", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.min_game_ply < 1:
        parser.error("--samples and --min-game-ply must be positive")

    runtime = args.runtime.expanduser().resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / ".prepare.lock").open("a") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_SH | fcntl.LOCK_NB)
        run_audit(args, runtime)


if __name__ == "__main__":
    main()
