#!/usr/bin/env python3
"""Diagnose fixed checkpoints on a verified train/holdout split, without search."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from benchmark import atomic_write_json, nonblocking_lock
from prepare import LOCK, REPO, sha256


PREDICTIONS = ("raw_float_cp", "quantized_float_cp", "inference_cp", "material_cp")


def strict_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected object in {path.name}")
        rows.append(row)
    return rows


def verified_split(dataset, split):
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_names = {f"{part}.{kind}.jsonl" for part in ("train", "holdout")
                      for kind in ("positions", "labels")}
    if set(manifest["files"]) != expected_names:
        raise ValueError("unexpected dataset file set")
    for name, info in manifest["files"].items():
        path = dataset / name
        if sha256(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError(f"dataset hash or size mismatch: {name}")
    identity = manifest["teacher_identity"]
    if not isinstance(identity, str) or not identity.startswith("external:") or len(identity) <= 9:
        raise ValueError("missing external teacher identity")
    positions = strict_rows(dataset / f"{split}.positions.jsonl")
    labels = strict_rows(dataset / f"{split}.labels.jsonl")
    if len(positions) != manifest["positions"][split] or not positions:
        raise ValueError("position count differs from manifest")
    sfens = [row.get("sfen") for row in positions]
    if any(not isinstance(sfen, str) for sfen in sfens) or len(set(sfens)) != len(sfens):
        raise ValueError("missing or duplicate position SFEN")
    cache = {}
    for row in labels:
        cp, sfen = row.get("score_cp"), row.get("sfen")
        if (row.get("teacher_identity") != identity or type(row.get("label_depth")) is not int
                or row["label_depth"] != 0 or type(cp) is not int or abs(cp) >= 30000
                or not isinstance(sfen, str) or sfen in cache):
            raise ValueError("invalid or duplicate external label")
        cache[sfen] = cp
    if set(sfens) != set(cache):
        raise ValueError("label cache must equal the complete position set")
    return manifest, [cache[sfen] for sfen in sfens]


def metrics(prediction, target):
    count = len(target)
    if count == 0 or len(prediction) != count:
        raise ValueError("metrics require equally sized nonempty vectors")
    pm, tm = statistics.fmean(prediction), statistics.fmean(target)
    ps, ts = statistics.pstdev(prediction), statistics.pstdev(target)
    covariance = statistics.fmean((p - pm) * (t - tm) for p, t in zip(prediction, target))
    return {"count": count,
            "mae_cp": statistics.fmean(abs(p - t) for p, t in zip(prediction, target)),
            "rmse_cp": math.sqrt(statistics.fmean((p - t) ** 2 for p, t in zip(prediction, target))),
            "bias_cp": pm - tm, "prediction_mean_cp": pm, "prediction_std_cp": ps,
            "prediction_min_cp": min(prediction), "prediction_max_cp": max(prediction),
            "target_mean_cp": tm, "target_std_cp": ts,
            "pearson_r": covariance / (ps * ts) if ps and ts else None}


def summarize(path, targets):
    rows = strict_rows(path)
    if len(rows) != len(targets):
        raise ValueError("diagnostic result count mismatch")
    for index, (row, target) in enumerate(zip(rows, targets)):
        if row.get("index") != index or row.get("teacher_cp_stm") != target:
            raise ValueError("diagnostic result ordering or label mismatch")
        for name in PREDICTIONS:
            value = row.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("missing or nonfinite diagnostic prediction")
    values = {name: [row[name] for row in rows] for name in PREDICTIONS}
    bridge_delta = [abs(a - b) for a, b in zip(values["quantized_float_cp"], values["inference_cp"])]
    if max(bridge_delta) >= 1.001:
        raise ValueError("dequantized float forward diverges from core inference by >= 1.001 cp")
    raw_delta = [abs(a - b) for a, b in zip(values["raw_float_cp"], values["quantized_float_cp"])]
    return {"count": len(rows), "metrics": {name: metrics(value, targets) for name, value in values.items()},
            "quantization_delta": {"mae_cp": statistics.fmean(raw_delta), "max_abs_cp": max(raw_delta)},
            "core_bridge_delta": {"mae_cp": statistics.fmean(bridge_delta), "max_abs_cp": max(bridge_delta)},
            "interpretation": "fixed static STM predictions on external pack labels; not the 1M-node adoption benchmark"}


def private_output(output, build_root):
    if not output.is_relative_to(build_root) or output == build_root:
        raise ValueError("diagnostic output must be inside its dedicated trainer runtime")
    if output.is_relative_to(build_root / "source") or output.is_relative_to(build_root / "build"):
        raise ValueError("diagnostic output cannot be inside the source or build directory")
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    check = subprocess.run(["git", "-C", str(ancestor), "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True)
    if check.returncode == 0 and check.stdout.strip() == "true":
        raise ValueError("per-position diagnostic output must remain outside every Git worktree")
    if output.exists():
        raise ValueError("diagnostic output exists; preserve it and select a new run")


def diagnose(args):
    dataset, build_root = args.dataset.expanduser().resolve(), args.trainer.expanduser().resolve()
    checkpoint, output = args.checkpoint.expanduser().resolve(), args.output.expanduser().resolve()
    if not 1 <= args.seconds <= 600:
        raise ValueError("diagnostic wall limit must be 1..600 seconds")
    private_output(output, build_root)
    manifest, targets = verified_split(dataset, args.split)
    build_path = build_root / "build-manifest.json"
    build = json.loads(build_path.read_text())
    if (build["upstream_commit"] != LOCK["sources"]["sekirei"]["commit"]
            or build["rustflags"] != LOCK["rustflags"]
            or build["patch_sha256"] != sha256(REPO / "patches/sekirei-train-external-labels.patch")
            or sha256(Path(build["binary"])) != build["binary_sha256"]):
        raise ValueError("trainer build identity mismatch")
    if sha256(build_root / "source/crates/sekirei-train/src/main.rs") != build["source_main_sha256"]:
        raise ValueError("trainer source identity mismatch")
    if checkpoint.suffix != ".bin":
        raise ValueError("checkpoint must be an inference .bin with matching .adam.json and .meta.json")
    adam, metadata = checkpoint.with_suffix(".adam.json"), checkpoint.with_suffix(".meta.json")
    meta = json.loads(metadata.read_text())
    if meta.get("nnue_output") != "absolute" or meta.get("teacher_identity") != manifest["teacher_identity"]:
        raise ValueError("checkpoint output mode or external teacher identity mismatch")
    if checkpoint.read_bytes()[:8] != b"SEKIRW01":
        raise ValueError("unexpected inference checkpoint format")
    input_paths = [dataset / "manifest.json", build_path, checkpoint, adam, metadata]
    input_paths += [dataset / name for name in manifest["files"]]
    hashes = {str(path): sha256(path) for path in input_paths}
    command = [build["binary"], "diagnose-external", "--positions", str(dataset / f"{args.split}.positions.jsonl"),
               "--teacher-cache", str(dataset / f"{args.split}.labels.jsonl"),
               "--external-teacher-id", manifest["teacher_identity"], "--weights", str(checkpoint), "--adam", str(adam)]
    output.mkdir(parents=True)
    record = {"schema_version": 1, "status": "running", "split": args.split,
              "count": len(targets), "argv": command, "input_sha256": hashes,
              "script_sha256": sha256(Path(__file__)), "trainer_build": build,
              "float_subnormal_policy": "x86-ftz-daz", "timeout_seconds": args.seconds}
    atomic_write_json(output / "run.json", record)
    started = time.monotonic()
    try:
        with nonblocking_lock(build_root / ".training.lock", exclusive=True):
            with (output / "predictions.jsonl").open("w") as predictions, (output / "diagnose.log").open("w") as log:
                process = subprocess.Popen(command, cwd=output, stdout=predictions, stderr=log,
                                           env=dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
                                                    SEKIREI_TRAIN_FTZ_DAZ="1"), start_new_session=True)
                try:
                    returncode = process.wait(timeout=args.seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    record["status"] = "timeout"
                    raise RuntimeError("fixed-checkpoint diagnosis exceeded its wall limit")
        record["returncode"] = returncode
        if returncode:
            raise RuntimeError(f"fixed-checkpoint diagnosis failed with exit {returncode}")
        for path, digest in hashes.items():
            if sha256(Path(path)) != digest:
                raise RuntimeError("diagnostic input changed during execution")
        summary = summarize(output / "predictions.jsonl", targets)
        summary.update(split=args.split, checkpoint_sha256=hashes[str(checkpoint)],
                       adam_sha256=hashes[str(adam)], dataset_manifest_sha256=hashes[str(dataset / "manifest.json")])
        atomic_write_json(output / "summary.json", summary)
        record.update(status="complete", predictions_sha256=sha256(output / "predictions.jsonl"),
                      summary_sha256=sha256(output / "summary.json"))
        print(json.dumps(summary, indent=2, allow_nan=False))
    except Exception as error:
        if record["status"] != "timeout":
            record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        atomic_write_json(output / "run.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--split", choices=("holdout", "train"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=120)
    diagnose(parser.parse_args())
