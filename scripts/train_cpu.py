#!/usr/bin/env python3
"""Run bounded CPU training using verified external labels, with provenance."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

from benchmark import atomic_write_json, nonblocking_lock
from pack_dataset import jsonl
from prepare import REPO, sha256


def train(args):
    dataset = args.dataset.expanduser().resolve()
    build_root = args.trainer.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError("training output exists; preserve it and choose a new run")
    if not 1 <= args.epochs <= 3 or not 1 <= args.seconds <= 3600:
        raise ValueError("initial run limits: epochs 1..3, wall seconds 1..3600")
    if shutil.disk_usage(output.parent).free < 2 * 2**30:
        raise RuntimeError("training requires 2 GiB free")
    manifest = json.loads((dataset / "manifest.json").read_text())
    build = json.loads((build_root / "build-manifest.json").read_text())
    if sha256(Path(build["binary"])) != build["binary_sha256"]:
        raise RuntimeError("trainer binary hash mismatch")
    if sha256(REPO / "patches/sekirei-train-external-labels.patch") != build["patch_sha256"]:
        raise RuntimeError("trainer patch identity mismatch")
    for name, info in manifest["files"].items():
        if Path(name).name != name or sha256(dataset / name) != info["sha256"]:
            raise RuntimeError("dataset hash mismatch")
    rows = [json.loads(line) for line in (dataset / "train.positions.jsonl").read_text().splitlines()]
    if args.max_positions:
        if not 1 <= args.max_positions <= len(rows):
            raise ValueError("max-positions exceeds the verified training dataset")
        rows = rows[:args.max_positions]
    output.mkdir()
    jsonl(output / "positions.jsonl", rows)
    command = [build["binary"], "--positions", str(output / "positions.jsonl"),
               "--strict-positions", "--external-teacher-id", manifest["teacher_identity"],
               "--teacher-cache", str(dataset / "train.labels.jsonl"),
               "--reuse-teacher-cache", "--cache-only", "--label-depth", "0",
               "--validation-ratio", "0", "--teacher-score-cap", "30000",
               "--exclude-mate-labels", "--nnue-output", "absolute",
               "--epochs", str(args.epochs), "--lr", str(args.lr),
               "--shuffle-seed", "42", "--seed", "42",
               "--output", str(output / "weights.bin"),
               "--checkpoint-dir", str(output / "checkpoints")]
    record = {"schema_version": 1, "status": "running", "argv": command,
              "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
              "trainer_build": build, "positions": len(rows),
              "epochs": args.epochs, "timeout_seconds": args.seconds,
              "positions_sha256": sha256(output / "positions.jsonl"),
              "script_sha256": sha256(Path(__file__)),
              "float_subnormal_policy": "x86-ftz-daz",
              "recipe": "external STM cp regression; random init; no WDL; no internal teacher search"}
    atomic_write_json(output / "run.json", record)
    started = time.monotonic()
    with nonblocking_lock(build_root / ".training.lock", exclusive=True):
        with (output / "train.log").open("w") as log:
            process = subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
                                                SEKIREI_TRAIN_FTZ_DAZ="1"),
                                       start_new_session=True)
            try:
                returncode = process.wait(timeout=args.seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = process.returncode
                record["status"] = "timeout"
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(returncode=returncode, wall_seconds=time.monotonic() - started,
                  cpu_seconds=usage.ru_utime + usage.ru_stime, max_rss_kib=usage.ru_maxrss)
    metadata = list((output / "checkpoints").glob("*.meta.json"))
    weights = output / "weights.bin"
    if returncode == 0 and record["status"] != "timeout":
        try:
            if len(metadata) != args.epochs or not weights.is_file():
                raise RuntimeError("missing epoch metadata or final weight")
            metas = [json.loads(p.read_text()) for p in sorted(metadata)]
            for meta in metas:
                if meta.get("teacher_identity") != manifest["teacher_identity"]:
                    raise RuntimeError("training teacher identity mismatch")
                if meta.get("cache_misses") != 0 or meta.get("cache_hits") != len(rows):
                    raise RuntimeError("training did not consume exactly the cached label set")
                if meta.get("train_count") != len(rows) or meta.get("teacher_eval") != "external":
                    raise RuntimeError("training sample count or label provenance mismatch")
                if meta.get("float_subnormal_policy") != record["float_subnormal_policy"]:
                    raise RuntimeError("training floating point policy mismatch")
            if weights.read_bytes()[:8] != b"SEKIRW01":
                raise RuntimeError("unexpected inference weight format")
            record.update(status="complete", weight_sha256=sha256(weights),
                          weight_bytes=weights.stat().st_size,
                          epoch_metadata=[str(p) for p in sorted(metadata)])
        except (OSError, ValueError, RuntimeError) as error:
            record.update(status="validation_failure", error=str(error))
    elif record["status"] != "timeout":
        record["status"] = "failed"
    record["output_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    atomic_write_json(output / "run.json", record)
    print(json.dumps({k: record.get(k) for k in ("status", "positions", "wall_seconds", "max_rss_kib",
                                               "weight_sha256", "output_bytes", "error")}, indent=2))
    if record["status"] != "complete":
        raise RuntimeError(f"training {record['status']}; inspect {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--lr", type=float, default=0.001)
    train(parser.parse_args())
