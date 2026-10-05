#!/usr/bin/env python3
"""Train one conditionally activated FanIn509 epoch3 with source-bound provenance.

This dedicated route preserves the existing trainer/export guards. Frozen
coordinates, raw Adam state and stored f32 budgets are checked independently;
none of these checks is a search-performance or adoption decision.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time

from benchmark import atomic_write_json, nonblocking_lock
from diagnose_weights import verified_split, verify_checkpoint_metadata
from export_nearest import fnv1a, load_json
from prepare import REPO, sha256
import fanin509_activation as activation
from fanin509_activation import validate_trigger_reference
from train_cpu import (TrainingCancelled, supervised_training, termination_guard,
                       verify_initial_weights, verify_initial_weights_unchanged)


SCHEMA = "sekirei.bounded-material-fanin509-preregistration.v1"
PLAN_SHA256 = "359cf9daa1a5e57be6c7a2d56b2ecfef5c9e24cfc7648e45f7930544b264ae6e"
DATASET_SHA256 = "ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6"
INITIAL_SHA256 = "bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40"
TEACHER = "external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d"
FIXED = {"epochs": 3, "selected_epoch": 3, "learning_rate": 0.0001,
         "min_lr": 0.0, "lr_schedule": "step-half", "lr_schedule_epochs": 3,
         "seed": 42, "shuffle_seed": 42, "timeout_seconds": 1200,
         "float_subnormal_policy": "x86-ftz-daz", "label_depth": 0,
         "teacher_score_cap": 30000, "positions": {"train": 112681, "holdout": 5895},
         "bounded_mode": "bounded-material-fanin509-v1", "real_residual_cap_cp": 99,
         "l2_effective_lr_denominator": 509, "out_effective_lr_denominator": 1,
         "l2_effective_lr_rule": "f32(epoch_lr)/f32(509)"}
HELPERS = {"train_fanin509.py", "prepare_fanin509.py", "diagnose_fanin509.py", "fanin509_activation.py",
           "bounded_material.py", "material_init.py", "train_cpu.py",
           "diagnose_weights.py", "export_nearest.py", "benchmark.py", "prepare.py",
           "diagnose_anchor.py", "functional_anchor.py", "pack_dataset.py",
           "prepare_bounded.py", "train_bounded.py", "diagnose_bounded.py"}


def exact_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact_value(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact_value(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_preregistration(prereg):
    if (type(prereg) is not dict or prereg.get("schema") != SCHEMA
            or prereg.get("status") != "frozen-before-training"
            or prereg.get("candidate") != "bounded-material-fanin509-100cp-e3-v1"
            or prereg.get("plan_sha256") != PLAN_SHA256
            or prereg.get("dataset_manifest_sha256") != DATASET_SHA256
            or prereg.get("teacher_identity") != TEACHER
            or prereg.get("initial_weights", {}).get("sha256") != INITIAL_SHA256):
        raise ValueError("bounded preregistration/plan/original teacher/init mismatch")
    validate_trigger_reference(prereg.get("trigger"))
    training = prereg.get("training")
    if type(training) is not dict or set(training) != set(FIXED):
        raise ValueError("bounded recipe field set differs from the fixed proposal")
    for key, expected in FIXED.items():
        if not exact_value(training[key], expected):
            raise ValueError(f"bounded recipe mismatch: {key}")
    helpers = prereg.get("source_helpers")
    if type(helpers) is not dict or not HELPERS.issubset(helpers):
        raise ValueError("bounded source helper binding missing")
    for key in ("build_manifest_sha256",):
        digest = prereg.get(key)
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid build manifest identity")
    for name, digest in helpers.items():
        if (type(name) is not str or Path(name).name != name or not name.endswith(".py")
                or type(digest) is not str or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid helper name or identity")
    return training


def verify_helpers(prereg):
    from prepare_fanin509 import verify_public_entry
    actual = verify_public_entry("train_fanin509.py", Path(__file__))
    if sha256(actual) != prereg["source_helpers"]["train_fanin509.py"]:
        raise ValueError("actual training entry source changed")
    activation.verify_public_module(prereg["source_helpers"]["fanin509_activation.py"])
    for name, expected in prereg["source_helpers"].items():
        if sha256(REPO / "scripts" / name) != expected:
            raise ValueError(f"bounded helper changed: {name}")


def training_argv(binary, output, dataset, initial):
    return [str(binary), "--positions", str(output / "positions.jsonl"),
            "--strict-positions", "--external-teacher-id", TEACHER,
            "--teacher-cache", str(dataset / "train.labels.jsonl"),
            "--reuse-teacher-cache", "--cache-only", "--label-depth", "0",
            "--validation-ratio", "0", "--teacher-score-cap", "30000",
            "--nnue-output", "absolute", "--epochs", "3",
            "--lr", "0.0001", "--lr-schedule", "step-half", "--min-lr", "0.0",
            "--lr-schedule-epochs", "3", "--shuffle-seed", "42", "--seed", "42",
            "--output", str(output / "weights.bin"), "--checkpoint-dir",
            str(output / "checkpoints"), "--init-weights", initial["path"],
            "--bounded-material-fanin509-v1"]


def validate_final_sidecar(data, metadata, epoch3, epoch3_metadata):
    from diagnose_fanin509 import validate_final_sidecar as validate_inference_sidecar
    if data != epoch3:
        raise ValueError("final inference bytes differ from selected epoch3")
    validate_inference_sidecar(data, metadata, epoch3_metadata)


def verify_epochs(output, manifest):
    from diagnose_fanin509 import validate_fanin509_adam, validate_epoch_metadata
    from bounded_material import validate_weights
    from export_nearest import validate_adam
    expected_paths = [output / "checkpoints" / f"weights.epoch{epoch}.meta.json"
                      for epoch in (1, 2, 3)]
    if sorted((output / "checkpoints").glob("*.meta.json")) != expected_paths:
        raise ValueError("exactly three fixed epoch metadata files required")
    receipts = []
    for epoch, path in enumerate(expected_paths, 1):
        native = path.with_suffix("").with_suffix(".bin")
        adam_path = native.with_suffix(".adam.json")
        meta = load_json(path.read_bytes())
        verify_checkpoint_metadata(native, meta, TEACHER)
        validate_epoch_metadata(meta, epoch, positions_path=output / "positions.jsonl")
        data = native.read_bytes()
        validation = validate_fanin509_adam(validate_adam(load_json(adam_path.read_bytes())),
                                           data, epoch * 112681)
        validate_weights(data)
        receipts.append({"epoch": epoch, "metadata_sha256": sha256(path),
                         "native_sha256": sha256(native), "adam_sha256": sha256(adam_path),
                         "bounded_adam_validation": validation})
    final = output / "weights.bin"
    validate_final_sidecar(final.read_bytes(), load_json(final.with_suffix(".meta.json").read_bytes()),
                           (output / "checkpoints/weights.epoch3.bin").read_bytes(), meta)
    return expected_paths, receipts


def stop_recorded_group(record):
    """Check descendants after the shared supervisor has reaped its leader."""
    import diagnose_anchor as lifecycle
    pgid = record.get("trainer_pgid")
    if pgid is None:
        return
    if type(pgid) is not int or pgid <= 0:
        raise ValueError("invalid recorded trainer process group")
    with lifecycle.blocked_termination():
        if lifecycle.group_exists(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5
            while lifecycle.group_exists(pgid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if lifecycle.group_exists(pgid):
                record["cleanup_status"] = "failed"
                raise RuntimeError("trainer process group remains after bounded cleanup")
            record["unexpected_descendants_terminated"] = True
        record["trainer_process_group_stopped"] = True


def save_failure(record, error, output, started):
    record.update(status="cancelled" if isinstance(error, TrainingCancelled) else "failed",
                  error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic() - started)
    try:
        stop_recorded_group(record)
    except BaseException as cleanup_error:
        record.update(status="cleanup_failure", cleanup_status="failed",
                      cleanup_error=f"{type(cleanup_error).__name__}: {cleanup_error}")
    finally:
        atomic_write_json(output / "run.json", record)


def train(args):
    from prepare_fanin509 import PRIVATE, canonical, outside_git, verify_build, require_runtime_ready, verify_public_entry
    require_runtime_ready()
    verify_public_entry("train_fanin509.py", Path(__file__))
    os.umask(0o077)
    prereg_path = canonical(args.preregistration, exists=True)
    if sha256(prereg_path) != args.preregistration_sha256:
        raise ValueError("preregistration file changed")
    prereg = load_json(prereg_path.read_bytes()); validate_preregistration(prereg)
    dataset = canonical(Path(prereg["dataset"]), exists=True)
    runtime = canonical(Path(prereg["trainer"]), exists=True)
    output = Path(prereg["output"])
    if (not output.is_absolute() or output.resolve() != output or not output.is_relative_to(runtime)
            or output == runtime or output.is_relative_to(runtime / "source")
            or output.is_relative_to(runtime / "build") or os.path.lexists(output)):
        raise ValueError("use a new private run inside the dedicated trainer runtime")
    if not runtime.is_relative_to(PRIVATE) or runtime == PRIVATE:
        raise ValueError("dedicated trainer must remain in the private runtime")
    initial_path = canonical(Path(prereg["initial_weights"]["path"]), exists=True)
    for path in (runtime, dataset, prereg_path.parent, initial_path.parent):
        outside_git(path)
    for protected in (dataset, prereg_path, initial_path, initial_path.with_suffix(".meta.json")):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("training output overlaps a frozen input")
    if shutil.disk_usage(runtime).free < 5 * 2**30:
        raise ValueError("5 GiB SSD budget required before bounded training")
    verify_helpers(prereg)
    with ExitStack() as locks:
        for name in (".build.lock", ".training.lock"):
            locks.enter_context(nonblocking_lock(runtime / name, exclusive=True))
        activation_proof = activation.verify_activation(prereg["trigger"])
        build = verify_build(runtime, prereg["build_manifest_sha256"])
        activation.validate_build_binding(build, prereg["trigger"],
            prereg["source_helpers"]["fanin509_activation.py"], activation_proof)
        if sha256(dataset / "manifest.json") != DATASET_SHA256:
            raise ValueError("original recovered dataset manifest changed")
        manifest, targets = verified_split(dataset, "train")
        verified_split(dataset, "holdout")
        if manifest["positions"] != FIXED["positions"] or len(targets) != 112681 or manifest["teacher_identity"] != TEACHER:
            raise ValueError("original O dataset split/identity changed")
        initial = verify_initial_weights(initial_path)
        if initial != prereg["initial_weights"]:
            raise ValueError("initial weights/sidecar identity changed")
        output.mkdir(mode=0o700)
        shutil.copyfile(dataset / "train.positions.jsonl", output / "positions.jsonl")
        positions_sha256 = sha256(output / "positions.jsonl")
        if positions_sha256 != manifest["files"]["train.positions.jsonl"]["sha256"]:
            raise ValueError("actual training position copy differs from the frozen dataset")
        command = training_argv(build["binary"], output, dataset, initial)
        record = {"schema": "sekirei.bounded-material-fanin509-training-run.v1", "schema_version": 1,
            "status": "running", "argv": command, "trainer_build": build,
            "trigger": prereg["trigger"], "activation_validation": activation_proof,
            "preregistration_sha256": args.preregistration_sha256,
            "dataset_manifest_sha256": DATASET_SHA256, "positions": 112681,
            "positions_sha256": positions_sha256,
            "script_sha256": sha256(Path(__file__)), "initial_weights": initial,
            "initialization": "inference weights; fresh Adam state",
            **{k: v for k, v in FIXED.items() if k != "positions"}}
        atomic_write_json(output / "run.json", record)
        with termination_guard() as outer_state:
            started = time.monotonic()
            try:
                with (output / "train.log").open("x") as log:
                    outcome = supervised_training(command, output, log, 1200, record)
                record.update(outcome, wall_seconds=time.monotonic() - started)
                stop_recorded_group(record)
                if record.get("unexpected_descendants_terminated"):
                    raise RuntimeError("trainer left descendants after exit")
                if outcome["returncode"] != 0 or outcome["status"] != "finished":
                    raise RuntimeError(f"bounded trainer ended: {outcome}")
                verify_initial_weights_unchanged(initial)
                verify_helpers(prereg)
                if sha256(prereg_path) != args.preregistration_sha256:
                    raise ValueError("preregistration changed during training")
                if not exact_value(activation.verify_activation(prereg["trigger"]), activation_proof):
                    raise ValueError("activation evidence changed during training")
                if not exact_value(verify_build(runtime, prereg["build_manifest_sha256"]), build):
                    raise ValueError("dedicated build changed during training")
                verified_split(dataset, "train"); verified_split(dataset, "holdout")
                if sha256(dataset / "manifest.json") != DATASET_SHA256:
                    raise ValueError("dataset manifest changed during training")
                if sha256(output / "positions.jsonl") != positions_sha256:
                    raise ValueError("actual training position input changed during training")
                paths, receipts = verify_epochs(output, manifest)
                usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                record.update(status="complete", initial_weights_unchanged=True,
                              epoch_metadata=[str(p) for p in paths], epoch_validation=receipts,
                              weight_sha256=sha256(output / "weights.bin"),
                              weight_bytes=(output / "weights.bin").stat().st_size,
                              max_rss_kib=usage.ru_maxrss, cpu_seconds=usage.ru_utime + usage.ru_stime,
                              source_input_helpers_unchanged=True, adoption_verified=False)
                record["output_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                outer_state["cleaning"] = True
                atomic_write_json(output / "run.json", record)
                outer_state["cleaning"] = False
                if outer_state["signal"] is not None:
                    raise TrainingCancelled("cancelled during final training receipt")
            except BaseException as error:
                outer_state["cleaning"] = True
                try:
                    save_failure(record, error, output, started)
                finally:
                    outer_state["cleaning"] = False
                raise
            print(json.dumps({k: record[k] for k in ("status", "weight_sha256", "wall_seconds", "output_bytes")}))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    train(parser.parse_args())
