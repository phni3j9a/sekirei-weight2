#!/usr/bin/env python3
"""Independently bind one activated FanIn509 E3 to the ORIGINAL static holdout.

A separate pinned build and preregistration are required before execution.
This module never changes existing source guards or expected-hash constants.
Its reusable Adam validator performs no I/O and never creates model artifacts.
An executed diagnostic is static evidence on the fixed 5,895 holdout rows;
it is not a search benchmark, a universal integer bound or adoption evidence.
"""
import argparse
from array import array
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import time

from benchmark import atomic_write_json, nonblocking_lock
import bounded_material as bounded
import diagnose_anchor as lifecycle
from diagnose_weights import verified_split, verify_checkpoint_metadata
from export_nearest import SOURCE_HASHES, load_json, post_export, validate_adam
import functional_anchor as original
from prepare import DEFAULT_RUNTIME, REPO, sha256
import fanin509_activation as activation
from train_cpu import float32, verify_initial_weights


SCHEMA = "sekirei.bounded-material-fanin509-diagnosis-run.v1"
TRAINING_SCHEMA = "sekirei.bounded-material-fanin509-training-run.v1"
PREREGISTRATION_SCHEMA = "sekirei.bounded-material-fanin509-preregistration.v1"
MODE = "bounded-material-fanin509-v1"
PLAN_SHA256 = "359cf9daa1a5e57be6c7a2d56b2ecfef5c9e24cfc7648e45f7930544b264ae6e"
DATASET_SHA256 = "ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6"
FANIN509_PATCH_SHA256 = "d5da76c8f316a6df4db3b23293feb47e9568bc75a7707a63fdce3c3b183d70c7"
SOURCE_MAIN_SHA256 = "21834567c71a2b6604422980a21f5b5577cc9af9ca839644e8e3abdc116955e2"
SOURCE_TRAINER_SHA256 = "0da66b6880785cb057f996c143bbab6347c23fbfc89f18196551b6a979db064b"
COUNTS = {"train": 112681, "holdout": 5895}
TEACHER = "external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d"
TRAINING = {"epochs": 3, "selected_epoch": 3, "learning_rate": 0.0001,
            "min_lr": 0.0, "lr_schedule": "step-half", "lr_schedule_epochs": 3,
            "seed": 42, "shuffle_seed": 42, "timeout_seconds": 1200,
            "float_subnormal_policy": "x86-ftz-daz", "label_depth": 0,
            "teacher_score_cap": 30000, "positions": COUNTS,
            "bounded_mode": MODE, "real_residual_cap_cp": 99,
            "l2_effective_lr_denominator": 509, "out_effective_lr_denominator": 1,
            "l2_effective_lr_rule": "f32(epoch_lr)/f32(509)"}
HELPERS = {"bounded_material.py", "material_init.py", "diagnose_weights.py",
           "diagnose_anchor.py", "export_nearest.py", "functional_anchor.py",
           "train_cpu.py", "prepare_fanin509.py", "train_fanin509.py",
           "diagnose_fanin509.py", "fanin509_activation.py", "prepare.py", "benchmark.py", "pack_dataset.py",
           "prepare_bounded.py", "train_bounded.py", "diagnose_bounded.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact_value(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact_value(a, b) for a, b in zip(actual, expected))
    return actual == expected


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(value):
    require(type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "invalid SHA-256")
    return value


def validate_fanin509_adam(adam, native, expected_step):
    from diagnose_bounded import validate_bounded_adam
    shared = validate_bounded_adam(adam, native, expected_step)
    return {"schema": "sekirei.fanin509-actual-adam-proof.v1",
            "bounded_mode": MODE, "expected_step": expected_step,
            "shared_parameter_invariant_proof": shared,
            "learning_rate_scaling_proven_by_checkpoint_alone": False,
            "adoption_claimed": False}


def validate_preregistration(prereg, expected_plan_sha256):
    require(type(prereg) is dict and prereg.get("schema") == PREREGISTRATION_SCHEMA
            and prereg.get("status") == "frozen-before-training", "wrong frozen preregistration")
    require(prereg.get("candidate") == "bounded-material-fanin509-100cp-e3-v1"
            and sha(expected_plan_sha256) == PLAN_SHA256
            and prereg.get("plan_sha256") == PLAN_SHA256, "fixed candidate/plan binding mismatch")
    from fanin509_activation import validate_trigger_reference
    validate_trigger_reference(prereg.get("trigger"))
    for name in ("build_manifest_sha256", "dataset_manifest_sha256"):
        sha(prereg.get(name))
    require(prereg.get("teacher_identity") == TEACHER, "original teacher identity mismatch")
    require(prereg.get("initial_weights", {}).get("sha256") == bounded.MATERIAL_SEED42_SHA256,
            "material seed42 initializer declaration changed")
    require(prereg.get("dataset_manifest_sha256") == DATASET_SHA256, "fixed original O manifest changed")
    training = prereg.get("training")
    require(type(training) is dict and set(training) == set(TRAINING), "fixed recipe field set mismatch")
    for name, expected in TRAINING.items():
        require(exact_value(training[name], expected),
                f"fixed bounded training recipe mismatch: {name}")
    helpers = prereg.get("source_helpers")
    require(type(helpers) is dict and HELPERS.issubset(helpers), "missing preregistered helper hashes")
    for name, value in helpers.items():
        require(type(name) is str and Path(name).name == name and name.endswith(".py"), "invalid helper basename")
        sha(value)
    return prereg


def training_argv(binary, train_root, dataset, initial):
    # Same common CLI/order as train_cpu; excluded-mate flag is omitted because
    # bounded mode forbids it and rejects out-of-range original CP explicitly.
    return [str(binary), "--positions", str(train_root / "positions.jsonl"),
            "--strict-positions", "--external-teacher-id", TEACHER,
            "--teacher-cache", str(dataset / "train.labels.jsonl"),
            "--reuse-teacher-cache", "--cache-only", "--label-depth", "0",
            "--validation-ratio", "0", "--teacher-score-cap", "30000",
            "--nnue-output", "absolute", "--epochs", "3", "--lr", "0.0001",
            "--lr-schedule", "step-half", "--min-lr", "0.0", "--lr-schedule-epochs", "3",
            "--shuffle-seed", "42", "--seed", "42", "--output", str(train_root / "weights.bin"),
            "--checkpoint-dir", str(train_root / "checkpoints"), "--init-weights", str(initial),
            "--bounded-material-fanin509-v1"]



MODE_FIELDS = {"mode": MODE, "mask_version": 1, "fixed_ft": "all",
    "fixed_l2_columns": [0, 1, 2, 3], "fixed_l2_rows": [0, 1, 256, 257],
    "fixed_bias_out": [0, 1, 2, 3], "fixed_out_bias": "+0", "auxiliary_units": 28,
    "residual_budget_cp": 99, "shrink": "uniform-aux-out-f32-margin-2^-20-max8",
    "shrink_moments": False, "optimizer": "fresh-adam", "resume_supported": False,
    "init_seed": 42, "initial_weights_fnv": "5ae69099a8bbcdad",
    "loaded_initial_binding": "core-save-weights-exact-bytes", "label_depth": 0,
    "target": "original-absolute-external-cp", "teacher_identity": TEACHER,
    "integer_core_bound_proven": False,
    "l2_effective_lr_denominator": 509, "out_effective_lr_denominator": 1}


def validate_fanin509_metadata(metadata):
    upper_fields = {"saved_f32_l1_upper", "saved_f32_budget_cp_upper"}
    require(type(metadata) is dict and set(metadata) == set(MODE_FIELDS) | upper_fields,
            "bounded metadata field set mismatch")
    for name, expected in MODE_FIELDS.items():
        require(exact_value(metadata[name], expected), f"bounded metadata mismatch: {name}")
    for name in upper_fields:
        value = metadata[name]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "nonfinite or negative bounded metadata upper estimate")
    require(metadata["saved_f32_budget_cp_upper"] <= 99,
            "bounded metadata residual estimate exceeds 99 cp")


def validate_epoch_metadata(meta, epoch, *, positions_path):
    require(type(epoch) is int and epoch in (1, 2, 3), "fixed epoch must be int 1..3")
    require(type(meta) is dict and "bounded_material_v1" not in meta, "legacy/new metadata mode collision")
    expected = {"epoch": epoch, "epochs": 3, "train_count": COUNTS["train"], "valid_count": 0,
                "cache_hits": COUNTS["train"], "cache_misses": 0, "init_seed": 42,
                "split_seed": 42, "shuffle_seed": 42, "lr_schedule_epochs": 3,
                "warmup_epochs": 0, "label_depth": 0, "source_cap": 0, "split_hash": 0,
                "nnue_output": "absolute", "teacher_eval": "external", "teacher_identity": TEACHER,
                "lr_schedule": "StepHalf", "float_subnormal_policy": "x86-ftz-daz",
                "architecture": "INPUT=2420 L1=256 L2=32", "cache_only": True,
                "exclude_mate_labels": False, "side_balance": False, "games_dir": None,
                "teacher_weights": None, "wdl_lambda": None, "phase_weights": {},
                "validation_ratio": 0.0, "teacher_score_cap": 30000.0,
                "search_target_weight": 1.0, "lr": float32(0.0001), "min_lr": float32(0.0),
                "positions": str(positions_path)}
    for name, value in expected.items():
        actual = meta.get(name)
        allowed = (int, float) if type(value) is float else (type(value),)
        require(name in meta and type(actual) in allowed and actual == value,
                f"epoch{epoch} metadata mismatch: {name}")
    bounded_meta = meta.get("bounded_material_fanin509_v1")
    validate_fanin509_metadata(bounded_meta)



def summarize_fanin509_rows(rows, targets):
    from diagnose_bounded import summarize_bounded_rows
    summary = summarize_bounded_rows(rows, targets)
    summary.update(bounded_mode=MODE, l2_effective_lr_denominator=509,
                   out_effective_lr_denominator=1,
                   learning_rate_scaling_proven_by_predictions_alone=False)
    return summary


def validate_final_sidecar(native, metadata, epoch3_metadata):
    """Final inference and epoch metadata remain distinct schemas."""
    from export_nearest import fnv1a
    require(type(metadata) is dict and set(metadata) == {
                "format", "nnue_output", "checkpoint_hash", "baseline", "bounded_material_fanin509_v1"}
            and metadata["baseline"] is None and metadata["format"] == "sekirei-nnue-output-v1"
            and metadata["nnue_output"] == "absolute" and metadata["checkpoint_hash"] == fnv1a(native),
            "final inference format/FNV/baseline differs from fresh absolute candidate")
    validate_fanin509_metadata(metadata["bounded_material_fanin509_v1"])
    require(exact_value(metadata["bounded_material_fanin509_v1"],
                        epoch3_metadata.get("bounded_material_fanin509_v1")),
            "final sidecar differs from verified E3 mode/teacher")


def diagnose(args):
    from prepare_fanin509 import require_runtime_ready
    require_runtime_ready()
    os.umask(0o077)
    holder = {"process": None}
    with lifecycle.termination_guard():
        with ExitStack() as resources:
            try:
                _diagnose(args, holder, resources)
            finally:
                if holder["process"] is not None:
                    lifecycle.cleanup_group(holder["process"])


def _diagnose(args, holder, resources):
    from prepare_fanin509 import require_runtime_ready, verify_public_entry
    require_runtime_ready()
    verify_public_entry("diagnose_fanin509.py", Path(__file__))
    # Import only the dedicated verifier. Existing generic builder/diagnostic
    # guard functions remain untouched; no expected-hash monkeypatch occurs.
    from prepare_fanin509 import verify_build
    require(type(args.seconds) is int and 1 <= args.seconds <= 600, "diagnostic timeout must be 1..600s")
    paths = {name: lifecycle.canonical_path(getattr(args, name), exists=True)
             for name in ("dataset", "trainer", "train_run", "preregistration", "private_output_root")}
    output = lifecycle.canonical_path(args.output)
    protected = [path for name, path in paths.items() if name != "private_output_root"]
    lifecycle.output_guard(output, paths["private_output_root"], protected)
    hashes, sizes = {}, {}

    def read(path):
        path = lifecycle.canonical_path(path, exists=True)
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "input must be a canonical regular file")
        data = path.read_bytes()
        actual = digest(data)
        require(str(path) not in hashes or hashes[str(path)] == actual, "input changed during preparation")
        hashes[str(path)], sizes[str(path)] = actual, len(data)
        return data

    def unchanged():
        require(exact_value(activation.verify_activation(prereg["trigger"]), activation_proof),
                "activation evidence changed during diagnosis")
        for name in hashes:
            path = lifecycle.canonical_path(Path(name), exists=True)
            require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_size == sizes[name]
                    and sha256(path) == hashes[name], "diagnostic input/source/build changed")
        require(verify_build(paths["trainer"], prereg["build_manifest_sha256"]) == build,
                "dedicated build changed during diagnosis")
        return dict(hashes)

    for name in (".build.lock", ".training.lock"):
        resources.enter_context(nonblocking_lock(paths["trainer"] / name, exclusive=True))
    prereg_bytes = read(paths["preregistration"])
    require(digest(prereg_bytes) == sha(args.expected_preregistration_sha256), "preregistration SHA mismatch")
    prereg = validate_preregistration(load_json(prereg_bytes), args.expected_plan_sha256)
    activation.verify_public_module(prereg["source_helpers"]["fanin509_activation.py"])
    activation_proof = activation.verify_activation(prereg["trigger"])
    for name in activation_proof["receipt_and_stopped_input_identities"]:
        read(Path(name))
    require(prereg.get("dataset") == str(paths["dataset"])
            and prereg.get("trainer") == str(paths["trainer"]), "preregistered input/runtime path mismatch")
    train_root = paths["train_run"].parent
    require(prereg.get("output") == str(train_root) and paths["train_run"] == train_root / "run.json",
            "training run path differs from frozen output")
    script_paths = {name: REPO / "scripts" / name for name in prereg["source_helpers"]}
    script_paths["diagnose_fanin509.py"] = lifecycle.canonical_path(Path(__file__), exists=True)
    for name, path in script_paths.items():
        require(digest(read(path)) == prereg["source_helpers"][name], f"preregistered helper changed: {name}")
    read(REPO / "config/toolchain.lock.json")
    # Preserve the original full source guard as an algorithm-reference check.
    # Bind the actual patched trainer independently, never substitute its hash
    # into SOURCE_HASHES or pretend it was the unmodified upstream trainer.
    reference_source = DEFAULT_RUNTIME / "sources/sekirei"
    for name, expected in SOURCE_HASHES.items():
        require(digest(read(reference_source / name)) == expected, f"original source guard failed: {name}")
    build_path = paths["trainer"] / "build-manifest.json"
    require(digest(read(build_path)) == prereg["build_manifest_sha256"], "dedicated build manifest SHA mismatch")
    build = verify_build(paths["trainer"], prereg["build_manifest_sha256"])
    activation.validate_build_binding(build, prereg["trigger"],
        prereg["source_helpers"]["fanin509_activation.py"], activation_proof)
    require(build.get("fanin509_patch_sha256") == FANIN509_PATCH_SHA256
            and build.get("source_main_sha256") == SOURCE_MAIN_SHA256
            and build.get("source_trainer_sha256") == SOURCE_TRAINER_SHA256,
            "actual build differs from the independently reviewed FanIn509 patch/source")
    read(Path(build["identity_document_path"]))
    for name in build.get("logs", {}):
        read(paths["trainer"] / name)
    for name in build.get("compiler_files", {}):
        read(Path(name))
    binary = lifecycle.canonical_path(Path(build["binary"]), exists=True)
    require(binary.is_relative_to(paths["trainer"] / "build") and os.access(binary, os.X_OK),
            "dedicated binary must be executable inside its actual build")
    read(binary)
    for name in build["source_files"]:
        read(paths["trainer"] / "source" / name)
    for name in build.get("deps_files", {}):
        read(paths["trainer"] / "source" / name)
    for name, expected in SOURCE_HASHES.items():
        if name != "crates/sekirei-train/src/trainer.rs":
            require(hashes[str(paths["trainer"] / "source" / name)] == expected,
                    f"actual fixed core/checkpoint source changed: {name}")
    manifest_bytes = read(paths["dataset"] / "manifest.json")
    original_files = {name: read(paths["dataset"] / name) for name in original.FILES}
    manifest, parsed = original.validate_original(manifest_bytes, original_files,
                            expected_manifest_sha256=prereg["dataset_manifest_sha256"])
    del parsed
    require(manifest["positions"] == COUNTS and manifest["teacher_identity"] == TEACHER,
            "original split/count/teacher differs from fixed O")
    verified, targets = verified_split(paths["dataset"], "holdout")
    require(verified == manifest and len(targets) == 5895, "holdout loader disagrees with fixed O")
    run = load_json(read(paths["train_run"]))
    initial_info = prereg.get("initial_weights")
    require(type(initial_info) is dict, "missing preregistered initial weights")
    initial_path = lifecycle.canonical_path(Path(initial_info["path"]), exists=True)
    initial_bytes = read(initial_path)
    read(initial_path.with_suffix(".meta.json"))
    initial = verify_initial_weights(initial_path)
    require(initial == initial_info and digest(initial_bytes) == bounded.MATERIAL_SEED42_SHA256,
            "initial material seed42/metadata binding mismatch")
    positions = read(train_root / "positions.jsonl")
    require(positions == original_files["train.positions.jsonl"], "training positions bytes/order changed")
    expected_argv = training_argv(binary, train_root, paths["dataset"], initial_path)
    expected_run = {"schema": TRAINING_SCHEMA, "status": "complete", "returncode": 0,
                    "dataset_manifest_sha256": prereg["dataset_manifest_sha256"], "trainer_build": build,
                    "trigger": prereg["trigger"], "activation_validation": activation_proof,
                    "argv": expected_argv, "positions": COUNTS["train"], "positions_sha256": digest(positions),
                    "learning_rate": 0.0001, "min_lr": 0.0, "lr_schedule": "step-half",
                    "lr_schedule_epochs": 3, "initial_weights": initial, "epochs": 3,
                    "timeout_seconds": 1200, "script_sha256": prereg["source_helpers"]["train_fanin509.py"],
                    "float_subnormal_policy": "x86-ftz-daz", "preregistration_sha256": digest(prereg_bytes),
                    "bounded_mode": MODE, "selected_epoch": 3, "real_residual_cap_cp": 99}
    expected_run.update({key: value for key, value in TRAINING.items() if key != "positions"})
    for name, value in expected_run.items():
        require(name in run and exact_value(run[name], value),
                f"completed training receipt mismatch: {name}")
    require(run.get("cleanup_status") == "ok" and run.get("initial_weights_unchanged") is True
            and run.get("source_input_helpers_unchanged") is True
            and run.get("trainer_process_group_stopped") is True,
            "training cleanup/initial/helper verification did not complete")
    require(type(run.get("trainer_pid")) is int and run["trainer_pid"] > 0
            and type(run.get("trainer_pgid")) is int and run["trainer_pgid"] == run["trainer_pid"],
            "missing actual training process identity")
    require(not lifecycle.group_exists(run["trainer_pgid"]), "training process group is still present")
    final = read(train_root / "weights.bin")
    require(run.get("weight_sha256") == digest(final) and run.get("weight_bytes") == len(final)
            and len(final) == bounded.WEIGHT_BYTES, "final weight byte identity mismatch")
    epochs, metadata_paths, expected_epoch_validation = {}, [], []
    for epoch in (1, 2, 3):
        checkpoint = train_root / f"checkpoints/weights.epoch{epoch}.bin"
        native = read(checkpoint)
        metadata_path, adam_path = checkpoint.with_suffix(".meta.json"), checkpoint.with_suffix(".adam.json")
        meta = load_json(read(metadata_path))
        verify_checkpoint_metadata(checkpoint, meta, TEACHER)
        validate_epoch_metadata(meta, epoch, positions_path=train_root / "positions.jsonl")
        epochs[epoch] = validate_fanin509_adam(load_json(read(adam_path)), native, epoch * COUNTS["train"])
        metadata_paths.append(str(metadata_path))
        expected_epoch_validation.append({"epoch": epoch,
            "metadata_sha256": hashes[str(metadata_path)], "native_sha256": digest(native),
            "adam_sha256": hashes[str(adam_path)], "bounded_adam_validation": epochs[epoch]})
        if epoch == 3:
            require(native == final, "selected E3 differs from final native weights")
    require(run.get("epoch_metadata") == metadata_paths, "training epoch metadata inventory mismatch")
    require(exact_value(run.get("epoch_validation"), expected_epoch_validation),
            "all three training checkpoint proof receipts differ from actual files")
    validate_final_sidecar(final, load_json(read(train_root / "weights.meta.json")), meta)
    require(sorted(path.name for path in (train_root / "checkpoints").glob("*.meta.json"))
            == [f"weights.epoch{epoch}.meta.json" for epoch in (1, 2, 3)], "unexpected checkpoint metadata files")
    checkpoint = train_root / "checkpoints/weights.epoch3.bin"
    lifecycle.output_guard(output, paths["private_output_root"], protected +
                           [initial_path, binary, reference_source, *script_paths.values()])
    command = lifecycle.diagnosis_argv(binary, paths["dataset"], checkpoint, TEACHER)
    unchanged()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    record = {"schema": SCHEMA, "status": "running", "selected_epoch": 3,
              "split": "original-holdout", "count": 5895, "teacher_identity": TEACHER,
              "bounded_mode": MODE, "preregistration_sha256": digest(prereg_bytes),
              "trigger": prereg["trigger"], "activation_validation": activation_proof,
              "argv": command, "input_sha256": hashes, "input_bytes": sizes,
              "actual_adam_proofs": epochs, "trainer_build": build,
              "float_subnormal_policy": "x86-ftz-daz", "timeout_seconds": args.seconds,
              "rust_native_reexport_verified": False, "cleanup_verified": False,
              "universal_integer_core_bound_proven": False, "adoption_claimed": False}
    atomic_write_json(output / "run.json", record)
    started = time.monotonic()
    try:
        with (output / "predictions.jsonl").open("x") as predictions, (output / "diagnose.log").open("x") as log:
            lifecycle.spawn_process(command, output, predictions, log, holder)
            process = holder["process"]
            try:
                returncode = process.wait(timeout=args.seconds)
            except subprocess.TimeoutExpired:
                record["status"] = "timeout"
                lifecycle.cleanup_group(process)
                raise RuntimeError("bounded original-holdout diagnostic exceeded wall limit")
        record["returncode"] = returncode
        if lifecycle.group_exists(process.pid):
            lifecycle.cleanup_group(process)
            raise RuntimeError("diagnostic descendants remained after process exit")
        record["cleanup_verified"] = True
        require(returncode == 0, f"bounded diagnostic exited with {returncode}")
        rows = [load_json(line) for line in (output / "predictions.jsonl").read_bytes().splitlines()]
        summary = summarize_fanin509_rows(rows, targets)
        unchanged()
        summary.update(checkpoint_sha256=hashes[str(checkpoint)],
                       adam_sha256=hashes[str(checkpoint.with_suffix(".adam.json"))],
                       dataset_manifest_sha256=prereg["dataset_manifest_sha256"], selected_epoch=3)
        atomic_write_json(output / "summary.json", summary)
        record.update(status="complete", rust_native_reexport_verified=True,
                      predictions_sha256=sha256(output / "predictions.jsonl"),
                      summary_sha256=sha256(output / "summary.json"))
    except BaseException as error:
        if isinstance(error, lifecycle.DiagnosticCancelled):
            record["status"] = "cancelled"
        elif record["status"] != "timeout":
            record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        try:
            if holder["process"] is not None:
                lifecycle.cleanup_group(holder["process"])
            record["cleanup_verified"] = True
            record["input_sha256_after"] = unchanged()
            record["input_unchanged"] = True
        except BaseException as error:
            record.update(status="final_verification_failure", input_unchanged=False, final_error=str(error))
        record["wall_seconds"] = time.monotonic() - started
        atomic_write_json(output / "run.json", record)
    require(record["status"] == "complete", "diagnostic did not satisfy all final guards")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "trainer", "train-run", "preregistration", "private-output-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--expected-plan-sha256", default=PLAN_SHA256)
    parser.add_argument("--seconds", type=int, default=120)
    diagnose(parser.parse_args())
