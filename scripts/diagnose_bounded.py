#!/usr/bin/env python3
"""Independently bind one bounded-material E3 to the ORIGINAL static holdout.

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
from train_cpu import float32, verify_initial_weights


SCHEMA = "sekirei.bounded-material-diagnosis-run.v1"
TRAINING_SCHEMA = "sekirei.bounded-material-training-run.v1"
PREREGISTRATION_SCHEMA = "sekirei.bounded-material-preregistration.v1"
MODE = "bounded-material-v1"
PLAN_SHA256 = "70d86307d5a23319d70dd6fa8cedb472954f178c3663d4265a7c982a85befd5e"
DATASET_SHA256 = "ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6"
BOUNDED_PATCH_SHA256 = "ee819bd80c41301ee78a0d3ce29efd976c3d762c1ddf07bf6cc259703d38933b"
SOURCE_MAIN_SHA256 = "1e4f89471713c4954de8100179f1fe797793471cf5fc355a047b2752dc2692df"
SOURCE_TRAINER_SHA256 = "b6447b10b0afbe7be4d0f275f5cbe6ab2b97384f8da5170966d2b6f71ff28a43"
COUNTS = {"train": 112681, "holdout": 5895}
TEACHER = "external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d"
TRAINING = {"epochs": 3, "selected_epoch": 3, "learning_rate": 0.0001,
            "min_lr": 0.0, "lr_schedule": "step-half", "lr_schedule_epochs": 3,
            "seed": 42, "shuffle_seed": 42, "timeout_seconds": 1200,
            "float_subnormal_policy": "x86-ftz-daz", "label_depth": 0,
            "teacher_score_cap": 30000, "positions": COUNTS,
            "bounded_mode": MODE, "real_residual_cap_cp": 99}
HELPERS = {"bounded_material.py", "material_init.py", "diagnose_weights.py",
           "diagnose_anchor.py", "export_nearest.py", "functional_anchor.py",
           "train_cpu.py", "prepare_bounded.py", "train_bounded.py",
           "diagnose_bounded.py", "prepare.py", "benchmark.py", "pack_dataset.py"}


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


def f32_bytes(values):
    result = array("f", values)
    require(result.itemsize == 4, "unsupported Python float array")
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def positive_zero(values, name):
    require(f32_bytes(values) == bytes(4 * len(values)), f"protected +0 state changed: {name}")


def validate_bounded_adam(adam, native, expected_step):
    """Pure actual checkpoint proof, independent of the trainer's receipt.

    Restore JSON to binary32 first. Compare all FT/bias raw binary32 values to
    exact dequantization of the fixed seed42 native integers, then inspect
    every protected moment. Native parameters/mask and exact Fraction99cp
    budget are independently verified by bounded_material.validate_weights.
    The input mapping is copied; validation does not mutate caller lists.
    """
    require(type(adam) is dict, "Adam checkpoint must be an object")
    require(type(expected_step) is int and expected_step > 0, "invalid expected Adam step")
    restored = validate_adam(dict(adam))
    require(restored["step"] == expected_step, "Adam step does not match fresh complete epochs")
    native_check = bounded.validate_weights(native)
    nearest, _ = post_export(restored, native)
    require(nearest == native, "actual raw FT has a native/nearest export difference")
    expected_ft = f32_bytes([value / 64.0 for (value,) in
                            struct.iter_unpack("<h", native[8:bounded.L2_OFFSET])])
    actual_ft = f32_bytes(restored["ft"] + restored["ft_bias"])
    require(actual_ft == expected_ft, "actual raw FT/bias binary32 differs from frozen seed42 reference")
    for name in ("ft_m", "ft_v", "bias_m", "bias_v"):
        positive_zero(restored[name], name)
    for name in ("l2_m", "l2_v"):
        values = restored[name]
        for row in range(512):
            start = row * 32
            positive_zero(values[start:start + 4], f"{name}:row{row}:material-columns")
            if row in bounded.MATERIAL_ROWS:
                positive_zero(values[start + 4:start + 32], f"{name}:row{row}:material-to-aux")
    for name in ("l2bias_m", "l2bias_v", "out_m", "out_v"):
        positive_zero(restored[name][:4], name)
    positive_zero([restored["obias_m"], restored["obias_v"]], "out-bias-moments")
    for name in ("ft_v", "bias_v", "l2_v", "l2bias_v", "out_v"):
        require(all(value >= 0.0 for value in restored[name]), f"negative Adam variance: {name}")
    require(restored["obias_v"] >= 0.0, "negative output-bias variance")
    return {"schema": "sekirei.bounded-material-actual-adam-proof.v1",
            "actual_raw_ft_binary32_fixed": True, "protected_moments_positive_zero": True,
            "raw_native_reexport_all_bytes_equal": True, "raw_nearest_export_all_bytes_equal": True,
            "expected_step": expected_step, "actual_step": restored["step"],
            "raw_ft_and_bias_sha256": digest(actual_ft), "native_validation": native_check,
            "intermediate_activations_verified": False,
            "universal_integer_core_bound_proven": False, "adoption_claimed": False}


def validate_preregistration(prereg, expected_plan_sha256):
    require(type(prereg) is dict and prereg.get("schema") == PREREGISTRATION_SCHEMA
            and prereg.get("status") == "frozen-before-training", "wrong frozen preregistration")
    require(prereg.get("candidate") == "bounded-material-residual-100cp-e3-v1"
            and sha(expected_plan_sha256) == PLAN_SHA256
            and prereg.get("plan_sha256") == PLAN_SHA256, "fixed candidate/plan binding mismatch")
    for name in ("build_manifest_sha256", "dataset_manifest_sha256"):
        sha(prereg.get(name))
    require(prereg.get("teacher_identity") == TEACHER, "original teacher identity mismatch")
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
            "--bounded-material-v1"]



MODE_FIELDS = {"mode": MODE, "mask_version": 1, "fixed_ft": "all",
    "fixed_l2_columns": [0, 1, 2, 3], "fixed_l2_rows": [0, 1, 256, 257],
    "fixed_bias_out": [0, 1, 2, 3], "fixed_out_bias": "+0", "auxiliary_units": 28,
    "residual_budget_cp": 99, "shrink": "uniform-aux-out-f32-margin-2^-20-max8",
    "shrink_moments": False, "optimizer": "fresh-adam", "resume_supported": False,
    "init_seed": 42, "initial_weights_fnv": "5ae69099a8bbcdad",
    "loaded_initial_binding": "core-save-weights-exact-bytes", "label_depth": 0,
    "target": "original-absolute-external-cp", "teacher_identity": TEACHER,
    "integer_core_bound_proven": False}


def validate_bounded_metadata(metadata):
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
    bounded_meta = meta.get("bounded_material_v1")
    validate_bounded_metadata(bounded_meta)



def summarize_bounded_rows(rows, targets):
    require(len(targets) == COUNTS["holdout"], "original holdout must contain exactly 5895 rows")
    summary = lifecycle.summarize_rows(rows, targets)
    for row in rows:
        require(row["raw_float_cp"] == row["quantized_float_cp"],
                "fixed FT raw/native floating prediction changed")
        require(abs(row["inference_cp"] - row["material_cp"]) <= 100,
                "observed core material difference exceeds optional100cp cap")
    summary.update(observed_integer_material_cap_cp=100,
                   maximum_observed_integer_material_difference_cp=max(
                       abs(row["inference_cp"] - row["material_cp"]) for row in rows),
                   raw_native_float_predictions_equal=True,
                   intermediate_activations_verified=False,
                   universal_integer_core_bound_proven=False, incremental_undo_verified=False,
                   search_benchmark_verified=False, adoption_claimed=False)
    return summary


def validate_final_sidecar(native, metadata, epoch3_metadata):
    """Final inference and epoch metadata are intentionally distinct schemas."""
    from export_nearest import fnv1a
    require(type(metadata) is dict and metadata.get("format") == "sekirei-nnue-output-v1"
            and metadata.get("nnue_output") == "absolute"
            and metadata.get("checkpoint_hash") == fnv1a(native)
            and exact_value(metadata.get("bounded_material_v1"), epoch3_metadata.get("bounded_material_v1"))
            and metadata.get("bounded_material_v1", {}).get("teacher_identity") == TEACHER,
            "final sidecar differs from verified E3/mode/teacher/native")


def diagnose(args):
    holder = {"process": None}
    with lifecycle.termination_guard():
        with ExitStack() as resources:
            try:
                _diagnose(args, holder, resources)
            finally:
                if holder["process"] is not None:
                    lifecycle.cleanup_group(holder["process"])


def _diagnose(args, holder, resources):
    # Import only the dedicated verifier. Existing generic builder/diagnostic
    # guard functions remain untouched; no expected-hash monkeypatch occurs.
    from prepare_bounded import verify_build
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
    require(prereg.get("dataset") == str(paths["dataset"])
            and prereg.get("trainer") == str(paths["trainer"]), "preregistered input/runtime path mismatch")
    train_root = paths["train_run"].parent
    require(prereg.get("output") == str(train_root) and paths["train_run"] == train_root / "run.json",
            "training run path differs from frozen output")
    script_paths = {name: REPO / "scripts" / name for name in prereg["source_helpers"]}
    script_paths["diagnose_bounded.py"] = lifecycle.canonical_path(Path(__file__), exists=True)
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
    require(build.get("bounded_patch_sha256") == BOUNDED_PATCH_SHA256
            and build.get("source_main_sha256") == SOURCE_MAIN_SHA256
            and build.get("source_trainer_sha256") == SOURCE_TRAINER_SHA256,
            "actual build differs from the independently reviewed v3 patch/source")
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
                    "argv": expected_argv, "positions": COUNTS["train"], "positions_sha256": digest(positions),
                    "learning_rate": 0.0001, "min_lr": 0.0, "lr_schedule": "step-half",
                    "lr_schedule_epochs": 3, "initial_weights": initial, "epochs": 3,
                    "timeout_seconds": 1200, "script_sha256": prereg["source_helpers"]["train_bounded.py"],
                    "float_subnormal_policy": "x86-ftz-daz", "preregistration_sha256": digest(prereg_bytes),
                    "bounded_mode": MODE, "selected_epoch": 3, "real_residual_cap_cp": 99}
    for name, value in expected_run.items():
        require(name in run and type(run[name]) is type(value) and run[name] == value,
                f"completed training receipt mismatch: {name}")
    require(run.get("cleanup_status") == "ok" and run.get("initial_weights_unchanged") is True
            and run.get("source_input_helpers_unchanged") is True,
            "training cleanup/initial/helper verification did not complete")
    require(type(run.get("trainer_pid")) is int and run["trainer_pid"] > 0
            and type(run.get("trainer_pgid")) is int and run["trainer_pgid"] == run["trainer_pid"],
            "missing actual training process identity")
    require(not lifecycle.group_exists(run["trainer_pgid"]), "training process group is still present")
    final = read(train_root / "weights.bin")
    require(run.get("weight_sha256") == digest(final) and run.get("weight_bytes") == len(final)
            and len(final) == bounded.WEIGHT_BYTES, "final weight byte identity mismatch")
    epochs, metadata_paths = {}, []
    for epoch in (1, 2, 3):
        checkpoint = train_root / f"checkpoints/weights.epoch{epoch}.bin"
        native = read(checkpoint)
        metadata_path, adam_path = checkpoint.with_suffix(".meta.json"), checkpoint.with_suffix(".adam.json")
        meta = load_json(read(metadata_path))
        verify_checkpoint_metadata(checkpoint, meta, TEACHER)
        validate_epoch_metadata(meta, epoch, positions_path=train_root / "positions.jsonl")
        epochs[epoch] = validate_bounded_adam(load_json(read(adam_path)), native, epoch * COUNTS["train"])
        metadata_paths.append(str(metadata_path))
        if epoch == 3:
            require(native == final, "selected E3 differs from final native weights")
    require(run.get("epoch_metadata") == metadata_paths, "training epoch metadata inventory mismatch")
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
        summary = summarize_bounded_rows(rows, targets)
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
