#!/usr/bin/env python3
"""Diagnose a preregistered anchor epoch3 against ORIGINAL teacher holdout.

The checkpoint's training target identity D and the evaluation teacher O are
validated separately. This route never changes diagnose_weights.py's guard or
the holdout labels. A completed result is a static diagnostic, not a real-data
generation preflight, a search benchmark, or an adoption decision.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path
import signal
import stat
import statistics
import subprocess
import time

from benchmark import atomic_write_json, nonblocking_lock
from diagnose_weights import PREDICTIONS, metrics, verified_split, verify_checkpoint_metadata
from export_nearest import SOURCE_HASHES, load_json, post_export, validate_adam
import functional_anchor as anchor
from prepare import LOCK, REPO, sha256
from train_cpu import NNUE_WEIGHT_BYTES, float32, verify_initial_weights


BINDING_SCHEMA = "sekirei.functional-anchor-diagnosis-binding.v1"
SHA_FIELDS = ("original_manifest_sha256", "spec_sha256", "trainer_build_manifest_sha256",
              "train_script_sha256", "initial_weights_sha256", "initial_metadata_sha256")
FIXED = {"epochs": 3, "selected_epoch": 3, "learning_rate": 0.0001, "min_lr": 0,
         "lr_schedule": "step-half", "lr_schedule_epochs": 3, "seed": 42, "shuffle_seed": 42,
         "float_subnormal_policy": "x86-ftz-daz", "timeout_seconds": 1200}


class DiagnosticCancelled(BaseException):
    """A termination request that must pass through process-group cleanup."""


@contextmanager
def termination_guard():
    signals = (signal.SIGTERM, signal.SIGINT)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    def cancel(sig, _frame):
        # A second termination request must not interrupt the bounded cleanup
        # and final receipt for the first request.
        for item in signals:
            signal.signal(item, signal.SIG_IGN)
        raise DiagnosticCancelled(f"diagnostic cancelled by signal {sig}")
    try:
        for sig in signals:
            signal.signal(sig, cancel)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def blocked_termination():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


@contextmanager
def deferred_termination():
    """Defer raising while Popen obtains its handle; do not mask the child."""
    signals = (signal.SIGTERM, signal.SIGINT)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    pending = []
    def defer(sig, _frame):
        pending.append(sig)
    try:
        for sig in signals:
            signal.signal(sig, defer)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if pending:
            sig = pending[0]
            if callable(previous[sig]):
                previous[sig](sig, None)
            raise DiagnosticCancelled(f"diagnostic cancelled during spawn by signal {sig}")


def validate_binding(preregistration, spec, original_manifest, derived_manifest):
    """Pure guard; caller binds the complete preregistration bytes before this."""
    binding = preregistration.get("diagnosis_binding")
    if type(binding) is not dict or binding.get("schema") != BINDING_SCHEMA:
        raise ValueError("missing versioned preregistered diagnosis binding")
    for name in SHA_FIELDS:
        anchor._sha(binding.get(name))
    for name, expected in FIXED.items():
        value = binding.get(name)
        numeric = name in ("learning_rate", "min_lr")
        if (type(value) not in ((int, float) if numeric else (type(expected),))
                or value != expected or (numeric and not math.isfinite(value))):
            raise ValueError(f"preregistered recipe mismatch: {name}")
    if (type(binding.get("positions")) is not dict or set(binding["positions"]) != {"train", "holdout"}
            or any(type(value) is not int or value < 1 for value in binding["positions"].values())):
        raise ValueError("invalid preregistered split counts")
    original, target = original_manifest["teacher_identity"], anchor.derived_identity(spec)
    if (binding["original_manifest_sha256"] != spec["original_manifest_sha256"]
            or binding["spec_sha256"] != anchor.spec_sha256(spec)
            or binding.get("original_teacher_identity") != original
            or binding.get("training_target_identity") != target or target == original
            or binding["positions"] != original_manifest["positions"]
            or derived_manifest["positions"] != original_manifest["positions"]
            or derived_manifest["teacher_identity"] != target
            or derived_manifest.get("split_teacher_identities") != {"train": target, "holdout": original}
            or binding["initial_weights_sha256"] != anchor.MATERIAL_INIT_SHA256):
        raise ValueError("preregistered input/spec/material or D/O identity mismatch")
    return binding


def training_argv(binary, train_root, dataset, initial_path, target_identity):
    """Exact existing train_cpu.py recipe; diagnostic commands never use it."""
    return [str(binary), "--positions", str(train_root / "positions.jsonl"),
            "--strict-positions", "--external-teacher-id", target_identity,
            "--teacher-cache", str(dataset / "train.labels.jsonl"),
            "--reuse-teacher-cache", "--cache-only", "--label-depth", "0",
            "--validation-ratio", "0", "--teacher-score-cap", "30000",
            "--exclude-mate-labels", "--nnue-output", "absolute", "--epochs", "3",
            "--lr", "0.0001", "--lr-schedule", "step-half", "--min-lr", "0.0",
            "--lr-schedule-epochs", "3", "--shuffle-seed", "42", "--seed", "42",
            "--output", str(train_root / "weights.bin"), "--checkpoint-dir",
            str(train_root / "checkpoints"), "--init-weights", str(initial_path)]


def validate_build(build, binding, *, manifest_sha256, binary_sha256, patch_sha256, main_sha256):
    """Pure build binding guard; disk/source/Git verification remains separate."""
    for digest in (manifest_sha256, binary_sha256, patch_sha256, main_sha256):
        anchor._sha(digest)
    if (manifest_sha256 != binding["trainer_build_manifest_sha256"]
            or build.get("upstream_commit") != LOCK["sources"]["sekirei"]["commit"]
            or build.get("rustflags") != LOCK["rustflags"]
            or binary_sha256 != build.get("binary_sha256")
            or patch_sha256 != build.get("patch_sha256")
            or main_sha256 != build.get("source_main_sha256")):
        raise ValueError("trainer build/commit/rustflags/binary/patch/main identity mismatch")


def validate_source_helpers(preregistration, actual_sha256, spec):
    required = {"functional_anchor.py", "material_init.py", "train_cpu.py", "diagnose_weights.py"}
    declared = preregistration.get("source_helpers")
    if (type(declared) is not dict or not required.issubset(declared)
            or not required.issubset(actual_sha256)):
        raise ValueError("missing preregistered helper implementation bindings")
    for name in required:
        if anchor._sha(declared[name]) != anchor._sha(actual_sha256[name]):
            raise ValueError(f"helper differs from preregistered implementation: {name}")
    if (actual_sha256["material_init.py"] != anchor.MATERIAL_IMPLEMENTATION_SHA256
            or actual_sha256["material_init.py"] != spec["material"]["implementation_sha256"]):
        raise ValueError("material implementation differs from fixed fallback spec")


def validate_training(run, binding, derived_manifest, build, initial, epochs, *,
                      derived_manifest_sha256, positions_sha256, final_sha256,
                      final_bytes, expected_argv, expected_metadata_paths):
    """Pure contract check after all checkpoint/Adam/FNV bytes are verified.

    epochs maps 1..3 to metadata/adam_step/weight_sha256. No claim about a
    checkpoint is made merely because a caller supplies these summaries.
    The filesystem adapter constructs them from actual files before this guard.
    """
    if (type(run.get("schema_version")) is not int or run["schema_version"] != 1
            or run.get("status") != "complete" or type(run.get("returncode")) is not int
            or run["returncode"] != 0 or run.get("initial_weights_unchanged") is not True
            or run.get("trainer_build") != build or run.get("argv") != expected_argv
            or run.get("dataset_manifest_sha256") != derived_manifest_sha256
            or run.get("positions_sha256") != positions_sha256
            or run.get("weight_sha256") != final_sha256
            or type(run.get("weight_bytes")) is not int or run["weight_bytes"] != final_bytes
            or final_bytes != NNUE_WEIGHT_BYTES
            or run.get("script_sha256") != binding["train_script_sha256"]
            or run.get("epoch_metadata") != expected_metadata_paths
            or run.get("initialization") != "inference weights; fresh Adam state"
            or run.get("initial_weights") != initial):
        raise ValueError("training run/provenance/argv/final weight mismatch")
    if type(run.get("positions")) is not int or run["positions"] != binding["positions"]["train"]:
        raise ValueError("training run position count mismatch")
    for name in ("epochs", "lr_schedule_epochs", "timeout_seconds"):
        if type(run.get(name)) is not int or run[name] != binding[name]:
            raise ValueError(f"training run integer mismatch: {name}")
    for name in ("learning_rate", "min_lr"):
        if type(run.get(name)) not in (int, float) or run[name] != binding[name]:
            raise ValueError(f"training run learning rate mismatch: {name}")
    if (run.get("lr_schedule") != binding["lr_schedule"]
            or run.get("float_subnormal_policy") != binding["float_subnormal_policy"]
            or initial.get("sha256") != binding["initial_weights_sha256"]
            or initial.get("metadata_sha256") != binding["initial_metadata_sha256"]
            or initial.get("nnue_output") != "absolute"
            or initial.get("optimizer") != "fresh Adam moments and step=0; inference parameters only"
            or set(epochs) != {1, 2, 3}):
        raise ValueError("training initialization/schedule/checkpoint set mismatch")
    n = binding["positions"]["train"]
    if derived_manifest["positions"]["train"] != n:
        raise ValueError("training dataset count mismatch")
    for epoch in (1, 2, 3):
        info, expected = epochs[epoch], {
            "epoch": epoch, "epochs": 3, "train_count": n, "valid_count": 0,
            "cache_hits": n, "cache_misses": 0, "init_seed": 42, "split_seed": 42,
            "shuffle_seed": 42, "lr_schedule_epochs": 3, "warmup_epochs": 0,
            "label_depth": 0, "source_cap": 0, "split_hash": 0,
            "nnue_output": "absolute", "teacher_eval": "external",
            "teacher_identity": binding["training_target_identity"],
            "lr_schedule": "StepHalf", "float_subnormal_policy": "x86-ftz-daz",
            "architecture": "INPUT=2420 L1=256 L2=32", "cache_only": True,
            "exclude_mate_labels": True, "side_balance": False,
            "games_dir": None, "teacher_weights": None, "wdl_lambda": None,
            "phase_weights": {}, "validation_ratio": 0.0,
            "teacher_score_cap": 30000.0, "search_target_weight": 1.0,
            "lr": float32(0.0001), "min_lr": float32(0.0),
            "positions": expected_argv[2]}
        meta = info["metadata"]
        for name, value in expected.items():
            actual = meta.get(name)
            # Rust JSON may serialize integral f32 as int or float. Bool is
            # never a numeric value, and integer counters must remain integers.
            numeric = type(value) is float
            if (name not in meta or type(actual) not in ((int, float) if numeric else (type(value),))
                    or actual != value):
                raise ValueError(f"epoch{epoch} metadata mismatch: {name}")
        if type(info.get("adam_step")) is not int or info["adam_step"] != epoch * n:
            raise ValueError("checkpoint Adam step does not match fresh complete epochs")
        anchor._sha(info.get("weight_sha256"))
    if epochs[3]["weight_sha256"] != final_sha256:
        raise ValueError("selected epoch3 differs from final training weights")


def diagnosis_argv(binary, original_dataset, checkpoint, original_identity):
    if (type(original_identity) is not str or not original_identity.startswith("external:")
            or len(original_identity) <= 9):
        raise ValueError("missing original external teacher identity")
    return [str(binary), "diagnose-external", "--positions",
            str(original_dataset / "holdout.positions.jsonl"), "--teacher-cache",
            str(original_dataset / "holdout.labels.jsonl"), "--external-teacher-id",
            original_identity, "--weights", str(checkpoint), "--adam",
            str(checkpoint.with_suffix(".adam.json"))]


def canonical_path(path, *, exists=False):
    path = path.expanduser()
    if not path.is_absolute() or path != path.resolve(strict=exists):
        raise ValueError("paths must be absolute and canonical, without symlink aliases")
    return path


def overlaps(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


def validate_output_paths(output, root, protected, *, root_private, inside_git, output_exists):
    """Pure path guard; the adapter supplies actual canonical paths/metadata."""
    if (not root_private or inside_git or output_exists or output == root
            or not output.is_relative_to(root)
            or any(overlaps(root, path) or overlaps(output, path) for path in protected)):
        raise ValueError("output requires a dedicated private Git-external new nonoverlapping root")


def output_guard(output, root, protected):
    canonical_path(root, exists=True)
    canonical_path(output)
    if not output.parent.is_dir():
        raise ValueError("diagnostic output parent must already exist")
    info = root.stat()
    root_private = (stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                    and stat.S_IMODE(info.st_mode) == 0o700)
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    git = subprocess.run(["git", "-C", str(ancestor), "rev-parse", "--is-inside-work-tree"],
                         capture_output=True, text=True, timeout=10,
                         env=dict(os.environ, LC_ALL="C"))
    if git.returncode != 0 and "not a git repository" not in git.stderr:
        raise ValueError("cannot verify diagnostic output is outside Git")
    validate_output_paths(output, root, protected, root_private=root_private,
                          inside_git=git.returncode == 0,
                          output_exists=output.exists() or output.is_symlink())


def summarize_rows(rows, targets):
    """Strict full original-order/static summary; no typed values are coerced."""
    if not targets or len(rows) != len(targets):
        raise ValueError("diagnostic result count mismatch")
    for index, (row, target) in enumerate(zip(rows, targets)):
        if (type(row) is not dict or type(row.get("index")) is not int or row["index"] != index
                or type(target) is not int or type(row.get("teacher_cp_stm")) is not int
                or row["teacher_cp_stm"] != target):
            raise ValueError("diagnostic result original ordering or label mismatch")
        for name in PREDICTIONS:
            value = row.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("missing or nonfinite diagnostic prediction")
        if type(row["inference_cp"]) is not int or type(row["material_cp"]) is not int:
            raise ValueError("core/material diagnostic values must be integers")
    values = {name: [row[name] for row in rows] for name in PREDICTIONS}
    bridge = [abs(a - b) for a, b in zip(values["quantized_float_cp"], values["inference_cp"])]
    if max(bridge) >= 1.001:
        raise ValueError("dequantized float forward diverges from core inference by >= 1.001 cp")
    quantization = [abs(a - b) for a, b in zip(values["raw_float_cp"], values["quantized_float_cp"])]
    deltas = {}
    for name in PREDICTIONS[:-1]:
        delta = [p - m for p, m in zip(values[name], values["material_cp"])]
        deltas[name] = {"count": len(delta), "mean_cp": statistics.fmean(delta),
                        "std_cp": statistics.pstdev(delta), "mean_abs_cp": statistics.fmean(map(abs, delta)),
                        "min_cp": min(delta), "max_cp": max(delta)}
    return {"count": len(rows), "metrics": {name: metrics(value, targets) for name, value in values.items()},
            "quantization_delta": {"mae_cp": statistics.fmean(quantization), "max_abs_cp": max(quantization)},
            "core_bridge_delta": {"mae_cp": statistics.fmean(bridge), "max_abs_cp": max(bridge)},
            "prediction_minus_fixed_material": deltas,
            "interpretation": "static STM predictions on original teacher holdout; neither blend loss nor 1M-node adoption evidence"}


def git_source_identity(source):
    def git(*args):
        return subprocess.run(["git", "-C", str(source), *args], check=True,
                              capture_output=True, timeout=10).stdout
    head = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all").decode()
    if head != LOCK["sources"]["sekirei"]["commit"] or status != " M crates/sekirei-train/src/main.rs\n":
        raise ValueError("trainer checkout is not the fixed commit with only the declared main.rs patch")
    names = git("ls-files", "-z").decode().split("\0")[:-1]
    return {"head": head, "status": status}, [source / name for name in names]


def group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _cleanup_group(process):
    """Bounded cleanup, including descendants that outlive the main process."""
    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
    if group_exists(process.pid):
        send(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if group_exists(process.pid):
            send(signal.SIGKILL)
        process.wait(timeout=5)
    if group_exists(process.pid):
        send(signal.SIGKILL)
        raise RuntimeError("diagnostic process group remained after bounded cleanup")


def cleanup_group(process):
    with blocked_termination():
        _cleanup_group(process)


def spawn_process(command, output, predictions, log, holder):
    """Close the spawn/assignment cancellation race, without launching in tests."""
    if type(holder) is not dict or set(holder) != {"process"} or holder["process"] is not None:
        raise ValueError("spawn requires an empty caller-owned process holder")
    try:
        with deferred_termination():
            holder["process"] = subprocess.Popen(command, cwd=output, stdout=predictions, stderr=log,
                env=dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", SEKIREI_TRAIN_FTZ_DAZ="1"),
                start_new_session=True)
    except BaseException:
        if holder["process"] is not None:
            cleanup_group(holder["process"])
        raise


def diagnose(args):
    holder = {"process": None}
    with termination_guard():
        with ExitStack() as resources:
            try:
                _diagnose(args, holder, resources)
            finally:
                # The real trainer lock remains owned by this stack until
                # outer group cleanup finishes, including cancellation during
                # the inner final receipt/cleanup.
                if holder["process"] is not None:
                    cleanup_group(holder["process"])


def _diagnose(args, holder, resources):
    if type(args.seconds) is not int or not 1 <= args.seconds <= 600:
        raise ValueError("diagnostic wall limit must be 1..600 seconds")
    paths = {name: canonical_path(getattr(args, name), exists=True) for name in
             ("original_dataset", "derived_dataset", "spec", "recipe", "train_run",
              "checkpoint", "trainer", "preregistration", "private_output_root")}
    output = canonical_path(args.output)
    build_root, train_root = paths["trainer"], paths["train_run"].parent
    checkpoint = paths["checkpoint"]
    if (paths["train_run"].name != "run.json"
            or checkpoint != train_root / "checkpoints/weights.epoch3.bin"):
        raise ValueError("diagnosis requires the complete train run's fixed weights.epoch3.bin")
    protected = [p for name, p in paths.items() if name != "private_output_root"]
    output_guard(output, paths["private_output_root"], protected)
    hashes, sizes = {}, {}

    def read(path):
        canonical_path(path, exists=True)
        if not path.is_file():
            raise ValueError("input must be a regular file")
        data = path.read_bytes()
        digest = anchor.sha256_bytes(data)
        if str(path) in hashes and hashes[str(path)] != digest:
            raise ValueError("input changed during diagnostic preparation")
        hashes[str(path)], sizes[str(path)] = digest, len(data)
        return data

    def unchanged():
        for path in hashes:
            canonical_path(Path(path), exists=True)
            if not stat.S_ISREG(Path(path).stat().st_mode):
                raise RuntimeError("diagnostic input ceased to be a regular canonical file")
        after = {path: sha256(Path(path)) for path in hashes}
        if after != hashes or any(Path(path).stat().st_size != sizes[path] for path in hashes):
            raise RuntimeError("diagnostic input/source/build changed during execution")
        return after

    process, record, started = None, None, time.monotonic()
    lock_path = canonical_path(build_root / ".training.lock")
    resources.enter_context(nonblocking_lock(lock_path, exclusive=True))
    prereg_bytes = read(paths["preregistration"])
    if anchor.sha256_bytes(prereg_bytes) != anchor._sha(args.expected_preregistration_sha256):
        raise ValueError("preregistration file hash mismatch")
    prereg = load_json(prereg_bytes)
    original_manifest_bytes = read(paths["original_dataset"] / "manifest.json")
    original_files = {name: read(paths["original_dataset"] / name) for name in anchor.FILES}
    spec = load_json(read(paths["spec"]))
    # Original hash is pinned by the externally SHA-bound preregistration;
    # derive/validate never infer it from a possibly changed dataset.
    binding_raw = prereg.get("diagnosis_binding", {})
    original_hash = binding_raw.get("original_manifest_sha256")
    original_manifest, _ = anchor.validate_original(original_manifest_bytes, original_files,
                                                   expected_manifest_sha256=original_hash)
    derived_manifest_bytes = read(paths["derived_dataset"] / "manifest.json")
    derived_manifest = load_json(derived_manifest_bytes)
    view = {"spec": spec, "manifest": derived_manifest,
            "recipe": load_json(read(paths["recipe"])),
            "files": {name: read(paths["derived_dataset"] / name) for name in anchor.FILES}}
    binding = validate_binding(prereg, spec, original_manifest, derived_manifest)
    pure_check = anchor.validate_train_view(view, original_manifest_bytes, original_files,
                 expected_manifest_sha256=binding["original_manifest_sha256"],
                 expected_spec_sha256=binding["spec_sha256"])
    verified_original, targets = verified_split(paths["original_dataset"], "holdout")
    if verified_original != original_manifest:
        raise ValueError("original holdout loader disagrees with pinned manifest")
    verified_split(paths["derived_dataset"], "train")
    build_path = build_root / "build-manifest.json"
    build = load_json(read(build_path))
    binary = canonical_path(Path(build["binary"]), exists=True)
    if not binary.is_relative_to(build_root / "build") or not os.access(binary, os.X_OK):
        raise ValueError("trainer binary must be executable inside its actual build root")
    read(binary)
    patch = REPO / "patches/sekirei-train-external-labels.patch"
    read(patch)
    main = build_root / "source/crates/sekirei-train/src/main.rs"
    read(main)
    validate_build(build, binding, manifest_sha256=hashes[str(build_path)],
                   binary_sha256=hashes[str(binary)], patch_sha256=hashes[str(patch)],
                   main_sha256=hashes[str(main)])
    source = build_root / "source"
    git_before, source_files = git_source_identity(source)
    for path in source_files:
        read(path)
    for name, digest in SOURCE_HASHES.items():
        if hashes[str(source / name)] != digest:
            raise ValueError(f"fixed core/checkpoint/trainer source mismatch: {name}")
    script_paths = [Path(__file__), REPO / "scripts/train_cpu.py", REPO / "scripts/functional_anchor.py",
                   REPO / "scripts/material_init.py", REPO / "scripts/diagnose_weights.py",
                   REPO / "scripts/export_nearest.py", REPO / "config/toolchain.lock.json"]
    for path in script_paths:
        read(path)
    if hashes[str(REPO / "scripts/train_cpu.py")] != binding["train_script_sha256"]:
        raise ValueError("training script is not the preregistered implementation")
    validate_source_helpers(prereg, {path.name: hashes[str(path)] for path in script_paths}, spec)
    run = load_json(read(paths["train_run"]))
    initial_path = canonical_path(Path(run["initial_weights"]["path"]), exists=True)
    read(initial_path); read(initial_path.with_suffix(".meta.json"))
    initial = verify_initial_weights(initial_path)
    # Add late-discovered inputs to the output overlap guard as well.
    output_guard(output, paths["private_output_root"], protected + [initial_path, binary, patch, *script_paths])
    train_positions = read(train_root / "positions.jsonl")
    if train_positions != view["files"]["train.positions.jsonl"]:
        raise ValueError("training positions differ in bytes/order from the complete derived view")
    final = read(train_root / "weights.bin")
    epochs, metadata_paths = {}, []
    for epoch in (1, 2, 3):
        native_path = train_root / f"checkpoints/weights.epoch{epoch}.bin"
        meta_path, adam_path = native_path.with_suffix(".meta.json"), native_path.with_suffix(".adam.json")
        native, meta = read(native_path), load_json(read(meta_path))
        verify_checkpoint_metadata(native_path, meta, binding["training_target_identity"])
        adam = validate_adam(load_json(read(adam_path)))
        # Full native reexport equality is independently checked here;
        # the actual pinned Rust diagnosis must check the selected pair too.
        post_export(adam, native)
        epochs[epoch] = {"metadata": meta, "adam_step": adam["step"],
                         "weight_sha256": hashes[str(native_path)]}
        metadata_paths.append(str(meta_path))
        del adam
    if len(list((train_root / "checkpoints").glob("*.meta.json"))) != 3:
        raise ValueError("training checkpoint metadata inventory is not exactly three epochs")
    validate_training(run, binding, derived_manifest, build, initial, epochs,
        derived_manifest_sha256=hashes[str(paths["derived_dataset"] / "manifest.json")],
        positions_sha256=hashes[str(train_root / "positions.jsonl")],
        final_sha256=hashes[str(train_root / "weights.bin")], final_bytes=len(final),
        expected_argv=training_argv(binary, train_root, paths["derived_dataset"], initial_path,
                                    binding["training_target_identity"]),
        expected_metadata_paths=metadata_paths)
    command = diagnosis_argv(binary, paths["original_dataset"], checkpoint, binding["original_teacher_identity"])
    unchanged()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    record = {"schema_version": 1, "status": "running", "split": "original-holdout",
              "training_target_identity": binding["training_target_identity"],
              "evaluation_teacher_identity": binding["original_teacher_identity"],
              "spec_sha256": binding["spec_sha256"], "selected_epoch": 3, "count": len(targets),
              "argv": command, "input_sha256": hashes, "input_bytes": sizes,
              "trainer_build": build, "source_git": git_before, "training_lock": str(lock_path),
              "pure_transformation_check": pure_check,
              "real_generation_preflight_claimed": False, "adoption_claimed": False,
              "float_subnormal_policy": "x86-ftz-daz", "timeout_seconds": args.seconds,
              "python_native_reexport_all_bytes_equal": True,
              "rust_native_reexport_verified": False, "cleanup_verified": False}
    atomic_write_json(output / "run.json", record)
    try:
        with (output / "predictions.jsonl").open("x") as predictions, (output / "diagnose.log").open("x") as log:
            spawn_process(command, output, predictions, log, holder)
            process = holder["process"]
            try:
                returncode = process.wait(timeout=args.seconds)
            except subprocess.TimeoutExpired:
                record["status"] = "timeout"
                cleanup_group(process)
                raise RuntimeError("anchor holdout diagnostic exceeded its wall limit")
        record["returncode"] = returncode
        if group_exists(process.pid):
            cleanup_group(process)
            raise RuntimeError("diagnostic left descendants after process exit")
        record["cleanup_verified"] = True
        if returncode != 0:
            raise RuntimeError(f"anchor holdout diagnostic failed with exit {returncode}")
        rows = [load_json(line) for line in (output / "predictions.jsonl").read_bytes().splitlines()]
        summary = summarize_rows(rows, targets)
        record["input_sha256_after"] = unchanged()
        if git_source_identity(source)[0] != git_before:
            raise RuntimeError("trainer source checkout changed during diagnosis")
        summary.update(training_target_identity=binding["training_target_identity"],
                       evaluation_teacher_identity=binding["original_teacher_identity"],
                       checkpoint_sha256=hashes[str(checkpoint)],
                       adam_sha256=hashes[str(checkpoint.with_suffix(".adam.json"))],
                       original_manifest_sha256=binding["original_manifest_sha256"],
                       spec_sha256=binding["spec_sha256"])
        atomic_write_json(output / "summary.json", summary)
        record.update(status="complete", input_unchanged=True, rust_native_reexport_verified=True,
                      predictions_sha256=sha256(output / "predictions.jsonl"),
                      summary_sha256=sha256(output / "summary.json"))
    except BaseException as error:
        if isinstance(error, DiagnosticCancelled):
            record["status"] = "cancelled"
        elif record["status"] != "timeout":
            record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        process = holder["process"]
        if process is not None:
            try:
                cleanup_group(process)
                record["cleanup_verified"] = True
            except DiagnosticCancelled as error:
                record.update(status="cancelled", error=str(error),
                              cleanup_verified=not group_exists(process.pid))
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                record.update(status="cleanup_failure", cleanup_verified=False, cleanup_error=str(error))
        try:
            record["input_sha256_after"] = unchanged()
            record["source_git_after"] = git_source_identity(source)[0]
            if record["source_git_after"] != git_before:
                raise RuntimeError("trainer source checkout changed during final verification")
            record["input_unchanged"] = True
        except DiagnosticCancelled as error:
            record.update(status="cancelled", input_unchanged=False, input_error=str(error))
        except Exception as error:
            record.update(status="input_changed", input_unchanged=False, input_error=str(error))
        record["wall_seconds"] = time.monotonic() - started
        atomic_write_json(output / "run.json", record)
    if record["status"] != "complete":
        raise RuntimeError("anchor holdout diagnostic did not satisfy all final guards")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-dataset", "derived-dataset", "spec", "recipe", "train-run", "checkpoint",
                 "trainer", "preregistration", "private-output-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--seconds", type=int, default=120)
    diagnose(parser.parse_args())
