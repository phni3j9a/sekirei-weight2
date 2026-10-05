#!/usr/bin/env python3
"""Supervise one fresh weekly E3 fit and verify its complete numerical state.

This parent owns process cleanup and immutable input checks. Its completion is
training evidence only; candidate proofs and the four formal evaluation stages
are separate prerequisites for adopting any model.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmark import nonblocking_lock
from prepare_bounded import cleanup
from train_cpu import termination_guard
from weekly_nonlinear_post_training import PhysicalBytes, epoch_observations, require, small


REPO = Path(__file__).resolve().parents[1]
GATE_DIR = REPO / "preparations/white-view-paired-nonlinear-math-v1/gate-v2"
NUMERIC_SOURCE_SHA = (
    "2cdc3ec2a36e9a7be75011456294602283f6819704cbf1912db78d8fac82584f",
    "f25a6763cbc6b4b7158fcd1072875f755d3b04e6c265604856f229bf6c39c6bd",
    "305d4a2148feecfc0b25ca7e6834c108a635e250cf5b42124511ab3c976b8144",
)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def physical_ref(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
            and path.is_file() and not path.is_symlink(), "canonical regular file required")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def info(path):
    return small(physical_ref(path))


def read_ref(reference):
    require(physical_ref(Path(reference["path"])) == reference, "changed full reference")
    raw = Path(reference["path"]).read_bytes()
    require({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} == small(reference),
            "input changed while reading")
    return raw


def merge(destination, values):
    for path, identity in values.items():
        require(path not in destination or destination[path] == identity,
                "conflicting immutable input identity")
        destination[path] = identity


def collect_refs(destination, value):
    if type(value) is dict:
        if set(value) == {"path", "bytes", "sha256"}:
            require(physical_ref(Path(value["path"])) == value, "transitive reference changed")
            merge(destination, {value["path"]: small(value)})
        else:
            for item in value.values():
                collect_refs(destination, item)
    elif type(value) is list:
        for item in value:
            collect_refs(destination, item)


def verify_map(values):
    actual = {path: info(Path(path)) for path in values}
    require(actual == values, "immutable input inventory changed")
    return actual


def source_map(root):
    names = []
    for flags in (["--cached"], ["--others", "--exclude-standard"]):
        raw = subprocess.run(["git", "-C", str(root), "ls-files", "-z", *flags],
                             check=True, stdout=subprocess.PIPE).stdout
        names.extend(raw.decode().split("\0")[:-1])
    return {str(root / name): info(root / name) for name in sorted(set(names))}


def write_new(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return physical_ref(path)


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def scan_heavy_processes():
    """Observe other local trainers, engines and experiment parents by process."""
    names = {"train", "sekirei", "sekirei-train", "yaneuraou", "shogiesa", "cargo", "rustc"}
    scripts = {"benchmark.py", "top3.py", "train_cpu.py", "train_bounded.py", "train_fanin509.py",
               "weekly_nonlinear_run.py", "weekly_nonlinear_build.py", "evaluate_candidate.py",
               "diverse_pack_dataset.py"}
    records = []
    for entry in sorted(Path("/proc").iterdir(), key=lambda p: p.name):
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().decode().split("\0")[:-1]
            comm = (entry / "comm").read_text().strip()
            if comm in names or any(Path(arg).name in scripts for arg in argv if arg and "\n" not in arg):
                records.append({"pid": int(entry.name), "comm": comm, "command": argv})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return records


def require_clear_processes():
    scans = []
    for _ in range(2):
        conflicts = scan_heavy_processes()
        scans.append({"conflicts": conflicts})
        require(not conflicts, "another heavy experiment process is running")
        time.sleep(0.05)
    return scans


def run_child(command, log, env, deadline, cwd):
    """Own the new process group through cleanup and durable failure evidence."""
    child = None
    started = time.monotonic()
    outcome = {"command": command, "pid": None, "pgid": None, "returncode": None,
               "waited": False, "reaped": False, "timed_out": False,
               "group_empty_scans": [False, False]}
    error = None
    with termination_guard() as state:
        try:
            with log.open("xb") as stream:
                try:
                    state["spawning"] = True
                    try:
                        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                    finally:
                        state["spawning"] = False
                    outcome.update(pid=child.pid, pgid=child.pid)
                    require(state["signal"] is None, "cancelled after child ownership")
                    try:
                        outcome["returncode"] = child.wait(timeout=deadline)
                        outcome["waited"] = True
                    except subprocess.TimeoutExpired:
                        outcome["timed_out"] = True
                        raise
                finally:
                    state["cleaning"] = True
                    try:
                        cleanup(child)
                        if child is not None:
                            outcome.update(pid=child.pid, pgid=child.pid, returncode=child.returncode,
                                           reaped=child.returncode is not None,
                                           waited=child.returncode is not None)
                            outcome["group_empty_scans"] = [not group_alive(child.pid)]
                            time.sleep(0.05)
                            outcome["group_empty_scans"].append(not group_alive(child.pid))
                            require(outcome["group_empty_scans"] == [True, True],
                                    "child process group remains")
                    finally:
                        state["cleaning"] = False
            require(state["signal"] is None, "cancelled child cannot complete")
        except BaseException as caught:
            error = caught
            caught.child_outcome = outcome
            caught.signal = state["signal"]
            raise
        finally:
            state["cleaning"] = True
            try:
                outcome["wall_seconds"] = time.monotonic() - started
                if log.exists():
                    outcome["log"] = physical_ref(log)
                record = {"schema": "sekirei.weekly-supervised-child.v1",
                          "status": "failed" if error is not None else "observed-exit",
                          "outcome": outcome, "signal": state["signal"]}
                if error is not None:
                    record.update(error_type=type(error).__name__, error=str(error))
                write_new(Path(str(log) + ".child-outcome.json"), record)
            finally:
                state["cleaning"] = False
        require(state["signal"] is None, "cancelled during child receipt")
    return outcome


def load_numeric_gate(build_contract):
    """Isolate immutable pure consumers; never enable their old outer gate."""
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    sources = [GATE_DIR / "paired_nonlinear_gate.py",
               GATE_DIR / "paired_nonlinear_proof_contract.py", build_contract]
    refs = {str(path): info(path) for path in sources}
    require(all(refs[str(path)]["sha256"] == sha for path, sha in zip(sources, NUMERIC_SOURCE_SHA)),
            "immutable numerical consumer source differs")
    probe = module("_weekly_numeric_probe", sources[1])
    contract = module("_weekly_numeric_build_contract", build_contract)
    names = {"paired_nonlinear_proof_contract": probe, "white_view_build_contract": contract}
    old = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules.update(names)
        gate = module("_weekly_numeric_gate", sources[0])
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    require(gate.PROTOTYPE_ONLY is True and gate.probe is probe and gate.build is contract,
            "pure numerical consumer identity differs")
    return gate, refs


def command(recipe_ref, recipe, binary, output):
    argv = [binary, "train-paired-nonlinear"]
    for name, reference in (("recipe", recipe_ref), ("source-binding", recipe["source_binding"]),
                            ("manifest", recipe["manifest"]), ("reference03", recipe["reference03"]),
                            ("positions", recipe["positions"]), ("labels", recipe["labels"])):
        argv.extend(["--" + name, reference["path"], "--" + name + "-bytes",
                     str(reference["bytes"]), "--" + name + "-sha256", reference["sha256"]])
    argv.extend(["--output", str(output)])
    require(len(argv) == 40, "literal fixed training argv required")
    return argv


def initialized_io(recipe_ref, recipe, binary, parent_dir, env, cwd, context, inputs, gate):
    """Validate this candidate's actual input route before committing CPU hours."""
    negative_dir = parent_dir / "missing-policy-output"
    negative_argv = command(recipe_ref, recipe, binary, negative_dir)
    negative_argv[1] = "verify-paired-nonlinear-inputs"
    negative_env = dict(env)
    negative_env.pop("SEKIREI_TRAIN_FTZ_DAZ", None)
    negative = run_child(negative_argv, parent_dir / "missing-policy.log", negative_env, 60, cwd)
    require(type(negative["returncode"]) is int and negative["returncode"] == 1
            and negative["waited"] is True and negative["reaped"] is True
            and negative["timed_out"] is False and negative["group_empty_scans"] == [True, True]
            and not negative_dir.exists(), "missing float policy must fail before output creation")
    require(b"SEKIREI_TRAIN_FTZ_DAZ missing or not Unicode" in read_ref(negative["log"]),
            "negative input diagnostic failed for an unexpected reason")
    directory = parent_dir / "initialized-io"
    argv = command(recipe_ref, recipe, binary, directory)
    argv[1] = "verify-paired-nonlinear-inputs"
    child = run_child(argv, parent_dir / "initialized-io.log", env, 3600, cwd)
    gate.lifecycle(child, argv, 3600)
    stage_ref = physical_ref(directory / "initialized-io-stage.json")
    stage = strict_json(read_ref(stage_ref))
    gate.obj(stage, {"schema", "status", "mode", "train_count", "global_step", "epochs_completed",
                     "candidate_model", "resume_allowed", "float_observations", "context", "outputs",
                     "inputs_before", "inputs_after", "all_state_bits_equal", "all_nearest_bytes_equal",
                     "compiled_native_reader_fullbytes_equal", "parent_reap_verified", "actual_fit_started",
                     "model_adopted", "final_used"})
    gate.fixed(stage, {
        "schema": "sekirei.white-view-paired-nonlinear-initialized-io-stage.v1",
        "status": "verified-original-inputs-and-initialized-codecs", "mode": recipe["mode"],
        "train_count": 112681, "global_step": 0, "epochs_completed": 0,
        "candidate_model": False, "resume_allowed": False, "context": context,
        "inputs_before": inputs, "inputs_after": inputs, "all_state_bits_equal": True,
        "all_nearest_bytes_equal": True, "compiled_native_reader_fullbytes_equal": True,
        "parent_reap_verified": False, "actual_fit_started": False, "model_adopted": False,
        "final_used": False})
    gate.obj(stage["float_observations"], {"before", "after_load", "after_io"})
    for snapshot in stage["float_observations"].values():
        gate.float_snapshot(snapshot)
    gate.obj(stage["outputs"], {"initialized_checkpoint", "initialized_native03"})
    for role, name in (("initialized_checkpoint", "initialized-fullstate.bits.json"),
                       ("initialized_native03", "initialized.nearest03.bin")):
        require(stage["outputs"][role] == physical_ref(directory / name),
                "initialized output path or bytes differ")
    receipt = write_new(parent_dir / "initialized-io-parent.json", {
        "schema": "sekirei.weekly-nonlinear-initialized-io-parent.v1", "status": "complete",
        "recipe": recipe_ref, "stage": stage_ref, "stage_outputs": stage["outputs"],
        "child": child, "missing_policy_child": negative,
        "inputs_before": inputs, "inputs_after": verify_map(inputs),
        "actual_fit_started": False, "model_adopted": False, "final_used": False})
    print(json.dumps({"stage": "initialized-io-verified", "receipt": receipt}), flush=True)
    return receipt


def run(args, state):
    from weekly_nonlinear_preflight import validate_source_binding
    for path in (args.recipe, args.output, args.receipt_dir, args.numeric_build_contract,
                 args.white_runtime, args.stock_runtime):
        require(path.is_absolute() and path.resolve() == path, "canonical absolute argument paths required")
    recipe_ref = physical_ref(args.recipe)
    require(recipe_ref["sha256"] == args.expected_recipe_sha256, "recipe pin differs")
    bound = validate_source_binding(recipe_ref)
    recipe, sb, parent, plan, build = (bound[key] for key in
                                     ("recipe", "source_binding", "parent", "plan", "build"))
    require(not args.output.exists() and args.output.parent.resolve(strict=True) == args.output.parent,
            "fresh canonical training output required; no resume")
    require(not args.receipt_dir.exists()
            and args.receipt_dir.parent.resolve(strict=True) == args.receipt_dir.parent,
            "fresh canonical parent receipt directory required")
    require(not args.output.is_relative_to(REPO) and not args.receipt_dir.is_relative_to(REPO),
            "private training paths required")
    deadline = datetime.fromisoformat(args.campaign_deadline_utc.replace("Z", "+00:00"))
    require(deadline.tzinfo is not None, "explicit UTC campaign deadline required")
    wall_limit = plan["resources"]["training_wall_limit_seconds"]
    require((deadline - datetime.now(timezone.utc)).total_seconds() > wall_limit + 7200,
            "campaign needs training plus two hours for evaluation and saving")
    source_root = Path(build["source_root"])
    training_root = Path(sb["training_binary"]["path"]).parents[2]
    lock_paths = sorted({args.receipt_dir.parent / ".heavy.lock",
                         training_root / ".build.lock", training_root / ".training.lock",
                         args.white_runtime / ".benchmark.lock",
                         args.stock_runtime / ".benchmark.lock"})
    args.receipt_dir.mkdir(mode=0o700)
    log = args.receipt_dir / "training.log"
    with ExitStack() as locks:
        lock_records = []
        for path in lock_paths:
            locks.enter_context(nonblocking_lock(path, exclusive=True))
            lock_records.append({"path": str(path), "exclusive": True,
                                 "nonblocking": True, "acquired": True})
        # Revalidate after acquiring locks; an earlier successful preflight is
        # not authorization to accept changed physical inputs at launch time.
        bound = validate_source_binding(recipe_ref)
        process_scans = require_clear_processes()
        require(source_map(source_root) == sb["training_source_files"], "source membership differs")
        gate, gate_sources = load_numeric_gate(args.numeric_build_contract)
        expected = dict(parent["source_inputs"])
        collect_refs(expected, [recipe_ref, recipe["source_binding"], sb["parent_preflight"]])
        merge(expected, gate_sources)
        for name in ("weekly_nonlinear_run.py", "weekly_nonlinear_post_training.py",
                     "weekly_nonlinear_preflight.py", "train_cpu.py", "prepare_bounded.py", "benchmark.py"):
            path = REPO / "scripts" / name
            merge(expected, {str(path): info(path)})
        before = verify_map(expected)
        child_inputs = dict(parent["source_inputs"])
        collect_refs(child_inputs, [recipe_ref, recipe["source_binding"], sb["parent_preflight"]])
        context = {"recipe": recipe_ref, "source_binding": recipe["source_binding"],
                   "manifest": recipe["manifest"], "reference03": recipe["reference03"],
                   "positions": recipe["positions"], "labels": recipe["labels"],
                   "source_files": child_inputs}
        available = shutil.disk_usage(training_root).free
        require(available >= plan["resources"]["minimum_ssd_remaining_bytes"] + 2 * 2**30,
                "insufficient SSD reserve and padded training output space")
        env = dict(os.environ)
        for key in tuple(env):
            if key.startswith(("CARGO_", "RUSTC", "RUSTDOC", "RUSTFLAGS")) or key == "RUSTUP_TOOLCHAIN":
                env.pop(key)
        env.update(SEKIREI_TRAIN_FTZ_DAZ="1", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", LC_ALL="C")
        diagnostic = initialized_io(recipe_ref, recipe, sb["training_binary"]["path"],
                                    args.receipt_dir, env, source_root, context, child_inputs, gate)
        collect_refs(before, [diagnostic, strict_json(read_ref(diagnostic))])
        before = verify_map(before)
        argv = command(recipe_ref, recipe, sb["training_binary"]["path"], args.output)
        started = write_new(args.receipt_dir / "started.json", {
            "schema": "sekirei.weekly-nonlinear-training-started.v1",
            "status": "validated-before-child-launch", "recipe": recipe_ref,
            "command": argv, "inputs_before": before, "lock_records": lock_records,
            "process_scan_records": process_scans,
            "available_ssd_bytes": available, "deadline_seconds": wall_limit,
            "campaign_deadline_utc": args.campaign_deadline_utc,
            "resume_used": False, "shuffle_used": False, "final_used": False,
            "adoption_claimed": False})
        print(json.dumps({"stage": "training-launched", "started": started}), flush=True)
        child = run_child(argv, log, env, wall_limit, source_root)
        gate.lifecycle(child, argv, wall_limit)
        refs = {"checkpoint": physical_ref(args.output / "fullstate.bits.json"),
                "native03": physical_ref(args.output / "weights.nearest03.bin"),
                "export_stage": physical_ref(args.output / "export-stage.json")}
        extra = dict(before)
        collect_refs(extra, [*refs.values(), child["log"]])
        reader = PhysicalBytes(extra, gate=gate, physical_ref=physical_ref)
        observed = epoch_observations(reader.read(child["log"]), gate=gate)
        native = reader.read(refs["native03"])
        stage = reader.json(refs["export_stage"])
        bounds = gate.checkpoint(reader.read(refs["checkpoint"]), context,
                                 reader.read(recipe["reference03"]), native)
        gate.export_stage(stage, refs, context, child_inputs, native)
        fp = {"schema": "sekirei.white-view-paired-nonlinear-float-policy-evidence.v1",
              "guard_source": physical_ref(source_root / "crates/sekirei-train/src/paired_nonlinear_float.rs"),
              "epoch_log": child["log"], "epoch_observations": observed,
              "export": stage["float_export"]}
        gate.float_evidence(fp, reader, child["log"], stage["float_export"])
        after = verify_map(before)
        require(source_map(source_root) == sb["training_source_files"], "source membership changed")
        require(physical_ref(Path(sb["training_binary"]["path"])) == sb["training_binary"],
                "training binary changed")
        after_scans = require_clear_processes()
        require(state["signal"] is None, "cancelled before completion publication")
        body = {"schema": "sekirei.weekly-nonlinear-training-completion.v1", "status": "complete",
                "mode": recipe["mode"], "plan": sb["selected_plan"], "source_binding": recipe["source_binding"],
                "parent_preflight": sb["parent_preflight"], "training_build": sb["training_build"],
                "engine_build": sb["engine_build"], "recipe": recipe_ref, "execution": child,
                "initialized_io_parent": diagnostic,
                "dataset_inputs": sb["dataset_inputs"], "reference03": recipe["reference03"],
                "outputs": refs, "fp_evidence": fp, "native_functional_bounds": bounds,
                "process_scan_records_before": process_scans, "process_scan_records_after": after_scans,
                "inputs_before": before, "inputs_after": after,
                "source_unchanged": True, "build_unchanged": True, "poisoned": False,
                "resume_used": False, "shuffle_used": False, "final_used": False,
                "core_fullrows_verified": False, "incremental_verified": False,
                "adoption_claimed": False}
        state["cleaning"] = True
        try:
            completion = write_new(args.receipt_dir / "completion.json", body)
            state["owned_completion"] = True
        finally:
            state["cleaning"] = False
        require(state["signal"] is None, "cancelled during completion publication")
        print(json.dumps({"status": "training-verified", "completion": completion,
                          "wall_seconds": child["wall_seconds"], "model_adopted": False}), flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--numeric-build-contract", type=Path, required=True)
    parser.add_argument("--white-runtime", type=Path, required=True)
    parser.add_argument("--stock-runtime", type=Path, required=True)
    parser.add_argument("--campaign-deadline-utc", required=True)
    args = parser.parse_args()
    with termination_guard() as state:
        state["owned_completion"] = False
        try:
            run(args, state)
            require(state["signal"] is None, "cancelled at parent exit")
        except BaseException as error:
            state["cleaning"] = True
            try:
                if state["owned_completion"]:
                    (args.receipt_dir / "completion.json").rename(args.receipt_dir / "rejected-completion.json")
                if args.receipt_dir.exists():
                    write_new(args.receipt_dir / "failure.json", {
                        "schema": "sekirei.weekly-nonlinear-training-failure.v1", "status": "failed",
                        "error_type": type(error).__name__, "error": str(error),
                        "signal": getattr(error, "signal", state["signal"]),
                        "child_outcome": getattr(error, "child_outcome", None),
                        "partial_resume_allowed": False, "model_adopted": False, "final_used": False})
            finally:
                state["cleaning"] = False
            raise


if __name__ == "__main__":
    main()
