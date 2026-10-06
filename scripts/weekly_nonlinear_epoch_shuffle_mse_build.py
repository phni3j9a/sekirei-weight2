#!/usr/bin/env python3
"""Build the weekly trainer and produce its typed, source-bound build receipt.

Source preparation must already be complete. Cargo uses the fixed installed
compiler, release/offline/locked/jobs2 and a fresh private target directory.
Training, model loading and search are separate stages.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from benchmark import nonblocking_lock
from prepare_bounded import cleanup
from prepare_weekly_nonlinear_epoch_shuffle_mse import (REPO, UPSTREAM, digest, pinned_json, raw_json, ref, run,
    SHUFFLE_PLAN_PIN, SHUFFLE_MODE, shuffle_variant_sources, require_shuffle_tests)
from train_cpu import termination_guard
from weekly_nonlinear_epoch_shuffle_mse_post_training import require, small


def info(path):
    return small(ref(Path(path)))


def verify_map(expected):
    require(type(expected) is dict and expected, "nonempty frozen input map required")
    actual = {path: info(path) for path in expected}
    require(actual == expected, "frozen immutable file identity changed")
    return actual


def merge(expected, values):
    for path, identity in values.items():
        require(path not in expected or expected[path] == identity, "conflicting input identity")
        expected[path] = identity


def write_exclusive(path, body):
    with path.open("xb") as stream:
        stream.write((json.dumps(body, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return ref(path)


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def supervised_command(argv, log, env, seconds, cwd):
    """Own the handle before cancellation; reap and check the owned process group."""
    child = None
    started = time.monotonic()
    outcome = {"command": argv, "pid": None, "pgid": None, "returncode": None,
               "waited": False, "reaped": False, "timed_out": False,
               "group_empty_scans": [False, False]}
    error = None
    with termination_guard() as state:
        try:
            with log.open("xb") as stream:
                try:
                    state["spawning"] = True
                    try:
                        child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
                    finally:
                        state["spawning"] = False
                    outcome.update(pid=child.pid, pgid=child.pid)
                    require(state["signal"] is None, "cancelled during child spawn")
                    try:
                        outcome["returncode"] = child.wait(timeout=seconds)
                        outcome["waited"] = True
                    except subprocess.TimeoutExpired:
                        outcome["timed_out"] = True
                        raise
                finally:
                    state["cleaning"] = True
                    try:
                        cleanup(child)
                        if child is not None:
                            outcome.update(returncode=child.returncode, waited=child.returncode is not None,
                                           reaped=child.returncode is not None)
                            outcome["group_empty_scans"] = [not group_alive(child.pid)]
                            time.sleep(0.05)
                            outcome["group_empty_scans"].append(not group_alive(child.pid))
                            require(outcome["group_empty_scans"] == [True, True], "owned child group remains")
                    finally:
                        state["cleaning"] = False
            require(state["signal"] is None, "cancelled after child cleanup")
            require(type(outcome["returncode"]) is int and outcome["returncode"] == 0
                    and outcome["waited"] and outcome["reaped"] and not outcome["timed_out"],
                    "child failed or timed out")
        except BaseException as caught:
            error = caught
            raise
        finally:
            state["cleaning"] = True
            try:
                outcome["wall_seconds"] = time.monotonic() - started
                if log.exists():
                    outcome["log"] = ref(log)
                record = {"schema": "sekirei.weekly-supervised-child-outcome.v1",
                          "status": "failed" if error is not None else "observed-complete",
                          "outcome": outcome, "signal": state["signal"]}
                if error is not None:
                    record.update(error_type=type(error).__name__, error=str(error))
                write_exclusive(Path(str(log) + ".child-outcome.json"), record)
            finally:
                state["cleaning"] = False
        require(state["signal"] is None, "cancelled while recording outcome")
    return outcome


def registry_dependencies(depfiles):
    registry = Path.home() / ".cargo/registry/src"
    roots = set()
    for dep in depfiles.glob("*.d"):
        for item in re.findall(r"(?:^|\s)(/[^\s:\\]+)", dep.read_text()):
            path = Path(item)
            if registry in path.parents:
                parts = path.relative_to(registry).parts
                require(len(parts) >= 3, "registry package path required")
                roots.add(registry / parts[0] / parts[1])
    require(roots, "actual registry dependencies absent from reference depfiles")
    return {str(path): info(path) for root in sorted(roots)
            for path in sorted(root.rglob("*")) if path.is_file()}


def parse_test_log(path):
    text = path.read_text()
    results = re.findall(r"^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", text, re.M)
    require(len(results) == 1, "one actual Rust harness summary required")
    passed, failed, ignored = map(int, results[0])
    names = re.findall(r"^test (.+) \.\.\. ok$", text, re.M)
    require(len(names) == len(set(names)) == passed and passed >= 24 and failed == ignored == 0,
            "actual Rust test membership or counts incomplete")
    for category in ("reader", "sha", "unique", "native", "adapter", "float_policy", "initialized", "weekly"):
        require(any(category in name for name in names), "required actual Rust test missing: " + category)
    return {"log": ref(path), "passed": passed, "failed": failed, "ignored": ignored,
            "required_test_names": names}


def build(args):
    preparation_path = args.source_preparation.resolve(strict=True)
    prep, _ = pinned_json(preparation_path, args.expected_source_preparation_sha256)
    require(prep.get("schema") == "sekirei.weekly-nonlinear-epoch-shuffle-mse-source-preparation.v1"
            and prep.get("status") == "source-only" and prep.get("compiled") is False,
            "uncompiled fresh source preparation required")
    source = Path(prep["source_root"])
    runtime = source.parent
    receipt_path, failure_path = runtime / "training-build.json", runtime / "training-build-failure.json"
    target = runtime / "build"
    require(not any(p.exists() for p in (receipt_path, failure_path, target)), "fresh build artifacts required")
    require(run(["git", "rev-parse", "HEAD"], source).decode().strip() == UPSTREAM,
            "prepared fixed upstream revision changed")
    engine_path = args.engine_manifest.resolve(strict=True)
    engine, _ = pinned_json(engine_path, args.expected_engine_manifest_sha256)
    compiler = engine["compiler_files"]
    verify_map(compiler)
    cargo = next(Path(p) for p in compiler if Path(p).name == "cargo")
    rustc = next(Path(p) for p in compiler if Path(p).name == "rustc")
    require(run([str(cargo), "--version"]).decode().strip() == engine["cargo"]
            and run([str(rustc), "--version"]).decode().strip() == engine["rustc"], "fixed compiler version changed")
    sources = verify_map(prep["source_files"])
    names = run(["git", "ls-files", "-z"], source).decode().split("\x00")[:-1]
    names += run(["git", "ls-files", "--others", "--exclude-standard", "-z"], source).decode().split("\x00")[:-1]
    require(set(sources) == {str(source / name) for name in names}, "prepared source membership changed")
    for name, identity in engine["source_files_after"].items():
        if not name.startswith("crates/sekirei-train/"):
            require(sources.get(str(source / name)) == identity, "fixed core/search source changed")
    plan_ref = prep["plan"]
    require(ref(Path(plan_ref["path"])) == plan_ref, "frozen plan identity changed")
    plan = raw_json(Path(plan_ref["path"]).read_bytes())
    require({key: plan_ref[key] for key in ("bytes", "sha256")} == SHUFFLE_PLAN_PIN and plan["mode"] == SHUFFLE_MODE,
            "fixed dedicated shuffle MSE selected plan required")
    variant = shuffle_variant_sources()
    require(prep.get("row_order_variant") == variant and prep.get("numeric_update_export_modules_preserved") is False,
            "explicit fixed row-order source delta required")
    budget = plan["resources"]
    require(shutil.disk_usage(runtime).free >= budget["minimum_ssd_remaining_bytes"] + budget["added_storage_budget_bytes"],
            "insufficient free SSD for declared build/output reserve")
    dependencies = registry_dependencies(args.dependency_reference.resolve(strict=True))
    before = dict(sources)
    merge(before, compiler)
    merge(before, dependencies)
    for path in (preparation_path, engine_path, Path(plan_ref["path"]), Path(__file__).resolve(),
                 REPO / "scripts/prepare_weekly_nonlinear_epoch_shuffle_mse.py", REPO / "scripts/weekly_nonlinear_epoch_shuffle_mse_post_training.py",
                 REPO / "scripts/prepare_bounded.py", REPO / "scripts/train_cpu.py", REPO / "scripts/benchmark.py",
                 REPO / "scripts/weekly_nonlinear_epoch_shuffle_mse_order.py"):
        merge(before, {str(path): info(path)})
    for path in (Path.home() / ".cargo/config", Path.home() / ".cargo/config.toml"):
        if path.exists():
            merge(before, {str(path): info(path)})
    for reference in [variant["manifest"], variant["patch"], *variant["postimages"].values()]:
        merge(before, {reference["path"]: small(reference)})
    before = verify_map(before)
    environment = {"CARGO_TARGET_DIR": str(target), "RUSTC": str(rustc),
                   "RUSTFLAGS": "-C target-cpu=x86-64-v3", "CARGO_BUILD_JOBS": "2"}
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CARGO_", "RUSTC", "RUSTDOC"))
           and key not in ("RUSTFLAGS", "RUSTUP_TOOLCHAIN")}
    env.update(environment)
    env.update(LC_ALL="C", CARGO_HOME=str(Path.home() / ".cargo"), CARGO_INCREMENTAL="0",
               RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", SEKIREI_TRAIN_FTZ_DAZ="1")
    env["PATH"] = str(rustc.parent) + os.pathsep + env.get("PATH", "")
    common = ["--package", "sekirei-train", "--bin", "train", "--release", "--offline", "--locked",
              "--jobs", "2", "--features", "nnue_white_view_aux_tied"]
    commands = [[str(cargo), "test", *common, "paired_nonlinear", "--", "--test-threads=1"],
                [str(cargo), "build", *common]]
    outcomes = []
    with ExitStack() as locks:
        locks.enter_context(nonblocking_lock(runtime / ".build.lock", True))
        locks.enter_context(nonblocking_lock(runtime / ".training.lock", True))
        coordination = {engine_path.parent / ".prepare.lock", engine_path.parent / ".benchmark.lock"}
        coordination.update(path.resolve(strict=True) for path in args.coordination_lock)
        for path in sorted(coordination):
            locks.enter_context(nonblocking_lock(path, True))
        try:
            for i, command in enumerate(commands):
                outcome = supervised_command(command, runtime / f"build-step-{i}.log", env, args.seconds, source)
                outcomes.append(outcome)
                print(json.dumps({"stage": i, "returncode": outcome["returncode"],
                                  "wall_seconds": outcome["wall_seconds"]}), flush=True)
            after = verify_map(before)
            final_names = run(["git", "ls-files", "-z"], source).decode().split("\x00")[:-1]
            final_names += run(["git", "ls-files", "--others", "--exclude-standard", "-z"], source).decode().split("\x00")[:-1]
            require(set(sources) == {str(source / name) for name in final_names}
                    and run(["git", "rev-parse", "HEAD"], source).decode().strip() == UPSTREAM,
                    "source membership or upstream revision changed during build")
            actual_deps = registry_dependencies(target / "release/deps")
            require(all(dependencies.get(path) == identity for path, identity in actual_deps.items()),
                    "new or changed registry dependency after compile")
            tests = parse_test_log(Path(outcomes[0]["log"]["path"]))
            require_shuffle_tests(tests)
            binary = ref(target / "release/train")
            require(os.access(binary["path"], os.X_OK), "built trainer must be executable")
            result = {"schema": "sekirei.weekly-nonlinear-epoch-shuffle-mse-training-build.v1", "status": "complete", "mode": plan["mode"],
                      "producer_source": ref(Path(__file__).resolve()), "source_root": str(source), "upstream_commit": UPSTREAM,
                      "source_head": run(["git", "rev-parse", "HEAD"], REPO).decode().strip(), "source_files": sources,
                      "compiler_files": compiler, "dependency_files": dependencies, "engine_build": ref(engine_path),
                      "training_binary": binary, "commands": commands, "environment": environment, "tests": tests,
                      "process_outcomes": outcomes, "inputs_before": before, "inputs_after": after,
                      "scalar_power_cache_used": False}
            saved = write_exclusive(receipt_path, result)
            print(json.dumps({"status": "complete", "build": saved, "tests_passed": tests["passed"]}), flush=True)
            return result
        except BaseException as error:
            write_exclusive(failure_path, {"schema": "sekirei.weekly-nonlinear-epoch-shuffle-mse-training-build-failure.v1", "status": "failed",
                "producer_source": ref(Path(__file__).resolve()), "process_outcomes": outcomes,
                "error_type": type(error).__name__, "error": str(error), "actual_fit_started": False})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preparation", type=Path, required=True)
    parser.add_argument("--expected-source-preparation-sha256", required=True)
    parser.add_argument("--engine-manifest", type=Path, required=True)
    parser.add_argument("--expected-engine-manifest-sha256", required=True)
    parser.add_argument("--dependency-reference", type=Path, required=True)
    parser.add_argument("--coordination-lock", type=Path, action="append", default=[])
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    require(1 <= args.seconds <= 1800, "build/test deadline must be in 1..1800 seconds")
    build(args)


if __name__ == "__main__":
    main()
