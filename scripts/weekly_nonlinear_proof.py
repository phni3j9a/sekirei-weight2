#!/usr/bin/env python3
"""Reconstruct this weekly candidate's numerical and fixed-core proof evidence.

The old pure numerical consumers remain immutable. This outer binds a fresh
weekly training context; it never calls their old candidate/origin outer gate.
All probe activation, compilation, split streams and process cleanup are saved.
Technical success is a prerequisite for evaluation, not model adoption.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from benchmark import nonblocking_lock
from prepare_bounded import cleanup
from train_cpu import termination_guard
from weekly_nonlinear_post_training import PhysicalBytes, epoch_observations, require, small
from weekly_nonlinear_run import (REPO, command, collect_refs, group_alive, info,
    load_numeric_gate, merge, physical_ref, read_ref, require_clear_processes,
    source_map, strict_json, verify_map, write_new)


PROOF_DIR = REPO / "preparations/white-view-paired-nonlinear-proof-v1"
PROOF_SOURCES = {
    "core": ("paired_nonlinear_core_pair_probe.rs", "ba362852b1e86a765ff06ab1036158eb06604bfbfeb04f8d3f831df9c116cf9b"),
    "incremental": ("paired_nonlinear_incremental_probe.rs", "5d6c1fede3e686841ff74d16e7c40bc822e394415d6ff14c37a69ba09a266e8b"),
    "native_contract": ("paired_nonlinear_native_contract.rs", "95507c5360cdd1b0929369b8a15ebaf3ea621c2c1ffaf50f69f2580c9e27a5b3"),
}
FIXTURES = REPO / "tests/fixtures/material_init.json"
FIXTURE_SHA = "18339a0fc5aa274b3f9cc2ae4c8d898aff5929b980cc947d709e3f6330cf1022"
COUNTS = {"train": 112681, "holdout": 5895, "fixtures": 15}
TOTAL_ROWS = 118591
OLD_PUBLIC_ROOT = Path("/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement")


def engine_input_inventory(manifest, identity, manifest_ref, identity_ref, gate):
    """Keep the historical manifest intact; bind relocated public bytes separately.

    Only raw inputs declared at both frozen build endpoints may be relocated.
    Runtime source, compiler, binaries, logs and dependency paths retain their
    literal identities. No missing private input can use this public mapping.
    """
    original = gate.build.immutable_inputs_for_runtime(
        {"manifest": manifest, "identity": identity}, manifest_ref)
    gate.identity_map(original)
    current, relocations = {}, {}
    for path, record in original.items():
        actual = Path(path)
        if actual.is_relative_to(OLD_PUBLIC_ROOT):
            relative = actual.relative_to(OLD_PUBLIC_ROOT)
            require(relative.parts and not {".", ".."} & set(relative.parts),
                    "invalid historical public relative path")
            require(gate.exact(manifest["inputs_before"].get(path), record)
                    and gate.exact(manifest["inputs_after"].get(path), record),
                    "only frozen build raw public inputs may relocate")
            actual = REPO / relative
            relocations[path] = {"path": str(actual), **record}
        merge(current, {str(actual): record})
    receipt = {"schema": "sekirei.weekly-nonlinear-engine-public-relocation.v1",
        "status": "same-byte-public-inputs-bound", "engine_manifest": manifest_ref,
        "engine_identity": identity_ref, "old_public_root": str(OLD_PUBLIC_ROOT),
        "current_public_root": str(REPO), "relocations": relocations,
        "original_inventory": original, "current_inventory": current,
        "historical_manifest_preserved": True}
    return current, receipt


def validate_engine_relocation(reference, reader, manifest, identity_ref, manifest_ref, gate):
    current, desired = engine_input_inventory(manifest, reader.json(identity_ref),
                                              manifest_ref, identity_ref, gate)
    value = reader.json(reference)
    gate.obj(value, set(desired))
    require(gate.exact(value, desired), "engine public relocation receipt differs")
    reader.refs_in(value)
    reader.map(current)
    return current


def raw_new(path, raw):
    require(type(raw) is bytes, "literal output bytes required")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return physical_ref(path)


def write_owned_json(path, value, state, key, ownership):
    """Remember ownership only after our exclusive open succeeds."""
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        state[key] = ownership
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return physical_ref(path)


def run_split(argv, stdin, stdout, stderr, env, limit, cwd):
    """Own a fresh session while preserving three independent bound streams."""
    input_ref = physical_ref(stdin)
    started, child, caught = time.monotonic(), None, None
    outcome = {"command": argv, "pid": None, "pgid": None, "returncode": None,
               "waited": False, "reaped": False, "timed_out": False,
               "group_empty_scans": [False, False]}
    with termination_guard() as state:
        try:
            with ExitStack() as streams:
                inp = streams.enter_context(stdin.open("rb"))
                out = streams.enter_context(stdout.open("xb"))
                err = streams.enter_context(stderr.open("xb"))
                try:
                    state["spawning"] = True
                    try:
                        child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=inp,
                            stdout=out, stderr=err, start_new_session=True)
                    finally:
                        state["spawning"] = False
                    outcome.update(pid=child.pid, pgid=child.pid)
                    require(state["signal"] is None, "cancelled after split child ownership")
                    try:
                        outcome["returncode"] = child.wait(timeout=limit)
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
                                waited=child.returncode is not None, reaped=child.returncode is not None)
                            outcome["group_empty_scans"] = [not group_alive(child.pid)]
                            time.sleep(0.05)
                            outcome["group_empty_scans"].append(not group_alive(child.pid))
                            require(outcome["group_empty_scans"] == [True, True], "split child group remains")
                    finally:
                        state["cleaning"] = False
            require(state["signal"] is None, "cancelled split child cannot complete")
            require(physical_ref(stdin) == input_ref, "split stdin changed")
        except BaseException as error:
            caught = error
            error.child_outcome = outcome
            error.child_signal = state["signal"]
            raise
        finally:
            state["cleaning"] = True
            try:
                outcome["wall_seconds"] = time.monotonic() - started
                if stderr.exists():
                    outcome["log"] = physical_ref(stderr)
                receipt = {"schema": "sekirei.weekly-nonlinear-split-child.v1",
                    "status": "failed" if caught is not None else "observed-exit",
                    "execution": outcome, "signal": state["signal"], "stdin_before": input_ref,
                    "stdin_after": physical_ref(stdin) if stdin.exists() else None,
                    "stdout": physical_ref(stdout) if stdout.exists() else None}
                if caught is not None:
                    receipt.update(error_type=type(caught).__name__, error=str(caught))
                write_new(Path(str(stderr) + ".child-outcome.json"), receipt)
            finally:
                state["cleaning"] = False
        require(state["signal"] is None, "cancelled while saving split child")
    return outcome


def enabled_probe(raw, kind):
    name, digest = PROOF_SOURCES[kind]
    require(hashlib.sha256(raw).hexdigest() == digest, "frozen proof source differs: " + name)
    if kind == "native_contract":
        return raw
    marker = b"const PROTOTYPE_ONLY:bool=true;"
    require(raw.count(marker) == 1, "sole probe activation bit required")
    return raw.replace(marker, b"const PROTOTYPE_ONLY:bool=false;", 1)


def validate_activation(raw, kind):
    if kind == "native_contract":
        require(hashlib.sha256(raw).hexdigest() == PROOF_SOURCES[kind][1], "native production source changed")
    else:
        marker = b"const PROTOTYPE_ONLY:bool=false;"
        require(raw.count(marker) == 1 and hashlib.sha256(raw.replace(marker,
            b"const PROTOTYPE_ONLY:bool=true;", 1)).hexdigest() == PROOF_SOURCES[kind][1],
            "enabled probe differs beyond sole activation bit")


def validate_scans(scans, allowlist, gate):
    gate.obj(allowlist, {"schema", "status", "services"})
    gate.fixed(allowlist, {"schema": "sekirei.weekly-nonlinear-heavy-process-allowlist.v1", "status": "frozen-before-preflight"})
    require(type(allowlist["services"]) is list, "typed allowed service inventory required")
    services = {}
    for service in allowlist["services"]:
        gate.obj(service, {"pid", "uid", "comm", "starttime_ticks", "cmdline_sha256"})
        for key, minimum in (("pid", 1), ("uid", 0), ("starttime_ticks", 1)):
            require(type(service[key]) is int and minimum <= service[key] <= 0xffffffffffffffff, "strict service integer required")
        require(type(service["comm"]) is str and service["comm"] and service["pid"] not in services, "duplicate/invalid allowed service")
        if service["cmdline_sha256"] is not None:
            gate.sha(service["cmdline_sha256"])
        services[service["pid"]] = service
    require(type(scans) is list and len(scans) == 2, "two process scans required")
    for scan in scans:
        gate.obj(scan, {"conflicts", "excluded_preexisting_services"})
        gate.fixed(scan, {"conflicts": []})
        require(type(scan["excluded_preexisting_services"]) is list, "typed observed service list required")
        seen = set()
        for service in scan["excluded_preexisting_services"]:
            require(type(service) is dict and type(service.get("pid")) is int and service["pid"] not in seen
                and gate.exact(service, services.get(service["pid"])), "unbound, changed or duplicate excluded service")
            seen.add(service["pid"])


def validate_completion(value, bound, completion_ref, reader, gate):
    recipe, sb, parent, plan = (bound[key] for key in ("recipe", "source_binding", "parent", "plan"))
    gate.obj(value, {"schema", "status", "mode", "plan", "source_binding", "parent_preflight",
        "training_build", "engine_build", "recipe", "execution", "initialized_io_parent", "dataset_inputs",
        "reference03", "outputs", "fp_evidence", "native_functional_bounds", "process_scan_records_before",
        "process_scan_records_after", "inputs_before", "inputs_after", "source_unchanged", "build_unchanged",
        "poisoned", "resume_used", "shuffle_used", "final_used", "core_fullrows_verified", "incremental_verified",
        "adoption_claimed"})
    gate.fixed(value, {"schema": "sekirei.weekly-nonlinear-training-completion.v1", "status": "complete",
        "mode": recipe["mode"], "plan": sb["selected_plan"], "source_binding": recipe["source_binding"],
        "parent_preflight": sb["parent_preflight"], "training_build": sb["training_build"],
        "engine_build": sb["engine_build"], "dataset_inputs": sb["dataset_inputs"], "reference03": recipe["reference03"],
        "source_unchanged": True, "build_unchanged": True, "poisoned": False, "resume_used": False,
        "shuffle_used": False, "final_used": False, "core_fullrows_verified": False,
        "incremental_verified": False, "adoption_claimed": False})
    require(gate.exact(value["inputs_before"], value["inputs_after"]), "training immutable inputs changed")
    reader.map(value["inputs_before"])
    reader.refs_in(value)
    gate.fullref(completion_ref)
    gate.obj(value["outputs"], {"checkpoint", "native03", "export_stage"})
    native_path = Path(value["outputs"]["native03"]["path"])
    output = native_path.parent
    require(native_path.name == "weights.nearest03.bin", "literal native export name required")
    for role, name in (("checkpoint", "fullstate.bits.json"), ("export_stage", "export-stage.json")):
        require(Path(value["outputs"][role]["path"]) == output / name, "training output tree differs")
    gate.lifecycle(value["execution"], command(value["recipe"], recipe, sb["training_binary"]["path"], output),
                   plan["resources"]["training_wall_limit_seconds"])
    child_inputs = dict(parent["source_inputs"])
    collect_refs(child_inputs, [value["recipe"], recipe["source_binding"], sb["parent_preflight"]])
    context = {"recipe": value["recipe"], "source_binding": recipe["source_binding"], "manifest": recipe["manifest"],
        "reference03": recipe["reference03"], "positions": recipe["positions"], "labels": recipe["labels"],
        "source_files": child_inputs}
    for path, identity in child_inputs.items():
        require(gate.exact(value["inputs_before"].get(path), identity), "training lost child context closure")
    for name in ("weekly_nonlinear_run.py", "weekly_nonlinear_post_training.py", "weekly_nonlinear_preflight.py",
                 "train_cpu.py", "prepare_bounded.py", "benchmark.py"):
        path = REPO / "scripts" / name
        require(gate.exact(value["inputs_before"].get(str(path)), info(path)), "training parent/helper source changed")
    native = reader.read(value["outputs"]["native03"])
    bounds = gate.checkpoint(reader.read(value["outputs"]["checkpoint"]), context,
                             reader.read(recipe["reference03"]), native)
    stage = reader.json(value["outputs"]["export_stage"])
    gate.export_stage(stage, value["outputs"], context, child_inputs, native)
    require(gate.exact(value["native_functional_bounds"], bounds), "completed numerical bounds differ")
    fp = value["fp_evidence"]
    require(gate.exact(fp["epoch_observations"], epoch_observations(reader.read(value["execution"]["log"]), gate=gate)),
            "training epoch observations differ")
    require(gate.exact(fp["guard_source"], physical_ref(Path(bound["build"]["source_root"]) /
            "crates/sekirei-train/src/paired_nonlinear_float.rs")), "training float guard source differs")
    gate.float_evidence(fp, reader, value["execution"]["log"], stage["float_export"])
    allowlist = reader.json(parent["process_evidence"]["allowlist"])
    for scans in (value["process_scan_records_before"], value["process_scan_records_after"]):
        validate_scans(scans, allowlist, gate)
    return context, native, bounds


def dataset_rows(reader, refs, gate):
    """Full weekly joins; the manifest is already bound by the new source reader."""
    manifest = reader.json(refs["manifest"])
    gate.fixed(manifest, {"schema_version": 1, "teacher_identity": gate.TEACHER,
                          "positions": {"train": COUNTS["train"], "holdout": COUNTS["holdout"]}})
    require(manifest["derivation"]["kind"] == "whole-pack-hash-ranked-frozen-holdout-v2",
            "weekly dataset derivation required")
    games = {row["game_id"]: row["split"] for row in manifest["games"]}
    result, boards = {}, {}
    for split in ("train", "holdout"):
        positions = gate.rows(reader.read(refs[split + "_positions"]))
        labels = gate.rows(reader.read(refs[split + "_labels"]))
        require(len(positions) == len(labels) == COUNTS[split], "complete split row counts differ")
        require(gate.exact(manifest["files"][split + ".positions.jsonl"], small(refs[split + "_positions"]))
                and gate.exact(manifest["files"][split + ".labels.jsonl"], small(refs[split + "_labels"])),
                "manifest split byte identities differ")
        cache, sfens = {}, []
        for row in labels:
            gate.obj(row, {"sfen", "score_cp", "label_depth", "teacher_identity"})
            gate.fixed(row, {"label_depth": 0, "teacher_identity": gate.TEACHER})
            require(type(row["score_cp"]) is int and abs(row["score_cp"]) < 30000, "teacher score type/domain differs")
            gate.material(row["sfen"])
            require(row["sfen"] not in cache, "duplicate split label")
            cache[row["sfen"]] = row["score_cp"]
        for row in positions:
            gate.obj(row, {"schema_version", "sfen", "source", "tags"})
            gate.fixed(row, {"schema_version": 1})
            gate.obj(row["source"], {"kind", "path", "ply"})
            gate.fixed(row["source"], {"kind": "gensfen-pack"})
            require(type(row["source"]["path"]) is str and games.get(row["source"]["path"]) == split,
                    "position game/split differs")
            require(type(row["source"]["ply"]) is int and row["source"]["ply"] >= 16
                    and row["sfen"].split(" ")[3] == str(row["source"]["ply"]), "position/SFEN ply differs")
            gate.obj(row["tags"], {"side_to_move", "phase"})
            gate.fixed(row["tags"], {"phase": "middlegame",
                "side_to_move": "black" if row["sfen"].split(" ")[1] == "b" else "white"})
            gate.material(row["sfen"])
            sfens.append(row["sfen"])
        require(len(set(sfens)) == len(sfens) and set(sfens) == set(cache), "full exact SFEN/cache join differs")
        boards[split] = {" ".join(s.split(" ")[:3]) for s in sfens}
        require(len(boards[split]) == len(sfens), "duplicate split board")
        result[split] = sfens
    require(not boards["train"] & boards["holdout"], "train/holdout board overlap")
    fixtures = reader.json(refs["fixtures"])
    require(type(fixtures) is list and len(fixtures) == 15, "fixed public15 required")
    for fixture in fixtures:
        require(type(fixture["expected_cp_stm"]) is int and gate.material(fixture["sfen"]) == fixture["expected_cp_stm"],
                "public fixture material differs")
    result["fixtures"] = [fixture["sfen"] for fixture in fixtures]
    return result, fixtures


def core_summary(raw, stderr, sfens, bounds, gate):
    gate.core_stderr(stderr, len(sfens), bounds)
    lines = raw.splitlines()
    require(len(lines) == len(sfens), "missing/extra core rows")
    maximum, bridge = 0, 0.0
    for index, (line, sfen) in enumerate(zip(lines, sfens)):
        row = gate.probe.core_row(line, index, gate.material(sfen))
        require(abs(row["native_quantized_float_cp"]) <= bounds["absolute_cp_upper"], "observed native bound exceeded")
        maximum = max(maximum, abs(row["native_core_cp"] - row["material_cp"]))
        bridge = max(bridge, abs(row["native_quantized_float_cp"] - row["native_core_cp"]))
    return {"count": len(sfens), "maximum_material_difference_cp": maximum,
            "maximum_stored_float_core_bridge_cp": bridge}


def incremental_summary(raw, bounds, gate):
    result = gate.probe.incremental_result(gate.strict_json(raw))
    require(result["native_q_l1_upper"] <= bounds["q_l1_upper"]
            and result["native_absolute_cp_upper"] <= bounds["absolute_cp_upper"], "incremental functional bounds differ")
    return result


def sidecar(native, completion_ref, completed, gate):
    require(type(native) is bytes and len(native) == gate.NATIVE_BYTES and native[:8] == b"SEKIRW03", "native03 ABI required")
    return {"format": "sekirei-nnue-output-v1", "nnue_output": "absolute", "checkpoint_hash": gate.fnv(native),
        "baseline": None, "schema": "sekirei.weekly-nonlinear-native-sidecar.v1", "mode": completed["mode"],
        "feature_schema": gate.FEATURE, "native_magic": "SEKIRW03", "native": completed["outputs"]["native03"],
        "weight_sha256": hashlib.sha256(native).hexdigest(), "weight_bytes": len(native),
        "checkpoint": completed["outputs"]["checkpoint"], "training_completion": completion_ref,
        "recipe": completed["recipe"], "source_binding": completed["source_binding"],
        "selected_epoch": 3, "epochs": 3, "adam_used": True, "global_step": 338043,
        "resume_used": False, "shuffle_used": False, "scalar_power_cache_used": False,
        "final_used": False, "model_adopted": False}


def compile_command(compiler, rlib, source, binary):
    return [compiler["path"], "--edition=2024", "-C", "target-cpu=x86-64-v3", "-C", "opt-level=3",
        "-C", "panic=abort", "--extern", "sekirei_core=" + rlib["path"], "-L",
        "dependency=" + str(Path(rlib["path"]).parent), source["path"], "-o", str(binary)]


def proof_environment():
    values = {key: value for key, value in os.environ.items()
              if not key.startswith(("CARGO_", "RUSTC", "RUSTDOC", "RUSTFLAGS")) and key != "RUSTUP_TOOLCHAIN"}
    values.update(SEKIREI_TRAIN_FTZ_DAZ="1", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", LC_ALL="C")
    return values


def validate_initialized(value, reader, recipe_ref, recipe, binary, context, inputs, gate):
    gate.obj(value, {"schema", "status", "recipe", "stage", "stage_outputs", "child", "missing_policy_child",
                     "inputs_before", "inputs_after", "actual_fit_started", "model_adopted", "final_used"})
    gate.fixed(value, {"schema": "sekirei.weekly-nonlinear-initialized-io-parent.v1", "status": "complete",
        "recipe": recipe_ref, "inputs_before": inputs, "inputs_after": inputs,
        "actual_fit_started": False, "model_adopted": False, "final_used": False})
    reader.refs_in(value)
    child, negative = value["child"], value["missing_policy_child"]
    argv = child["command"]
    require(type(argv) is list and len(argv) == 40 and argv[1] == "verify-paired-nonlinear-inputs",
            "actual initialized diagnostic argv required")
    directory = Path(value["stage"]["path"]).parent
    desired = command(recipe_ref, recipe, binary, directory)
    desired[1] = "verify-paired-nonlinear-inputs"
    require(gate.exact(argv, desired), "initialized diagnostic literal recipe/binary argv differs")
    negative_argv = command(recipe_ref, recipe, binary, directory.parent / "missing-policy-output")
    negative_argv[1] = "verify-paired-nonlinear-inputs"
    require(gate.exact(negative["command"], negative_argv), "negative diagnostic literal argv differs")
    require(not Path(negative_argv[-1]).exists(), "negative diagnostic created an output tree")
    gate.lifecycle(child, argv, 3600)
    gate.obj(negative, set(child))
    gate.fixed(negative, {"returncode": 1, "waited": True, "reaped": True, "timed_out": False,
                         "group_empty_scans": [True, True]})
    require(type(negative["pid"]) is int and negative["pid"] > 0 and negative["pid"] == negative["pgid"],
            "negative diagnostic session differs")
    require(type(negative["wall_seconds"]) in (int, float) and 0 <= negative["wall_seconds"] < 60,
            "negative diagnostic deadline differs")
    require(b"SEKIREI_TRAIN_FTZ_DAZ missing or not Unicode" in reader.read(negative["log"]),
            "negative diagnostic failed for an unexpected reason")
    stage = reader.json(value["stage"])
    gate.obj(stage, {"schema", "status", "mode", "train_count", "global_step", "epochs_completed", "candidate_model",
        "resume_allowed", "float_observations", "context", "outputs", "inputs_before", "inputs_after",
        "all_state_bits_equal", "all_nearest_bytes_equal", "compiled_native_reader_fullbytes_equal",
        "parent_reap_verified", "actual_fit_started", "model_adopted", "final_used"})
    gate.fixed(stage, {"schema": "sekirei.white-view-paired-nonlinear-initialized-io-stage.v1",
        "status": "verified-original-inputs-and-initialized-codecs", "mode": recipe["mode"], "train_count": 112681,
        "global_step": 0, "epochs_completed": 0, "candidate_model": False, "resume_allowed": False,
        "context": context, "inputs_before": inputs, "inputs_after": inputs, "outputs": value["stage_outputs"],
        "all_state_bits_equal": True, "all_nearest_bytes_equal": True, "compiled_native_reader_fullbytes_equal": True,
        "parent_reap_verified": False, "actual_fit_started": False, "model_adopted": False, "final_used": False})
    gate.obj(stage["float_observations"], {"before", "after_load", "after_io"})
    for snapshot in stage["float_observations"].values():
        gate.float_snapshot(snapshot)
    gate.obj(value["stage_outputs"], {"initialized_checkpoint", "initialized_native03"})
    for key, name in (("initialized_checkpoint", "initialized-fullstate.bits.json"), ("initialized_native03", "initialized.nearest03.bin")):
        require(Path(value["stage_outputs"][key]["path"]) == directory / name, "initialized output path differs")


def validate_split_receipt(reference, reader, execution, stdin, stdout, stderr, gate):
    value = reader.json(reference)
    gate.obj(value, {"schema", "status", "execution", "signal", "stdin_before", "stdin_after", "stdout"})
    gate.fixed(value, {"schema": "sekirei.weekly-nonlinear-split-child.v1", "status": "observed-exit",
        "execution": execution, "signal": None, "stdin_before": stdin, "stdin_after": stdin, "stdout": stdout})
    require(gate.exact(execution["log"], stderr), "split executed stderr differs")
    reader.refs_in(value)


def validate_technical_proof(value, reader, bound, completed, completion_ref, manifest, identity_ref, gate):
    """Dedicated weekly outer consumer; reparse raw observations, never old roles."""
    keys = {"schema", "status", "mode", "producer_source", "training_completion", "recipe", "source_binding", "plan",
        "training_build", "engine_manifest", "engine_identity", "context", "native03", "checkpoint", "reference03",
        "engine_input_relocation", "metadata", "source_activation", "compilations", "counts", "total_core_rows", "core", "incremental",
        "native_functional_bounds", "inputs_before", "inputs_after", "original_inputs_before", "original_inputs_after",
        "lock_records", "process_scan_records_before", "process_scan_records_after", "source_unchanged", "build_unchanged",
        "core_fullrows_verified", "incremental_verified", "protected_material_ft_bits_preserved", "whole_board_ft_preservation_claimed",
        "universal_integer_bound_claimed", "native_bitexact_covariance_claimed", "resume_used", "shuffle_used", "final_used",
        "model_adopted", "adoption_claimed"}
    gate.obj(value, keys)
    recipe, sb = bound["recipe"], bound["source_binding"]
    context, native, bounds = validate_completion(completed, bound, completion_ref, reader, gate)
    gate.fixed(value, {"schema": "sekirei.weekly-nonlinear-model-technical-proof.v1", "status": "complete",
        "mode": recipe["mode"], "training_completion": completion_ref, "recipe": completed["recipe"],
        "source_binding": recipe["source_binding"], "plan": sb["selected_plan"], "training_build": sb["training_build"],
        "engine_manifest": sb["engine_build"], "engine_identity": identity_ref, "context": context,
        "native03": completed["outputs"]["native03"], "checkpoint": completed["outputs"]["checkpoint"],
        "reference03": recipe["reference03"], "counts": COUNTS, "total_core_rows": TOTAL_ROWS,
        "native_functional_bounds": bounds, "source_unchanged": True, "build_unchanged": True,
        "core_fullrows_verified": True, "incremental_verified": True, "protected_material_ft_bits_preserved": True,
        "whole_board_ft_preservation_claimed": False, "universal_integer_bound_claimed": False,
        "native_bitexact_covariance_claimed": False, "resume_used": False, "shuffle_used": False,
        "final_used": False, "model_adopted": False, "adoption_claimed": False})
    for before, after in (("inputs_before", "inputs_after"), ("original_inputs_before", "original_inputs_after")):
        require(gate.exact(value[before], value[after]), "proof raw immutable inventory changed")
        reader.map(value[before])
    reader.refs_in(value)
    engine_inputs = validate_engine_relocation(value["engine_input_relocation"], reader, manifest,
                                               identity_ref, value["engine_manifest"], gate)
    for path, record in engine_inputs.items():
        require(gate.exact(value["original_inputs_before"].get(path), record),
                "proof lost fixed engine current input closure")
    require(gate.exact(value["producer_source"], physical_ref(Path(__file__).resolve())), "proof producer source changed")
    require(gate.exact(reader.json(value["metadata"]), sidecar(native, completion_ref, completed, gate)),
            "native inference sidecar context/SHA/FNV differs")
    validate_initialized(reader.json(completed["initialized_io_parent"]), reader, completed["recipe"], recipe,
                         sb["training_binary"]["path"], context, context["source_files"], gate)
    activation = reader.json(value["source_activation"])
    gate.obj(activation, {"schema", "status", "mode", "training_completion", "sources", "native_contract_model_mode",
        "candidate_identity_bound_by_outer_context", "compiled", "probe_started", "model_adopted", "final_used"})
    gate.fixed(activation, {"schema": "sekirei.weekly-nonlinear-proof-source-activation.v1", "status": "source-prepared",
        "mode": recipe["mode"], "training_completion": completion_ref, "native_contract_model_mode": gate.MODE,
        "candidate_identity_bound_by_outer_context": True, "compiled": False, "probe_started": False,
        "model_adopted": False, "final_used": False})
    gate.obj(activation["sources"], set(PROOF_SOURCES))
    for kind, item in activation["sources"].items():
        gate.obj(item, {"prototype", "enabled", "production_code_preserved", "prototype_only", "sole_delta"})
        gate.fixed(item, {"production_code_preserved": True, "prototype_only": False if kind != "native_contract" else None,
            "sole_delta": "PROTOTYPE_ONLY true -> false" if kind != "native_contract" else "none"})
        require(item["prototype"]["sha256"] == PROOF_SOURCES[kind][1], "prototype SHA differs")
        require(reader.read(item["enabled"]) == enabled_probe(reader.read(item["prototype"]), kind),
                "actual activated source differs beyond authorized bit")
        require(Path(item["enabled"]["path"]).name == PROOF_SOURCES[kind][0]
                and Path(item["enabled"]["path"]).parent ==
                    Path(activation["sources"]["native_contract"]["enabled"]["path"]).parent,
                "probe and exact adjacent native module paths required")
    gate.obj(value["compilations"], {"core", "incremental"})
    for kind, compilation in value["compilations"].items():
        gate.obj(compilation, {"schema", "status", "kind", "engine_manifest", "engine_identity", "compiler", "core_rlib",
            "release_dependencies_before", "release_dependencies_after", "sources", "binary", "command", "execution",
            "stdin", "stdout", "stderr", "child_receipt", "inputs_before", "inputs_after"})
        gate.fixed(compilation, {"schema": "sekirei.weekly-nonlinear-proof-compilation.v1", "status": "complete", "kind": kind,
            "engine_manifest": sb["engine_build"], "engine_identity": identity_ref, "core_rlib": manifest["core_link"]["rlib"],
            "release_dependencies_before": manifest["release_dependencies_after_probes"],
            "release_dependencies_after": manifest["release_dependencies_after_probes"],
            "sources": {"probe": activation["sources"][kind]["enabled"],
                        "native_contract": activation["sources"]["native_contract"]["enabled"]}})
        require(gate.exact(compilation["inputs_before"], compilation["inputs_after"]), "compile inputs changed")
        reader.map(compilation["inputs_before"])
        for path, record in manifest["release_dependencies_after_probes"].items():
            require(gate.exact(compilation["inputs_before"].get(path), record), "compile lost full linked dependency closure")
        compiler = compilation["compiler"]
        require(Path(compiler["path"]).name == "rustc" and gate.exact(manifest["compiler_files"].get(compiler["path"]), small(compiler)),
                "proof compiler not fixed White compiler")
        argv = compile_command(compiler, compilation["core_rlib"], compilation["sources"]["probe"], Path(compilation["binary"]["path"]))
        require(gate.exact(compilation["command"], argv), "literal proof compile argv differs")
        gate.lifecycle(compilation["execution"], argv, 1200)
        require(reader.read(compilation["stdin"]) == b"", "compile stdin must be empty")
        validate_split_receipt(compilation["child_receipt"], reader, compilation["execution"], compilation["stdin"],
                               compilation["stdout"], compilation["stderr"], gate)
    refs = {"manifest": recipe["manifest"], "fixtures": physical_ref(FIXTURES)}
    for split in ("train", "holdout"):
        for role in ("positions", "labels"):
            refs[split + "_" + role] = sb["dataset_inputs"][split + "." + role + ".jsonl"]
    positions, fixtures = dataset_rows(reader, refs, gate)
    gate.obj(value["core"], set(COUNTS))
    for split, item in value["core"].items():
        gate.obj(item, {"positions", "stdin", "stdout", "stderr", "execution", "inputs_before", "inputs_after",
            "child_receipt", "count", "maximum_material_difference_cp", "maximum_stored_float_core_bridge_cp"})
        require(gate.exact(item["positions"], refs["fixtures" if split == "fixtures" else split + "_positions"]), "core split source differs")
        require(gate.exact(item["inputs_before"], item["inputs_after"]), "core inputs changed")
        reader.map(item["inputs_before"])
        require(reader.read(item["stdin"]) == "".join(s + "\n" for s in positions[split]).encode(), "core full input order differs")
        argv = [value["compilations"]["core"]["binary"]["path"], value["native03"]["path"], value["reference03"]["path"], "x86-ftz-daz"]
        gate.lifecycle(item["execution"], argv, 600)
        validate_split_receipt(item["child_receipt"], reader, item["execution"], item["stdin"], item["stdout"], item["stderr"], gate)
        gate.fixed(item, core_summary(reader.read(item["stdout"]), reader.read(item["stderr"]), positions[split], bounds, gate))
    inc = value["incremental"]
    gate.obj(inc, {"fixture_tsv", "stdin", "stdout", "stderr", "execution", "child_receipt", "result", "inputs_before", "inputs_after"})
    require(gate.exact(inc["inputs_before"], inc["inputs_after"]), "incremental inputs changed")
    reader.map(inc["inputs_before"])
    require(reader.read(inc["stdin"]) == reader.read(inc["stderr"]) == b"", "incremental stdin/stderr must be empty")
    require(reader.read(inc["fixture_tsv"]) == "".join(str(f["expected_cp_stm"]) + "\t" + f["sfen"] + "\n" for f in fixtures).encode(),
            "public15 incremental fixture bytes/order differ")
    argv = [value["compilations"]["incremental"]["binary"]["path"], value["native03"]["path"], value["reference03"]["path"], inc["fixture_tsv"]["path"]]
    gate.lifecycle(inc["execution"], argv, 1200)
    validate_split_receipt(inc["child_receipt"], reader, inc["execution"], inc["stdin"], inc["stdout"], inc["stderr"], gate)
    require(gate.exact(inc["result"], incremental_summary(reader.read(inc["stdout"]), bounds, gate)), "incremental raw result differs")
    for scans in (value["process_scan_records_before"], value["process_scan_records_after"]):
        validate_scans(scans, reader.json(bound["parent"]["process_evidence"]["allowlist"]), gate)
    require(type(value["lock_records"]) is list and len(value["lock_records"]) >= 5, "proof lock closure required")
    seen = set()
    for lock in value["lock_records"]:
        gate.obj(lock, {"path", "exclusive", "nonblocking", "acquired"})
        gate.absolute(lock["path"])
        gate.fixed(lock, {"exclusive": True, "nonblocking": True, "acquired": True})
        require(lock["path"] not in seen, "duplicate proof lock")
        seen.add(lock["path"])
    return bounds


def run(args, state):
    from weekly_nonlinear_preflight import validate_source_binding
    completion_ref = physical_ref(args.completion)
    require(completion_ref["sha256"] == args.expected_completion_sha256, "completion external SHA differs")
    completed = strict_json(read_ref(completion_ref))
    bound = validate_source_binding(completed["recipe"])
    recipe, sb, build = (bound[key] for key in ("recipe", "source_binding", "build"))
    gate, pure_sources = load_numeric_gate(args.numeric_build_contract)
    identity_ref = physical_ref(args.engine_identity)
    require(identity_ref["sha256"] == args.expected_engine_identity_sha256, "fixed engine identity external SHA differs")
    identity = strict_json(read_ref(identity_ref))
    manifest_ref = sb["engine_build"]
    manifest = strict_json(read_ref(manifest_ref))
    require(gate.exact(manifest["identity"], identity_ref), "engine identity path or raw identity differs")
    gate.build.validate_manifest(manifest, identity, identity_ref)
    expected = dict(completed["inputs_before"])
    collect_refs(expected, [completion_ref, completed, identity_ref, manifest_ref])
    engine_inputs, relocation = engine_input_inventory(manifest, identity, manifest_ref, identity_ref, gate)
    merge(expected, engine_inputs)
    merge(expected, pure_sources)
    for name in ("weekly_nonlinear_proof.py", "weekly_nonlinear_run.py", "weekly_nonlinear_post_training.py",
                 "weekly_nonlinear_preflight.py", "train_cpu.py", "prepare_bounded.py", "benchmark.py"):
        merge(expected, {str(REPO / "scripts" / name): info(REPO / "scripts" / name)})
    for name, digest in PROOF_SOURCES.values():
        reference = physical_ref(PROOF_DIR / name)
        require(reference["sha256"] == digest, "public proof source changed")
        collect_refs(expected, reference)
    fixtures_ref = physical_ref(FIXTURES)
    require(fixtures_ref["sha256"] == FIXTURE_SHA, "public fixture bytes changed")
    collect_refs(expected, fixtures_ref)
    for path in (args.completion, args.engine_identity, args.numeric_build_contract, args.output, args.stock_runtime):
        require(path.is_absolute() and path.resolve() == path, "canonical absolute arguments required")
    require(not args.output.exists() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.is_relative_to(REPO), "fresh private proof output required")
    for path in expected:
        require(not args.output.is_relative_to(path) and not Path(path).is_relative_to(args.output), "proof output/input overlap")
    source_root = Path(build["source_root"])
    runtime = Path(sb["training_binary"]["path"]).parents[2]
    allowlist = strict_json(read_ref(bound["parent"]["process_evidence"]["allowlist"]))
    locks = sorted({args.output.parent / ".heavy.lock", runtime / ".build.lock", runtime / ".training.lock",
        Path(manifest_ref["path"]).parent / ".prepare.lock", Path(manifest_ref["path"]).parent / ".benchmark.lock",
        args.stock_runtime / ".benchmark.lock"})
    with ExitStack() as stack:
        for path in locks:
            stack.enter_context(nonblocking_lock(path, exclusive=True))
        scans_before = require_clear_processes(allowlist)
        validate_source_binding(completed["recipe"])
        require(source_map(source_root) == sb["training_source_files"], "training source membership changed")
        before = verify_map(expected)
        reader = PhysicalBytes(expected, gate=gate, physical_ref=physical_ref)
        context, native, bounds = validate_completion(completed, bound, completion_ref, reader, gate)
        refs = {"manifest": recipe["manifest"], "reference03": recipe["reference03"], "fixtures": fixtures_ref}
        for split in ("train", "holdout"):
            for role in ("positions", "labels"):
                refs[split + "_" + role] = sb["dataset_inputs"][split + "." + role + ".jsonl"]
        positions, fixtures = dataset_rows(reader, refs, gate)
        require(shutil.disk_usage(args.output.parent).free >= bound["plan"]["resources"]["minimum_ssd_remaining_bytes"] + 2**30,
                "insufficient SSD reserve and padded proof space")
        args.output.mkdir(mode=0o700)
        (args.output / "src").mkdir(mode=0o700)
        relocation_ref = write_new(args.output / "engine-public-input-relocation.json", relocation)
        collect_refs(expected, relocation_ref)
        validate_engine_relocation(relocation_ref, PhysicalBytes(expected, gate=gate,
            physical_ref=physical_ref), manifest, identity_ref, manifest_ref, gate)
        activation = {}
        for kind, (name, _) in PROOF_SOURCES.items():
            prototype = physical_ref(PROOF_DIR / name)
            raw = enabled_probe(read_ref(prototype), kind)
            enabled = raw_new(args.output / "src" / name, raw)
            validate_activation(read_ref(enabled), kind)
            activation[kind] = {"prototype": prototype, "enabled": enabled,
                "production_code_preserved": True, "prototype_only": False if kind != "native_contract" else None,
                "sole_delta": "PROTOTYPE_ONLY true -> false" if kind != "native_contract" else "none"}
            collect_refs(expected, enabled)
        activation_ref = write_new(args.output / "source-activation.json", {
            "schema": "sekirei.weekly-nonlinear-proof-source-activation.v1", "status": "source-prepared",
            "mode": recipe["mode"], "training_completion": completion_ref, "sources": activation,
            "native_contract_model_mode": gate.MODE, "candidate_identity_bound_by_outer_context": True,
            "compiled": False, "probe_started": False, "model_adopted": False, "final_used": False})
        collect_refs(expected, activation_ref)
        if args.prepare_only:
            verify_map(before)
            print(json.dumps({"status": "source-only", "source_activation": activation_ref}), flush=True)
            return
        env = proof_environment()
        empty = raw_new(args.output / "stdin.empty", b"")
        collect_refs(expected, empty)
        compiler = physical_ref(Path(next(path for path in identity["compiler_files"] if Path(path).name == "rustc")))
        rlib = manifest["core_link"]["rlib"]
        compilations = {}
        for kind in ("core", "incremental"):
            snapshot = verify_map(expected)
            argv = compile_command(compiler, rlib, activation[kind]["enabled"], args.output / (kind + "-probe"))
            execution = run_split(argv, Path(empty["path"]), args.output / (kind + "-compile.stdout"),
                args.output / (kind + "-compile.stderr"), env, 1200, args.output)
            gate.lifecycle(execution, argv, 1200)
            binary = physical_ref(args.output / (kind + "-probe"))
            after = verify_map(snapshot)
            require(os.access(binary["path"], os.X_OK), "compiled proof binary not executable")
            compilations[kind] = {"schema": "sekirei.weekly-nonlinear-proof-compilation.v1", "status": "complete",
                "kind": kind, "engine_manifest": manifest_ref, "engine_identity": identity_ref, "compiler": compiler,
                "core_rlib": rlib, "release_dependencies_before": manifest["release_dependencies_after_probes"],
                "release_dependencies_after": manifest["release_dependencies_after_probes"],
                "sources": {"probe": activation[kind]["enabled"], "native_contract": activation["native_contract"]["enabled"]},
                "binary": binary, "command": argv, "execution": execution,
                "stdin": empty, "stdout": physical_ref(args.output / (kind + "-compile.stdout")),
                "stderr": execution["log"], "child_receipt": physical_ref(Path(execution["log"]["path"] + ".child-outcome.json")),
                "inputs_before": snapshot, "inputs_after": after}
            collect_refs(expected, [binary, execution, physical_ref(Path(execution["log"]["path"] + ".child-outcome.json")),
                                    physical_ref(args.output / (kind + "-compile.stdout"))])
            print(json.dumps({"stage": "probe-compiled", "kind": kind, "binary": binary}), flush=True)
        core = {}
        for split, sfens in positions.items():
            inp = raw_new(args.output / (split + ".sfens"), "".join(s + "\n" for s in sfens).encode())
            collect_refs(expected, inp)
            snapshot = verify_map(expected)
            out, err = args.output / (split + ".stdout.jsonl"), args.output / (split + ".stderr.log")
            argv = [compilations["core"]["binary"]["path"], completed["outputs"]["native03"]["path"], recipe["reference03"]["path"], "x86-ftz-daz"]
            execution = run_split(argv, Path(inp["path"]), out, err, env, 600, args.output)
            gate.lifecycle(execution, argv, 600)
            summary = core_summary(out.read_bytes(), err.read_bytes(), sfens, bounds, gate)
            core[split] = {"positions": refs["fixtures" if split == "fixtures" else split + "_positions"],
                "stdin": inp, "stdout": physical_ref(out), "stderr": physical_ref(err), "execution": execution,
                "child_receipt": physical_ref(Path(str(err) + ".child-outcome.json")),
                "inputs_before": snapshot, "inputs_after": verify_map(snapshot), **summary}
            collect_refs(expected, [core[split], physical_ref(Path(str(err) + ".child-outcome.json"))])
            print(json.dumps({"stage": "core-rows-verified", "split": split, **summary}), flush=True)
        tsv = raw_new(args.output / "fixtures.tsv", "".join(str(f["expected_cp_stm"]) + "\t" + f["sfen"] + "\n" for f in fixtures).encode())
        collect_refs(expected, tsv)
        snapshot = verify_map(expected)
        argv = [compilations["incremental"]["binary"]["path"], completed["outputs"]["native03"]["path"], recipe["reference03"]["path"], tsv["path"]]
        out, err = args.output / "incremental.stdout.json", args.output / "incremental.stderr.log"
        execution = run_split(argv, Path(empty["path"]), out, err, env, 1200, args.output)
        gate.lifecycle(execution, argv, 1200)
        require(err.read_bytes() == b"", "incremental stderr must be empty")
        incremental = {"fixture_tsv": tsv, "stdin": empty, "stdout": physical_ref(out), "stderr": physical_ref(err),
            "child_receipt": physical_ref(Path(str(err) + ".child-outcome.json")),
            "execution": execution, "result": incremental_summary(out.read_bytes(), bounds, gate),
            "inputs_before": snapshot, "inputs_after": verify_map(snapshot)}
        collect_refs(expected, [incremental, physical_ref(Path(str(err) + ".child-outcome.json"))])
        metadata_path = Path(completed["outputs"]["native03"]["path"]).with_suffix(".meta.json")
        require(not metadata_path.exists(), "fresh native inference sidecar required")
        state["cleaning"] = True
        try:
            metadata = write_owned_json(metadata_path, sidecar(native, completion_ref, completed, gate),
                                        state, "owned_metadata", metadata_path)
        finally:
            state["cleaning"] = False
        require(state["signal"] is None, "cancelled during sidecar publication")
        collect_refs(expected, metadata)
        after = verify_map(expected)
        require(source_map(source_root) == sb["training_source_files"], "training source membership changed during proofs")
        scans_after = require_clear_processes(allowlist)
        require(state["signal"] is None, "cancelled before technical completion")
        body = {"schema": "sekirei.weekly-nonlinear-model-technical-proof.v1", "status": "complete", "mode": recipe["mode"],
            "producer_source": physical_ref(Path(__file__).resolve()), "training_completion": completion_ref,
            "recipe": completed["recipe"], "source_binding": recipe["source_binding"], "plan": sb["selected_plan"],
            "training_build": sb["training_build"], "engine_manifest": manifest_ref, "engine_identity": identity_ref,
            "engine_input_relocation": relocation_ref,
            "context": context, "native03": completed["outputs"]["native03"], "checkpoint": completed["outputs"]["checkpoint"],
            "reference03": recipe["reference03"], "metadata": metadata, "source_activation": activation_ref,
            "compilations": compilations, "counts": COUNTS, "total_core_rows": TOTAL_ROWS, "core": core,
            "incremental": incremental, "native_functional_bounds": bounds,
            "inputs_before": dict(after), "inputs_after": dict(after), "original_inputs_before": before,
            "original_inputs_after": verify_map(before), "lock_records": [{"path": str(p), "exclusive": True,
                "nonblocking": True, "acquired": True} for p in locks], "process_scan_records_before": scans_before,
            "process_scan_records_after": scans_after, "source_unchanged": True, "build_unchanged": True,
            "core_fullrows_verified": True, "incremental_verified": True, "protected_material_ft_bits_preserved": True,
            "whole_board_ft_preservation_claimed": False, "universal_integer_bound_claimed": False,
            "native_bitexact_covariance_claimed": False, "resume_used": False, "shuffle_used": False,
            "final_used": False, "model_adopted": False, "adoption_claimed": False}
        consumer = PhysicalBytes(expected, gate=gate, physical_ref=physical_ref)
        validate_technical_proof(body, consumer, bound, completed, completion_ref, manifest, identity_ref, gate)
        verify_map(expected)
        require(state["signal"] is None, "cancelled after technical consumer")
        state["cleaning"] = True
        try:
            saved = write_owned_json(args.output / "technical-proof.json", body, state, "owned_completion", True)
        finally:
            state["cleaning"] = False
        require(state["signal"] is None, "cancelled during technical publication")
        print(json.dumps({"status": "complete", "technical_proof": saved, "metadata": metadata,
                          "model_adopted": False}), flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--engine-identity", type=Path, required=True)
    parser.add_argument("--expected-engine-identity-sha256", required=True)
    parser.add_argument("--numeric-build-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stock-runtime", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    with termination_guard() as state:
        state.update(owned_completion=False, owned_metadata=None)
        try:
            run(args, state)
            require(state["signal"] is None, "cancelled at proof parent exit")
        except BaseException as error:
            state["cleaning"] = True
            try:
                if state["owned_completion"] and (args.output / "technical-proof.json").exists():
                    (args.output / "technical-proof.json").rename(args.output / "rejected-technical-proof.json")
                if state["owned_metadata"] is not None and state["owned_metadata"].exists():
                    state["owned_metadata"].rename(args.output / "rejected-native-sidecar.json")
                if args.output.exists():
                    write_new(args.output / "failure.json", {"schema": "sekirei.weekly-nonlinear-proof-failure.v1",
                        "status": "failed", "error_type": type(error).__name__, "error": str(error),
                        "signal": state["signal"] or getattr(error, "child_signal", None),
                        "child_outcome": getattr(error, "child_outcome", None), "model_adopted": False, "final_used": False})
            finally:
                state["cleaning"] = False
            raise


if __name__ == "__main__":
    main()
