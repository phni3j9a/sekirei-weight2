#!/usr/bin/env python3
"""Save a pinned functional-anchor view after source/core evidence checks.

This adapter never runs a probe, teacher, trainer or search. The pure recipe
remains pure-transformation-only/real_generation_ready=False. Only the separate
generation receipt can certify this adapter's verified private file output.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil

from benchmark import atomic_write_json
from diagnose_anchor import (DiagnosticCancelled, canonical_path, output_guard,
                             termination_guard, validate_binding, validate_source_helpers)
import functional_anchor as anchor
from prepare import REPO, sha256


SOURCE_SCHEMA = "sekirei.functional-anchor-source-preflight.v1"
CORE_SCHEMA = "sekirei.functional-anchor-core-proof.v1"
GENERATION_SCHEMA = "sekirei.functional-anchor-generation.v1"
SOURCE_CHECKS = {"original_input_hashes", "teacher_packs", "frozen_holdout_derivation",
                 "whole_pool_exclusion_replay", "whole_pool_intersection_zero",
                 "fixed_source_build", "inputs_unchanged", "source_unchanged"}
CORE_CHECKS = {"every_native_equals_python_material", "every_nearest_equals_python_material",
               "every_float_equals_python_material", "every_ft_prefix_in_range"}
RAW_FIELDS = {"index", "native_core_cp", "nearest_core_cp", "material_cp",
              "native_quantized_float_cp", "nearest_quantized_float_cp"}


def file_info(files):
    return {name: {"bytes": len(data), "sha256": anchor.sha256_bytes(data)}
            for name, data in files.items()}


def validate_source_receipt(receipt, binding, preregistration_sha256):
    if (type(receipt) is not dict or receipt.get("schema") != SOURCE_SCHEMA
            or receipt.get("status") != "complete"
            or receipt.get("preregistration_sha256") != preregistration_sha256
            or receipt.get("original_manifest_sha256") != binding["original_manifest_sha256"]
            or receipt.get("spec_sha256") != binding["spec_sha256"]
            or receipt.get("counts") != binding["positions"]
            or any(type(value) is not int for value in receipt["counts"].values())
            or type(receipt.get("checks")) is not dict
            or any(receipt["checks"].get(name) is not True for name in SOURCE_CHECKS)):
        raise ValueError("source preflight receipt identity/counts/explicit checks mismatch")


def validate_core_receipt(receipt, binding, original_files, preregistration_sha256):
    if (type(receipt) is not dict or receipt.get("schema") != CORE_SCHEMA
            or receipt.get("status") != "complete"
            or receipt.get("preregistration_sha256") != preregistration_sha256
            or receipt.get("original_manifest_sha256") != binding["original_manifest_sha256"]
            or receipt.get("spec_sha256") != binding["spec_sha256"]
            or receipt.get("initializer_sha256") != anchor.MATERIAL_INIT_SHA256
            or receipt.get("material_implementation_sha256") != anchor.MATERIAL_IMPLEMENTATION_SHA256
            or receipt.get("original_files") != file_info(original_files)
            or receipt.get("float_policy") != "x86-ftz-daz"
            or receipt.get("inputs_unchanged") is not True or receipt.get("source_unchanged") is not True
            or type(receipt.get("results")) is not dict or set(receipt["results"]) != {"train", "holdout"}):
        raise ValueError("core proof receipt provenance mismatch")
    anchor._sha(receipt.get("probe_sha256"))
    for name, info in receipt["original_files"].items():
        if type(info.get("bytes")) is not int or info["bytes"] != len(original_files[name]):
            raise ValueError("core original file byte count must be an integer")
    for split, result in receipt["results"].items():
        if (type(result) is not dict or type(result.get("count")) is not int
                or result["count"] != binding["positions"][split]
                or result.get("positions_sha256") != anchor.sha256_bytes(original_files[f"{split}.positions.jsonl"])
                or type(result.get("returncode")) is not int or result["returncode"] != 0
                or result.get("timeout") is not False or result.get("cleanup_status") != "ok"
                or any(result.get(name) is not True for name in CORE_CHECKS)):
            raise ValueError(f"incomplete or invalid core result: {split}")
        for name in ("stdin_sha256", "stdout_sha256", "stderr_sha256"):
            anchor._sha(result.get(name))


def validate_core_raw(result, positions, stdin_bytes, stdout_bytes, stderr_bytes):
    sfens = [row["sfen"] for row in positions]
    expected_stdin = "".join(sfen + "\n" for sfen in sfens).encode("utf-8")
    if (stdin_bytes != expected_stdin or anchor.sha256_bytes(stdin_bytes) != result["stdin_sha256"]
            or anchor.sha256_bytes(stdout_bytes) != result["stdout_sha256"]
            or anchor.sha256_bytes(stderr_bytes) != result["stderr_sha256"]
            or len(set(sfens)) != len(sfens)):
        raise ValueError("core raw hash or exact original SFEN order mismatch")
    rows = anchor._rows(stdout_bytes)
    if len(rows) != len(sfens) or len(rows) != result["count"]:
        raise ValueError("core raw complete count mismatch")
    for index, (row, sfen) in enumerate(zip(rows, sfens)):
        if set(row) != RAW_FIELDS or type(row.get("index")) is not int or row["index"] != index:
            raise ValueError("core raw duplicate/missing/out-of-order index")
        material = anchor.fixed_material_cp(sfen)
        for name in ("native_core_cp", "nearest_core_cp", "material_cp"):
            if type(row[name]) is not int or row[name] != material:
                raise ValueError("core integer prediction differs from fixed Python material")
        for name in ("native_quantized_float_cp", "nearest_quantized_float_cp"):
            if type(row[name]) not in (int, float) or not math.isfinite(row[name]) or row[name] != material:
                raise ValueError("core float prediction differs from fixed Python material")
    stderr = stderr_bytes.decode("utf-8").splitlines()
    policy = re.fullmatch(r"float_subnormal_policy=x86-ftz-daz; mxcsr=0x([0-9a-fA-F]+)", stderr[0]) if stderr else None
    if (len(stderr) != 2 or policy is None or int(policy[1], 16) & 0x8040 != 0x8040
            or int(policy[1], 16) & 0x6000 != 0
            or stderr[1] != f"complete count={len(sfens)}; absolute STM; checked both float bridges and FT prefixes"):
        raise ValueError("core stderr policy/MXCSR/completion/prefix evidence mismatch")


def prepare(args):
    previous_umask = os.umask(0o077)
    try:
        _prepare(args)
    finally:
        os.umask(previous_umask)


def _prepare(args):
    paths = {name: canonical_path(getattr(args, name), exists=True) for name in
             ("original_dataset", "spec", "preregistration", "source_preflight", "core_proof_dir", "private_output_root")}
    output = canonical_path(args.output)
    protected = [path for name, path in paths.items() if name != "private_output_root"]
    output_guard(output, paths["private_output_root"], protected)
    if shutil.disk_usage(paths["private_output_root"]).free < 2 * 2**30:
        raise ValueError("anchor generation requires 2 GiB free on its private output filesystem")
    inputs = {}
    def read(path):
        canonical_path(path, exists=True)
        if not path.is_file():
            raise ValueError("adapter inputs must be canonical regular files")
        data = path.read_bytes()
        info = {"bytes": len(data), "sha256": anchor.sha256_bytes(data)}
        if str(path) in inputs and inputs[str(path)] != info:
            raise ValueError("input changed during adapter validation")
        inputs[str(path)] = info
        return data
    def unchanged():
        after = {}
        for name in inputs:
            path = canonical_path(Path(name), exists=True)
            if not path.is_file():
                raise ValueError("adapter input ceased to be a regular file")
            after[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if after != inputs:
            raise ValueError("adapter inputs changed during generation")
        return after
    prereg_bytes = read(paths["preregistration"])
    prereg_sha = anchor.sha256_bytes(prereg_bytes)
    if prereg_sha != anchor._sha(args.expected_preregistration_sha256):
        raise ValueError("preregistration SHA mismatch")
    prereg, spec = anchor._json(prereg_bytes), anchor._json(read(paths["spec"]))
    binding = prereg["diagnosis_binding"]
    if (anchor.spec_sha256(spec) != binding["spec_sha256"]
            or inputs[str(paths["spec"])]["sha256"] != prereg["spec_file_sha256"]):
        raise ValueError("preregistered spec SHA mismatch")
    manifest_bytes = read(paths["original_dataset"] / "manifest.json")
    original_files = {name: read(paths["original_dataset"] / name) for name in anchor.FILES}
    original, parsed = anchor.validate_original(manifest_bytes, original_files,
                        expected_manifest_sha256=binding["original_manifest_sha256"])
    source_bytes = read(paths["source_preflight"])
    core_bytes = read(paths["core_proof_dir"] / "receipt.json")
    if (anchor.sha256_bytes(source_bytes) != anchor._sha(args.expected_source_preflight_sha256)
            or anchor.sha256_bytes(core_bytes) != anchor._sha(args.expected_core_receipt_sha256)):
        raise ValueError("source/core receipt SHA mismatch")
    source_receipt, core_receipt = anchor._json(source_bytes), anchor._json(core_bytes)
    validate_source_receipt(source_receipt, binding, prereg_sha)
    validate_core_receipt(core_receipt, binding, original_files, prereg_sha)
    for split in ("train", "holdout"):
        validate_core_raw(core_receipt["results"][split], parsed[f"{split}.positions.jsonl"],
            read(paths["core_proof_dir"] / f"{split}.sfens"),
            read(paths["core_proof_dir"] / f"{split}.stdout.jsonl"),
            read(paths["core_proof_dir"] / f"{split}.stderr.txt"))
    helper_hashes = {}
    for name in ("functional_anchor.py", "material_init.py", "train_cpu.py", "diagnose_weights.py", "prepare_anchor.py"):
        path = REPO / "scripts" / name
        read(path)
        helper_hashes[name] = inputs[str(path)]["sha256"]
    validate_source_helpers(prereg, helper_hashes, spec)
    view = anchor.derive_train_view(manifest_bytes, original_files, spec,
                                   expected_manifest_sha256=binding["original_manifest_sha256"])
    validate_binding(prereg, spec, original, view["manifest"])
    anchor.validate_train_view(view, manifest_bytes, original_files,
        expected_manifest_sha256=binding["original_manifest_sha256"], expected_spec_sha256=binding["spec_sha256"])
    unchanged()
    output_guard(output, paths["private_output_root"], protected + [REPO / "scripts"])
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    record = {"schema": GENERATION_SCHEMA, "status": "running", "preregistration_sha256": prereg_sha,
              "source_preflight_sha256": anchor.sha256_bytes(source_bytes), "core_receipt_sha256": anchor.sha256_bytes(core_bytes),
              "original_manifest_sha256": binding["original_manifest_sha256"], "spec_sha256": binding["spec_sha256"],
              "training_target_identity": view["manifest"]["teacher_identity"],
              "original_teacher_identity": original["teacher_identity"], "counts": original["positions"],
              "input_files_before": inputs, "script_sha256": helper_hashes["prepare_anchor.py"],
              "probe_executed_by_adapter": False, "training_executed_by_adapter": False, "adoption_claimed": False}
    atomic_write_json(output / "generation.json", record)
    try:
        outputs = dict(view["files"])
        outputs.update({name + ".json": anchor.canonical_json_bytes(view[name]) + b"\n" for name in ("manifest", "recipe")})
        for name, data in outputs.items():
            with (output / name).open("xb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
        saved = {"spec": spec, "manifest": anchor._json((output / "manifest.json").read_bytes()),
                 "recipe": anchor._json((output / "recipe.json").read_bytes()),
                 "files": {name: (output / name).read_bytes() for name in anchor.FILES}}
        anchor.validate_train_view(saved, manifest_bytes, original_files,
            expected_manifest_sha256=binding["original_manifest_sha256"], expected_spec_sha256=binding["spec_sha256"])
        if any((output / name).read_bytes() != data for name, data in outputs.items()):
            raise ValueError("generated output bytes changed or failed write verification")
        record.update(input_files_after=unchanged(), inputs_unchanged=True,
                      output_files=file_info(outputs), pure_recipe_real_generation_ready=False,
                      core_raw_reverified=True, source_preflight_verified=True, status="complete")
        atomic_write_json(output / "generation.json", record)
    except BaseException as error:
        record.update(status="cancelled" if isinstance(error, DiagnosticCancelled) else "failed", error=str(error))
        atomic_write_json(output / "generation.json", record)
        raise
    print(json.dumps({"status": "complete", "output": str(output), "counts": record["counts"],
                      "target_identity": record["training_target_identity"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-dataset", "spec", "preregistration", "source-preflight", "core-proof-dir", "private-output-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-preregistration-sha256", "expected-source-preflight-sha256", "expected-core-receipt-sha256"):
        parser.add_argument("--" + name, required=True)
    with termination_guard():
        prepare(parser.parse_args())
