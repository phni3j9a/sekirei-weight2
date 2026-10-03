"""Small synthetic adapter fixtures; no real dataset, probe, teacher or training."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_anchor as adapter
import functional_anchor as anchor
from test_diagnose_anchor import contract
from test_functional_anchor import fixture


def encoded(value):
    return anchor.canonical_json_bytes(value) + b"\n"


def proof_fixture():
    prereg, spec, _, _ = contract()
    manifest_bytes, files, _ = fixture()
    binding = prereg["diagnosis_binding"]
    source = {"schema": adapter.SOURCE_SCHEMA, "status": "complete",
              "preregistration_sha256": "f" * 64,
              "original_manifest_sha256": binding["original_manifest_sha256"],
              "spec_sha256": binding["spec_sha256"], "counts": binding["positions"],
              "checks": {name: True for name in adapter.SOURCE_CHECKS}}
    core = {"schema": adapter.CORE_SCHEMA, "status": "complete", "preregistration_sha256": "f" * 64,
            "original_manifest_sha256": binding["original_manifest_sha256"], "spec_sha256": binding["spec_sha256"],
            "original_files": adapter.file_info(files), "initializer_sha256": anchor.MATERIAL_INIT_SHA256,
            "material_implementation_sha256": anchor.MATERIAL_IMPLEMENTATION_SHA256,
            "probe_sha256": "e" * 64, "float_policy": "x86-ftz-daz", "inputs_unchanged": True,
            "source_unchanged": True, "results": {}}
    raw = {}
    for split in ("train", "holdout"):
        positions = anchor._rows(files[f"{split}.positions.jsonl"])
        stdin = "".join(row["sfen"] + "\n" for row in positions).encode()
        rows = []
        for index, position in enumerate(positions):
            material = anchor.fixed_material_cp(position["sfen"])
            rows.append({"index": index, "native_core_cp": material, "nearest_core_cp": material,
                         "material_cp": material, "native_quantized_float_cp": float(material),
                         "nearest_quantized_float_cp": float(material)})
        stdout = b"".join(encoded(row) for row in rows)
        stderr = ("float_subnormal_policy=x86-ftz-daz; mxcsr=0x9fc0\n"
                  f"complete count={len(rows)}; absolute STM; checked both float bridges and FT prefixes\n").encode()
        result = {"count": len(rows), "positions_sha256": anchor.sha256_bytes(files[f"{split}.positions.jsonl"]),
                  "stdin_sha256": anchor.sha256_bytes(stdin), "stdout_sha256": anchor.sha256_bytes(stdout),
                  "stderr_sha256": anchor.sha256_bytes(stderr), "returncode": 0, "timeout": False,
                  "cleanup_status": "ok", **{name: True for name in adapter.CORE_CHECKS}}
        core["results"][split] = result
        raw[split] = positions, stdin, stdout, stderr
    return prereg, spec, manifest_bytes, files, source, core, raw


def disk_fixture(root):
    prereg, spec, manifest, files, source, core, raw = proof_fixture()
    original, proof, private = root / "original", root / "proof", root / "outputs"
    for directory in (original, proof, private):
        directory.mkdir(mode=0o700)
    (original / "manifest.json").write_bytes(manifest)
    for name, data in files.items():
        (original / name).write_bytes(data)
    spec_bytes = encoded(spec)
    (root / "spec.json").write_bytes(spec_bytes)
    prereg["spec_file_sha256"] = anchor.sha256_bytes(spec_bytes)
    prereg["source_helpers"] = {name: anchor.sha256_bytes((adapter.REPO / "scripts" / name).read_bytes())
        for name in ("functional_anchor.py", "material_init.py", "train_cpu.py", "diagnose_weights.py")}
    prereg["diagnosis_binding"]["train_script_sha256"] = prereg["source_helpers"]["train_cpu.py"]
    prereg_bytes = encoded(prereg)
    prereg_sha = anchor.sha256_bytes(prereg_bytes)
    (root / "preregistration.json").write_bytes(prereg_bytes)
    source["preregistration_sha256"] = core["preregistration_sha256"] = prereg_sha
    source_bytes, core_bytes = encoded(source), encoded(core)
    (root / "source.json").write_bytes(source_bytes)
    (proof / "receipt.json").write_bytes(core_bytes)
    for split, (_, stdin, stdout, stderr) in raw.items():
        for suffix, data in (("sfens", stdin), ("stdout.jsonl", stdout), ("stderr.txt", stderr)):
            (proof / f"{split}.{suffix}").write_bytes(data)
    args = SimpleNamespace(original_dataset=original, spec=root / "spec.json", preregistration=root / "preregistration.json",
        source_preflight=root / "source.json", core_proof_dir=proof, private_output_root=private, output=private / "new",
        expected_preregistration_sha256=prereg_sha, expected_source_preflight_sha256=anchor.sha256_bytes(source_bytes),
        expected_core_receipt_sha256=anchor.sha256_bytes(core_bytes))
    return args, files


class PrepareAnchorTests(unittest.TestCase):
    def test_complete_synthetic_adapter_keeps_original_holdout_and_pure_recipe(self):
        with tempfile.TemporaryDirectory() as temp:
            args, files = disk_fixture(Path(temp).resolve())
            with redirect_stdout(io.StringIO()):
                adapter.prepare(args)
            receipt = anchor._json((args.output / "generation.json").read_bytes())
            self.assertEqual(receipt["status"], "complete")
            self.assertTrue(receipt["inputs_unchanged"])
            self.assertEqual(receipt["input_files_before"], receipt["input_files_after"])
            for name in ("train.positions.jsonl", "holdout.positions.jsonl", "holdout.labels.jsonl"):
                self.assertEqual((args.output / name).read_bytes(), files[name])
            recipe = anchor._json((args.output / "recipe.json").read_bytes())
            self.assertIs(recipe["real_generation_ready"], False)
            self.assertEqual(recipe["scope"], "pure-transformation-only")
            self.assertEqual(set(receipt["output_files"]), anchor.FILES | {"manifest.json", "recipe.json"})
            self.assertEqual(args.output.stat().st_mode & 0o777, 0o700)
            self.assertEqual(len(list(args.output.iterdir())), 7)
            for path in args.output.iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_source_identity_counts_and_all_explicit_checks(self):
        prereg, _, _, _, source, _, _ = proof_fixture()
        binding = prereg["diagnosis_binding"]
        adapter.validate_source_receipt(source, binding, "f" * 64)
        for name in adapter.SOURCE_CHECKS:
            changed = copy.deepcopy(source)
            changed["checks"][name] = False
            with self.assertRaises(ValueError):
                adapter.validate_source_receipt(changed, binding, "f" * 64)
        changed = copy.deepcopy(source)
        changed["counts"]["train"] = True
        with self.assertRaises(ValueError):
            adapter.validate_source_receipt(changed, binding, "f" * 64)
        with self.assertRaises(ValueError):
            adapter.validate_source_receipt(source, binding, "0" * 64)

    def test_core_provenance_and_typed_completion_required(self):
        prereg, _, _, files, _, core, _ = proof_fixture()
        binding = prereg["diagnosis_binding"]
        adapter.validate_core_receipt(core, binding, files, "f" * 64)
        for name, value in (("initializer_sha256", "0" * 64), ("inputs_unchanged", False),
                            ("float_policy", "gradual"), ("material_implementation_sha256", "0" * 64)):
            changed = dict(core, **{name: value})
            with self.assertRaises(ValueError):
                adapter.validate_core_receipt(changed, binding, files, "f" * 64)
        for name, value in (("count", True), ("returncode", False), ("timeout", True),
                            ("cleanup_status", "failed"), ("every_ft_prefix_in_range", False)):
            changed = copy.deepcopy(core)
            changed["results"]["train"][name] = value
            with self.assertRaises(ValueError):
                adapter.validate_core_receipt(changed, binding, files, "f" * 64)

    def test_raw_requires_exact_sfen_order_hash_and_unique_indices(self):
        _, _, _, _, _, core, raw = proof_fixture()
        positions, stdin, stdout, stderr = raw["train"]
        result = core["results"]["train"]
        adapter.validate_core_raw(result, positions, stdin, stdout, stderr)
        with self.assertRaises(ValueError):
            adapter.validate_core_raw(result, positions, b"".join(reversed(stdin.splitlines(keepends=True))), stdout, stderr)
        rows = anchor._rows(stdout)
        rows[1]["index"] = rows[0]["index"]
        changed = b"".join(encoded(row) for row in rows)
        with self.assertRaises(ValueError):
            adapter.validate_core_raw(dict(result, stdout_sha256=anchor.sha256_bytes(changed)), positions, stdin, changed, stderr)

    def test_raw_rejects_wrong_material_bool_nan_and_missing_rows(self):
        _, _, _, _, _, core, raw = proof_fixture()
        positions, stdin, stdout, stderr = raw["train"]
        result = core["results"]["train"]
        for name, value in (("index", False), ("native_core_cp", True), ("nearest_core_cp", -999),
                            ("material_cp", 100.), ("native_quantized_float_cp", float("nan")),
                            ("nearest_quantized_float_cp", False)):
            rows = anchor._rows(stdout)
            rows[0][name] = value
            changed = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
            with self.assertRaises(ValueError):
                adapter.validate_core_raw(dict(result, stdout_sha256=anchor.sha256_bytes(changed)), positions, stdin, changed, stderr)
        changed = stdout.splitlines(keepends=True)[0]
        with self.assertRaises(ValueError):
            adapter.validate_core_raw(dict(result, stdout_sha256=anchor.sha256_bytes(changed)), positions, stdin, changed, stderr)

    def test_stderr_mxcsr_float_policy_prefix_and_complete_count(self):
        _, _, _, _, _, core, raw = proof_fixture()
        positions, stdin, stdout, stderr = raw["train"]
        result = core["results"]["train"]
        for changed in (stderr.replace(b"0x9fc0", b"0x1f80"), stderr.replace(b"0x9fc0", b"0xbfc0"),
                        stderr.replace(b"count=2", b"count=1"), stderr + b"failed: test\n", b""):
            with self.assertRaises(ValueError):
                adapter.validate_core_raw(dict(result, stderr_sha256=anchor.sha256_bytes(changed)), positions, stdin, stdout, changed)

    def test_wrong_receipt_sha_and_low_capacity_never_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = disk_fixture(Path(temp).resolve())
            args.expected_core_receipt_sha256 = "0" * 64
            with self.assertRaises(ValueError):
                adapter.prepare(args)
            self.assertFalse(args.output.exists())
            with patch.object(adapter.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaises(ValueError):
                    adapter.prepare(args)
            self.assertFalse(args.output.exists())

    def test_cancellation_leaves_failed_generation_receipt_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            args, files = disk_fixture(Path(temp).resolve())
            original_open = Path.open
            def cancel(path, mode="r", *positional, **keywords):
                if path == args.output / "train.labels.jsonl" and mode == "xb":
                    raise adapter.DiagnosticCancelled("synthetic cancellation during save")
                return original_open(path, mode, *positional, **keywords)
            with patch.object(Path, "open", new=cancel):
                with self.assertRaises(adapter.DiagnosticCancelled):
                    adapter.prepare(args)
            self.assertEqual(anchor._json((args.output / "generation.json").read_bytes())["status"], "cancelled")
            for name, data in files.items():
                self.assertEqual((args.original_dataset / name).read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
