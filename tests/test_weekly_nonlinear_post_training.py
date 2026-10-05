"""Public fixtures for explicit validator ownership and strict post-fit reads."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "preparations/white-view-paired-nonlinear-math-v1/gate-v2"))
import weekly_nonlinear_post_training as checks

spec = importlib.util.spec_from_file_location("weekly_fixture_gate",
    REPO / "preparations/white-view-paired-nonlinear-math-v1/gate-v2/paired_nonlinear_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def physical_ref(path):
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("canonical regular file required")
    raw = path.read_bytes()
    return {"path": str(path), **gate.digest(raw)}


def snapshot(status=0):
    return {"schema": "sekirei.white-view-paired-nonlinear-float-snapshot.v1",
            "float_policy": "x86-ftz-daz", "environment_variable": "SEKIREI_TRAIN_FTZ_DAZ",
            "environment_value": "1", "mxcsr_raw_bits": 0x9fc0 | status,
            "mxcsr_control_bits": 0x9fc0, "mxcsr_status_bits": status,
            "mxcsr_status_mask": 0x3f, "required_control_bits": 0x9fc0}


def log_rows():
    rows = []
    for epoch in (1, 2, 3):
        rows.append((b"PAIRED_EPOCH_BEGIN ", {"epoch": epoch,
            "start_step": (epoch - 1) * 112681, "float": snapshot()}))
        rows.append((b"PAIRED_EPOCH_COMPLETE ", {"epoch": epoch,
            "positions": 112681, "end_step": epoch * 112681, "float": snapshot(1)}))
    return rows


def encode(rows):
    return b"public noise\n" + b"".join(prefix + json.dumps(body).encode() + b"\n"
                                      for prefix, body in rows)


class PostTrainingTests(unittest.TestCase):
    def test_local_validator_can_read_and_parse_after_training(self):
        # No module-global g is installed. This is the former failure boundary.
        self.assertFalse(hasattr(checks, "g"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_bytes(b'{"value": 3}\n')
            reference = physical_ref(path)
            expected = {str(path): checks.small(reference)}
            reader = checks.PhysicalBytes(expected, gate=gate, physical_ref=physical_ref)
            expected.clear()
            self.assertEqual(reader.json(reference), {"value": 3})
            reader.map({str(path): checks.small(reference)})
            reader.refs_in({"nested": [reference]}, {str(path): checks.small(reference)})
            with self.assertRaises(ValueError):
                reader.refs_in(reference, {})

    def test_changed_file_or_hash_and_duplicate_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_bytes(b'{"value": 3}\n')
            reference = physical_ref(path)
            reader = checks.PhysicalBytes({str(path): checks.small(reference)},
                gate=gate, physical_ref=physical_ref)
            with self.assertRaises(ValueError):
                reader.read(dict(reference, bytes=True))
            path.write_bytes(b'{"value": 4}\n')
            with self.assertRaises(ValueError):
                reader.read(reference)
            path.write_bytes(b'{"value": 3, "value": 3}\n')
            reference = physical_ref(path)
            reader = checks.PhysicalBytes({str(path): checks.small(reference)},
                gate=gate, physical_ref=physical_ref)
            with self.assertRaises(ValueError):
                reader.json(reference)

    def test_mutation_during_physical_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            path.write_bytes(b"before")
            reference = physical_ref(path)
            def changing_ref(file):
                result = physical_ref(file)
                file.write_bytes(b"after!")
                return result
            reader = checks.PhysicalBytes({str(path): checks.small(reference)},
                gate=gate, physical_ref=changing_ref)
            with self.assertRaises(ValueError):
                reader.read(reference)

    def test_exact_three_epoch_evidence_uses_supplied_gate(self):
        observations = checks.epoch_observations(encode(log_rows()), gate=gate)
        self.assertEqual([row["end_step"] for row in observations], [112681, 225362, 338043])
        self.assertEqual(observations[-1]["after"], snapshot(1))

    def test_partial_duplicate_reordered_and_extra_epochs_fail(self):
        rows = log_rows()
        bad_sets = [rows[:-1], rows[1:], rows[:1] + rows, rows[2:4] + rows[:2] + rows[4:],
                    rows + rows[:2]]
        for bad in bad_sets:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                checks.epoch_observations(encode(bad), gate=gate)

    def test_bad_counts_types_control_or_duplicate_fields_fail(self):
        changes = [(0, "epoch", True), (0, "start_step", 1), (1, "positions", 112680),
                   (1, "end_step", 112682), (1, "float", snapshot(64))]
        for index, key, value in changes:
            rows = deepcopy(log_rows())
            rows[index][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checks.epoch_observations(encode(rows), gate=gate)
        raw = encode(log_rows()).replace(b'"epoch": 1,', b'"epoch": 1, "epoch": 1,', 1)
        with self.assertRaises(ValueError):
            checks.epoch_observations(raw, gate=gate)


class NativeRepairSourceTests(unittest.TestCase):
    def test_native_repair_only_changes_test_fixture(self):
        base = REPO / "preparations/white-view-paired-nonlinear-proof-v1/paired_nonlinear_native_contract.rs"
        root = REPO / "preparations/white-view-paired-nonlinear-runtime-v2"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["base"]["sha256"], hashlib.sha256(base.read_bytes()).hexdigest())
        old, new = base.read_bytes(), (root / base.name).read_bytes()
        self.assertEqual(old.split(b"#[cfg(test)]")[0], new.split(b"#[cfg(test)]")[0])
        for name, identity in manifest["files"].items():
            raw = (root / name).read_bytes()
            self.assertEqual(identity, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        self.assertIn(b"copy_is_bit_exact_and_owns_matrix_storage", new)
        self.assertNotIn(b"let mut w=r.clone()", new)


if __name__ == "__main__":
    unittest.main()
