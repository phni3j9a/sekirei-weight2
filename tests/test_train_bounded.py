import copy
import sys
import json
import tempfile
import time
from unittest.mock import patch
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_bounded as bounded


def prereg():
    return {"schema": bounded.SCHEMA, "status": "frozen-before-training",
            "candidate": "bounded-material-residual-100cp-e3-v1",
            "plan_sha256": bounded.PLAN_SHA256,
            "dataset_manifest_sha256": bounded.DATASET_SHA256,
            "teacher_identity": bounded.TEACHER,
            "initial_weights": {"sha256": bounded.INITIAL_SHA256},
            "build_manifest_sha256": "a" * 64,
            "source_helpers": {name: "b" * 64 for name in bounded.HELPERS},
            "training": copy.deepcopy(bounded.FIXED)}


class BoundedTrainingBindingTests(unittest.TestCase):
    def test_original_fixed_recipe_passes_without_io(self):
        self.assertEqual(bounded.validate_preregistration(prereg()), bounded.FIXED)

    def test_other_teacher_dataset_initializer_or_plan_rejected(self):
        for key in ("teacher_identity", "dataset_manifest_sha256", "plan_sha256"):
            data = prereg(); data[key] = "c" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                bounded.validate_preregistration(data)
        data = prereg(); data["initial_weights"]["sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            bounded.validate_preregistration(data)

    def test_cap_label_depth_epoch_and_schedule_are_fixed(self):
        for key, value in (("real_residual_cap_cp", 100), ("label_depth", 1),
                           ("selected_epoch", 2), ("teacher_score_cap", 600),
                           ("lr_schedule", "constant"), ("timeout_seconds", 3600)):
            data = prereg(); data["training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bounded.validate_preregistration(data)

    def test_boolean_depth_extra_or_missing_recipe_keys_rejected(self):
        data = prereg(); data["training"]["label_depth"] = False
        with self.assertRaises(ValueError):
            bounded.validate_preregistration(data)
        for operation in (lambda d: d["training"].pop("seed"),
                          lambda d: d["training"].update(resume=True)):
            data = prereg(); operation(data)
            with self.assertRaises(ValueError):
                bounded.validate_preregistration(data)
        data = prereg(); data["training"]["positions"]["train"] = 112681.0
        with self.assertRaises(ValueError):
            bounded.validate_preregistration(data)

    def test_missing_or_traversing_helpers_and_invalid_manifest_rejected(self):
        data = prereg(); data["source_helpers"].pop("diagnose_bounded.py")
        with self.assertRaises(ValueError):
            bounded.validate_preregistration(data)
        for name in ("../secret.py", "..", "/absolute.py"):
            data = prereg(); data["source_helpers"][name] = "e" * 64
            with self.subTest(name=name), self.assertRaises(ValueError):
                bounded.validate_preregistration(data)
        data = prereg(); data["build_manifest_sha256"] = "G" * 64
        with self.assertRaises(ValueError):
            bounded.validate_preregistration(data)

    def test_argv_cache_only_without_internal_teacher_or_resume(self):
        command = bounded.training_argv("/public/train", Path("/public/run"),
                                         Path("/public/data"), {"path": "/public/init.bin"})
        self.assertEqual(command[-1], "--bounded-material-v1")
        self.assertIn("--cache-only", command)
        for flag, value in (("--label-depth", "0"), ("--teacher-score-cap", "30000"),
                            ("--epochs", "3"), ("--external-teacher-id", bounded.TEACHER)):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertFalse(any(x.startswith("--resume") or x.startswith("--teacher-weights")
                             for x in command))
        self.assertNotIn("--exclude-mate-labels", command)

    def test_final_inference_sidecar_uses_nested_teacher_and_exact_epoch3(self):
        data = b"public synthetic identity fixture"
        mode = {"teacher_identity": bounded.TEACHER, "mode": "bounded-material-v1", "mask_version": 1}
        meta = {"format": "sekirei-nnue-output-v1", "nnue_output": "absolute",
                "checkpoint_hash": bounded.fnv1a(data), "bounded_material_v1": mode}
        bounded.validate_final_sidecar(data, meta, data, {"bounded_material_v1": mode})
        for key, value in (("checkpoint_hash", "0" * 16), ("nnue_output", "residual-material")):
            broken = copy.deepcopy(meta); broken[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bounded.validate_final_sidecar(data, broken, data, {"bounded_material_v1": mode})
        with self.assertRaises(ValueError):
            bounded.validate_final_sidecar(data, meta, data + b"changed", {"bounded_material_v1": mode})
        broken = copy.deepcopy(meta); broken["bounded_material_v1"]["teacher_identity"] = "other"
        with self.assertRaises(ValueError):
            bounded.validate_final_sidecar(data, broken, data, {"bounded_material_v1": mode})
        for version in (True, 1.0):
            broken = copy.deepcopy(meta); broken["bounded_material_v1"]["mask_version"] = version
            with self.subTest(version=version), self.assertRaises(ValueError):
                bounded.validate_final_sidecar(data, broken, data, {"bounded_material_v1": mode})

    def test_cleanup_failure_still_records_original_error_and_failed_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); record = {"status": "running", "cleanup_status": "ok"}
            with patch.object(bounded, "stop_recorded_group", side_effect=PermissionError("public fixture")):
                bounded.save_failure(record, ValueError("original public failure"), root, time.monotonic())
            saved = json.loads((root / "run.json").read_text())
            self.assertEqual(saved["status"], "cleanup_failure")
            self.assertEqual(saved["cleanup_status"], "failed")
            self.assertIn("original public failure", saved["error"])
            self.assertIn("PermissionError", saved["cleanup_error"])


if __name__ == "__main__":
    unittest.main()
