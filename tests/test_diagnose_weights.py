import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_weights import metrics, summarize, verified_split


class DiagnosisContractTests(unittest.TestCase):
    def make_dataset(self, root, duplicate=False, missing=False):
        for split in ("train", "holdout"):
            positions = [{"sfen": f"fixture-{split}-1"}, {"sfen": f"fixture-{split}-2"}]
            labels = [{"sfen": p["sfen"], "score_cp": i * 100,
                       "label_depth": 0, "teacher_identity": "external:fixture"}
                      for i, p in enumerate(positions)]
            if duplicate:
                labels[1] = labels[0]
            if missing:
                labels.pop()
            for kind, rows in (("positions", positions), ("labels", labels)):
                (root / f"{split}.{kind}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest = {"positions": {"train": 2, "holdout": 2}, "teacher_identity": "external:fixture",
                    "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                       "bytes": p.stat().st_size}
                              for p in root.glob("*.jsonl")}}
        (root / "manifest.json").write_text(json.dumps(manifest))

    def test_split_requires_full_matching_unique_cache(self):
        for mode in ("good", "duplicate", "missing"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_dataset(root, duplicate=mode == "duplicate", missing=mode == "missing")
                if mode == "good":
                    self.assertEqual(verified_split(root, "holdout")[1], [0, 100])
                else:
                    with self.assertRaises(ValueError):
                        verified_split(root, "holdout")

    def test_tampering_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            with (root / "holdout.labels.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
                verified_split(root, "holdout")

    def test_summary_preserves_labels_and_checks_actual_core_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = [{"index": i, "teacher_cp_stm": target, "raw_float_cp": target,
                     "quantized_float_cp": target + .5, "inference_cp": target,
                     "material_cp": 0} for i, target in enumerate((-100, 100))]
            def save():
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            save()
            result = summarize(path, [-100, 100])
            self.assertEqual(result["metrics"]["raw_float_cp"]["mae_cp"], 0)
            self.assertEqual(result["quantization_delta"]["max_abs_cp"], .5)
            rows[1]["inference_cp"] = 80
            save()
            with self.assertRaisesRegex(ValueError, "core inference"):
                summarize(path, [-100, 100])
            rows[1]["teacher_cp_stm"] = -100
            save()
            with self.assertRaisesRegex(ValueError, "ordering or label"):
                summarize(path, [-100, 100])

    def test_constant_prediction_has_no_spurious_correlation(self):
        self.assertIsNone(metrics([0, 0], [-100, 100])["pearson_r"])


if __name__ == "__main__":
    unittest.main()
