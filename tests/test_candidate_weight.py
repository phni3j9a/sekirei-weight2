"""The benchmark must prove NNUE activation, not just receive readyok."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark as b


class CandidateWeightTests(unittest.TestCase):
    def test_load_proof_required_in_live_attempt(self):
        fixture = Path(__file__).parent / "fixtures/fake_usi.py"
        for mode, expected in (("exact", "config_failure"),
                               ("bad-weight", "config_failure"),
                               ("loaded-weight", "exact_cp")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                with patch.dict(os.environ, {"FAKE_USI_MODE": mode}):
                    result = b.run_engine_attempt(
                        fixture, "position startpos", "black", {"EvalFile": "/tmp/weight.bin"},
                        cwd=Path(d), requested_nodes=1000, timeout_seconds=5,
                    )
                self.assertEqual(result["status"], expected)

    def test_fallback_cannot_load_weight(self):
        config = copy.deepcopy(b.load_config())
        config["engines"]["sekirei"]["options"]["EvalFile"] = "/tmp/hidden.bin"
        with self.assertRaises(b.ConfigurationError):
            b._model_identity(config)

    def test_raw_lifecycle_weight_proof(self):
        with self.assertRaises(b.ConfigurationError):
            b._validate_weight_loaded([
                {"direction": "receive", "line": "readyok"},
            ], {"EvalFile": "/tmp/weight.bin"})

    def test_report_cannot_attach_a_different_candidate(self):
        config = copy.deepcopy(b.load_config())
        manifest = {"runtime_identity": {"candidate_model": b._model_identity(config)},
                    "fingerprint_payload": {"engines": config["engines"],
                                            "resolved_options": {"sekirei": {}}}}
        b._validate_candidate_configuration(manifest, config)
        with tempfile.TemporaryDirectory() as d:
            weight = Path(d) / "weights.bin"
            weight.write_bytes(b"model B")
            config["candidate_model"] = {"kind": "nnue", "path": str(weight)}
            with self.assertRaisesRegex(b.BenchmarkError, "candidate model"):
                b._validate_candidate_configuration(manifest, config)

    def test_report_rejects_unbound_activation_path(self):
        config = copy.deepcopy(b.load_config())
        with tempfile.TemporaryDirectory() as d:
            weight = Path(d) / "weights.bin"
            weight.write_bytes(b"model A")
            config["candidate_model"] = {"kind": "nnue", "path": str(weight)}
            manifest = {"runtime_identity": {"candidate_model": b._model_identity(config)},
                        "fingerprint_payload": {"engines": config["engines"],
                                                "resolved_options": {"sekirei": {
                                                    "EvalFile": str(weight), "NnueOutput": "absolute"}}}}
            b._validate_candidate_configuration(manifest, config)
            manifest["fingerprint_payload"]["resolved_options"]["sekirei"]["EvalFile"] = "/wrong"
            with self.assertRaisesRegex(b.BenchmarkError, "NNUE options"):
                b._validate_candidate_configuration(manifest, config)


if __name__ == "__main__":
    unittest.main()
