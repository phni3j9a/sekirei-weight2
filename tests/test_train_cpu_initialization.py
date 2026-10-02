"""Public synthetic checks for inference initialization and provenance."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_cpu


def fnv1a(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def write_initial(root, name="initial.bin"):
    path = root / name
    data = b"SEKIRW01" + bytes(train_cpu.NNUE_WEIGHT_BYTES - 8)
    path.write_bytes(data)
    path.with_suffix(".meta.json").write_text(json.dumps({
        "format": "sekirei-nnue-output-v1", "nnue_output": "absolute",
        "baseline": None, "checkpoint_hash": fnv1a(data)}))
    return path


class InitialWeightsTests(unittest.TestCase):
    def test_canonical_path_and_content_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_initial(root)
            alias = root / "alias.bin"
            alias.symlink_to(path)
            result = train_cpu.verify_initial_weights(alias)
            self.assertEqual(result["path"], str(path.resolve()))
            self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result["metadata_sha256"], hashlib.sha256(path.with_suffix(".meta.json").read_bytes()).hexdigest())
            train_cpu.verify_initial_weights_unchanged(result)
            self.assertIsNone(train_cpu.verify_initial_weights(None))

    def test_truncated_nonfinite_and_legacy_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_initial(root)
            good = path.read_bytes()
            invalid = [good[:-4], b"JANOSW03" + good[8:],
                       good[:train_cpu.NNUE_FLOAT_OFFSET] + struct.pack("<f", float("nan"))
                       + good[train_cpu.NNUE_FLOAT_OFFSET + 4:]]
            for data in invalid:
                with self.subTest(length=len(data), header=data[:8]):
                    path.write_bytes(data)
                    with self.assertRaises(ValueError):
                        train_cpu.verify_initial_weights(path)

    def test_sidecar_mode_format_and_hash_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_initial(root)
            sidecar = path.with_suffix(".meta.json")
            good = json.loads(sidecar.read_text())
            for key, value in (("format", "unknown"), ("nnue_output", "residual-material"), ("checkpoint_hash", "bad")):
                with self.subTest(key=key):
                    sidecar.write_text(json.dumps(dict(good, **{key: value})))
                    with self.assertRaises(ValueError):
                        train_cpu.verify_initial_weights(path)
            sidecar.unlink()
            with self.assertRaises(FileNotFoundError):
                train_cpu.verify_initial_weights(path)

    def test_post_training_mutation_of_either_input_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for changed in ("weights", "sidecar"):
                with self.subTest(changed=changed):
                    path = write_initial(root)
                    identity = train_cpu.verify_initial_weights(path)
                    target = path if changed == "weights" else path.with_suffix(".meta.json")
                    with target.open("ab") as stream:
                        stream.write(b" ")
                    with self.assertRaises(RuntimeError):
                        train_cpu.verify_initial_weights_unchanged(identity)

    def test_wrapper_records_fresh_optimizer_and_refuses_changed_input(self):
        # No engine or learning is run. The stub emits one completed epoch to
        # exercise the real wrapper's initialization argv and final validation.
        for initialized, mutate in ((False, False), (True, False), (True, True)):
            with self.subTest(initialized=initialized, mutate=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset, trainer, output = root / "dataset", root / "trainer", root / "run"
                dataset.mkdir(); trainer.mkdir()
                positions = dataset / "train.positions.jsonl"
                positions.write_text('{}\n')
                teacher = "external:fixture"
                (dataset / "manifest.json").write_text(json.dumps({"teacher_identity": teacher,
                    "files": {positions.name: {"sha256": train_cpu.sha256(positions)}}}))
                binary = trainer / "train"
                binary.write_bytes(b"public fake executable identity")
                (trainer / "build-manifest.json").write_text(json.dumps({
                    "binary": str(binary), "binary_sha256": train_cpu.sha256(binary),
                    "patch_sha256": train_cpu.sha256(train_cpu.REPO / "patches/sekirei-train-external-labels.patch")}))
                initial = write_initial(root) if initialized else None
                args = Namespace(dataset=dataset, trainer=trainer, output=output,
                    epochs=1, seconds=1, lr=.001, min_lr=0., lr_schedule="constant",
                    lr_schedule_epochs=None, max_positions=None, init_weights=initial)
                seen = []
                def fake_popen(command, **kwargs):
                    seen.append(command)
                    self.assertEqual(command[command.index("--seed") + 1], "42")
                    self.assertNotIn("--resume-adam", command)
                    self.assertNotIn("--resume-checkpoint", command)
                    checkpoints = output / "checkpoints"
                    checkpoints.mkdir()
                    (checkpoints / "weights.epoch1.meta.json").write_text(json.dumps({
                        "teacher_identity": teacher, "cache_misses": 0, "cache_hits": 1,
                        "train_count": 1, "teacher_eval": "external", "float_subnormal_policy": "x86-ftz-daz",
                        "lr_schedule": "Constant", "lr": train_cpu.float32(.001), "min_lr": 0.,
                        "lr_schedule_epochs": 1, "nnue_output": "absolute"}))
                    (output / "weights.bin").write_bytes(b"SEKIRW01")
                    if mutate:
                        initial.with_suffix(".meta.json").write_text('{}')
                    return Namespace(wait=lambda timeout: 0)
                with patch.object(train_cpu.subprocess, "Popen", side_effect=fake_popen), patch("builtins.print"):
                    if mutate:
                        with self.assertRaisesRegex(RuntimeError, "validation_failure"):
                            train_cpu.train(args)
                    else:
                        train_cpu.train(args)
                record = json.loads((output / "run.json").read_text())
                self.assertEqual(record["initial_weights_unchanged"], not mutate)
                self.assertEqual("--init-weights" in seen[0], initialized)
                if initialized:
                    self.assertEqual(record["initial_weights"]["path"], str(initial.resolve()))
                    self.assertEqual(record["initialization"], "inference weights; fresh Adam state")
                else:
                    self.assertIsNone(record["initial_weights"])
                    self.assertEqual(record["initialization"], "random seed 42; fresh Adam state")


if __name__ == "__main__":
    unittest.main()
