import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_nearest as e


def synthetic_weights():
    values = {"ft": [0.0] * (e.INPUT * e.L1), "ft_bias": [0.0] * e.L1,
              "l2": [0.0] * (2 * e.L1 * e.L2), "l2_bias": [0.0] * e.L2,
              "out": [0.0] * e.L2, "out_bias": -0.0}
    values["ft"][:9] = [e.f32(x / 64) for x in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 32768, -32768, 1.1)]
    values["ft_bias"][0] = e.f32(3.5 / 64)
    # Smallest positive f32 subnormal and signed zero must survive tail copy.
    values["l2"][0] = struct.unpack("<f", bytes.fromhex("01000000"))[0]
    values["l2"][1] = -0.0
    return values


def native_bytes(adam):
    values = adam["ft"] + adam["ft_bias"]
    return e.MAGIC + struct.pack(f"<{e.FT_COUNT}h", *(e.quantize(v, False) for v in values)) + e.float_tail(adam)


class ExportTests(unittest.TestCase):
    def test_rounding_ties_and_clamps(self):
        self.assertEqual([e.quantize(x / 64, True) for x in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5)],
                         [0, 2, 2, 0, -2, -2])
        self.assertEqual(e.quantize(e.f32(3.4028234663852886e38), True), 32767)
        self.assertEqual(e.quantize(e.f32(-3.4028234663852886e38), False), -32767)

    def test_restore_f32_before_scaling(self):
        # This decimal is below a tie as f64 but restores to the exact f32 tie.
        restored = e.f32(0.0234374999)
        self.assertEqual(restored * 64, 1.5)
        self.assertEqual(e.quantize(restored, True), 2)
        self.assertEqual(round(0.0234374999 * 64), 1)

    def test_preserves_every_non_ft_byte_and_changes_bias(self):
        adam = synthetic_weights()
        native = native_bytes(adam)
        output, stats = e.post_export(adam, native)
        self.assertEqual(len(output), 1305356)
        self.assertEqual(e.TAIL_OFFSET, 1239560)
        self.assertEqual(output[:8], native[:8])
        self.assertEqual(output[e.TAIL_OFFSET:], native[e.TAIL_OFFSET:])
        self.assertEqual(output[e.TAIL_OFFSET:e.TAIL_OFFSET + 8].hex(), "0100000000000080")
        self.assertEqual(stats["ft_bias"]["changed"], 1)
        self.assertEqual(stats["ft"]["half_ties"], 6)
        self.assertEqual(struct.unpack_from("<h", output, 8 + 2 * e.INPUT * e.L1)[0], 4)
        self.assertLessEqual(stats["ft"]["nearest_mean_abs_weight_error"], stats["ft"]["native_mean_abs_weight_error"])

    def test_native_mismatch_in_ft_or_tail_fails(self):
        adam = synthetic_weights()
        native = native_bytes(adam)
        for offset in (8, e.TAIL_OFFSET):
            bad = bytearray(native)
            bad[offset] ^= 1
            with self.assertRaisesRegex(ValueError, "native trunc export differs"):
                e.post_export(adam, bytes(bad))

    def test_repeat_is_identical(self):
        adam = synthetic_weights()
        native = native_bytes(adam)
        self.assertEqual(e.post_export(adam, native), e.post_export(adam, native))

    def test_nonfinite_duplicate_and_f32_overflow_rejected(self):
        for text in ('{"x": NaN}', '{"x": Infinity}', '{"x": 1, "x": 2}'):
            with self.assertRaises(ValueError):
                e.load_json(text)
        for value in (float("nan"), float("inf"), 1e39, True):
            with self.assertRaises(ValueError):
                e.f32(value)

    def test_cli_public_synthetic_only(self):
        adam = synthetic_weights()
        for name, length in e.LENGTHS.items():
            if name not in adam:
                adam[name] = [0.0] * length
        adam.update(schema="sekirei.adam-checkpoint.v1", version=1, step=0, obias_m=0.0, obias_v=0.0)
        with tempfile.TemporaryDirectory(prefix="nearest-public-fixture-") as directory:
            base = Path(directory)
            native = native_bytes(adam)
            (base / "input.bin").write_bytes(native)
            (base / "input.adam.json").write_text(json.dumps(adam, allow_nan=False))
            (base / "input.meta.json").write_text(json.dumps({"nnue_output": "absolute",
                "teacher_identity": "external:public-synthetic-fixture", "checkpoint_hash": e.fnv1a(native)}))
            # Test source-hash enforcement with synthetic public files; CI never
            # needs a user's private checkout or teacher/runtime artifacts.
            source = base / "source"
            expected_hashes = {}
            for name in e.SOURCE_HASHES:
                file = source / name
                file.parent.mkdir(parents=True, exist_ok=True)
                data = ("public test source: " + name).encode()
                file.write_bytes(data)
                expected_hashes[name] = e.sha256(data)
            args = SimpleNamespace(source=source,
                native=base / "input.bin", adam=base / "input.adam.json", metadata=base / "input.meta.json",
                teacher_identity="external:public-synthetic-fixture", output=base / "derived")
            with mock.patch.object(e, "SOURCE_HASHES", expected_hashes), contextlib.redirect_stdout(io.StringIO()):
                e.main(args)
            output = (args.output / "nearest.bin").read_bytes()
            sidecar = json.loads((args.output / "nearest.meta.json").read_text())
            recipe = json.loads((args.output / "recipe.json").read_text())
            self.assertEqual(sidecar, {"format": "sekirei-nnue-output-v1", "nnue_output": "absolute",
                                      "checkpoint_hash": e.fnv1a(output), "baseline": None})
            self.assertEqual(recipe["status"], "complete")
            self.assertFalse(recipe["engine_verified"])
            with mock.patch.object(e, "SOURCE_HASHES", expected_hashes):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                    e.main(args)
                repo = base / "public-repo"
                subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
                args.output = repo / "must-not-write"
                with self.assertRaisesRegex(ValueError, "outside every Git worktree"):
                    e.main(args)
                self.assertFalse(args.output.exists())
                first_source = source / next(iter(expected_hashes))
                first_source.write_text("changed source")
                with self.assertRaisesRegex(ValueError, "pinned source mismatch"):
                    e.main(args)


if __name__ == "__main__":
    unittest.main()
