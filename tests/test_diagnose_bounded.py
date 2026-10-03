"""Pure synthetic fixtures for the bounded diagnostic checkpoint contract.

No checkpoint files, private data, engine, trainer or external diagnostic are
opened or executed. The material reference and full-shaped Adam are generated
in memory. These tests establish validator behavior, not actual training,
intermediate activation safety or a universal 100 cp integer-core bound.

When this prototype lives in /tmp, set PYTHONPATH to the repository scripts
directory. When copied into tests/, the relative scripts path is sufficient.
"""
import hashlib
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bounded_material as bounded
import diagnose_bounded as diagnose
from export_nearest import LENGTHS, post_export, validate_adam
import material_init


MATERIAL_SEED42_SHA256 = "bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40"
TEACHER = "external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d"
EXPECTED_STEP = 3 * 112681


def metadata_fixture():
    # Independent literal receipt: do not obtain valid fields from MODE_FIELDS.
    return {
        "mode": "bounded-material-v1", "mask_version": 1, "fixed_ft": "all",
        "fixed_l2_columns": [0, 1, 2, 3], "fixed_l2_rows": [0, 1, 256, 257],
        "fixed_bias_out": [0, 1, 2, 3], "fixed_out_bias": "+0", "auxiliary_units": 28,
        "residual_budget_cp": 99, "shrink": "uniform-aux-out-f32-margin-2^-20-max8",
        "shrink_moments": False, "optimizer": "fresh-adam", "resume_supported": False,
        "init_seed": 42, "initial_weights_fnv": "5ae69099a8bbcdad",
        "loaded_initial_binding": "core-save-weights-exact-bytes", "label_depth": 0,
        "target": "original-absolute-external-cp", "teacher_identity": TEACHER,
        "integer_core_bound_proven": False,
        "saved_f32_l1_upper": 6.0, "saved_f32_budget_cp_upper": 11.90625,
    }


def edit(adam, changes):
    """Copy only changed arrays; never alter the shared full-shaped fixture."""
    result = dict(adam)
    for name, indices in changes.items():
        result[name] = list(adam[name])
        for index, value in indices.items():
            result[name][index] = value
    return result


def native_with_parameters(native, adam):
    """Independent tail serialization; frozen native FT stays byte-for-byte."""
    tail = adam["l2"] + adam["l2_bias"] + adam["out"] + [adam["out_bias"]]
    return native[:bounded.L2_OFFSET] + struct.pack(f"<{len(tail)}f", *tail)


class ActualAdamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = material_init.encode(material_init.build_weights(42))
        if (len(cls.native) != 1305356
                or hashlib.sha256(cls.native).hexdigest() != MATERIAL_SEED42_SHA256):
            raise AssertionError("the synthetic initializer is not the pinned full native reference")
        decoded = material_init.decode(cls.native)
        cls.adam = {
            "schema": "sekirei.adam-checkpoint.v1", "version": 1, "step": EXPECTED_STEP,
            "ft": [value / 64.0 for value in decoded["ft"]],
            "ft_bias": [value / 64.0 for value in decoded["ft_bias"]],
            "l2": list(decoded["l2"]), "l2_bias": list(decoded["l2_bias"]),
            "out": list(decoded["out"]), "out_bias": decoded["out_bias"],
            "obias_m": 0.0, "obias_v": 0.0,
        }
        # Every moment is present at the actual checkpoint dimensions. Reusing
        # zero float objects is safe; each key owns its own mutable list.
        for name, count in LENGTHS.items():
            if name not in cls.adam:
                cls.adam[name] = [0.0] * count
        cls.aux_adam = edit(cls.adam, {
            "l2": {2 * 32 + 4: 3.25, 258 * 32 + 5: -0.75},
            "l2_bias": {4: -20.0, 5: 1.5}, "out": {4: 4.0, 5: -2.0},
            "l2_m": {2 * 32 + 4: -0.125}, "l2_v": {2 * 32 + 4: 0.25},
            "l2bias_m": {4: -0.25}, "l2bias_v": {4: 0.5},
            "out_m": {4: 0.125}, "out_v": {4: 0.75},
        })
        cls.aux_native = native_with_parameters(cls.native, cls.aux_adam)

    def validate(self, adam=None, native=None, expected_step=EXPECTED_STEP):
        return diagnose.validate_bounded_adam(
            self.adam if adam is None else adam,
            self.native if native is None else native,
            expected_step,
        )

    def test_full_seed42_and_allowed_auxiliary_state(self):
        for adam, native, norm in ((self.adam, self.native, 0),
                                   (self.aux_adam, self.aux_native, 6)):
            with self.subTest(auxiliary_l1=norm):
                lists_before = {name: id(adam[name]) for name in LENGTHS}
                proof = self.validate(adam, native)
                self.assertEqual(proof["expected_step"], EXPECTED_STEP)
                self.assertEqual(proof["actual_step"], EXPECTED_STEP)
                self.assertTrue(proof["actual_raw_ft_binary32_fixed"])
                self.assertTrue(proof["protected_moments_positive_zero"])
                self.assertTrue(proof["raw_native_reexport_all_bytes_equal"])
                self.assertTrue(proof["raw_nearest_export_all_bytes_equal"])
                self.assertEqual(proof["native_validation"]["auxiliary_out_l1"],
                                 {"numerator": norm, "denominator": 1})
                self.assertFalse(proof["intermediate_activations_verified"])
                self.assertFalse(proof["universal_integer_core_bound_proven"])
                self.assertFalse(proof["adoption_claimed"])
                self.assertEqual({name: id(adam[name]) for name in LENGTHS}, lists_before)
        self.assertEqual(self.aux_adam["l2_m"][2 * 32 + 4], -0.125)
        self.assertEqual(self.adam["out"][4], 0.0)

    def test_raw_ft_one_ulp_drift_rejected_despite_equal_native_and_nearest(self):
        original = self.adam["ft"][0]
        self.assertGreater(original, 0.0)
        bits = struct.unpack("<I", struct.pack("<f", original))[0]
        changed = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
        self.assertGreater(changed, original)
        drift = edit(self.adam, {"ft": {0: changed}})
        # This precondition proves that checking either inference artifact alone
        # would accept this microscopic raw-FT movement.
        nearest, stats = post_export(validate_adam(dict(drift)), self.native)
        self.assertEqual(nearest, self.native)
        self.assertEqual(stats["ft"]["changed"], 0)
        self.assertGreater(stats["ft"]["native_mean_abs_weight_error"], 0)
        with self.assertRaisesRegex(ValueError, "actual raw FT/bias binary32 differs"):
            self.validate(drift)
        self.assertEqual(self.adam["ft"][0], original)

    def test_protected_moments_and_signed_zero_are_rejected(self):
        cases = (
            ("ft_m", 19, 0.125),
            ("l2_v", 256 * 32 + 4, 0.25),  # blocked material-to-aux row
            ("out_m", 0, -0.0),              # frozen output bit pattern
        )
        for name, index, value in cases:
            with self.subTest(field=name, index=index, value=value):
                with self.assertRaisesRegex(ValueError, "protected \\+0 state changed"):
                    self.validate(edit(self.adam, {name: {index: value}}))

    def test_step_must_match_fresh_complete_epochs_and_have_exact_type(self):
        cases = ((dict(self.adam, step=EXPECTED_STEP - 1), EXPECTED_STEP,
                  "Adam step does not match"),
                 (dict(self.adam, step=True), EXPECTED_STEP, "invalid Adam step"),
                 (self.adam, True, "invalid expected Adam step"))
        for adam, step, message in cases:
            with self.subTest(actual_step=adam["step"], expected_step=step):
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(adam, expected_step=step)

    def test_native_tail_must_reproduce_actual_adam_all_bytes(self):
        with self.assertRaisesRegex(ValueError, "raw Adam native trunc export differs"):
            self.validate(self.aux_adam, self.native)

    def test_absolute_output_norm_rejects_cancelling_coefficients(self):
        excessive = edit(self.adam, {"out": {4: 32.0, 5: -32.0}})
        native = native_with_parameters(self.native, excessive)
        self.assertEqual(sum(excessive["out"][4:]), 0.0)
        # Native and raw agree; rejection is the L1 budget, not serialization.
        self.assertEqual(post_export(validate_adam(dict(excessive)), native)[0], native)
        with self.assertRaisesRegex(ValueError, "auxiliary real residual budget exceeds 99 cp"):
            self.validate(excessive, native)

    def test_mutable_variances_must_be_finite_and_nonnegative(self):
        cases = (("l2_v", 2 * 32 + 4, -0.25, "negative Adam variance: l2_v"),
                 ("out_v", 4, float("inf"), "checkpoint values must be finite numbers"))
        for name, index, value, message in cases:
            with self.subTest(field=name, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(edit(self.aux_adam, {name: {index: value}}), self.aux_native)

    def test_nonfinite_output_and_incomplete_full_checkpoint_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(output_value=value):
                invalid = edit(self.adam, {"out": {4: value}})
                with self.assertRaisesRegex(ValueError, "checkpoint values must be finite numbers"):
                    self.validate(invalid, native_with_parameters(self.native, invalid))
        short = dict(self.adam, ft_v=[0.0])
        with self.assertRaisesRegex(ValueError, "invalid ft_v length; expected 619520"):
            self.validate(short)


class ModeMetadataTests(unittest.TestCase):
    def test_complete_independent_mode_receipt_is_accepted(self):
        diagnose.validate_bounded_metadata(metadata_fixture())

    def test_mode_requires_exact_scalar_nested_types_and_initial_load_binding(self):
        cases = (
            ("residual_budget_cp", 99.0),
            ("mask_version", True),
            ("label_depth", 1),
            ("fixed_l2_columns", [0, True, 2, 3]),
            ("integer_core_bound_proven", True),
        )
        for name, value in cases:
            with self.subTest(field=name, value=value):
                receipt = metadata_fixture()
                receipt[name] = value
                with self.assertRaisesRegex(ValueError, f"bounded metadata mismatch: {name}"):
                    diagnose.validate_bounded_metadata(receipt)
        missing = metadata_fixture()
        del missing["loaded_initial_binding"]
        with self.assertRaisesRegex(ValueError, "bounded metadata field set mismatch"):
            diagnose.validate_bounded_metadata(missing)

    def test_upper_estimates_reject_nonfinite_negative_bool_and_overbudget(self):
        for name in ("saved_f32_l1_upper", "saved_f32_budget_cp_upper"):
            for value in (float("nan"), float("inf"), -0.125, True):
                with self.subTest(field=name, value=value):
                    receipt = metadata_fixture()
                    receipt[name] = value
                    with self.assertRaisesRegex(ValueError, "nonfinite or negative bounded metadata"):
                        diagnose.validate_bounded_metadata(receipt)
        over = metadata_fixture()
        over["saved_f32_budget_cp_upper"] = 99.00000000000001
        with self.assertRaisesRegex(ValueError, "bounded metadata residual estimate exceeds 99 cp"):
            diagnose.validate_bounded_metadata(over)


if __name__ == "__main__":
    unittest.main()
