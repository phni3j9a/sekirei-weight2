"""Synthetic in-memory bytes only; no core, dataset, engine or training run."""
from fractions import Fraction
import hashlib
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bounded_material as bounded
import material_init


class BoundedMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = material_init.encode(material_init.build_weights(42))

    def f32_at(self, offset, value, data=None):
        result = bytearray(self.initial if data is None else data)
        struct.pack_into("<f", result, offset, value)
        return bytes(result)

    def i16_at(self, offset, data=None):
        result = bytearray(self.initial if data is None else data)
        old = struct.unpack_from("<h", result, offset)[0]
        struct.pack_into("<h", result, offset, old + 1)
        return bytes(result)

    def test_initializer_passes_but_recipe_never_claims_real_success(self):
        result = bounded.validate_weights(self.initial)
        self.assertEqual(hashlib.sha256(self.initial).hexdigest(), bounded.MATERIAL_SEED42_SHA256)
        self.assertEqual(result["auxiliary_real_residual_bound_cp"], {"numerator": 0, "denominator": 1})
        self.assertEqual(result["real_residual_cap_cp"], 99)
        recipe = bounded.candidate_recipe(self.initial)
        self.assertEqual(recipe["status"], "technical_preparation_only")
        self.assertEqual(recipe["candidate_contract"]["proposed_integer_cap_cp"], 100)
        for key in ("preregistered", "real_data_preflight_verified", "trainer_mask_verified",
                    "optimizer_verified", "core_integer_difference_verified", "engine_verified",
                    "training_verified", "adoption_verified"):
            self.assertIs(recipe[key], False)
        self.assertEqual(self.initial, material_init.encode(material_init.build_weights(42)))

    def test_immutable_bytes_length_and_magic_are_strict(self):
        for data in (None, True, bytearray(self.initial), memoryview(self.initial),
                     self.initial[:-1], self.initial + b"\0", b"WRONGMAG" + self.initial[8:]):
            with self.subTest(kind=type(data), length=len(data) if hasattr(data, "__len__") else None):
                with self.assertRaises(bounded.BoundedMaterialError):
                    bounded.validate_weights(data)

    def test_full_ft_and_bias_groups_are_frozen_including_auxiliary_features(self):
        for offset in (bounded.FT_OFFSET, bounded.FT_OFFSET + 2 * 2,
                       bounded.FT_END - 2, bounded.FT_END, bounded.L2_OFFSET - 2):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "frozen FT"):
                bounded.validate_weights(self.i16_at(offset))
        with self.assertRaisesRegex(ValueError, "frozen FT"):
            bounded.validate_weights(material_init.encode(material_init.build_weights(43)))

    def test_every_frozen_l2_row_and_column_rejects_mutation(self):
        for row in range(512):
            for column in range(4):
                offset = bounded.L2_OFFSET + (row * 32 + column) * 4
                old = struct.unpack_from("<f", self.initial, offset)[0]
                with self.subTest(row=row, column=column), self.assertRaisesRegex(ValueError, "material L2"):
                    bounded.validate_weights(self.f32_at(offset, old + 0.25))

    def test_material_l2_bias_and_out_coordinates_are_frozen(self):
        for name, base in (("bias", bounded.L2_BIAS_OFFSET), ("out", bounded.OUT_OFFSET)):
            for coordinate in range(4):
                offset = base + coordinate * 4
                old = struct.unpack_from("<f", self.initial, offset)[0]
                with self.subTest(name=name, coordinate=coordinate), self.assertRaises(ValueError):
                    bounded.validate_weights(self.f32_at(offset, old + 0.25))

    def test_every_material_to_auxiliary_connection_is_positive_zero(self):
        for row in bounded.MATERIAL_ROWS:
            for column in bounded.AUXILIARY_COLUMNS:
                offset = bounded.L2_OFFSET + (row * 32 + column) * 4
                for value in (0.125, -0.0):
                    with self.subTest(row=row, column=column, value=value), self.assertRaisesRegex(ValueError, "positive-zero"):
                        bounded.validate_weights(self.f32_at(offset, value))

    def test_positive_zero_is_byte_strict_for_all_protected_float_zeros(self):
        for offset in (bounded.L2_OFFSET + (2 * 32) * 4, bounded.L2_BIAS_OFFSET,
                       bounded.OUT_BIAS_OFFSET):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                bounded.validate_weights(self.f32_at(offset, -0.0))
        with self.assertRaisesRegex(ValueError, "out bias"):
            bounded.validate_weights(self.f32_at(bounded.OUT_BIAS_OFFSET, 0.125))

    def test_nan_and_both_infinities_are_rejected_in_every_float_group(self):
        offsets = (bounded.L2_OFFSET + (2 * 32 + 4) * 4,
                   bounded.L2_BIAS_OFFSET + 4 * 4, bounded.OUT_OFFSET + 4 * 4,
                   bounded.OUT_BIAS_OFFSET)
        for offset in offsets:
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(offset=offset, value=value), self.assertRaisesRegex(ValueError, "nonfinite"):
                    bounded.validate_weights(self.f32_at(offset, value))

    def test_auxiliary_changes_and_unprotected_signed_zero_are_allowed(self):
        data = self.f32_at(bounded.L2_OFFSET + (2 * 32 + 4) * 4, 3.25)
        data = self.f32_at(bounded.L2_BIAS_OFFSET + 4 * 4, -20.0, data)
        data = self.f32_at(bounded.OUT_OFFSET + 4 * 4, 4.0, data)
        data = self.f32_at(bounded.OUT_OFFSET + 5 * 4, -2.0, data)
        data = self.f32_at(bounded.OUT_OFFSET + 6 * 4, -0.0, data)
        data = self.f32_at(bounded.L2_OFFSET + (3 * 32 + 7) * 4, -0.0, data)
        result = bounded.validate_weights(data)
        self.assertEqual(result["auxiliary_out_l1"], {"numerator": 6, "denominator": 1})
        self.assertEqual(result["auxiliary_real_residual_bound_cp"], {"numerator": 381, "denominator": 32})
        self.assertEqual(data[bounded.FT_OFFSET:bounded.L2_OFFSET],
                         self.initial[bounded.FT_OFFSET:bounded.L2_OFFSET])

    def test_budget_uses_absolute_sum_not_signed_cancellation(self):
        data = self.f32_at(bounded.OUT_OFFSET + 4 * 4, 32.0)
        data = self.f32_at(bounded.OUT_OFFSET + 5 * 4, -32.0, data)
        with self.assertRaisesRegex(ValueError, "budget"):
            bounded.validate_weights(data)

    def test_exact_fraction_budget_boundary_uses_adjacent_actual_f32s(self):
        # 99*64/127 is not dyadic, so no binary32 sum equals the limit exactly.
        # The greatest f32 below it must pass; its immediate neighbour must fail.
        threshold = Fraction(99 * 64, 127)
        approx = struct.pack("<f", float(threshold))
        bits = struct.unpack("<I", approx)[0]
        if Fraction.from_float(struct.unpack("<f", approx)[0]) > threshold:
            bits -= 1
        below = struct.unpack("<f", struct.pack("<I", bits))[0]
        above = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
        self.assertLess(Fraction.from_float(below), threshold)
        self.assertGreater(Fraction.from_float(above), threshold)
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                result = bounded.validate_weights(self.f32_at(bounded.OUT_OFFSET + 4 * 4, sign * below))
                record = result["auxiliary_real_residual_bound_cp"]
                self.assertEqual(Fraction(record["numerator"], record["denominator"]),
                                 Fraction.from_float(below) * Fraction(127, 64))
                with self.assertRaisesRegex(ValueError, "budget"):
                    bounded.validate_weights(self.f32_at(bounded.OUT_OFFSET + 4 * 4, sign * above))

    def test_fraction_sum_retains_binary32_small_terms_without_tolerance(self):
        data = self.f32_at(bounded.OUT_OFFSET + 4 * 4, 1.0)
        data = self.f32_at(bounded.OUT_OFFSET + 5 * 4, -(2 ** -30), data)
        result = bounded.validate_weights(data)
        record = result["auxiliary_out_l1"]
        self.assertEqual(Fraction(record["numerator"], record["denominator"]),
                         Fraction(1) + Fraction(1, 2 ** 30))

    def test_standard_initializer_native_ft_orbit_is_complete_and_not_a_core_test(self):
        result = bounded.validate_native_ft_roundtrip(self.initial)
        self.assertEqual(result["ft_and_bias_values"], 619776)
        self.assertEqual(result["ft_and_bias_sha256"],
                         hashlib.sha256(self.initial[8:bounded.L2_OFFSET]).hexdigest())
        self.assertIs(result["native_ft_bytes_equal"], True)
        self.assertIs(result["reference_native_roundtrip_has_no_nearest_difference"], True)
        self.assertIs(result["raw_checkpoint_ft_verified"], False)
        self.assertIs(result["engine_verified"], False)
        self.assertIs(result["training_verified"], False)
        # Stored pawn weight=50, training float=50/64=100/128, not 50/128.
        stored = struct.unpack_from("<h", self.initial, bounded.FT_OFFSET)[0]
        self.assertEqual(stored, 50)
        self.assertEqual(stored / 64, 100 / 128)
        with self.assertRaisesRegex(ValueError, "frozen FT"):
            bounded.validate_native_ft_roundtrip(self.i16_at(bounded.FT_END))


if __name__ == "__main__":
    unittest.main()
