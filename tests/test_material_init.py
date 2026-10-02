import json
from pathlib import Path
import struct
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import material_init as m

FIXTURES = json.loads((Path(__file__).resolve().parent / "fixtures/material_init.json").read_text())


class MaterialInitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = m.build_weights(42)
        cls.binary = m.encode(cls.weights)

    def test_format_round_trip_and_reproducibility(self):
        self.assertEqual(len(self.binary), 1305356)
        self.assertEqual(self.binary[:8], b"SEKIRW01")
        self.assertEqual(m.encode(m.decode(self.binary)), self.binary)
        self.assertEqual(m.encode(m.build_weights(42)), self.binary)
        self.assertNotEqual(m.encode(m.build_weights(43)), self.binary)
        self.assertEqual(struct.unpack_from("<h", self.binary, 8)[0], 50)
        self.assertEqual(m.fnv1a(b""), "cbf29ce484222325")
        self.assertEqual(m.fnv1a(b"a"), "af63dc4c8601ec8c")

    def test_every_board_feature_has_correct_group_color_and_value(self):
        for square in range(81):
            for kind, value in enumerate(m.VALUES):
                for opponent in range(2):
                    offset = (square * 28 + kind * 2 + opponent) * m.L1
                    expected = [0, 0]
                    if not opponent and kind != 7:
                        expected[m.group(kind)] = value // 2
                    self.assertEqual(list(self.weights["ft"][offset:offset + 2]), expected)

    def test_every_hand_threshold_has_correct_own_perspective(self):
        for color in range(2):
            for perspective in range(2):
                for kind, count in enumerate(m.HAND_MAX):
                    for n in range(1, count + 1):
                        feature = m.BOARD_INPUT + (color * 2 + perspective) * 38 + m.HAND_OFFSETS[kind] + n - 1
                        offset = feature * m.L1
                        expected = [0, 0]
                        if color == perspective:
                            expected[m.group(kind)] = m.VALUES[kind] // 2
                        self.assertEqual(list(self.weights["ft"][offset:offset + 2]), expected)

    def test_auxiliary_path_is_live_but_has_zero_initial_output(self):
        self.assertEqual(list(self.weights["out"][4:]), [0.0] * 28)
        self.assertEqual(set(self.weights["ft_bias"]), {64})
        for row in range(2 * m.L1):
            for column in range(4):
                expected = 1.0 if (row, column) in ((0, 0), (1, 1), (256, 2), (257, 3)) else 0.0
                self.assertEqual(self.weights["l2"][row * 32 + column], expected)
        # Bounds hold for all <=40-piece inventory-valid positions, independent
        # of RNG draws, board arrangement, side to move or held pieces.
        ft_lower, ft_upper = (64 - 40) / 64, (64 + 40) / 64
        l2_radius = 2 * 254 * ft_upper / 256
        self.assertGreater(ft_lower, 0)
        self.assertLess(ft_upper, 127)
        self.assertGreater(4 - l2_radius, 0)
        self.assertLess(4 + l2_radius, 127)
        # Distinct columns avoid the all-identical initializer pathology.
        columns = {tuple(self.weights["l2"][row * 32 + col] for row in range(512))
                   for col in range(4, 32)}
        self.assertEqual(len(columns), 28)

    def test_full_binary_forward_matches_reachable_fixtures(self):
        result = m.verify_fixtures(m.decode(self.binary), FIXTURES)
        self.assertEqual(result["fixture_count"], 15)
        self.assertFalse(result["engine_verified"])
        self.assertEqual({row["score_cp"] for row in result["results"]}, {-700, -500, 0, 500})
        alternate = m.build_weights(43)
        for fixture in FIXTURES[9:]:
            p = m.parse_sfen(fixture["sfen"])
            self.assertEqual(m.reference_forward(alternate, p)["float_cp"], m.material(p))

    def test_upper_bounds_are_below_all_clips_and_float_exactness_limit(self):
        pawn_max = 18 * max(m.VALUES[0], m.VALUES[8])
        other_max = sum(n * max(m.VALUES[k], m.VALUES[p])
                        for k, p, n in ((1, 9, 4), (2, 10, 4), (3, 11, 4),
                                        (4, 4, 4), (5, 12, 2), (6, 13, 2)))
        self.assertEqual((pawn_max, other_max), (10800, 14980))
        self.assertLess(64 + max(pawn_max, other_max) // 2, 127 * 64)
        self.assertLess(64 * (pawn_max + other_max) + 4 * 8192, 2 ** 24)
        self.assertTrue(all(value % 2 == 0 for value in m.VALUES))

    def test_overfull_inventory_is_rejected(self):
        for sfen in ("4k4/9/9/9/9/9/9/9/4K4 b 19P 1",
                     "4k4/9/9/9/9/9/9/9/4K4 b 3R 1",
                     "9/9/9/9/9/9/9/9/4K4 b - 1"):
            with self.assertRaises(ValueError):
                m.parse_sfen(sfen)

    def test_fixture_sfen_matches_pawn_only_replay_from_startpos(self):
        # These hand-authored public fixtures only use pawn moves, captures,
        # optional promotions and pawn drops. Full-engine legality still must
        # be checked separately; this check prevents SFEN/transcription drift.
        start = m.parse_sfen(FIXTURES[0]["sfen"])
        def square(text):
            return (9 - int(text[0])) * 9 + ord(text[1]) - ord("a")
        for fixture in FIXTURES:
            board = {sq: (kind, color) for sq, kind, color in start["pieces"]}
            hands = [[0] * 7 for _ in range(2)]
            stm = 0
            for move in fixture["from_startpos_usi"]:
                to = square(move[2:4])
                if move[1] == "*":
                    self.assertEqual(move[0], "P")
                    self.assertNotIn(to, board)
                    self.assertGreater(hands[stm][0], 0)
                    self.assertFalse(any(k == 0 and c == stm and sq // 9 == to // 9
                                         for sq, (k, c) in board.items()))
                    self.assertNotEqual(to % 9, 0 if stm == 0 else 8)
                    hands[stm][0] -= 1
                    kind = 0
                else:
                    frm = square(move[:2])
                    kind, color = board.pop(frm)
                    self.assertEqual((kind, color), (0, stm))
                    self.assertEqual(frm // 9, to // 9)
                    self.assertEqual(to % 9 - frm % 9, -1 if stm == 0 else 1)
                    captured = board.pop(to, None)
                    if captured:
                        self.assertNotEqual(captured[1], stm)
                        self.assertNotEqual(captured[0], 7)
                        hands[stm][m.BASE[captured[0]]] += 1
                    if move.endswith("+"):
                        self.assertTrue(to % 9 <= 2 if stm == 0 else to % 9 >= 6)
                        kind = 8
                board[to] = (kind, stm)
                stm = 1 - stm
            parsed = m.parse_sfen(fixture["sfen"])
            self.assertEqual(sorted(parsed["pieces"]), sorted((sq, k, c) for sq, (k, c) in board.items()))
            self.assertEqual(parsed["hands"], hands)
            self.assertEqual(parsed["stm"], stm)


if __name__ == "__main__":
    unittest.main()
