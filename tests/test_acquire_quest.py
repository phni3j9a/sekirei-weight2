"""Tests for Shogi Quest corpus filtering and normalization."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from acquire_quest import (
    STANDARD_BOARD,
    anonymized_csa,
    looks_like_bot,
    parse_csa,
    parse_official_attrs,
)


class HumanMarkerTests(unittest.TestCase):
    def test_official_human_attribute(self):
        html = '<script>x.attrs=["opp:human","senkei:aigakari"];x.id="abc"</script>'
        self.assertIn("opp:human", parse_official_attrs(html))

    def test_bot_markers(self):
        self.assertTrue(looks_like_bot({"id": ":shogi8bot"}))
        self.assertTrue(looks_like_bot({"id": "x", "avatar": "bot_s"}))
        self.assertFalse(looks_like_bot({"id": "human", "avatar": "s_02"}))


class CsaTests(unittest.TestCase):
    def fixture(self, moves=("+7776FU", "-3334FU")):
        return "\n".join((
            "'Shogi Quest",
            "N+Alice(1500)",
            "N-Bob(1600)",
            *STANDARD_BOARD,
            "+",
            *moves,
            "",
        ))

    def test_normalization_omits_identity(self):
        parsed = parse_csa(self.fixture())
        self.assertEqual(parsed["plies"], 2)
        self.assertNotIn("Alice", parsed["canonical"])
        snapshot = anonymized_csa(parsed)
        self.assertIn("N+black", snapshot)
        self.assertNotIn("1500", snapshot)

    def test_rejects_broken_turn_order(self):
        with self.assertRaisesRegex(ValueError, "turn order"):
            parse_csa(self.fixture(("+7776FU", "+2726FU")))

    def test_rejects_nonstandard_board(self):
        with self.assertRaisesRegex(ValueError, "standard-even"):
            parse_csa(self.fixture().replace("P1-KY", "P1 * ", 1))


if __name__ == "__main__":
    unittest.main()
