"""Tests for Shogi Quest corpus filtering and normalization."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from acquire_quest import (
    STANDARD_BOARD,
    anonymized_csa,
    choose_snapshot,
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


class SnapshotSelectionTests(unittest.TestCase):
    def test_fixed_buckets_have_no_repeated_players(self):
        games = {}
        for index in range(5):
            for game_type in ("shogi10", "shogi"):
                game_id = f"game{game_type}{index}"
                games[game_id] = {
                    "disposition": "accepted",
                    "canonical_sha256": f"{index:02x}" * 32 + game_type,
                    "plies": 80 + index,
                    "metadata": {
                        "game_type": game_type,
                        "players": [
                            {"id": f"{game_type}-black-{index}", "oldR": 1800},
                            {"id": f"{game_type}-white-{index}", "oldR": 1900},
                        ],
                    },
                }
        selected = choose_snapshot({"games": games})
        self.assertEqual(len(selected), 10)
        players = []
        for _split, _rank, _game_id, game in selected:
            players.extend(player["id"] for player in game["metadata"]["players"])
        self.assertEqual(len(players), len(set(players)))


if __name__ == "__main__":
    unittest.main()
