"""Top3 membership must not be manufactured from stale or invalid ranks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from top3 import parse_top3


def line(rank, move, depth=12):
    return f"info multipv {rank} depth {depth} score cp {100-rank} nodes 1000000 pv {move}"


class Top3Tests(unittest.TestCase):
    moves = ["7g7f", "2g2f", "5g5f"]

    def events(self):
        return ["go nodes 1000000"] + [line(i, m) for i, m in enumerate(self.moves, 1)] + ["bestmove 7g7f"]

    def test_complete_legal_distinct_roots(self):
        self.assertEqual(parse_top3(self.events(), "7g7f", self.moves), self.moves)

    def test_new_primary_cannot_borrow_previous_secondary_ranks(self):
        values = self.events()[:-1] + [line(1, "7g7f", depth=13), "bestmove 7g7f"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parse_top3(values, "7g7f", self.moves)

    def test_mixed_depths_are_invalid(self):
        values = self.events()
        values[2] = line(2, "2g2f", depth=11)
        with self.assertRaisesRegex(ValueError, "depth"):
            parse_top3(values, "7g7f", self.moves)

    def test_duplicate_illegal_and_bound_are_invalid(self):
        for replacement in (line(3, "7g7f"), line(3, "9a9b"),
                            line(3, "5g5f").replace("nodes", "lowerbound nodes")):
            values = self.events()
            values[3] = replacement
            with self.assertRaises(ValueError):
                parse_top3(values, "7g7f", self.moves)

    def test_previous_go_and_post_bestmove_do_not_supply_ranks(self):
        values = self.events() + ["go nodes 1000000", line(1, "7g7f"), "bestmove 7g7f",
                                  line(2, "2g2f"), line(3, "5g5f")]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parse_top3(values, "7g7f", self.moves)


if __name__ == "__main__":
    unittest.main()
