"""Guard against accepting misleading USI scores as successful observations."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from smoke import parse_result


class ScoreAcceptanceTests(unittest.TestCase):
    def test_white_score_is_converted_to_sente(self):
        result = parse_result([
            "info depth 10 score cp 125 nodes 1000000 pv 8c8d 2f2e",
            "bestmove 8c8d ponder 2f2e",
        ], "white")
        self.assertEqual(result["score_cp_sente"], -125)

    def test_final_bound_does_not_reuse_an_earlier_exact_score(self):
        with self.assertRaisesRegex(RuntimeError, "bound"):
            parse_result([
                "info depth 9 score cp 10 nodes 400000 pv 8c8d",
                "info depth 10 score cp 20 lowerbound nodes 1000000 pv 8c8d",
                "bestmove 8c8d",
            ], "white")

    def test_mismatched_bestmove_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            parse_result([
                "info depth 10 score cp 10 nodes 1000000 pv 8c8d",
                "bestmove 4a3b",
            ], "white")

    def test_unexpected_mate_is_not_coerced_to_cp(self):
        with self.assertRaisesRegex(RuntimeError, "finite cp"):
            parse_result([
                "info depth 10 score mate 3 nodes 1000000 pv 8c8d",
                "bestmove 8c8d",
            ], "white")

    def test_runner_up_is_not_used_as_primary_score(self):
        result = parse_result([
            "info multipv 1 depth 10 score cp 20 nodes 1000000 pv 8c8d",
            "info multipv 2 depth 10 score cp -300 nodes 1000000 pv 4a3b",
            "bestmove 8c8d",
        ], "black")
        self.assertEqual(result["score_cp_sente"], 20)


if __name__ == "__main__":
    unittest.main()
