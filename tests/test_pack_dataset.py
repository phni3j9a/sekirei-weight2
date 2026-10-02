"""Game-level separation and cross-split board leakage protection."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pack_dataset import game_identity, position_key, remove_shared_positions, split_for_game


class DatasetTests(unittest.TestCase):
    def test_ply_counter_does_not_hide_shared_positions(self):
        groups = {"train": [{"sfen": "a b - 4"}, {"sfen": "c w P 7"}],
                  "holdout": [{"sfen": "a b - 99"}, {"sfen": "d w - 8"}]}
        kept, count = remove_shared_positions(groups)
        self.assertEqual(count, 1)
        self.assertEqual(kept["train"], [{"sfen": "c w P 7"}])
        self.assertEqual(kept["holdout"], [{"sfen": "d w - 8"}])

    def test_side_and_hand_remain_part_of_leakage_identity(self):
        self.assertNotEqual(position_key("a b P 1"), position_key("a w P 1"))
        self.assertNotEqual(position_key("a b P 1"), position_key("a b - 1"))

    def test_same_game_in_different_packs_keeps_split(self):
        a = [{"sfen": "a b - 1", "selected_move": "7g7f", "pack_sha256": "x"},
             {"sfen": "c w - 2", "selected_move": "3c3d", "pack_sha256": "x"}]
        b = [dict(row, pack_sha256="y", game_index=45) for row in a]
        self.assertEqual(game_identity(a), game_identity(b))
        self.assertEqual(split_for_game(game_identity(a), "seed"), split_for_game(game_identity(b), "seed"))

    def test_within_split_duplicates_keep_only_one_label(self):
        groups = {"train": [{"sfen": "a b - 4", "label": 1},
                            {"sfen": "a b - 5", "label": 2}], "holdout": []}
        kept, count = remove_shared_positions(groups)
        self.assertEqual(count, 0)
        self.assertEqual(len(kept["train"]), 1)


if __name__ == "__main__":
    unittest.main()
