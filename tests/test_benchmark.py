"""Tests for the fixed development universe, USI boundary, and report masks."""

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark import (  # noqa: E402
    BenchmarkError,
    CleanupError,
    ConfigurationError,
    NODE_POLICY,
    NODE_POLICY_ID,
    NODE_POLICY_VERSION,
    OUTCOME_FIELDS,
    PARSER_ID,
    POSITION_CLASSIFICATION_TYPES,
    atomic_write,
    atomic_write_json,
    build_universe,
    compare_cshogi,
    development_csa_hash,
    execute_run,
    load_config,
    load_development_games,
    load_position_classifications,
    make_plan,
    expected_attempt_matrix,
    json_sha256,
    observed_max_reported_nodes,
    parse_csa_text,
    parse_usi_observation,
    positive_node_evidence,
    node_reporting_limit,
    run_engine_attempt,
    sha256,
    USIProcess,
    validate_formal_gate,
    universe_hash,
)
from benchmark_report import (  # noqa: E402
    export_public,
    load_run,
    report_from_run,
    render_svg,
    score_observations,
)


FAKE = Path(__file__).resolve().parent / "fixtures/fake_usi.py"


class UniverseTests(unittest.TestCase):
    def test_fixed_development_universe_has_570_occurrences(self):
        config = load_config()
        universe, games = build_universe(config)
        self.assertEqual([game["plies"] for game in games], [86, 126, 127, 108, 123])
        self.assertEqual(sum(len(game["occurrences"]) for game in universe["games"]), 570)
        self.assertEqual(development_csa_hash(games), config["hashes"]["development_csa_aggregate_sha256"])
        self.assertEqual(universe_hash(universe), config["hashes"]["universe_sha256"])

    def test_node_policy_is_named_integer_and_shown_in_both_plans(self):
        config = load_config()
        self.assertEqual(config["node_policy"], NODE_POLICY)
        self.assertEqual(NODE_POLICY_ID, "one-sided-1-percent")
        self.assertEqual(NODE_POLICY_VERSION, 1)
        self.assertEqual(node_reporting_limit(config["requested_nodes"]), 1_010_000)
        for run_type in ("pilot", "formal"):
            plan, _ = make_plan(run_type, config)
            self.assertEqual(plan["max_reported_nodes"], 1_010_000)

    def test_pilot_points_are_first_middle_last_and_no_final_path(self):
        config = load_config()
        from benchmark import make_plan

        plan, _ = make_plan("pilot", config)
        self.assertEqual(len(plan["positions"]), 17)
        for index in range(0, 15, 3):
            values = [row["ply"] for row in plan["positions"][index:index + 3]]
            length = config["universe"]["plies"][plan["positions"][index]["game_id"]]
            self.assertEqual(values, [1, (length + 1) // 2, length])
        self.assertTrue(all("final" not in row["position"] for row in plan["positions"]))
        self.assertEqual(
            [(row["game_id"], row["ply"]) for row in plan["positions"][-2:]],
            [("development-04", 77), ("development-05", 122)],
        )
        self.assertTrue(all(row["selection"] == "regression" for row in plan["positions"][-2:]))
        self.assertTrue(all(row["selection"] == "canonical" for row in plan["positions"][:15]))

    def test_current_pilot_and_formal_attempt_matrices_are_exact(self):
        config = load_config()
        pilot, _ = make_plan("pilot", config)
        formal, _ = make_plan("formal", config)
        self.assertEqual(len(pilot["positions"]), 17)
        self.assertEqual(len(expected_attempt_matrix(pilot, config)), 102)
        self.assertEqual(len(formal["positions"]), 570)
        self.assertEqual(len(expected_attempt_matrix(formal, config)), 1140)
        self.assertEqual(pilot["max_reported_nodes"], 1_010_000)
        self.assertEqual(formal["max_reported_nodes"], 1_010_000)

    def test_node_policy_tampering_fails_even_when_limits_are_coordinated(self):
        config = copy.deepcopy(load_config())
        config["node_policy"]["rate_numerator"] = 2
        config["pilot"]["max_reported_nodes"] = 1_020_000
        config["formal"]["max_reported_nodes"] = 1_020_000
        with self.assertRaisesRegex(ConfigurationError, "node_policy"):
            make_plan("pilot", config)

    def test_conversion_handles_promotion_capture_and_drop(self):
        game = next(game for game in load_development_games(load_config()) if game["game_id"] == "development-05")
        self.assertTrue(any(move.startswith("P*") for move in game["usi_moves"]))
        self.assertTrue(any(move.endswith("+") for move in game["usi_moves"]))
        self.assertEqual(game["boards_after"][-1].side, -1)

    def test_malformed_side_and_ownership_transition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "side-to-move"):
            parse_csa_text("PI\n+\n-7776FU\n")
        with self.assertRaisesRegex(ValueError, "other side"):
            parse_csa_text("PI\n+\n+3334FU\n")

    def test_optional_cshogi_check_does_not_make_ci_depend_on_it(self):
        config = load_config()
        games = load_development_games(config)
        result = compare_cshogi(games)
        self.assertIn("available", result)
        if result["available"]:
            self.assertEqual(result["checked_legal_moves"], 46668)
            self.assertEqual(result["checked_occurrences"], 570)
            self.assertEqual(
                result["classification_counts"],
                {"normal": 563, "mate_in_one_available": 1, "forced_single_legal_move": 5, "terminal_checkmate": 1, "terminal_stalemate": 0},
            )

    def test_position_types_are_hash_bound_and_leave_terminal_in_universe(self):
        config = load_config()
        universe, _games = build_universe(config)
        values = {
            occurrence["occurrence_id"]: occurrence
            for game in universe["games"]
            for occurrence in game["occurrences"]
        }
        self.assertEqual(
            {kind: sum(occurrence["position_type"] == kind for occurrence in values.values()) for kind in POSITION_CLASSIFICATION_TYPES},
            {"normal": 563, "mate_in_one_available": 1, "forced_single_legal_move": 5, "terminal_checkmate": 1, "terminal_stalemate": 0},
        )
        self.assertEqual(values["development-05:123"]["position_type"], "terminal_checkmate")
        self.assertEqual(values["development-04:106"]["classification"]["sole_legal_move"], "2h1h")
        self.assertEqual(values["development-05:122"]["classification"]["mating_moves"], ["2g4g"])
        self.assertEqual(load_position_classifications(config)["counts"], {
            "normal": 563,
            "mate_in_one_available": 1,
            "forced_single_legal_move": 5,
            "terminal_checkmate": 1,
            "terminal_stalemate": 0,
        })

    def test_position_metadata_hash_mismatch_fails_closed(self):
        config = copy.deepcopy(load_config())
        config["position_classifications"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ConfigurationError, "hash mismatch"):
            load_position_classifications(config)


class USIParserTests(unittest.TestCase):
    def test_scores_are_bounded_by_this_go_and_info_string_is_free_text(self):
        events = [
            {"offset_ns": 1, "direction": "receive", "line": "info depth 1 score cp 999 nodes 1 pv 7g7f"},
            {"offset_ns": 2, "direction": "receive", "line": "bestmove 7g7f"},
            {"offset_ns": 3, "direction": "send", "line": "go nodes 1000"},
            {"offset_ns": 4, "direction": "receive", "line": "info string score cp 777 nodes 999 pv 3c3d"},
            {"offset_ns": 5, "direction": "receive", "line": "info depth 10 score cp 20 nodes 1000 pv 7g7f"},
            {"offset_ns": 6, "direction": "receive", "line": "bestmove 7g7f"},
        ]
        result = parse_usi_observation(events, "black", requested_nodes=1000)
        self.assertEqual(result["status"], "exact_cp")
        self.assertEqual(result["score_cp_sente"], 20)
        self.assertEqual(result["reported_nodes_at_score"], 1000)
        self.assertEqual(result["info_count"], 1)

    def test_plain_line_window_starts_at_the_latest_go_nodes(self):
        result = parse_usi_observation([
            "info depth 1 score cp 999 nodes 2000 pv 7g7f",
            "bestmove 7g7f",
            "go nodes 1000",
            "info depth 10 score cp 20 nodes 1000 pv 7g7f",
            "bestmove 7g7f",
        ], "black", requested_nodes=1000, max_reported_nodes=1000)
        self.assertEqual(result["status"], "exact_cp")
        self.assertEqual(result["node_evidence_count"], 1)
        self.assertEqual(result["max_reported_nodes_evidence"], 1000)

    def test_info_string_consumes_the_remainder_at_any_position(self):
        cases = (
            "info depth 1 string diagnostic score cp 77 nodes 1000 pv 7g7f",
            "info depth 1 currmove 7g7f string diagnostic score cp 77 nodes 1000 pv 7g7f",
            "info score cp 12 string diagnostic pv 7g7f nodes 1000",
        )
        for line in cases:
            result = parse_usi_observation([line, "bestmove 7g7f"], "black", requested_nodes=1000)
            self.assertEqual(result["status"], "no_score", line)
            self.assertIsNone(result["score_kind"], line)
            self.assertEqual(result["info_count"], 0, line)
        result = parse_usi_observation(
            ["info score cp 12 nodes 1000 pv 7g7f string diagnostic score cp 77 nodes 1000", "bestmove 7g7f"],
            "black",
            requested_nodes=1000,
        )
        self.assertEqual(result["status"], "exact_cp")
        self.assertEqual(result["score_cp_sente"], 12)

    def test_malformed_string_placement_cannot_create_late_score(self):
        result = parse_usi_observation(
            ["info depth 1 string score cp 77 nodes 1000 pv 7g7f", "bestmove 7g7f"],
            "black",
            requested_nodes=1000,
        )
        self.assertEqual(result["status"], "no_score")
        result = parse_usi_observation(
            ["info depth 1 score cp 12 string free text score mate +3 pv 7g7f", "bestmove 7g7f"],
            "black",
            requested_nodes=1000,
        )
        self.assertEqual(result["status"], "no_score")
        self.assertIsNone(result["pv"])

    def test_bare_mate_signs_preserve_lexical_winner_without_distance(self):
        for raw, winner in (("+", "black"), ("-", "white")):
            result = parse_usi_observation(
                [f"info depth 10 score mate {raw} nodes 1000 pv 7g7f", "bestmove 7g7f"],
                "black",
                requested_nodes=1000,
            )
            self.assertEqual(result["status"], "mate")
            self.assertEqual(result["score_mate_stm"], raw)
            self.assertIsNone(result["mate_distance"])
            self.assertFalse(result["mate_distance_known"])
            self.assertEqual(result["winner_sente"], winner)

    def test_cp_sign_and_bound_direction_are_converted_from_stm(self):
        result = parse_usi_observation(
            ["info depth 10 score cp 125 nodes 1000 pv 7g7f", "bestmove 7g7f"],
            "white",
            requested_nodes=1000,
        )
        self.assertEqual(result["status"], "exact_cp")
        self.assertEqual(result["score_cp_sente"], -125)
        result = parse_usi_observation(
            ["info depth 10 score cp -63 upperbound nodes 1000 pv 7g7f", "bestmove 7g7f"],
            "white",
            requested_nodes=1000,
        )
        self.assertEqual(result["status"], "bound_cp")
        self.assertEqual(result["score_bound_stm"], "upperbound")
        self.assertEqual(result["score_bound_sente"], "lowerbound")

    def test_final_bound_wins_over_earlier_exact_and_multipv_is_filtered(self):
        result = parse_usi_observation([
            "info multipv 1 depth 9 score cp 10 nodes 400 pv 7g7f",
            "info multipv 2 depth 9 score cp -300 nodes 400 pv 3c3d",
            "info multipv 1 depth 10 score cp 20 lowerbound nodes 1000 pv 7g7f",
            "bestmove 7g7f",
        ], "black", requested_nodes=1000)
        self.assertEqual(result["status"], "bound_cp")
        self.assertIsNone(result["score_cp_sente"])
        self.assertEqual(result["reported_cp_stm"], 20)

    def test_all_current_go_nodes_drive_maximum_before_selected_score(self):
        result = parse_usi_observation([
            "info depth 9 score cp 10 nodes 2000 pv 7g7f",
            "info depth 10 score cp 20 nodes 1000 pv 7g7f",
            "bestmove 7g7f",
        ], "black", requested_nodes=1000, max_reported_nodes=1000)
        self.assertEqual(result["status"], "node_budget_failure")
        self.assertEqual(result["failure_reason"], "node_count_above_policy_limit")
        self.assertEqual(result["reported_nodes_at_score"], 1000)
        self.assertEqual(result["last_reported_nodes"], 1000)
        self.assertEqual(result["max_reported_nodes_evidence"], 2000)

    def test_all_current_go_nodes_drive_maximum_after_selected_score(self):
        result = parse_usi_observation([
            "info depth 10 score cp 20 nodes 1000 pv 7g7f",
            "info depth 11 nodes 2000",
            "info depth 12 nodes 1000",
            "bestmove 7g7f",
        ], "black", requested_nodes=1000, max_reported_nodes=1000)
        self.assertEqual(result["status"], "node_budget_failure")
        self.assertEqual(result["failure_reason"], "node_count_above_policy_limit")
        self.assertEqual(result["reported_nodes_at_score"], 1000)
        self.assertEqual(result["last_reported_nodes"], 1000)
        self.assertEqual(result["max_reported_nodes_evidence"], 2000)

    def test_invalid_node_evidence_cannot_be_hidden_by_a_later_valid_value(self):
        for bad_line, reason in (
            ("info depth 9 nodes -1", "negative_node_evidence"),
            ("info depth 9 nodes -0", "negative_node_evidence"),
            ("info depth 9 nodes xyz", "malformed_node_evidence"),
            ("info depth 9 nodes True", "malformed_node_evidence"),
            ("info depth 9 nodes 1.5", "malformed_node_evidence"),
        ):
            result = parse_usi_observation([
                bad_line,
                "info depth 10 score cp 20 nodes 1000 pv 7g7f",
                "bestmove 7g7f",
            ], "black", requested_nodes=1000)
            self.assertEqual(result["status"], "node_budget_failure", bad_line)
            self.assertEqual(result["failure_reason"], reason, bad_line)
            self.assertEqual(result["last_reported_nodes"], 1000, bad_line)

    def test_malformed_score_or_unknown_field_cannot_hide_later_nodes(self):
        cases = (
            ("info score bogus 0 nodes 2000", "protocol_failure", "malformed_score", 2000),
            ("info score bogus 0 nodes -1", "protocol_failure", "malformed_score", 1000),
            ("info vendor_noise payload nodes 2000", "node_budget_failure", "node_count_above_policy_limit", 2000),
        )
        for preceding, status, reason, expected_max in cases:
            result = parse_usi_observation(
                [
                    preceding,
                    "info depth 10 score cp 20 nodes 1000 pv 7g7f",
                    "bestmove 7g7f",
                ],
                "black",
                requested_nodes=1000,
                max_reported_nodes=1000,
            )
            self.assertEqual(result["status"], status, preceding)
            self.assertEqual(result["failure_reason"], reason, preceding)
            self.assertEqual(result["max_reported_nodes_evidence"], expected_max, preceding)

    def test_payload_words_in_malformed_score_slots_cannot_hide_later_nodes(self):
        cases = (
            "info score pv 0 nodes 2000",
            "info score bogus pv nodes 2000",
            "info score cp pv nodes 2000",
            "info score bogus string nodes -1",
            "info score refutation 0 nodes 2000",
            "info score cp currline nodes 2000",
        )
        for preceding in cases:
            result = parse_usi_observation(
                [
                    preceding,
                    "info depth 10 score cp 20 nodes 1000 pv 7g7f",
                    "bestmove 7g7f",
                ],
                "black",
                requested_nodes=1000,
                max_reported_nodes=1000,
            )
            self.assertEqual(result["status"], "protocol_failure", preceding)
            self.assertEqual(result["failure_reason"], "malformed_score", preceding)
            expected_max = 2000 if "2000" in preceding else 1000
            self.assertEqual(result["max_reported_nodes_evidence"], expected_max, preceding)

    def test_truncated_score_subcommands_fail_closed_even_with_a_later_score(self):
        cases = (
            "info score",
            "info score cp",
            "info score mate",
            "info score cp lowerbound",
            "info score mate pv nodes 2000",
            "info score cp nodes 2000",
        )
        for preceding in cases:
            result = parse_usi_observation(
                [
                    preceding,
                    "info depth 10 score cp 20 nodes 1000 pv 7g7f",
                    "bestmove 7g7f",
                ],
                "black",
                requested_nodes=1000,
                max_reported_nodes=1000,
            )
            self.assertEqual(result["status"], "protocol_failure", preceding)
            self.assertEqual(result["failure_reason"], "malformed_score", preceding)

    def test_valid_score_payload_boundary_remains_compatible(self):
        for payload in ("pv 7g7f nodes 2000", "string diagnostic nodes 2000"):
            result = parse_usi_observation(
                [
                    f"info score cp 20 {payload}",
                    "info depth 10 score cp 30 nodes 1000 pv 7g7f",
                    "bestmove 7g7f",
                ],
                "black",
                requested_nodes=1000,
                max_reported_nodes=1000,
            )
            self.assertEqual(result["status"], "exact_cp", payload)
            self.assertEqual(result["score_cp_sente"], 30, payload)
            self.assertEqual(result["max_reported_nodes_evidence"], 1000, payload)

    def test_unknown_info_noise_without_nodes_does_not_invent_evidence(self):
        result = parse_usi_observation(
            [
                "info vendor_noise payload",
                "info depth 10 score cp 20 nodes 1000 pv 7g7f",
                "bestmove 7g7f",
            ],
            "black",
            requested_nodes=1000,
            max_reported_nodes=1000,
        )
        self.assertEqual(result["status"], "exact_cp")
        self.assertEqual(result["node_evidence_count"], 1)
        self.assertEqual(result["max_reported_nodes_evidence"], 1000)

    def test_nodes_after_known_payload_boundary_are_not_scanned(self):
        result = parse_usi_observation(
            [
                "info vendor_noise payload pv 7g7f nodes 2000",
                "info depth 10 score cp 20 nodes 1000 pv 7g7f",
                "bestmove 7g7f",
            ],
            "black",
            requested_nodes=1000,
            max_reported_nodes=1000,
        )
        self.assertEqual(result["status"], "exact_cp")
        self.assertEqual(result["node_evidence_count"], 1)
        self.assertEqual(result["max_reported_nodes_evidence"], 1000)

    def test_nodes_in_string_and_pv_payload_are_not_structured_evidence(self):
        result = parse_usi_observation([
            "info string score cp 99 nodes 3000 pv 3c3d",
            "info depth 10 score cp 20 nodes 1000 pv 7g7f nodes 4000",
            "bestmove 7g7f",
        ], "black", requested_nodes=1000, max_reported_nodes=1000)
        self.assertEqual(result["status"], "exact_cp")
        self.assertEqual(result["node_evidence_count"], 1)
        self.assertEqual(result["max_reported_nodes_evidence"], 1000)

    def test_last_and_max_node_evidence_are_distinct_for_valid_values(self):
        result = parse_usi_observation([
            "info depth 9 nodes 400",
            "info depth 10 score cp 20 nodes 1000 pv 7g7f",
            "info depth 11 nodes 700",
            "bestmove 7g7f",
        ], "black", requested_nodes=1000, max_reported_nodes=1000)
        self.assertEqual(result["status"], "exact_cp")
        self.assertEqual(result["reported_nodes_at_score"], 1000)
        self.assertEqual(result["last_reported_nodes"], 700)
        self.assertEqual(result["max_reported_nodes_evidence"], 1000)
        self.assertEqual(result["node_evidence_count"], 3)

    def test_duplicate_nodes_on_one_line_fail_closed(self):
        result = parse_usi_observation([
            "info depth 10 score cp 20 nodes 1000 nodes 1001 pv 7g7f",
            "bestmove 7g7f",
        ], "black", requested_nodes=1000, max_reported_nodes=1001)
        self.assertEqual(result["status"], "node_budget_failure")
        self.assertEqual(result["failure_reason"], "ambiguous_node_evidence")
        self.assertEqual(result["node_evidence_count"], 2)
        self.assertEqual(result["max_reported_nodes_evidence"], 1001)

    def test_zero_in_earlier_current_go_evidence_is_not_hidden_by_later_positive(self):
        result = parse_usi_observation([
            "info depth 9 nodes 0",
            "info depth 10 score cp 20 nodes 1000 pv 7g7f",
            "bestmove 7g7f",
        ], "black", requested_nodes=1000)
        self.assertEqual(result["status"], "node_budget_failure")
        self.assertEqual(result["failure_reason"], "zero_nodes_without_verified_exception")
        self.assertEqual(result["max_reported_nodes_evidence"], 1000)

    def test_mate_lexical_forms_are_retained(self):
        for raw, sign, winner in (("+3", "+", "black"), ("-3", "-", "white"), ("-0", "-", "white"), ("0", "", None)):
            result = parse_usi_observation(
                [f"info depth 10 score mate {raw} nodes 1000 pv 7g7f", "bestmove 7g7f"],
                "black",
                requested_nodes=1000,
            )
            self.assertEqual(result["status"], "mate")
            self.assertEqual(result["score_mate_stm"], raw)
            self.assertEqual(result["mate_sign"], sign)
            self.assertEqual(result["winner_sente"], winner)

    def test_special_no_score_and_pv_mismatch_remain_distinct(self):
        result = parse_usi_observation(["bestmove resign"], "black")
        self.assertEqual((result["status"], result["bestmove_kind"]), ("no_score", "resign"))
        result = parse_usi_observation(["bestmove 0000"], "black")
        self.assertEqual(result["bestmove_kind"], "other_special")
        result = parse_usi_observation([
            "info depth 10 score cp 12 nodes 1000 pv 7g7f", "bestmove 3c3d"
        ], "black", requested_nodes=1000)
        self.assertEqual(result["status"], "protocol_failure")
        self.assertEqual(result["failure_reason"], "pv_bestmove_mismatch")

    def test_forced_single_sekirei_zero_nodes_requires_the_sole_move(self):
        classification = {
            "type": "forced_single_legal_move",
            "in_check": True,
            "legal_move_count": 1,
            "sole_legal_move": "7g7f",
        }
        result = parse_usi_observation(
            ["info score cp 20 nodes 0 pv 7g7f", "bestmove 7g7f"],
            "black",
            requested_nodes=1000,
            engine_id="sekirei",
            position_classification=classification,
        )
        self.assertEqual(result["status"], "exact_cp")
        self.assertEqual(result["position_type"], "forced_single_legal_move")
        result = parse_usi_observation(
            ["info score cp 20 nodes 0 pv 7g7f", "bestmove 7g7f"],
            "black",
            requested_nodes=1000,
            engine_id="sekirei",
            position_classification={**classification, "sole_legal_move": "3c3d"},
        )
        self.assertEqual(result["status"], "node_budget_failure")
        self.assertEqual(result["failure_reason"], "zero_nodes_without_verified_exception")

    def test_zero_nodes_is_not_accepted_for_an_unexplained_normal_score(self):
        result = parse_usi_observation(
            ["info score cp 20 nodes 0 pv 7g7f", "bestmove 7g7f"],
            "black",
            requested_nodes=1000,
            engine_id="sekirei",
        )
        self.assertEqual(result["status"], "node_budget_failure")
        self.assertEqual(result["failure_reason"], "zero_nodes_without_verified_exception")

    def test_mate_in_one_zero_nodes_require_the_frozen_sekirei_contract(self):
        classification = {
            "type": "mate_in_one_available",
            "in_check": False,
            "legal_move_count": 217,
            "mating_moves": ["2g4g"],
            "mating_moves_sha256": json_sha256(["2g4g"]),
        }
        valid = parse_usi_observation(
            ["info score mate 1 nodes 0 pv 2g4g", "bestmove 2g4g"],
            "black",
            requested_nodes=1000,
            engine_id="sekirei",
            position_classification=classification,
        )
        self.assertEqual(valid["status"], "mate")
        self.assertEqual(valid["max_reported_nodes_evidence"], 0)
        wrong_side = parse_usi_observation(
            ["info score mate 1 nodes 0 pv 2g4g", "bestmove 2g4g"],
            "white",
            requested_nodes=1000,
            engine_id="sekirei",
            position_classification=classification,
        )
        self.assertEqual(wrong_side["status"], "node_budget_failure")
        rejected = (
            ("info score mate +1 nodes 0 pv 2g4g", "2g4g"),
            ("info score mate -2 nodes 0 pv 2g4g", "2g4g"),
            ("info score mate 1 upperbound nodes 0 pv 2g4g", "2g4g"),
            ("info score cp 20 nodes 0 pv 2g4g", "2g4g"),
            ("info score mate 1 nodes 0 pv 2g4g 3c3d", "2g4g"),
            ("info score mate 1 nodes 0 pv 3c3d", "3c3d"),
        )
        for line, bestmove in rejected:
            result = parse_usi_observation(
                [line, f"bestmove {bestmove}"],
                "black",
                requested_nodes=1000,
                engine_id="sekirei",
                position_classification=classification,
            )
            self.assertEqual(result["status"], "node_budget_failure", line)
        resign = parse_usi_observation(
            ["info score mate 1 nodes 0 pv resign", "bestmove resign"],
            "black",
            requested_nodes=1000,
            engine_id="sekirei",
            position_classification=classification,
        )
        self.assertEqual(resign["status"], "no_score")
        for engine_id in ("teacher", None):
            kwargs = {"engine_id": engine_id} if engine_id else {}
            result = parse_usi_observation(
                ["info score mate 1 nodes 0 pv 2g4g", "bestmove 2g4g"],
                "black",
                requested_nodes=1000,
                position_classification=classification,
                **kwargs,
            )
            self.assertEqual(result["status"], "node_budget_failure", engine_id)
        with self.assertRaisesRegex(ValueError, "move list hash"):
            parse_usi_observation(
                ["info score mate 1 nodes 0 pv 2g4g", "bestmove 2g4g"],
                "black",
                requested_nodes=1000,
                engine_id="sekirei",
                position_classification={**classification, "mating_moves_sha256": "0" * 64},
            )
        mixed = parse_usi_observation(
            [
                "info score mate 1 nodes 0 pv 2g4g",
                "info nodes 1",
                "bestmove 2g4g",
            ],
            "black",
            requested_nodes=1000,
            engine_id="sekirei",
            position_classification=classification,
        )
        self.assertEqual(mixed["status"], "node_budget_failure")
        missing = parse_usi_observation(
            [
                "info depth 9 score mate 1 pv 2g4g",
                "info depth 10 score mate 1 nodes 0 pv 2g4g",
                "bestmove 2g4g",
            ],
            "black",
            requested_nodes=1000,
            engine_id="sekirei",
            position_classification=classification,
        )
        self.assertEqual(missing["status"], "node_budget_failure")
        for extra in ({"side_to_move": "white"}, {"occurrence_id": "development-04:77"}):
            result = parse_usi_observation(
                ["info score mate 1 nodes 0 pv 2g4g", "bestmove 2g4g"],
                "black",
                requested_nodes=1000,
                engine_id="sekirei",
                position_classification={**classification, **extra},
            )
            self.assertEqual(result["status"], "node_budget_failure", extra)

    def test_policy_boundary_is_inclusive_and_earlier_overrun_is_retained(self):
        limit = node_reporting_limit(1_000_000)
        for value, expected in ((limit, "exact_cp"), (limit + 1, "node_budget_failure")):
            result = parse_usi_observation(
                [f"info score cp 20 nodes {value} pv 7g7f", "bestmove 7g7f"],
                "black",
                requested_nodes=1_000_000,
                max_reported_nodes=limit,
            )
            self.assertEqual(result["status"], expected, value)
        result = parse_usi_observation(
            [
                f"info depth 9 score cp 10 nodes {limit + 1} pv 7g7f",
                f"info depth 10 score cp 20 nodes {limit} pv 7g7f",
                "bestmove 7g7f",
            ],
            "black",
            requested_nodes=1_000_000,
            max_reported_nodes=limit,
        )
        self.assertEqual(result["status"], "node_budget_failure")
        self.assertEqual(result["max_reported_nodes_evidence"], limit + 1)

    def test_missing_negative_and_malformed_nodes_are_technical_failures(self):
        for line, reason in (
            ("info score cp 20 pv 7g7f", "missing_node_evidence"),
            ("info score cp 20 nodes pv 7g7f", "missing_node_evidence"),
            ("info score cp 20 nodes -1 pv 7g7f", "negative_node_evidence"),
            ("info score cp 20 nodes invalid pv 7g7f", "malformed_node_evidence"),
        ):
            result = parse_usi_observation(
                [line, "bestmove 7g7f"],
                "black",
                requested_nodes=1000,
                engine_id="teacher",
            )
            self.assertEqual(result["status"], "node_budget_failure", line)
            self.assertEqual(result["failure_reason"], reason, line)

    def test_zero_bound_is_only_accepted_for_the_forced_sekirei_exception(self):
        classification = {
            "type": "forced_single_legal_move",
            "in_check": True,
            "legal_move_count": 1,
            "sole_legal_move": "7g7f",
        }
        result = parse_usi_observation(
            ["info score cp 20 upperbound nodes 0 pv 7g7f", "bestmove 7g7f"],
            "black",
            requested_nodes=1000,
            engine_id="sekirei",
            position_classification=classification,
        )
        self.assertEqual(result["status"], "bound_cp")
        result = parse_usi_observation(
            ["info score cp 20 upperbound nodes 0 pv 7g7f", "bestmove 7g7f"],
            "black",
            requested_nodes=1000,
            engine_id="teacher",
        )
        self.assertEqual(result["status"], "node_budget_failure")

    def test_terminal_teacher_mate_resign_is_typed_and_strict(self):
        classification = {
            "type": "terminal_checkmate",
            "in_check": True,
            "legal_move_count": 0,
        }
        result = parse_usi_observation(
            ["info score mate -1 nodes 0 pv resign", "bestmove resign"],
            "white",
            requested_nodes=1000,
            engine_id="teacher",
            position_classification=classification,
        )
        self.assertEqual(result["status"], "mate")
        self.assertEqual(result["score_mate_stm"], "-1")
        self.assertEqual(result["position_type"], "terminal_checkmate")
        for lines, kwargs, expected in (
            (["bestmove resign"], {}, "no_score"),
            (["info score mate +1 nodes 0 pv resign", "bestmove resign"], {"engine_id": "teacher", "position_classification": classification}, "no_score"),
            (["info score mate -2 nodes 0 pv resign", "bestmove resign"], {"engine_id": "teacher", "position_classification": classification}, "no_score"),
            (["info score mate -1 nodes 0 pv resign", "bestmove resign"], {"engine_id": "teacher", "position_classification": classification, "lifecycle_valid": False}, "no_score"),
            (["info score cp 20 nodes 1000 pv 7g7f", "bestmove resign"], {"engine_id": "teacher", "position_type": "normal"}, "no_score"),
        ):
            value = parse_usi_observation(lines, "white", requested_nodes=1000, **kwargs)
            self.assertEqual(value["status"], expected, lines)

    def test_node_policy_limit_applies_to_positive_evidence(self):
        result = parse_usi_observation(
            ["info score cp 20 nodes 1001 pv 7g7f", "bestmove 7g7f"],
            "black",
            requested_nodes=1000,
            max_reported_nodes=1000,
        )
        self.assertEqual(result["status"], "node_budget_failure")
        self.assertEqual(result["failure_reason"], "node_count_above_policy_limit")

    def test_protocol_pv_mismatch_is_not_rescued_by_zero_node_exception(self):
        classification = {
            "type": "forced_single_legal_move",
            "in_check": True,
            "legal_move_count": 1,
            "sole_legal_move": "7g7f",
        }
        result = parse_usi_observation(
            ["info score cp 20 nodes 0 pv 3c3d", "bestmove 7g7f"],
            "black",
            requested_nodes=1000,
            engine_id="sekirei",
            position_classification=classification,
        )
        self.assertEqual(result["status"], "protocol_failure")


@contextmanager
def fake_mode(mode, child_pid_file=None):
    values = {"FAKE_USI_MODE": mode}
    if child_pid_file:
        values["FAKE_USI_CHILD_PID_FILE"] = str(child_pid_file)
    with mock.patch.dict(os.environ, values, clear=False):
        yield


class USIProcessTests(unittest.TestCase):
    def run_fake(self, mode, *, timeout=2, ceiling=0.02, child_pid_file=None):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "raw.jsonl"
            with fake_mode(mode, child_pid_file=child_pid_file):
                result = run_engine_attempt(
                    FAKE,
                    "position startpos moves 7g7f",
                    "white",
                    {"Threads": "1"},
                    cwd=temp,
                    requested_nodes=1000,
                    timeout_seconds=timeout,
                    anomaly_ceiling=ceiling,
                    raw_log_path=log,
                )
            return result, log.read_text()

    def test_fake_engine_success_bound_multipv_and_atomic_raw_log(self):
        for mode, expected in (("exact", "exact_cp"), ("bound", "bound_cp"), ("multipv", "exact_cp")):
            result, log_text = self.run_fake(mode)
            self.assertEqual(result["status"], expected)
            self.assertTrue(log_text)
            events = [json.loads(line) for line in log_text.splitlines()]
            self.assertGreater(len(events), 0)
            self.assertEqual(sorted(event["offset_ns"] for event in events), [event["offset_ns"] for event in events])
            self.assertTrue(all("direction" in event and "line" in event for event in events))

    def test_fake_engine_failure_categories_and_node_anomaly(self):
        for mode, expected in (("mismatch", "protocol_failure"), ("unknown-option", "config_failure"), ("startup-error", "startup_failure"), ("exit", "exit_failure"), ("node-anomaly", "node_budget_failure"), ("no-score", "no_score")):
            result, _ = self.run_fake(mode)
            self.assertEqual(result["status"], expected, mode)
            if mode == "exit":
                self.assertEqual(result["engine_returncode"], 7)
                self.assertEqual(result["supervisor_status"], "ok")

    def test_timeout_kills_process_group_and_does_not_salvage(self):
        with tempfile.TemporaryDirectory() as temp:
            child_file = Path(temp) / "child.pid"
            log = Path(temp) / "raw.jsonl"
            with fake_mode("timeout", child_pid_file=child_file):
                result = run_engine_attempt(
                    FAKE,
                    "position startpos moves 7g7f",
                    "white",
                    {"Threads": "1"},
                    cwd=temp,
                    requested_nodes=1000,
                    timeout_seconds=0.3,
                    raw_log_path=log,
                )
            self.assertEqual(result["status"], "timeout")
            self.assertTrue(log.is_file())
            if child_file.exists():
                child_pid = int(child_file.read_text())
                for _ in range(30):
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                else:
                    # A killed child can briefly remain a zombie until init
                    # reaps it; it is no longer an executable process.
                    with self.assertRaises(OSError):
                        os.kill(child_pid, signal.SIGKILL)

    def test_parent_exit_still_cleans_a_surviving_child_group(self):
        for mode in ("parent-exit-child", "timeout-parent-exit"):
            with tempfile.TemporaryDirectory() as temp:
                child_file = Path(temp) / "child.pid"
                log = Path(temp) / "raw.jsonl"
                with fake_mode(mode, child_pid_file=child_file):
                    result = run_engine_attempt(
                        FAKE,
                        "position startpos moves 7g7f",
                        "white",
                        {"Threads": "1"},
                        cwd=temp,
                        requested_nodes=1000,
                        timeout_seconds=0.3,
                        raw_log_path=log,
                    )
                self.assertEqual(result["status"], "timeout")
                self.assertTrue(log.is_file())
                child_pid = int(child_file.read_text())
                with self.assertRaises(OSError):
                    os.kill(child_pid, signal.SIGKILL)

    def test_immediate_parent_exit_child_is_cleaned_promptly_repeatedly(self):
        with tempfile.TemporaryDirectory() as temp:
            for index in range(8):
                child_file = Path(temp) / f"immediate-{index}.pid"
                log = Path(temp) / f"immediate-{index}.jsonl"
                started = time.monotonic()
                with fake_mode("immediate-parent-exit-child", child_pid_file=child_file):
                    result = run_engine_attempt(
                        FAKE,
                        "position startpos moves 7g7f",
                        "white",
                        {"Threads": "1"},
                        cwd=temp,
                        requested_nodes=1000,
                        timeout_seconds=0.3,
                        raw_log_path=log,
                    )
                elapsed = time.monotonic() - started
                self.assertEqual(result["status"], "timeout")
                self.assertLess(elapsed, 1.5)
                child_pid = int(child_file.read_text())
                for _ in range(50):
                    stat = USIProcess._read_proc_stat(child_pid)
                    if stat is None or stat["state"] == "Z":
                        break
                    time.sleep(0.01)
                else:
                    self.fail(f"immediate descendant remains alive: {child_pid}")

    def test_immediate_setsid_parent_exit_child_is_reaped_without_touching_sentinel(self):
        """The subreaper must catch a child that leaves the engine's session."""
        sentinel = subprocess.Popen(["sleep", "60"])
        try:
            with tempfile.TemporaryDirectory() as temp:
                for index in range(8):
                    child_file = Path(temp) / f"immediate-setsid-{index}.pid"
                    log = Path(temp) / f"immediate-setsid-{index}.jsonl"
                    started = time.monotonic()
                    with fake_mode("immediate-setsid-parent-exit-child", child_pid_file=child_file):
                        result = run_engine_attempt(
                            FAKE,
                            "position startpos moves 7g7f",
                            "white",
                            {"Threads": "1"},
                            cwd=temp,
                            requested_nodes=1000,
                            timeout_seconds=0.3,
                            raw_log_path=log,
                        )
                    elapsed = time.monotonic() - started
                    self.assertLess(elapsed, 2.0)
                    child_pid = int(child_file.read_text())
                    if result["supervisor_status"] == "ok":
                        self.assertEqual(result["cleanup_status"], "ok")
                        for _ in range(50):
                            stat = USIProcess._read_proc_stat(child_pid)
                            if stat is None:
                                break
                            time.sleep(0.01)
                        else:
                            self.fail(f"setsid descendant remains alive: {child_pid}")
                    else:
                        # Unsupported enumeration/pidfd capabilities must be
                        # visible as cleanup failure rather than success.
                        self.assertNotEqual(result["cleanup_status"], "ok")
                    self.assertIsNone(sentinel.poll())
        finally:
            if sentinel.poll() is None:
                sentinel.terminate()
            sentinel.wait(timeout=2)

    def test_unknown_reused_group_is_not_signalled(self):
        process = USIProcess.__new__(USIProcess)
        process.leader_pid = 123
        process.leader_starttime = 10
        process.pgid = 44
        process.session_id = 55
        process._proc_identity_available = True
        process._ownership_ambiguous = False
        process._known_group_identities = {123: 10}
        process._identity_lock = mock.MagicMock()
        process._refresh_group_members = mock.Mock(return_value=[
            (999, {"state": "R", "starttime": 88, "pgrp": 44, "session": 55}),
        ])
        process.process = mock.Mock()
        with mock.patch.object(USIProcess, "_read_proc_stat", return_value=None):
            with mock.patch("benchmark.os.killpg") as killpg:
                with self.assertRaises(CleanupError):
                    process.terminate_group()
                killpg.assert_not_called()

    def test_popen_startup_failure_has_durable_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "missing.jsonl"
            result = run_engine_attempt(
                Path(temp) / "does-not-exist",
                "position startpos moves 7g7f",
                "white",
                {},
                cwd=temp,
                requested_nodes=1000,
                timeout_seconds=1,
                raw_log_path=log,
            )
            self.assertEqual(result["status"], "startup_failure")
            self.assertTrue(log.is_file())
            events = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(events[-1]["event"], "runner")
            self.assertEqual(events[-1]["status"], "startup_failure")
            self.assertEqual(result["raw_log_sha256"], sha256(log))

    def test_cleanup_failure_preserves_original_outcome_and_aborts_safely(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "cleanup.jsonl"
            with fake_mode("exact"):
                with mock.patch.object(USIProcess, "_owned_group_members", return_value=None):
                    result = run_engine_attempt(
                        FAKE,
                        "position startpos moves 7g7f",
                        "white",
                        {"Threads": "1"},
                        cwd=temp,
                        requested_nodes=1000,
                        timeout_seconds=2,
                        raw_log_path=log,
                    )
            self.assertEqual(result["status"], "cleanup_failure")
            self.assertEqual(result["original_status"], "exact_cp")
            self.assertEqual(result["cleanup_status"], "failure")
            event = json.loads(log.read_text().splitlines()[-1])
            self.assertEqual(event["status"], "cleanup_failure")
            self.assertEqual(event["original_status"], "exact_cp")

    def test_run_resume_skips_recorded_failures_and_rejects_fingerprint_mismatch(self):
        config = copy.deepcopy(load_config())
        # A temporary prepared runtime is enough to exercise the immutable
        # run protocol; its fake output is intentionally illegal at most
        # benchmark points and therefore records protocol failures.
        config["engines"]["teacher"]["binary"] = "fake-usi.py"
        config["engines"]["teacher"]["weight_required"] = False
        config["engines"]["sekirei"]["binary"] = "fake-usi.py"
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            binary = runtime / "bin/fake-usi.py"
            binary.parent.mkdir(parents=True)
            shutil.copy2(FAKE, binary)
            binary.chmod(0o755)
            atomic_write_json(runtime / "build-manifest.json", {
                "lock_sha256": sha256(Path(__file__).resolve().parents[1] / "config/toolchain.lock.json"),
                "binaries": {"fake-usi.py": {"path": str(binary), "sha256": sha256(binary)}},
            })
            with mock.patch.dict(os.environ, {"FAKE_USI_MODE": "exact"}, clear=False):
                with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                    first = execute_run(runtime, config, "pilot", run_id="resume-fixture")
            self.assertEqual(first["status"], "complete")
            attempt_files = sorted((runtime / "runs/resume-fixture/attempts").glob("*.json"))
            self.assertEqual(len(attempt_files), 102)
            self.assertEqual(list((runtime / "runs/resume-fixture").rglob("*.tmp")), [])
            target = attempt_files[0]
            original_record = target.read_text()
            original_record_value = json.loads(original_record)
            corrupted_record = json.loads(original_record)
            corrupted_record["position"] += " "
            atomic_write_json(target, corrupted_record)
            with self.assertRaisesRegex(BenchmarkError, "position hash"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            target.write_text(original_record)
            log_path = runtime / "runs/resume-fixture/logs" / json.loads(original_record)["raw_log"]
            original_log = log_path.read_text()

            for field in (
                "schema_version",
                "reported_nodes_at_score",
                "last_reported_nodes",
                "max_reported_nodes_evidence",
                "node_evidence_count",
                "node_evidence_valid_count",
                "node_evidence_invalid_count",
                "node_evidence_positive_count",
                "node_evidence_zero_count",
                "node_evidence_errors",
                "position_type",
                "position_classification",
            ):
                missing_field = copy.deepcopy(original_record_value)
                self.assertIn(field, missing_field["result"])
                del missing_field["result"][field]
                atomic_write_json(target, missing_field)
                with self.assertRaises(BenchmarkError):
                    execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
                if field == "max_reported_nodes_evidence":
                    with self.assertRaisesRegex(ValueError, "raw observation schema"):
                        report_from_run(runtime / "runs/resume-fixture")
                target.write_text(original_record)

            def rewrite_raw(mutator, *, record=None):
                values = [json.loads(line) for line in original_log.splitlines()]
                mutator(values)
                atomic_write(
                    log_path,
                    "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
                )
                updated = copy.deepcopy(original_record_value if record is None else record)
                updated["raw_log_sha256"] = sha256(log_path)
                updated["result"]["raw_log_sha256"] = updated["raw_log_sha256"]
                atomic_write_json(target, updated)
                return updated

            log_path.write_text(original_log + "corruption\n")
            with self.assertRaisesRegex(BenchmarkError, "raw log hash"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            log_path.write_text(original_log)
            rewrite_raw(
                lambda values: next(
                    value.__setitem__("line", value["line"] + " ")
                    for value in values
                    if value.get("direction") == "send" and value["line"].startswith("position ")
                )
            )
            with self.assertRaisesRegex(BenchmarkError, "send lifecycle"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            target.write_text(original_record)
            log_path.write_text(original_log)
            rewrite_raw(
                lambda values: next(
                    value.__setitem__("line", "go nodes 999")
                    for value in values
                    if value.get("direction") == "send" and value["line"].startswith("go nodes ")
                )
            )
            with self.assertRaisesRegex(BenchmarkError, "send lifecycle"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            target.write_text(original_record)
            log_path.write_text(original_log)
            rewrite_raw(
                lambda values: next(
                    value.__setitem__("line", value["line"].replace("value 1", "value 2"))
                    for value in values
                    if value.get("direction") == "send" and value["line"].startswith("setoption ")
                )
            )
            with self.assertRaisesRegex(BenchmarkError, "send lifecycle"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            target.write_text(original_record)
            log_path.write_text(original_log)
            def move_setoption_before_usiok(values):
                usiok_index = next(
                    index
                    for index, value in enumerate(values)
                    if value.get("direction") == "receive" and value.get("line") == "usiok"
                )
                setoption_index = next(
                    index
                    for index, value in enumerate(values)
                    if value.get("direction") == "send" and value.get("line", "").startswith("setoption ")
                )
                self.assertGreater(setoption_index, usiok_index)
                usiok_event = copy.deepcopy(values[usiok_index])
                setoption_event = copy.deepcopy(values[setoption_index])
                usiok_offset = usiok_event["offset_ns"]
                setoption_offset = setoption_event["offset_ns"]
                values[usiok_index] = setoption_event
                values[setoption_index] = usiok_event
                # Keep the original offset multiset and monotonic transcript
                # timing while changing only lifecycle order.
                values[usiok_index]["offset_ns"] = usiok_offset
                values[setoption_index]["offset_ns"] = setoption_offset

            rewrite_raw(move_setoption_before_usiok)
            with self.assertRaisesRegex(BenchmarkError, "setoption.*usiok"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            target.write_text(original_record)
            log_path.write_text(original_log)
            binary_tampered = copy.deepcopy(original_record_value)
            binary_tampered["result"]["binary_identity"]["sha256"] = "0" * 64
            atomic_write_json(target, binary_tampered)
            with self.assertRaisesRegex(BenchmarkError, "binary identity"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            target.write_text(original_record)
            runner_tampered = copy.deepcopy(original_record_value)
            rewrite_raw(
                lambda values: next(
                    value.__setitem__("failure_reason", "tampered")
                    for value in values
                    if value.get("event") == "runner"
                ),
                record=runner_tampered,
            )
            with self.assertRaisesRegex(BenchmarkError, "outcome"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            target.write_text(original_record)
            log_path.write_text(original_log)
            corrupted_record = json.loads(original_record)
            corrupted_record["result"]["score_cp_sente"] = corrupted_record["result"].get("score_cp_sente", 0) + 1
            atomic_write_json(target, corrupted_record)
            with self.assertRaisesRegex(BenchmarkError, "raw observation score_cp_sente"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            target.write_text(original_record)
            corrupted_record = json.loads(original_record)
            corrupted_record["result"]["status"] = "no_score"
            atomic_write_json(target, corrupted_record)
            with self.assertRaisesRegex(BenchmarkError, "status mismatch"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            target.write_text(original_record)
            original_events = [json.loads(line) for line in original_log.splitlines()]
            go_index = next(
                index
                for index, value in enumerate(original_events)
                if value.get("direction") == "send" and value["line"].startswith("go nodes ")
            )
            late_events = original_events[:go_index + 1]
            late_offset = late_events[-1]["offset_ns"] + 1
            late_events.extend(
                (
                    {"direction": "receive", "line": "info depth 99 score cp 777 nodes 1000 pv 7g7f", "offset_ns": late_offset},
                    {"direction": "receive", "line": "bestmove 7g7f", "offset_ns": late_offset + 1},
                )
            )
            timeout_record = copy.deepcopy(original_record_value)
            timeout_result = timeout_record["result"]
            for field in (
                "bestmove", "bestmove_kind", "pv", "pv_head", "score_kind", "score_cp_stm",
                "score_cp_sente", "reported_cp_stm", "reported_cp_sente", "score_bound_stm",
                "score_bound_sente", "score_mate_stm", "mate_distance", "mate_distance_known",
                "mate_sign", "winner", "winner_stm", "winner_sente", "reported_nodes_at_score",
                "last_reported_nodes", "max_reported_nodes_evidence", "node_evidence_count",
                "node_evidence_valid_count", "node_evidence_invalid_count", "node_evidence_positive_count",
                "node_evidence_zero_count", "node_evidence_errors", "engine_time_ms", "engine_time",
                "wall_go_to_bestmove_ns",
                "wall_go_to_bestmove_seconds", "raw_score_line", "raw_bestmove_line", "info_count",
            ):
                timeout_result[field] = None
            timeout_result.update(
                status="timeout",
                failure_reason="deadline",
                outcome_phase="go",
                completed_before_deadline=False,
                cleanup_status="ok",
                cleanup_failure=None,
                original_status=None,
                original_failure_reason=None,
                original_outcome_phase=None,
            )
            timeout_record["status"] = "timeout"
            timeout_record["outcome"] = {field: timeout_result.get(field) for field in OUTCOME_FIELDS}
            runner = copy.deepcopy(original_events[-1])
            runner.update(timeout_record["outcome"])
            runner["offset_ns"] = late_offset + 2
            late_events.append(runner)
            atomic_write(
                log_path,
                "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in late_events),
            )
            timeout_record["raw_log_sha256"] = sha256(log_path)
            timeout_record["result"]["raw_log_sha256"] = timeout_record["raw_log_sha256"]
            atomic_write_json(target, timeout_record)
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                resumed_late = execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            self.assertEqual(resumed_late["status"], "complete")
            target.write_text(original_record)
            log_path.write_text(original_log)
            extra = target.parent / "unexpected.json"
            atomic_write_json(extra, {})
            with self.assertRaisesRegex(BenchmarkError, "extra"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            extra.unlink()
            other = attempt_files[1]
            original_other = other.read_bytes()
            other.write_bytes(Path(target).read_bytes())
            with self.assertRaisesRegex(BenchmarkError, "identity/position"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            other.write_bytes(original_other)
            missing_log = runtime / "runs/resume-fixture/logs" / json.loads(original_record)["raw_log"]
            missing_log.unlink()
            target.unlink()
            with self.assertRaisesRegex(BenchmarkError, "attempt set mismatch"):
                execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            missing_log.write_text(original_log)
            target.write_text(original_record)
            before = {path.name: path.read_bytes() for path in attempt_files}
            with mock.patch.dict(os.environ, {"FAKE_USI_MODE": "timeout"}, clear=False):
                with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                    resumed = execute_run(runtime, config, "pilot", run_id="resume-fixture", resume=True)
            self.assertEqual(resumed["status"], "complete")
            after = {path.name: path.read_bytes() for path in attempt_files}
            self.assertEqual(before, after)
            report = report_from_run(runtime / "runs/resume-fixture", Path(temp) / "report")
            self.assertEqual(report["universe"]["total_occurrences"], 17)
            self.assertEqual(report["repeatability"]["attempt_count"], 102)
            self.assertTrue(report["validity"]["evidence_validator_passed"])
            self.assertTrue((Path(temp) / "report/evaluation.svg").is_file())
            changed = copy.deepcopy(config)
            changed["timeout_seconds"] = config["timeout_seconds"] + 1
            with self.assertRaisesRegex(BenchmarkError, "fingerprint mismatch"):
                execute_run(runtime, changed, "pilot", run_id="resume-fixture", resume=True)


    def test_malformed_score_artifact_is_revalidated_and_rejected_by_gate(self):
        config = copy.deepcopy(load_config())
        config["requested_nodes"] = 1000
        config["pilot"]["max_reported_nodes"] = node_reporting_limit(config["requested_nodes"])
        config["engines"]["teacher"]["binary"] = "fake-usi.py"
        config["engines"]["teacher"]["weight_required"] = False
        config["engines"]["sekirei"]["binary"] = "fake-usi.py"
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            binary = runtime / "bin/fake-usi.py"
            binary.parent.mkdir(parents=True)
            shutil.copy2(FAKE, binary)
            binary.chmod(0o755)
            atomic_write_json(runtime / "build-manifest.json", {
                "lock_sha256": sha256(Path(__file__).resolve().parents[1] / "config/toolchain.lock.json"),
                "binaries": {"fake-usi.py": {"path": str(binary), "sha256": sha256(binary)}},
            })
            with mock.patch.dict(os.environ, {"FAKE_USI_MODE": "malformed-score"}, clear=False):
                with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                    pilot = execute_run(runtime, config, "pilot", run_id="malformed-score-fixture")
            records = [
                json.loads(path.read_text())
                for path in (runtime / "runs/malformed-score-fixture/attempts").glob("*.json")
            ]
            self.assertEqual(len(records), 102)
            self.assertTrue(all(record["status"] == "protocol_failure" for record in records))
            self.assertTrue(all(record["result"]["failure_reason"] == "malformed_score" for record in records))
            self.assertTrue(all(record["result"]["max_reported_nodes_evidence"] == 2000 for record in records))
            with mock.patch("benchmark_report.load_config", return_value=config):
                report = report_from_run(runtime / "runs/malformed-score-fixture")
            self.assertEqual(report["validity"]["technical_failure_count"], 102)
            self.assertFalse(report["validity"]["complete_evidence_valid"])
            config["formal"]["max_reported_nodes"] = node_reporting_limit(config["requested_nodes"])
            config["formal"]["pilot_evidence"] = {
                "pilot_run_id": "malformed-score-fixture",
                "pilot_fingerprint": pilot["fingerprint"],
                "observed_max_reported_nodes": 2000,
                "max_reported_nodes": node_reporting_limit(config["requested_nodes"]),
            }
            with self.assertRaisesRegex(ConfigurationError, "technical failures|node gate"):
                validate_formal_gate(runtime, config)


    def test_formal_run_is_refused_until_pilot_policy_evidence_is_frozen(self):
        from benchmark import execute_run

        config = copy.deepcopy(load_config())
        config["formal"]["pilot_evidence"] = None
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ConfigurationError, "node policy"):
                execute_run(temp, config, "formal", run_id="formal-before-pilot")
            self.assertFalse((Path(temp) / "runs/formal-before-pilot").exists())

    def test_formal_gate_requires_matching_complete_pilot_evidence(self):
        config = copy.deepcopy(load_config())
        config["requested_nodes"] = 1000
        config["pilot"]["max_reported_nodes"] = node_reporting_limit(config["requested_nodes"])
        config["engines"]["teacher"]["binary"] = "fake-usi.py"
        config["engines"]["teacher"]["weight_required"] = False
        config["engines"]["sekirei"]["binary"] = "fake-usi.py"
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            binary = runtime / "bin/fake-usi.py"
            binary.parent.mkdir(parents=True)
            shutil.copy2(FAKE, binary)
            binary.chmod(0o755)
            atomic_write_json(runtime / "build-manifest.json", {
                "lock_sha256": sha256(Path(__file__).resolve().parents[1] / "config/toolchain.lock.json"),
                "binaries": {"fake-usi.py": {"path": str(binary), "sha256": sha256(binary)}},
            })
            with mock.patch.dict(os.environ, {"FAKE_USI_MODE": "gate"}, clear=False):
                with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                    pilot = execute_run(runtime, config, "pilot", run_id="gate-fixture")
            records = [json.loads(path.read_text()) for path in (runtime / "runs/gate-fixture/attempts").glob("*.json")]
            observed = max(
                value
                for record in records
                for value in (record["result"].get("reported_nodes_at_score"), record["result"].get("last_reported_nodes"))
                if value is not None
            )
            config["formal"]["max_reported_nodes"] = node_reporting_limit(config["requested_nodes"])
            config["formal"]["pilot_evidence"] = {
                "pilot_run_id": "gate-fixture",
                "pilot_fingerprint": pilot["fingerprint"],
                "observed_max_reported_nodes": observed,
                "max_reported_nodes": node_reporting_limit(config["requested_nodes"]),
            }
            self.assertEqual(validate_formal_gate(runtime, config)["pilot_run_id"], "gate-fixture")
            target = next(
                path for path in (runtime / "runs/gate-fixture/attempts").glob("*.json")
                if path.is_file()
            )
            original_record = target.read_text()
            missing_max = json.loads(original_record)
            del missing_max["result"]["max_reported_nodes_evidence"]
            atomic_write_json(target, missing_max)
            with self.assertRaisesRegex(BenchmarkError, "raw observation schema"):
                validate_formal_gate(runtime, config)
            target.write_text(original_record)
            changed_option = copy.deepcopy(config)
            changed_option["engines"]["sekirei"]["options"]["Hash"] = "64"
            with self.assertRaisesRegex(ConfigurationError, "execution identity"):
                validate_formal_gate(runtime, changed_option)
            config["formal"]["pilot_evidence"]["observed_max_reported_nodes"] = observed + 1
            with self.assertRaisesRegex(ConfigurationError, "observed maximum mismatch"):
                validate_formal_gate(runtime, config)

    def test_formal_gate_allows_observed_nodes_below_requested(self):
        config = copy.deepcopy(load_config())
        config["engines"]["teacher"]["binary"] = "fake-usi.py"
        config["engines"]["teacher"]["weight_required"] = False
        config["engines"]["sekirei"]["binary"] = "fake-usi.py"
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            binary = runtime / "bin/fake-usi.py"
            binary.parent.mkdir(parents=True)
            shutil.copy2(FAKE, binary)
            binary.chmod(0o755)
            atomic_write_json(runtime / "build-manifest.json", {
                "lock_sha256": sha256(Path(__file__).resolve().parents[1] / "config/toolchain.lock.json"),
                "binaries": {"fake-usi.py": {"path": str(binary), "sha256": sha256(binary)}},
            })
            with mock.patch.dict(os.environ, {"FAKE_USI_MODE": "gate-early"}, clear=False):
                with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                    pilot = execute_run(runtime, config, "pilot", run_id="early-gate-fixture")
            records = [
                json.loads(path.read_text())
                for path in (runtime / "runs/early-gate-fixture/attempts").glob("*.json")
            ]
            observed = max(
                value
                for record in records
                for value in (
                    record["result"].get("reported_nodes_at_score"),
                    record["result"].get("last_reported_nodes"),
                )
                if value is not None
            )
            self.assertLess(observed, config["requested_nodes"])
            config["formal"]["max_reported_nodes"] = node_reporting_limit(config["requested_nodes"])
            config["formal"]["pilot_evidence"] = {
                "pilot_run_id": "early-gate-fixture",
                "pilot_fingerprint": pilot["fingerprint"],
                "observed_max_reported_nodes": observed,
                "max_reported_nodes": node_reporting_limit(config["requested_nodes"]),
            }
            self.assertEqual(validate_formal_gate(runtime, config)["pilot_run_id"], "early-gate-fixture")



class NodeEvidenceTests(unittest.TestCase):
    def test_positive_node_evidence_is_per_engine_and_ignores_technical_rows(self):
        records = [
            {"engine_id": "teacher", "result": {"status": "exact_cp", "reported_nodes_at_score": 0, "last_reported_nodes": 0}},
            {"engine_id": "sekirei", "result": {"status": "exact_cp", "reported_nodes_at_score": 1, "last_reported_nodes": 1}},
            {"engine_id": "teacher", "result": {"status": "node_budget_failure", "reported_nodes_at_score": 9999, "last_reported_nodes": 9999}},
            {"engine_id": "teacher", "status": "node_budget_failure", "result": {"status": "exact_cp", "reported_nodes_at_score": 9999, "last_reported_nodes": 9999}},
        ]
        summary = positive_node_evidence(records)
        self.assertFalse(summary["teacher"]["has_positive"])
        self.assertTrue(summary["sekirei"]["has_positive"])
        self.assertEqual(summary["sekirei"]["positive_observation_count"], 2)
        self.assertEqual(summary["teacher"]["attempt_count"], 3)

    def test_observed_max_uses_all_evidence_maximum(self):
        self.assertEqual(
            observed_max_reported_nodes([
                {
                    "engine_id": "teacher",
                    "result": {
                        "reported_nodes_at_score": 1000,
                        "last_reported_nodes": 700,
                        "max_reported_nodes_evidence": 2000,
                    },
                }
            ]),
            2000,
        )

    def test_strict_node_helpers_do_not_fallback_from_missing_maximum(self):
        records = [{
            "engine_id": "teacher",
            "status": "exact_cp",
            "result": {
                "status": "exact_cp",
                "reported_nodes_at_score": 1000,
                "last_reported_nodes": 1000,
                "node_evidence_positive_count": 2,
                "node_evidence_zero_count": 0,
                "node_evidence_errors": [],
                "node_evidence_count": 2,
            },
        }]
        self.assertIsNone(observed_max_reported_nodes(records, strict=True))
        self.assertIsNone(positive_node_evidence(records, ("teacher",), strict=True)["teacher"]["observed_max"])

    def test_formal_gate_rejects_coordinated_policy_tampering(self):
        import benchmark

        config = copy.deepcopy(load_config())
        derived = node_reporting_limit(config["requested_nodes"])
        positions = [
            {
                "game_id": f"game-{index:02d}",
                "ply": 1,
                "position": "position startpos moves 7g7f",
                "side_to_move": "white",
            }
            for index in range(1, 16)
        ]
        plan = {"positions": positions, "repetitions": 3}
        manifest = {
            "fingerprint": "pilot-fingerprint",
            "fingerprint_payload": {"execution_identity": {}},
        }
        records = []
        for engine_id in ("teacher", "sekirei"):
            records.extend(
                {
                    "engine_id": engine_id,
                    "status": "exact_cp",
                    "result": {
                        "status": "exact_cp",
                        "reported_nodes_at_score": config["requested_nodes"],
                        "last_reported_nodes": config["requested_nodes"],
                        "max_reported_nodes_evidence": config["requested_nodes"],
                    },
                }
                for _ in range(45)
            )
        config["formal"]["max_reported_nodes"] = derived
        config["formal"]["pilot_evidence"] = {
            "pilot_run_id": "pilot-test",
            "pilot_fingerprint": "pilot-fingerprint",
            "observed_max_reported_nodes": config["requested_nodes"],
            "max_reported_nodes": derived,
        }
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp)
            (runtime / "runs/pilot-test").mkdir(parents=True)
            with mock.patch.object(
                benchmark,
                "validate_run_artifacts",
                return_value=(manifest, plan, {}, records, {}, []),
            ):
                with mock.patch.object(benchmark, "make_plan", return_value=(plan, [])):
                    with mock.patch.object(benchmark, "execution_identity", return_value={}):
                        self.assertEqual(validate_formal_gate(runtime, config)["pilot_run_id"], "pilot-test")
                        for tampered in (derived - 1, derived + 1):
                            tampered_config = copy.deepcopy(config)
                            tampered_config["formal"]["max_reported_nodes"] = tampered
                            tampered_config["formal"]["pilot_evidence"]["max_reported_nodes"] = tampered
                            with self.assertRaisesRegex(ConfigurationError, "node policy limit"):
                                validate_formal_gate(runtime, tampered_config)

    def test_formal_gate_rejects_all_zero_or_one_engine_positive_evidence(self):
        import benchmark

        positions = [
            {"game_id": f"game-{index:02d}", "ply": 1, "position": "position startpos moves 7g7f", "side_to_move": "white"}
            for index in range(1, 16)
        ]
        plan = {"positions": positions, "repetitions": 3}
        manifest = {"fingerprint": "pilot-fingerprint", "fingerprint_payload": {"execution_identity": {}}}
        for positive_engine in ("teacher", "sekirei"):
            records = []
            for engine_id in ("teacher", "sekirei"):
                nodes = 1000 if engine_id == positive_engine else 0
                records.extend(
                    {
                        "engine_id": engine_id,
                        "status": "exact_cp",
                        "result": {
                            "status": "exact_cp",
                            "reported_nodes_at_score": nodes,
                            "last_reported_nodes": nodes,
                        },
                    }
                    for _ in range(45)
                )
            config = copy.deepcopy(load_config())
            config["requested_nodes"] = 1000
            config["formal"]["max_reported_nodes"] = node_reporting_limit(config["requested_nodes"])
            config["formal"]["pilot_evidence"] = {
                "pilot_run_id": "pilot-test",
                "pilot_fingerprint": "pilot-fingerprint",
                "observed_max_reported_nodes": 1000,
                "max_reported_nodes": node_reporting_limit(config["requested_nodes"]),
            }
            with tempfile.TemporaryDirectory() as temp:
                runtime = Path(temp)
                (runtime / "runs/pilot-test").mkdir(parents=True)
                with mock.patch.object(benchmark, "validate_run_artifacts", return_value=(manifest, plan, {}, records, {}, [])):
                    with mock.patch.object(benchmark, "make_plan", return_value=(plan, [])):
                        with mock.patch.object(benchmark, "execution_identity", return_value={}):
                            with self.assertRaisesRegex(ConfigurationError, "positive node evidence"):
                                validate_formal_gate(runtime, config)


def small_plan():
    positions = []
    for index in range(1, 6):
        game_id = f"game-{index:02d}"
        for ply in (1, 2):
            positions.append({
                "game_id": game_id,
                "ply": ply,
                "position": f"position startpos moves {ply}",
                "side_to_move": "white" if ply == 1 else "black",
            })
    return {
        "schema_version": 1,
        "benchmark_id": "fixture",
        "run_type": "pilot",
        "repetitions": 1,
        "positions": positions,
        "hashes": {"universe_sha256": "u", "development_csa_aggregate_sha256": "c"},
    }


def row(game_id, ply, status, **values):
    result = {"status": status}
    result.update(values)
    return {"game_id": game_id, "ply": ply, **result}


class ReportTests(unittest.TestCase):
    def test_report_exposes_position_types_and_per_engine_positive_nodes(self):
        plan = small_plan()
        plan["positions"][0].update(
            position_type="forced_single_legal_move",
            classification={
                "type": "forced_single_legal_move",
                "in_check": True,
                "legal_move_count": 1,
                "sole_legal_move": "7g7f",
            },
        )
        plan["positions"][-1].update(
            position_type="terminal_checkmate",
            classification={
                "type": "terminal_checkmate",
                "in_check": True,
                "legal_move_count": 0,
            },
        )
        teacher = [
            row(game_id, ply, "exact_cp", score_cp_sente=10, reported_nodes_at_score=1000, last_reported_nodes=1000)
            for game_id in (f"game-{index:02d}" for index in range(1, 6))
            for ply in (1, 2)
        ]
        candidate = [
            row(game_id, ply, "exact_cp", score_cp_sente=10, reported_nodes_at_score=900, last_reported_nodes=900)
            for game_id in (f"game-{index:02d}" for index in range(1, 6))
            for ply in (1, 2)
        ]
        report = score_observations(plan, teacher, candidate, accuracy_thresholds=[0])
        self.assertEqual(report["position_type_diagnostics"]["counts"]["forced_single_legal_move"], 1)
        self.assertEqual(report["position_type_diagnostics"]["counts"]["terminal_checkmate"], 1)
        for engine_id in ("teacher", "sekirei"):
            self.assertTrue(report["node_evidence"][engine_id]["has_positive"])
        svg = render_svg(report)
        self.assertIn('data-position-type="forced_single_legal_move"', svg)
        with tempfile.TemporaryDirectory() as temp:
            export_public(temp, report)
            public_json = (Path(temp) / "validation.json").read_text()
            self.assertIn("forced_single_legal_move", public_json)
            self.assertIn("terminal_checkmate", public_json)
            self.assertNotIn("position startpos moves", public_json)

    def test_pilot_retains_all_repetitions_and_reports_repeatability(self):
        plan = small_plan()
        plan["repetitions"] = 3
        teacher, candidate = [], []
        for repetition in (1, 2, 3):
            for index in range(1, 6):
                for ply in (1, 2):
                    teacher.append(row(f"game-{index:02d}", ply, "exact_cp", repetition=repetition, score_cp_sente=100, reported_nodes_at_score=1000 + repetition, last_reported_nodes=1000 + repetition, engine_time_ms=4, wall_go_to_bestmove_ns=10))
                    candidate.append(row(f"game-{index:02d}", ply, "exact_cp", repetition=repetition, score_cp_sente=105, reported_nodes_at_score=1000 + repetition, last_reported_nodes=1000 + repetition, engine_time_ms=5, wall_go_to_bestmove_ns=11))
        report = score_observations(plan, teacher, candidate, accuracy_thresholds=[0])
        self.assertEqual(report["attempt_count"], 60)
        self.assertEqual(report["repeatability"]["attempt_count"], 60)
        self.assertEqual(len(report["repetition_reports"]), 3)
        self.assertFalse(report["headline"]["formal"])
        self.assertIsNone(report["headline"]["mae_cp"])
        self.assertEqual(report["repeatability"]["observed_max_reported_nodes"], 1003)
        self.assertEqual(report["repeatability"]["engines"]["teacher"]["stable_position_count"], 10)

    def test_report_keeps_deterministic_all_evidence_m_distribution(self):
        plan = small_plan()
        plan["requested_nodes"] = 1000
        plan["max_reported_nodes"] = 1010
        values = [0, 1000, 1005, 1010, 1011, 0, 1000, 1005, 1010, 1011]
        teacher = []
        candidate = []
        for value, occurrence in zip(values, plan["positions"]):
            payload = {
                "score_cp_sente": 10,
                "reported_nodes_at_score": value,
                "last_reported_nodes": value,
                "max_reported_nodes_evidence": value,
            }
            teacher.append(row(occurrence["game_id"], occurrence["ply"], "exact_cp", **payload))
            candidate.append(row(occurrence["game_id"], occurrence["ply"], "exact_cp", **payload))
        first = score_observations(plan, teacher, candidate, accuracy_thresholds=[0])
        second = score_observations(plan, teacher, candidate, accuracy_thresholds=[0])
        summary = first["node_evidence"]["all_evidence_m"]["teacher"]
        self.assertEqual(summary["evidence_count"], 10)
        self.assertEqual(summary["positive_count"], 8)
        self.assertEqual(summary["zero_count"], 2)
        self.assertEqual(summary["missing_count"], 0)
        self.assertEqual(summary["invalid_count"], 0)
        self.assertEqual(summary["p50"], 1005)
        self.assertEqual(summary["p95"], 1011)
        self.assertEqual(summary["p99"], 1011)
        self.assertEqual(summary["max"], 1011)
        self.assertEqual(summary["count_above_requested_nodes"], 6)
        self.assertEqual(summary["count_above_policy_limit"], 2)
        self.assertEqual(summary["max_positive_overrun"], 11)
        self.assertEqual(summary["max_positive_overrun_rate"], 0.011)
        self.assertEqual(
            first["node_evidence"]["all_evidence_m"],
            second["node_evidence"]["all_evidence_m"],
        )
        with tempfile.TemporaryDirectory() as temp:
            export_public(temp, first)
            public_json = (Path(temp) / "validation.json").read_text()
            self.assertIn("all_evidence_m", public_json)
            self.assertNotIn("position startpos moves", public_json)

    def test_report_propagates_explicit_all_evidence_maximum(self):
        plan = small_plan()
        teacher, candidate = [], []
        for index in range(1, 6):
            for ply in (1, 2):
                teacher.append(row(
                    f"game-{index:02d}",
                    ply,
                    "exact_cp",
                    score_cp_sente=100,
                    reported_nodes_at_score=1000,
                    last_reported_nodes=1000,
                    max_reported_nodes_evidence=2000 if (index, ply) == (1, 1) else 1000,
                ))
                candidate.append(row(
                    f"game-{index:02d}",
                    ply,
                    "exact_cp",
                    score_cp_sente=100,
                    reported_nodes_at_score=900,
                    last_reported_nodes=900,
                    max_reported_nodes_evidence=1500,
                ))
        report = score_observations(plan, teacher, candidate, accuracy_thresholds=[0])
        self.assertEqual(report["repeatability"]["observed_max_reported_nodes"], 2000)
        self.assertEqual(
            report["repeatability"]["engines"]["teacher"]["max_reported_nodes_evidence"]["max_reported_nodes_evidence"],
            2000,
        )
        self.assertEqual(
            report["series"]["game-01"][0]["max_reported_nodes_evidence"],
            2000,
        )

    def test_fixed_denominator_headline_and_accuracy_curve(self):
        plan = small_plan()
        teacher = [row(f"game-{index:02d}", ply, "exact_cp", score_cp_sente=100)
                   for index in range(1, 6) for ply in (1, 2)]
        candidate = [row(f"game-{index:02d}", 1, "exact_cp", score_cp_sente=110)
                     for index in range(1, 6)]
        report = score_observations(plan, teacher, candidate, accuracy_thresholds=[0, 9, 10])
        self.assertFalse(report["headline"]["valid"])
        self.assertEqual(report["e_exact_coverage"]["candidate_exact_count"], 5)
        self.assertEqual(report["e_exact_coverage"]["teacher_e_count"], 10)
        self.assertEqual([point["equal_game_mean"] for point in report["accuracy_curve"]], [0, 0, 0.5])
        self.assertEqual(report["diagnostic_error"]["partial_numerator_abs_cp"], 50)

    def test_bound_and_mate_diagnostics_are_not_combined_into_mae(self):
        plan = small_plan()
        teacher = [row("game-01", 1, "bound_cp", reported_cp_stm=100, reported_cp_sente=100, score_bound_sente="lowerbound"),
                   row("game-02", 1, "mate", winner_sente="black", mate_distance=3)]
        candidate = [row("game-01", 1, "exact_cp", score_cp_sente=90),
                     row("game-02", 1, "mate", winner_sente="white", mate_distance=5)]
        report = score_observations(plan, teacher, candidate, accuracy_thresholds=[0])
        self.assertEqual(report["bound_diagnostic"]["lower_violation_count"], 1)
        self.assertEqual(report["mate_diagnostic"]["winner_counts"]["opposite"], 1)
        self.assertIsNone(report["headline"]["mae_cp"])

    def test_svg_is_deterministic_shared_and_keeps_gaps(self):
        plan = small_plan()
        teacher = [row(game, ply, "exact_cp", score_cp_sente=ply * 10)
                   for game in (f"game-{i:02d}" for i in range(1, 6)) for ply in (1, 2)]
        candidate = [row(game, 1, "exact_cp", score_cp_sente=15)
                     for game in (f"game-{i:02d}" for i in range(1, 6))]
        report = score_observations(plan, teacher, candidate, accuracy_thresholds=[0, 10])
        first, second = render_svg(report), render_svg(report)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("<svg "))
        root = ET.fromstring(first)
        self.assertEqual(root.attrib["data-panel-count"], "5")
        self.assertEqual(root.attrib["data-y-min"], "-20")
        self.assertEqual(root.attrib["data-y-max"], "20")
        self.assertIn("y ±20 cp", first)
        self.assertIn("candidate", first)
        self.assertIn("gap", first)
        self.assertEqual(first.count('class="exact"'), 15)

    def test_public_export_is_redacted_and_hashed(self):
        plan = small_plan()
        teacher = [row(game, ply, "exact_cp", score_cp_sente=ply * 10)
                   for game in (f"game-{i:02d}" for i in range(1, 6)) for ply in (1, 2)]
        candidate = [row(game, ply, "exact_cp", score_cp_sente=ply * 10)
                     for game in (f"game-{i:02d}" for i in range(1, 6)) for ply in (1, 2)]
        report = score_observations(plan, teacher, candidate, accuracy_thresholds=[0])
        with tempfile.TemporaryDirectory() as temp:
            hashes = export_public(temp, report)
            self.assertIn("validation.md", hashes["files"])
            for name in ("validation.md", "reviewed.svg", "validation.json", "manifest.json"):
                text = (Path(temp) / name).read_text()
                self.assertNotIn("/home/", text)
                self.assertNotIn("position startpos moves", text)
                self.assertNotIn("source_game_id", text)
            self.assertIn("Game 01", (Path(temp) / "reviewed.svg").read_text())

    def test_public_export_is_empty_root_and_strips_free_form_cshogi(self):
        plan = small_plan()
        teacher = [row(game, ply, "exact_cp", score_cp_sente=ply * 10)
                   for game in (f"game-{i:02d}" for i in range(1, 6)) for ply in (1, 2)]
        candidate = [row(game, ply, "exact_cp", score_cp_sente=ply * 10)
                     for game in (f"game-{i:02d}" for i in range(1, 6)) for ply in (1, 2)]
        report = score_observations(plan, teacher, candidate, accuracy_thresholds=[0])
        report["cshogi"] = {"available": False, "reason": "free-form source_game_id leak"}
        with tempfile.TemporaryDirectory() as temp:
            hashes = export_public(temp, report)
            self.assertEqual(set(hashes["files"]), {"validation.md", "reviewed.svg", "validation.json"})
            self.assertEqual(sorted(path.name for path in Path(temp).iterdir()), ["manifest.json", "reviewed.svg", "validation.json", "validation.md"])
            public_json = (Path(temp) / "validation.json").read_text()
            self.assertNotIn("reason", public_json)
            self.assertNotIn("source_game_id", public_json)
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "already-there").write_text("x")
            with self.assertRaisesRegex(ValueError, "empty"):
                export_public(temp, report)

    def test_report_rejects_synthetic_final_plan_without_reading_final_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_write_json(root / "manifest.json", {
                "schema_version": 1,
                "run_id": "synthetic-final",
                "run_type": "pilot",
                "benchmark_id": "final-holdout",
                "split": "final",
                "fingerprint": "synthetic",
            })
            atomic_write_json(root / "plan.json", {"schema_version": 1, "run_type": "pilot"})
            with self.assertRaisesRegex(ValueError, "development"):
                load_run(root)


if __name__ == "__main__":
    unittest.main()
