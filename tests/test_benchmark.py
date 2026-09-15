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
    OUTCOME_FIELDS,
    atomic_write,
    atomic_write_json,
    build_universe,
    compare_cshogi,
    development_csa_hash,
    execute_run,
    load_config,
    load_development_games,
    parse_csa_text,
    parse_usi_observation,
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

    def test_pilot_points_are_first_middle_last_and_no_final_path(self):
        config = load_config()
        from benchmark import make_plan

        plan, _ = make_plan("pilot", config)
        self.assertEqual(len(plan["positions"]), 15)
        for index in range(0, 15, 3):
            values = [row["ply"] for row in plan["positions"][index:index + 3]]
            length = config["universe"]["plies"][plan["positions"][index]["game_id"]]
            self.assertEqual(values, [1, (length + 1) // 2, length])
        self.assertTrue(all("final" not in row["position"] for row in plan["positions"]))

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
            self.assertEqual(result["checked_moves"], 570)


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
            self.assertEqual(len(attempt_files), 90)
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
                "last_reported_nodes", "engine_time_ms", "engine_time", "wall_go_to_bestmove_ns",
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
            self.assertEqual(report["universe"]["total_occurrences"], 15)
            self.assertEqual(report["repeatability"]["attempt_count"], 90)
            self.assertTrue(report["validity"]["evidence_validator_passed"])
            self.assertTrue((Path(temp) / "report/evaluation.svg").is_file())
            changed = copy.deepcopy(config)
            changed["timeout_seconds"] = config["timeout_seconds"] + 1
            with self.assertRaisesRegex(BenchmarkError, "fingerprint mismatch"):
                execute_run(runtime, changed, "pilot", run_id="resume-fixture", resume=True)

    def test_formal_run_is_refused_until_pilot_ceiling_is_frozen(self):
        from benchmark import execute_run

        config = load_config()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ConfigurationError, "ceiling"):
                execute_run(temp, config, "formal", run_id="formal-before-pilot")
            self.assertFalse((Path(temp) / "runs/formal-before-pilot").exists())

    def test_formal_gate_requires_matching_complete_pilot_evidence(self):
        config = copy.deepcopy(load_config())
        config["requested_nodes"] = 1000
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
                    pilot = execute_run(runtime, config, "pilot", run_id="gate-fixture")
            records = [json.loads(path.read_text()) for path in (runtime / "runs/gate-fixture/attempts").glob("*.json")]
            observed = max(
                value
                for record in records
                for value in (record["result"].get("reported_nodes_at_score"), record["result"].get("last_reported_nodes"))
                if value is not None
            )
            config["formal"]["max_reported_nodes"] = observed
            config["formal"]["pilot_evidence"] = {
                "pilot_run_id": "gate-fixture",
                "pilot_fingerprint": pilot["fingerprint"],
                "observed_max_reported_nodes": observed,
                "max_reported_nodes": observed,
            }
            self.assertEqual(validate_formal_gate(runtime, config)["pilot_run_id"], "gate-fixture")
            changed_option = copy.deepcopy(config)
            changed_option["engines"]["sekirei"]["options"]["Hash"] = "64"
            with self.assertRaisesRegex(ConfigurationError, "execution identity"):
                validate_formal_gate(runtime, changed_option)
            config["formal"]["pilot_evidence"]["observed_max_reported_nodes"] = observed + 1
            config["formal"]["pilot_evidence"]["max_reported_nodes"] = observed + 1
            config["formal"]["max_reported_nodes"] = observed + 1
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
            with mock.patch.dict(os.environ, {"FAKE_USI_MODE": "early"}, clear=False):
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
            config["formal"]["max_reported_nodes"] = config["requested_nodes"]
            config["formal"]["pilot_evidence"] = {
                "pilot_run_id": "early-gate-fixture",
                "pilot_fingerprint": pilot["fingerprint"],
                "observed_max_reported_nodes": observed,
                "max_reported_nodes": config["requested_nodes"],
            }
            self.assertEqual(validate_formal_gate(runtime, config)["pilot_run_id"], "early-gate-fixture")


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
