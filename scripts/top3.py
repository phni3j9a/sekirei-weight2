#!/usr/bin/env python3
"""Measure teacher MultiPV=1 bestmove membership in candidate MultiPV=3."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import benchmark as b
from audit_pack import require_decoder
from prepare import DEFAULT_RUNTIME, REPO, sha256

POLICY_PATH = REPO / "config/top3-benchmark.json"


def parse_top3(events, bestmove, legal_moves):
    """Reject stale/mixed/incomplete ranks; the primary parser checks node evidence."""
    latest = {}
    for info in b._structured_info_parses(events):
        if info.get("parse_error"):
            raise ValueError("malformed MultiPV info")
        if "score_kind" not in info:
            continue
        rank = info.get("multipv")
        if rank not in (1, 2, 3) or info.get("bound_stm") != "exact":
            raise ValueError("missing/invalid MultiPV rank or bounded candidate score")
        # A new primary line starts a new iteration; never reuse older ranks.
        if rank == 1:
            latest = {}
        if rank in latest:
            raise ValueError("duplicate MultiPV rank")
        latest[rank] = info
    if set(latest) != {1, 2, 3}:
        raise ValueError("incomplete Top3")
    depths = {x.get("depth") for x in latest.values()}
    if len(depths) != 1 or not isinstance(next(iter(depths)), int) or next(iter(depths)) <= 0:
        raise ValueError("mixed or invalid MultiPV depths")
    moves = []
    for rank in (1, 2, 3):
        pv = latest[rank].get("pv")
        if not pv or pv[0] not in legal_moves:
            raise ValueError("illegal Top3 root")
        moves.append(pv[0])
    if len(set(moves)) != 3 or moves[0] != bestmove:
        raise ValueError("duplicate Top3 root or bestmove mismatch")
    return moves


def context(runtime, config, mae_run):
    cshogi, _numpy, versions = require_decoder()
    parent, formal, _u, records, _expected, games = b.validate_run_artifacts(
        mae_run, config, require_complete=True, expected_run_type="formal")
    base_identity = b.execution_identity(config, formal, runtime)
    if parent["fingerprint_payload"]["execution_identity"] != base_identity:
        raise ValueError("MAE parent and Top3 must use the same model and execution settings")
    if any(r["status"] in b.TECHNICAL_FAILURE_STATUSES for r in records):
        raise ValueError("MAE reference run contains technical failures")
    teacher = {(r["game_id"], r["ply"]): r["result"]
               for r in records if r["engine_id"] == "teacher"}
    legal = {}
    for game in games:
        board = cshogi.Board()
        for ply, move in enumerate(game["usi_moves"], 1):
            parsed = board.move_from_usi(move)
            if not board.is_legal(parsed):
                raise ValueError("illegal fixed development history")
            board.push(parsed)
            legal[(game["game_id"], ply)] = sorted(cshogi.move_to_usi(m) for m in board.legal_moves)
    eligible = [p for p in formal["positions"]
                if p["position_type"] == "normal" and len(legal[(p["game_id"], p["ply"])]) >= 4]
    for p in eligible:
        key = (p["game_id"], p["ply"])
        if teacher[key].get("bestmove") not in legal[key]:
            raise ValueError("teacher reference lacks a legal normal bestmove")
    options = dict(b._runtime_options(config, runtime)["sekirei"], MultiPV="3")
    identity = {
        "schema_version": 1, "metric": "teacher_best_in_candidate_top3",
        "script_sha256": sha256(Path(__file__)), "policy_sha256": sha256(POLICY_PATH),
        "base_execution_identity": base_identity,
        "mae_run_fingerprint": parent["fingerprint"],
        "mae_manifest_sha256": sha256(mae_run / "manifest.json"),
        "options": options, "dependencies": versions,
        "eligible_occurrences": [p["occurrence_id"] for p in eligible],
        "legal_moves_sha256": b.json_sha256({p["occurrence_id"]: legal[(p["game_id"], p["ply"])]
                                            for p in eligible}),
    }
    return identity, eligible, teacher, legal, games


def expected(occurrence, repetition):
    return {"attempt_id": b._attempt_id("sekirei", occurrence, repetition),
            "engine_id": "sekirei", "game_id": occurrence["game_id"], "ply": occurrence["ply"],
            "repetition": repetition, "position": occurrence["position"],
            "side_to_move": occurrence["side_to_move"], "position_type": occurrence["position_type"],
            "position_classification": occurrence["classification"]}


def select_positions(eligible, run_type, config):
    if run_type == "formal":
        return eligible, 1
    pilot, _ = b.make_plan("pilot", config)
    ids = {p["occurrence_id"] for p in pilot["positions"]}
    return [p for p in eligible if p["occurrence_id"] in ids], 3


def inspect(run_dir, identity, eligible, teacher, legal, games, config):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    run_type = manifest["run_type"]
    if run_type not in ("pilot", "formal") or manifest["run_id"] != run_dir.name:
        raise ValueError("invalid Top3 run identity")
    payload = {"identity": identity, "run_type": run_type}
    fingerprint = b.json_sha256(payload)
    if manifest.get("payload") != payload or manifest.get("fingerprint") != fingerprint:
        raise ValueError("Top3 execution identity mismatch")
    if run_type == "formal":
        pilot_id = manifest.get("pilot_run_id")
        b._validate_run_id(pilot_id)
        if pilot_id == run_dir.name:
            raise ValueError("Top3 pilot cannot reference itself")
        pilot_dir = run_dir.parent / pilot_id
        pilot_manifest = json.loads((pilot_dir / "manifest.json").read_text())
        if pilot_manifest.get("run_type") != "pilot":
            raise ValueError("Top3 evidence must reference a pilot")
        pilot = inspect(pilot_dir, identity, eligible, teacher, legal, games, config)
        if not pilot["valid"] or pilot["fingerprint"] != manifest.get("pilot_fingerprint"):
            raise ValueError("Top3 pilot evidence changed or is invalid")
    positions, repetitions = select_positions(eligible, run_type, config)
    board_map = b._board_map(games)
    binary = identity["base_execution_identity"]["runtime"]["binaries"]["sekirei"]
    expected_ids = {expected(p, rep)["attempt_id"] for p in positions for rep in range(1, repetitions + 1)}
    actual_ids = {p.stem for p in (run_dir / "attempts").glob("*.json")}
    if actual_ids - expected_ids:
        raise ValueError("unexpected Top3 attempts")
    hits = defaultdict(list)
    roots = defaultdict(list)
    failures = []
    for p in positions:
        key = (p["game_id"], p["ply"])
        for rep in range(1, repetitions + 1):
            exp = expected(p, rep)
            path = b._attempt_path(run_dir, exp["attempt_id"])
            if not path.exists():
                failures.append({"attempt_id": exp["attempt_id"], "reason": "missing"})
                continue
            b._validate_recorded_attempt(
                path, fingerprint, exp, expected_options=identity["options"],
                expected_binary_identity=binary, requested_nodes=1000000, max_reported_nodes=1010000,
                board=board_map[key].clone())
            record = json.loads(path.read_text())
            try:
                if record["status"] not in ("exact_cp", "mate"):
                    raise ValueError("candidate outcome: " + record["status"])
                events = b._read_raw_event_log(b._log_path(run_dir, exp["attempt_id"]))
                moves = parse_top3(events, record["result"]["bestmove"], legal[key])
                hit = teacher[key]["bestmove"] in moves
                hits[p["game_id"]].append(hit)
                roots[p["occurrence_id"]].append(moves)
            except ValueError as error:
                failures.append({"attempt_id": exp["attempt_id"], "reason": str(error)})
    stable = all(len(values) == repetitions and all(v == values[0] for v in values) for values in roots.values())
    valid = not failures and stable and len(actual_ids) == len(expected_ids)
    per_game = {}
    for game_id in config["universe"]["plies"]:
        denominator = sum(p["game_id"] == game_id for p in positions) * repetitions
        observed = hits[game_id]
        per_game[game_id] = {"hits": sum(observed), "denominator": denominator,
                             "observed": len(observed),
                             "rate": sum(observed) / denominator if denominator and len(observed) == denominator else None}
    rates = [x["rate"] for x in per_game.values()]
    report = {"metric": "teacher_best_in_candidate_top3", "run_type": run_type,
              "valid": valid, "stable": stable, "attempts": len(expected_ids),
              "eligible_positions": len(positions), "per_game": per_game,
              "top3_rate": sum(rates) / 5 if valid and all(x is not None for x in rates) else None,
              "failures": failures, "fingerprint": fingerprint}
    return report


def run(args):
    runtime = args.runtime.expanduser().resolve()
    config = b.load_config(args.mae_config)
    b.validate_runtime(runtime, config)
    mae_run = args.mae_run.expanduser().resolve()
    ctx = context(runtime, config, mae_run)
    identity, eligible, teacher, legal, games = ctx
    b._validate_run_id(args.run_id)
    if "final" in args.run_id.casefold():
        raise ValueError("only development runs are supported")
    run_dir = runtime / "runs" / args.run_id
    if args.action == "report":
        report = inspect(run_dir, *ctx, config)
        b.atomic_write_json(run_dir / "top3-report.json", report)
        print(json.dumps(report, indent=2))
        return
    run_type = args.action
    if run_type == "formal":
        if not args.pilot_run_id:
            raise ValueError("formal Top3 needs its own reviewed pilot")
        b._validate_run_id(args.pilot_run_id)
        pilot_dir = runtime / "runs" / args.pilot_run_id
        pilot = inspect(pilot_dir, *ctx, config)
        if pilot["run_type"] != "pilot" or not pilot["valid"]:
            raise ValueError("Top3 pilot is incomplete, unstable, or invalid")
    with b.nonblocking_lock(runtime / ".prepare.lock", exclusive=False), \
         b.nonblocking_lock(runtime / ".benchmark.lock", exclusive=True):
        payload = {"identity": identity, "run_type": run_type}
        fingerprint = b.json_sha256(payload)
        if run_dir.exists():
            if not args.resume:
                raise ValueError("Top3 run exists; use --resume")
            previous = inspect(run_dir, *ctx, config)
            if any(f["reason"] == "candidate outcome: cleanup_failure" for f in previous["failures"]):
                raise ValueError("cannot resume after a cleanup failure")
        else:
            (run_dir / "attempts").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            b.atomic_write_json(run_dir / "manifest.json", {
                "run_id": args.run_id, "run_type": run_type, "fingerprint": fingerprint,
                "payload": payload, "created_at": b.utc_now(),
                "pilot_run_id": args.pilot_run_id,
                "pilot_fingerprint": pilot["fingerprint"] if run_type == "formal" else None,
                "mae_run": str(mae_run)})
        positions, repetitions = select_positions(eligible, run_type, config)
        board_map = b._board_map(games)
        completed = 0
        total = len(positions) * repetitions
        for p in positions:
            for rep in range(1, repetitions + 1):
                exp = expected(p, rep)
                path = b._attempt_path(run_dir, exp["attempt_id"])
                if not path.exists():
                    result = b.run_engine_attempt(
                        runtime / "bin/sekirei", p["position"], p["side_to_move"], identity["options"],
                        cwd=run_dir, requested_nodes=1000000, timeout_seconds=config["timeout_seconds"],
                        max_reported_nodes=1010000, board=board_map[(p["game_id"], p["ply"])].clone(),
                        raw_log_path=b._log_path(run_dir, exp["attempt_id"]), engine_id="sekirei",
                        position_type=p["position_type"], position_classification=p["classification"])
                    record = dict(exp, schema_version=1, run_fingerprint=fingerprint,
                                  position_sha256=b.digest_bytes(p["position"].encode()),
                                  status=result["status"], result=result, outcome=b._outcome_payload(result),
                                  raw_log=exp["attempt_id"] + ".jsonl", raw_log_sha256=result["raw_log_sha256"])
                    b.atomic_write_json(path, record)
                    if result["status"] == "cleanup_failure":
                        raise RuntimeError("Top3 cleanup failure; aborting")
                completed += 1
                print(f"[{completed}/{total}] {exp['attempt_id']}", flush=True)
        report = inspect(run_dir, *ctx, config)
        b.atomic_write_json(run_dir / "top3-report.json", report)
        print(json.dumps(report, indent=2))
        if not report["valid"]:
            raise RuntimeError("Top3 measurement invalid; inspect preserved report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pilot", "formal", "report"))
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--mae-config", type=Path, default=b.CONFIG_PATH)
    parser.add_argument("--mae-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot-run-id")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
