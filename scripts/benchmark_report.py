#!/usr/bin/env python3
"""Score a completed development benchmark and make deterministic reports.

This module consumes only local run observations.  It never reads the final
benchmark and does not turn mate or bound scores into artificial centipawn
values.  The public-export command deliberately builds a much smaller,
redacted view from the aggregate report.
"""

import argparse
import csv
import copy
import html
import json
import math
import os
from pathlib import Path
import re
import sys

from benchmark import (
    OBSERVATION_STATUSES,
    POSITION_CLASSIFICATION_TYPES,
    TECHNICAL_FAILURE_STATUSES,
    atomic_write,
    atomic_write_json,
    development_manifest_entries,
    digest_bytes,
    load_config,
    node_reporting_limit,
    observed_max_reported_nodes,
    positive_node_evidence,
    plan_hash,
    sha256,
    validate_run_artifacts,
    universe_hash,
)


FAILURE_STATUSES = set(TECHNICAL_FAILURE_STATUSES)
SUCCESS_TYPES = {"exact_cp", "bound_cp", "mate"}
CROSS_CATEGORIES = ("exact_cp", "bound_cp", "mate", "no_score", "failure", "missing")
OPERATIONAL_CATEGORIES = ("operational", "failure", "missing")
KNOWN_STATUSES = tuple(sorted(OBSERVATION_STATUSES | TECHNICAL_FAILURE_STATUSES))
NODE_QUANTILE_DEFINITION = "sorted finite M values; linear interpolation at (n - 1) * p"
SELECTION_TYPES = ("canonical", "regression", "formal")


def _result(row):
    if row is None:
        return {}
    nested = row.get("result") if isinstance(row, dict) else None
    return nested if isinstance(nested, dict) else row


def _status(row):
    result = _result(row)
    return result.get("status") or row.get("status") if isinstance(row, dict) else None


def _value(row, key):
    value = _result(row).get(key)
    if value is None and isinstance(row, dict):
        value = row.get(key)
    return value


def _finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _cp(row):
    value = _value(row, "score_cp_sente")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reported_cp(row):
    value = _value(row, "reported_cp_sente")
    if value is None:
        value = _value(row, "reported_cp_stm")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bound(row):
    return _value(row, "score_bound_sente")


def _winner(row):
    value = _value(row, "winner_sente")
    if value is None:
        value = _value(row, "winner")
    return value if value in ("black", "white") else None


def _mate_distance(row):
    value = _value(row, "mate_distance")
    return value if isinstance(value, int) and value >= 0 else None


def _category(row, missing=False):
    if missing or row is None:
        return "missing"
    status = _status(row)
    if status in SUCCESS_TYPES or status == "no_score":
        return status
    return "failure"


def _operational_category(row):
    if row is None:
        return "missing"
    return "failure" if _status(row) in FAILURE_STATUSES or _status(row) is None else "operational"


def _position_type(value):
    """Return a typed position label while keeping old synthetic fixtures usable."""
    if isinstance(value, dict):
        result = _value(value, "position_type")
        if result is None:
            classification = value.get("classification") or value.get("position_classification")
            if isinstance(classification, dict):
                result = classification.get("type")
    else:
        result = None
    return result if result in POSITION_CLASSIFICATION_TYPES else "normal"


def _report_node_policy(plan, config):
    policy = dict(config.get("node_policy") or {})
    policy.update({
        "requested_nodes": plan.get("requested_nodes"),
        "max_reported_nodes": plan.get("max_reported_nodes"),
    })
    return policy


def _selection_diagnostics(plan):
    counts = {selection: 0 for selection in SELECTION_TYPES}
    unknown_count = 0
    for occurrence in plan.get("positions", []):
        selection = occurrence.get("selection", "formal") if isinstance(occurrence, dict) else None
        if selection in counts:
            counts[selection] += 1
        else:
            unknown_count += 1
    return {
        "counts": counts,
        "unknown_count": unknown_count,
        "total_occurrences": sum(counts.values()) + unknown_count,
    }


def _node_evidence_diagnostics(
    teacher_rows,
    candidate_rows,
    *,
    requested_nodes=None,
    policy_limit=None,
):
    records = [
        *(_with_engine(teacher_rows, "teacher")),
        *(_with_engine(candidate_rows, "sekirei")),
    ]
    result = positive_node_evidence(records)
    if policy_limit is None and isinstance(requested_nodes, int) and not isinstance(requested_nodes, bool):
        policy_limit = node_reporting_limit(requested_nodes)
    result["all_evidence_m"] = {
        engine_id: _all_evidence_m_distribution(
            rows,
            requested_nodes=requested_nodes,
            policy_limit=policy_limit,
        )
        for engine_id, rows in (
            ("teacher", teacher_rows),
            ("sekirei", candidate_rows),
        )
    }
    result["units"] = {
        "reported_nodes_at_score": "nodes",
        "last_reported_nodes": "nodes",
        "max_reported_nodes_evidence": "nodes",
    }
    return result


def _max_node_evidence(row):
    """Read the all-current-go maximum, with compatibility for old fixtures."""
    result = _result(row)
    if "max_reported_nodes_evidence" in result:
        value = result.get("max_reported_nodes_evidence")
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    values = [
        result.get(field)
        for field in ("reported_nodes_at_score", "last_reported_nodes")
        if isinstance(result.get(field), int)
        and not isinstance(result.get(field), bool)
        and result.get(field) >= 0
    ]
    return max(values) if values else None


def _all_evidence_m_state(row):
    """Classify one attempt's all-current-go maximum without hiding errors."""
    result = _result(row)
    errors = result.get("node_evidence_errors")
    if errors is not None and not isinstance(errors, list):
        return "invalid", None
    errors = errors or []
    invalid_count = result.get("node_evidence_invalid_count")
    if invalid_count is not None and (
        not isinstance(invalid_count, int)
        or isinstance(invalid_count, bool)
        or invalid_count < 0
    ):
        return "invalid", None
    if invalid_count:
        return "invalid", None

    if "max_reported_nodes_evidence" in result:
        value = result.get("max_reported_nodes_evidence")
        if not (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        ):
            return (
                "invalid"
                if value is not None or any(error != "missing_nodes_value" for error in errors)
                else "missing",
                None,
            )
        if errors:
            return "invalid", None
        return "evidence", value

    if errors:
        return ("invalid" if any(error != "missing_nodes_value" for error in errors) else "missing"), None
    fallback = _max_node_evidence(row)
    if fallback is not None:
        return "evidence", fallback
    result_values = [
        result.get(field)
        for field in ("reported_nodes_at_score", "last_reported_nodes")
    ]
    if any(value is not None for value in result_values):
        return "invalid", None
    return "missing", None


def _all_evidence_m_distribution(rows, *, requested_nodes=None, policy_limit=None):
    values = []
    counts = {
        "attempt_count": len(rows),
        "evidence_count": 0,
        "positive_count": 0,
        "zero_count": 0,
        "missing_count": 0,
        "invalid_count": 0,
    }
    for row in rows:
        state, value = _all_evidence_m_state(row)
        if state == "evidence":
            counts["evidence_count"] += 1
            values.append(value)
            if value > 0:
                counts["positive_count"] += 1
            else:
                counts["zero_count"] += 1
        else:
            counts[f"{state}_count"] += 1

    distribution = _distribution(values)
    if requested_nodes is None:
        above_requested = None
        overrun_values = []
    else:
        above_requested = sum(value > requested_nodes for value in values)
        overrun_values = [value - requested_nodes for value in values if value > requested_nodes]
    if policy_limit is None:
        above_policy = None
    else:
        above_policy = sum(value > policy_limit for value in values)
    max_overrun = max(overrun_values, default=None)
    return {
        **counts,
        "requested_nodes": requested_nodes,
        "policy_limit": policy_limit,
        "distribution": distribution,
        "p50": distribution["p50"],
        "p95": distribution["p95"],
        "p99": distribution["p99"],
        "max": distribution["max"],
        "count_above_requested_nodes": above_requested,
        "count_above_policy_limit": above_policy,
        "max_positive_overrun": max_overrun,
        "max_positive_overrun_rate": (
            max_overrun / requested_nodes
            if max_overrun is not None and isinstance(requested_nodes, int) and requested_nodes > 0
            else None
        ),
        "quantile_definition": NODE_QUANTILE_DEFINITION,
    }


def _position_type_diagnostics(plan, teacher_rows, candidate_rows):
    positions = list(_plan_occurrences(plan).values())
    by_key = {
        (occurrence["game_id"], int(occurrence["ply"])): _position_type(occurrence)
        for occurrence in positions
    }
    counts = {position_type: 0 for position_type in POSITION_CLASSIFICATION_TYPES}
    for position_type in by_key.values():
        counts[position_type] += 1
    by_engine = {
        "teacher": teacher_rows,
        "sekirei": candidate_rows,
    }
    by_type = {}
    for position_type in POSITION_CLASSIFICATION_TYPES:
        keys = {key for key, value in by_key.items() if value == position_type}
        coverage = {}
        for engine_id, rows in by_engine.items():
            selected = [
                row for row in rows
                if (row.get("game_id"), int(row["ply"])) in keys
            ]
            node = positive_node_evidence(_with_engine(selected, engine_id), (engine_id,))[engine_id]
            coverage[engine_id] = {
                "attempt_count": len(selected),
                "status_counts": {
                    status: sum(1 for row in selected if _status(row) == status)
                    for status in sorted(OBSERVATION_STATUSES | TECHNICAL_FAILURE_STATUSES)
                },
                "exact_cp_count": sum(1 for row in selected if _status(row) == "exact_cp"),
                "bound_cp_count": sum(1 for row in selected if _status(row) == "bound_cp"),
                "mate_count": sum(1 for row in selected if _status(row) == "mate"),
                "positive_attempt_count": node["positive_attempt_count"],
                "positive_observation_count": node["positive_observation_count"],
                "has_positive": node["has_positive"],
                "observed_max": node["observed_max"],
            }
        by_type[position_type] = {
            "occurrence_count": counts[position_type],
            "coverage": coverage,
        }
    return {
        "counts": counts,
        "total_occurrences": len(positions),
        "by_type": by_type,
    }


def load_run(run_dir):
    """Load a run manifest, plan, and all immutable attempt records."""
    try:
        manifest, plan, _universe, attempts, _expected, _games = validate_run_artifacts(
            run_dir,
            load_config(),
            require_complete=True,
        )
    except Exception as error:
        # Keep the reporter's historical ValueError-facing API while making
        # the underlying validator the same one used by resume/export.
        if isinstance(error, (OSError, ValueError)):
            raise
        raise ValueError(str(error)) from error
    return manifest, plan, attempts


def _index_attempts(attempts, engine_id):
    """Index observations; repeated pilot rows remain lists, never cached across games."""
    indexed = {}
    for record in attempts:
        if record.get("engine_id") != engine_id:
            continue
        key = (record.get("game_id"), int(record.get("ply")))
        indexed.setdefault(key, []).append(record)
    for values in indexed.values():
        values.sort(key=lambda row: int(row.get("repetition", 1)))
    return indexed


def _first(index, key):
    values = index.get(key, [])
    return values[0] if values else None


def _plan_game_order(plan):
    seen = []
    for occurrence in plan.get("positions", []):
        game_id = occurrence["game_id"]
        if game_id not in seen:
            seen.append(game_id)
    return seen


def _plan_occurrences(plan):
    return {(row["game_id"], int(row["ply"])): row for row in plan.get("positions", [])}


def _teacher_masks(plan, teacher_index):
    masks = {game_id: {"E": [], "B": [], "M": [], "U": []} for game_id in _plan_game_order(plan)}
    for key, occurrence in _plan_occurrences(plan).items():
        game_id, ply = key
        row = _first(teacher_index, key)
        status = _status(row)
        masks[game_id]["U"].append(ply)
        if status == "exact_cp":
            masks[game_id]["E"].append(ply)
        elif status == "bound_cp":
            masks[game_id]["B"].append(ply)
        elif status == "mate":
            masks[game_id]["M"].append(ply)
    return masks


def _cross_table(plan, teacher_index, candidate_index):
    table = {teacher: {candidate: 0 for candidate in CROSS_CATEGORIES} for teacher in CROSS_CATEGORIES}
    for key in _plan_occurrences(plan):
        teacher_row = _first(teacher_index, key)
        candidate_row = _first(candidate_index, key)
        table[_category(teacher_row)][_category(candidate_row)] += 1
    return table


def _operational_type_cross_table(plan, teacher_index, candidate_index):
    table = {
        teacher: {candidate: 0 for candidate in CROSS_CATEGORIES}
        for teacher in ("operational", "failure", "missing")
    }
    for key in _plan_occurrences(plan):
        table[_operational_category(_first(teacher_index, key))][
            _category(_first(candidate_index, key))
        ] += 1
    return table


def _operational_cross_table(plan, teacher_index, candidate_index):
    table = {
        teacher: {candidate: 0 for candidate in OPERATIONAL_CATEGORIES}
        for teacher in OPERATIONAL_CATEGORIES
    }
    for key in _plan_occurrences(plan):
        table[_operational_category(_first(teacher_index, key))][
            _operational_category(_first(candidate_index, key))
        ] += 1
    return table


def _per_game_metrics(game_id, e_plies, teacher_index, candidate_index):
    numerator = 0.0
    exact_count = 0
    errors = []
    rows = []
    for ply in e_plies:
        key = (game_id, ply)
        teacher = _first(teacher_index, key)
        candidate = _first(candidate_index, key)
        teacher_cp = _cp(teacher)
        candidate_cp = _cp(candidate)
        error = None if teacher_cp is None or candidate_cp is None else candidate_cp - teacher_cp
        if error is not None:
            numerator += abs(error)
            errors.append(error)
            exact_count += 1
        rows.append({
            "ply": ply,
            "teacher_status": _status(teacher) if teacher is not None else "missing",
            "teacher_cp_sente": teacher_cp,
            "candidate_status": _status(candidate) if candidate is not None else "missing",
            "candidate_cp_sente": candidate_cp,
            "signed_error_cp": error,
        })
    complete = exact_count == len(e_plies) and len(e_plies) > 0
    return {
        "game_id": game_id,
        "e_count": len(e_plies),
        "candidate_exact_count": exact_count,
        "partial_numerator_abs_cp": numerator,
        "partial_mae_cp": numerator / exact_count if exact_count else None,
        "partial_mae_over_teacher_e_cp": numerator / len(e_plies) if e_plies else None,
        "headline_defined": complete,
        "mae_cp": numerator / len(e_plies) if complete else None,
        "signed_error_mean_cp": sum(errors) / len(errors) if errors else None,
        "max_abs_error_cp": max((abs(value) for value in errors), default=None),
        "rows": rows,
    }


def _accuracy_curve(game_ids, masks, teacher_index, candidate_index, thresholds):
    values = []
    for threshold in thresholds:
        if not _finite_number(threshold) or threshold < 0:
            raise ValueError("accuracy curve thresholds must be finite non-negative numbers")
        per_game = {}
        defined = []
        for game_id in game_ids:
            e_plies = masks[game_id]["E"]
            if not e_plies:
                per_game[game_id] = None
                continue
            count = 0
            for ply in e_plies:
                candidate = _first(candidate_index, (game_id, ply))
                teacher = _first(teacher_index, (game_id, ply))
                candidate_cp, teacher_cp = _cp(candidate), _cp(teacher)
                if candidate_cp is not None and teacher_cp is not None and abs(candidate_cp - teacher_cp) <= threshold:
                    count += 1
            per_game[game_id] = count / len(e_plies)
            defined.append(per_game[game_id])
        values.append({
            "x_cp": threshold,
            "per_game": per_game,
            "equal_game_mean": sum(defined) / len(defined) if len(defined) == len(game_ids) else None,
            "defined_games": len(defined),
        })
    return values


def _bound_diagnostics(plan, teacher_index, candidate_index):
    total = 0
    exact_candidate = 0
    lower_violations = 0
    upper_violations = 0
    unknown = 0
    for key in _plan_occurrences(plan):
        teacher = _first(teacher_index, key)
        if _status(teacher) != "bound_cp":
            continue
        total += 1
        candidate = _first(candidate_index, key)
        candidate_cp = _cp(candidate)
        teacher_bound_cp = _reported_cp(teacher)
        if _status(candidate) != "exact_cp" or candidate_cp is None or teacher_bound_cp is None:
            unknown += 1
            continue
        exact_candidate += 1
        bound = _bound(teacher)
        if bound == "lowerbound" and candidate_cp < teacher_bound_cp:
            lower_violations += 1
        elif bound == "upperbound" and candidate_cp > teacher_bound_cp:
            upper_violations += 1
        else:
            # An exact point inside a one-sided interval is not a violation.
            pass
    return {
        "teacher_bound_count": total,
        "candidate_exact_count": exact_candidate,
        "lower_violation_count": lower_violations,
        "upper_violation_count": upper_violations,
        "unknown_or_nonexact_candidate_count": unknown,
    }


def _mate_diagnostics(plan, teacher_index, candidate_index):
    counts = {name: 0 for name in ("same", "opposite", "nonmate", "unknown")}
    distances = []
    teacher_mates = 0
    for key in _plan_occurrences(plan):
        teacher = _first(teacher_index, key)
        if _status(teacher) != "mate":
            continue
        teacher_mates += 1
        candidate = _first(candidate_index, key)
        if candidate is None or _status(candidate) in FAILURE_STATUSES or _status(candidate) in (None, "no_score"):
            counts["unknown"] += 1
            continue
        if _status(candidate) != "mate":
            counts["nonmate"] += 1
            continue
        teacher_winner, candidate_winner = _winner(teacher), _winner(candidate)
        if teacher_winner is None or candidate_winner is None:
            counts["unknown"] += 1
        elif teacher_winner == candidate_winner:
            counts["same"] += 1
        else:
            counts["opposite"] += 1
        teacher_distance, candidate_distance = _mate_distance(teacher), _mate_distance(candidate)
        if teacher_distance is not None and candidate_distance is not None:
            distances.append(candidate_distance - teacher_distance)
    return {
        "teacher_mate_count": teacher_mates,
        "winner_counts": counts,
        "distance_known_pairs": len(distances),
        "distance_signed_deltas": distances,
        "distance_mean_signed_delta": sum(distances) / len(distances) if distances else None,
        "distance_mean_abs_delta": sum(abs(value) for value in distances) / len(distances) if distances else None,
    }


def _series(plan, teacher_index, candidate_index, *, repetition=1):
    series = {}
    for game_id in _plan_game_order(plan):
        points = []
        for key, occurrence in _plan_occurrences(plan).items():
            if key[0] != game_id:
                continue
            teacher = _first(teacher_index, key)
            candidate = _first(candidate_index, key)
            for engine, row in (("teacher", teacher), ("sekirei", candidate)):
                point_cp = _cp(row)
                if point_cp is None and _status(row) == "bound_cp":
                    point_cp = _reported_cp(row)
                points.append({
                    "engine": engine,
                    "repetition": repetition,
                    "ply": occurrence["ply"],
                    "selection": occurrence.get("selection", "formal"),
                    "position_type": _position_type(occurrence),
                    "status": _status(row) if row is not None else "missing",
                    "cp_sente": point_cp,
                    "bound_sente": _bound(row),
                    "winner_sente": _winner(row),
                    "mate_distance": _mate_distance(row),
                    "reported_nodes_at_score": _value(row, "reported_nodes_at_score"),
                    "last_reported_nodes": _value(row, "last_reported_nodes"),
                    "max_reported_nodes_evidence": _value(row, "max_reported_nodes_evidence"),
                })
        series[game_id] = points
    return series


def _score_single(plan, teacher_rows, candidate_rows, *, accuracy_thresholds=None, config=None, repetition=1):
    """Score one repetition using the fixed teacher-E/B/M masks."""
    config = config or load_config()
    teacher_index = _index_attempts(teacher_rows, "teacher")
    candidate_index = _index_attempts(candidate_rows, "sekirei")
    # Direct callers may pass bare result records with explicit engine_id
    # omitted; accept that useful fixture shape as well.
    if not teacher_index and teacher_rows:
        teacher_index = {(row.get("game_id"), int(row["ply"])): [row] for row in teacher_rows}
    if not candidate_index and candidate_rows:
        candidate_index = {(row.get("game_id"), int(row["ply"])): [row] for row in candidate_rows}
    game_ids = _plan_game_order(plan)
    masks = _teacher_masks(plan, teacher_index)
    per_game = {}
    headline_values = []
    all_errors = []
    for game_id in game_ids:
        metric = _per_game_metrics(game_id, masks[game_id]["E"], teacher_index, candidate_index)
        per_game[game_id] = metric
        if metric["headline_defined"]:
            headline_values.append(metric["mae_cp"])
        for row in metric["rows"]:
            if row["signed_error_cp"] is not None:
                all_errors.append(row["signed_error_cp"])
    if accuracy_thresholds is None:
        unique_errors = sorted({abs(value) for value in all_errors})
        accuracy_thresholds = [0.0, *unique_errors]
    curve = _accuracy_curve(game_ids, masks, teacher_index, candidate_index, accuracy_thresholds)
    exact_total = sum(len(masks[game_id]["E"]) for game_id in game_ids)
    exact_candidate_total = sum(per_game[game_id]["candidate_exact_count"] for game_id in game_ids)
    headline_valid = len(headline_values) == len(game_ids) and len(game_ids) == 5
    series_data = _series(plan, teacher_index, candidate_index, repetition=repetition)
    position_type_diagnostics = _position_type_diagnostics(plan, teacher_rows, candidate_rows)
    node_evidence = _node_evidence_diagnostics(
        teacher_rows,
        candidate_rows,
        requested_nodes=plan.get("requested_nodes"),
        policy_limit=plan.get("max_reported_nodes"),
    )
    graph_y_min, graph_y_max = shared_y_domain({"series": series_data})
    return {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "run_type": plan.get("run_type"),
        "repetitions": plan.get("repetitions"),
        "node_policy": _report_node_policy(plan, config),
        "selection_diagnostics": _selection_diagnostics(plan),
        "hashes": dict(plan.get("hashes", {})),
        "cshogi": plan.get("cshogi"),
        "universe": {
            "game_count": len(game_ids),
            "game_ids": game_ids,
            "total_occurrences": len(plan.get("positions", [])),
        },
        "teacher_masks": masks,
        "headline": {
            "candidate": "sekirei",
            "mae_cp": sum(headline_values) / len(headline_values) if headline_valid else None,
            "valid": headline_valid,
            "reason": None if headline_valid else "candidate lacks exact cp at one or more teacher-E points or a game is incomplete",
            "defined_game_count": len(headline_values),
        },
        "per_game": per_game,
        "e_exact_coverage": {
            "teacher_e_count": exact_total,
            "candidate_exact_count": exact_candidate_total,
            "fraction": exact_candidate_total / exact_total if exact_total else None,
            "per_game": {
                game_id: {
                    "numerator": per_game[game_id]["candidate_exact_count"],
                    "denominator": per_game[game_id]["e_count"],
                    "fraction": per_game[game_id]["candidate_exact_count"] / per_game[game_id]["e_count"]
                    if per_game[game_id]["e_count"] else None,
                }
                for game_id in game_ids
            },
        },
        "diagnostic_error": {
            "overlap_count": len(all_errors),
            "micro_mae_cp": sum(abs(value) for value in all_errors) / len(all_errors) if all_errors else None,
            "mean_signed_error_cp": sum(all_errors) / len(all_errors) if all_errors else None,
            "max_abs_error_cp": max((abs(value) for value in all_errors), default=None),
            "partial_numerator_abs_cp": sum(abs(value) for value in all_errors),
        },
        "accuracy_curve": curve,
        "bound_diagnostic": _bound_diagnostics(plan, teacher_index, candidate_index),
        "mate_diagnostic": _mate_diagnostics(plan, teacher_index, candidate_index),
        "position_type_diagnostics": position_type_diagnostics,
        "node_evidence": node_evidence,
        "all_u_cross_table": _cross_table(plan, teacher_index, candidate_index),
        "all_u_operational_cross_table": _operational_cross_table(plan, teacher_index, candidate_index),
        "all_u_operational_type_cross_table": _operational_type_cross_table(plan, teacher_index, candidate_index),
        "graph": {
            "panel_count": len(game_ids),
            "y_min": graph_y_min,
            "y_max": graph_y_max,
            "x": "after_move_ply",
            "y": "raw_sente_cp",
        },
        "series": series_data,
    }


def _with_engine(rows, engine_id):
    values = []
    for row in rows:
        value = dict(row)
        value.setdefault("engine_id", engine_id)
        values.append(value)
    return values


def _rows_for_repetition(rows, repetition, repetitions):
    if repetitions == 1:
        return list(rows)
    return [row for row in rows if row.get("repetition") == repetition]


def _distribution(values):
    values = sorted(value for value in values if _finite_number(value))
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    def percentile(fraction):
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "count": len(values),
        "min": values[0],
        "p50": percentile(0.5),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": values[-1],
    }


def _repeatability_state(row):
    score_kind = _value(row, "score_kind")
    score_value = _cp(row)
    if score_value is None and _status(row) == "bound_cp":
        score_value = _reported_cp(row)
    return {
        "status": _status(row),
        "score_kind": score_kind,
        "score_value": score_value,
        "score_bound_sente": _bound(row),
        "score_mate_stm": _value(row, "score_mate_stm"),
        "bestmove": _value(row, "bestmove"),
        "bestmove_kind": _value(row, "bestmove_kind"),
        "max_reported_nodes_evidence": _value(row, "max_reported_nodes_evidence"),
    }


REPEATABILITY_FIELDS = (
    "reported_nodes_at_score",
    "last_reported_nodes",
    "max_reported_nodes_evidence",
    "engine_time_ms",
    "wall_go_to_bestmove_ns",
)


def _repeatability_diagnostics(plan, teacher_rows, candidate_rows):
    repetitions = int(plan.get("repetitions", 1))
    positions = list(_plan_occurrences(plan).values())
    by_engine = {
        "teacher": teacher_rows,
        "sekirei": candidate_rows,
    }
    positive_by_engine = _node_evidence_diagnostics(
        teacher_rows,
        candidate_rows,
        requested_nodes=plan.get("requested_nodes"),
        policy_limit=plan.get("max_reported_nodes"),
    )
    engines = {}
    for engine_id, rows in by_engine.items():
        position_rows = {}
        for row in rows:
            position_rows.setdefault((row.get("game_id"), int(row.get("ply"))), []).append(row)
        position_reports = []
        stable_count = 0
        unstable_count = 0
        for occurrence in positions:
            key = (occurrence["game_id"], int(occurrence["ply"]))
            values = sorted(position_rows.get(key, []), key=lambda row: int(row.get("repetition", 1)))
            states = [_repeatability_state(row) for row in values]
            distinct_states = []
            seen_states = set()
            for state in states:
                encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if encoded not in seen_states:
                    seen_states.add(encoded)
                    distinct_states.append(state)
            stable = len(values) == repetitions and len(distinct_states) == 1
            if stable:
                stable_count += 1
            else:
                unstable_count += 1
            position_report = {
                "game_id": occurrence["game_id"],
                "ply": occurrence["ply"],
                "selection": occurrence.get("selection", "formal"),
                "position_type": _position_type(occurrence),
                "expected_repetitions": repetitions,
                "attempt_count": len(values),
                "status_counts": {
                    status: sum(1 for row in values if _status(row) == status)
                    for status in KNOWN_STATUSES
                },
                "distinct_state_count": len(distinct_states),
                "distinct_states": distinct_states,
                "stable": stable,
                "distributions": {
                    field: _distribution([_value(row, field) for row in values])
                    for field in REPEATABILITY_FIELDS
                },
                "max_reported_nodes_evidence": max(
                    (
                        value
                        for row in values
                        for value in (_max_node_evidence(row),)
                        if value is not None
                    ),
                    default=None,
                ),
            }
            position_reports.append(position_report)
        status_counts = {
            status: sum(1 for row in rows if _status(row) == status)
            for status in KNOWN_STATUSES
        }
        field_distributions = {
            field: _distribution([_value(row, field) for row in rows])
            for field in REPEATABILITY_FIELDS
        }
        node_fields = {
            field: max(
                (
                    _value(row, field)
                    for row in rows
                    if isinstance(_value(row, field), int)
                    and not isinstance(_value(row, field), bool)
                    and _value(row, field) >= 0
                ),
                default=None,
            )
            for field in ("reported_nodes_at_score", "last_reported_nodes")
        }
        node_fields["max_reported_nodes_evidence"] = max(
            (value for row in rows for value in (_max_node_evidence(row),) if value is not None),
            default=None,
        )
        node_fields["max_both_fields"] = max(
            (value for value in node_fields.values() if value is not None),
            default=None,
        )
        engines[engine_id] = {
            "expected_attempts": len(positions) * repetitions,
            "attempt_count": len(rows),
            "status_counts": status_counts,
            "stable_position_count": stable_count,
            "unstable_position_count": unstable_count,
            "position_count": len(positions),
            "distributions": field_distributions,
            "max_reported_nodes_evidence": node_fields,
            "positive_node_evidence": positive_by_engine[engine_id],
            "all_evidence_m": positive_by_engine["all_evidence_m"][engine_id],
            "positions": position_reports,
        }
    all_nodes = [
        value
        for rows in by_engine.values()
        for row in rows
        for value in (_max_node_evidence(row),)
        if value is not None
    ]
    return {
        "expected_repetitions": repetitions,
        "expected_positions": len(positions),
        "expected_attempts": len(positions) * repetitions * len(by_engine),
        "attempt_count": sum(len(rows) for rows in by_engine.values()),
        "observed_max_reported_nodes": max(all_nodes, default=None),
        "positive_node_evidence": positive_by_engine,
        "all_evidence_m": positive_by_engine["all_evidence_m"],
        "units": {
            "reported_nodes_at_score": "nodes",
            "last_reported_nodes": "nodes",
            "max_reported_nodes_evidence": "nodes",
            "engine_time_ms": "milliseconds",
            "wall_go_to_bestmove_ns": "nanoseconds",
        },
        "engines": engines,
    }


def _formal_validity(plan, teacher_rows, candidate_rows, *, evidence_validator_passed=False):
    repetitions = int(plan.get("repetitions", 1))
    positions = list(_plan_occurrences(plan).values())
    expected_keys = {
        (engine_id, occurrence["game_id"], int(occurrence["ply"]), repetition)
        for engine_id in ("teacher", "sekirei")
        for occurrence in positions
        for repetition in range(1, repetitions + 1)
    }
    rows_by_engine = {
        "teacher": teacher_rows,
        "sekirei": candidate_rows,
    }
    observed_keys = []
    status_counts = {}
    for engine_id, rows in rows_by_engine.items():
        status_counts[engine_id] = {
            status: sum(1 for row in rows if _status(row) == status)
            for status in KNOWN_STATUSES
        }
        for row in rows:
            observed_keys.append((
                engine_id,
                row.get("game_id"),
                int(row["ply"]) if isinstance(row.get("ply"), int) else row.get("ply"),
                row.get("repetition", 1),
            ))
    duplicate_keys = sorted({key for key in observed_keys if observed_keys.count(key) > 1}, key=str)
    observed_set = set(observed_keys)
    missing_keys = sorted(expected_keys - observed_set, key=str)
    extra_keys = sorted(observed_set - expected_keys, key=str)
    matrix_valid = not duplicate_keys and not missing_keys and not extra_keys and len(observed_keys) == len(expected_keys)
    technical_count = sum(
        1 for rows in rows_by_engine.values() for row in rows if _status(row) in FAILURE_STATUSES
    )
    teacher_reference_categories = {
        category: 0 for category in (*SUCCESS_TYPES, "no_score", "failure", "missing")
    }
    for occurrence in positions:
        for repetition in range(1, repetitions + 1):
            matching = [
                row for row in teacher_rows
                if row.get("game_id") == occurrence["game_id"]
                and row.get("ply") == occurrence["ply"]
                and row.get("repetition", 1) == repetition
            ]
            teacher_reference_categories[_category(matching[0] if matching else None)] += 1
    missing_count = len(missing_keys)
    complete_valid = matrix_valid and technical_count == 0 and missing_count == 0 and evidence_validator_passed
    return {
        "evidence_validator_passed": evidence_validator_passed,
        "exact_attempt_matrix": {
            "expected_count": len(expected_keys),
            "observed_count": len(observed_keys),
            "unique_count": len(observed_set),
            "valid": matrix_valid,
            "missing_count": len(missing_keys),
            "extra_count": len(extra_keys),
            "duplicate_count": len(duplicate_keys),
            "missing": missing_keys,
            "extra": extra_keys,
            "duplicates": duplicate_keys,
        },
        "status_counts": status_counts,
        "technical_failure_count": technical_count,
        "missing_count": missing_count,
        "teacher_reference_categories": teacher_reference_categories,
        "positive_node_evidence": _node_evidence_diagnostics(
            teacher_rows,
            candidate_rows,
            requested_nodes=plan.get("requested_nodes"),
            policy_limit=plan.get("max_reported_nodes"),
        ),
        "complete_evidence_valid": complete_valid,
        "formal_run_valid": complete_valid and plan.get("run_type") == "formal" and repetitions == 1,
    }


def _sum_cross_tables(reports, key, rows, columns=None):
    columns = rows if columns is None else columns
    result = {row: {column: 0 for column in columns} for row in rows}
    for report in reports:
        for row in result:
            for column in result[row]:
                result[row][column] += report.get(key, {}).get(row, {}).get(column, 0)
    return result


def _aggregate_mate_diagnostics(reports):
    counts = {name: 0 for name in ("same", "opposite", "nonmate", "unknown")}
    distances = []
    teacher_mates = 0
    for report in reports:
        value = report.get("mate_diagnostic", {})
        teacher_mates += value.get("teacher_mate_count", 0)
        for key in counts:
            counts[key] += value.get("winner_counts", {}).get(key, 0)
        distances.extend(value.get("distance_signed_deltas", []))
    return {
        "teacher_mate_count": teacher_mates,
        "winner_counts": counts,
        "distance_known_pairs": len(distances),
        "distance_signed_deltas": distances,
        "distance_mean_signed_delta": sum(distances) / len(distances) if distances else None,
        "distance_mean_abs_delta": sum(abs(value) for value in distances) / len(distances) if distances else None,
    }


def _aggregate_bound_diagnostics(reports):
    keys = (
        "teacher_bound_count",
        "candidate_exact_count",
        "lower_violation_count",
        "upper_violation_count",
        "unknown_or_nonexact_candidate_count",
    )
    return {key: sum(report.get("bound_diagnostic", {}).get(key, 0) for report in reports) for key in keys}


def score_observations(plan, teacher_rows, candidate_rows, *, accuracy_thresholds=None, config=None):
    """Score formal runs once, or retain every pilot repetition diagnostically."""
    config = config or load_config()
    repetitions = plan.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("plan repetitions must be a positive integer")
    teacher_rows = _with_engine(teacher_rows, "teacher")
    candidate_rows = _with_engine(candidate_rows, "sekirei")
    repetition_reports = []
    for repetition in range(1, repetitions + 1):
        repetition_reports.append(
            _score_single(
                plan,
                _rows_for_repetition(teacher_rows, repetition, repetitions),
                _rows_for_repetition(candidate_rows, repetition, repetitions),
                accuracy_thresholds=accuracy_thresholds,
                config=config,
                repetition=repetition,
            )
        )
    validity = _formal_validity(plan, teacher_rows, candidate_rows)
    repeatability = _repeatability_diagnostics(plan, teacher_rows, candidate_rows)
    if repetitions == 1:
        report = repetition_reports[0]
        report["repetition_reports"] = repetition_reports
        report["repeatability"] = repeatability
        report["attempt_count"] = repeatability["attempt_count"]
        report["validity"] = validity
        report["run_validity"] = validity
        report["headline"]["formal"] = True
        if not validity["complete_evidence_valid"]:
            report["headline"]["valid"] = False
            report["headline"]["mae_cp"] = None
            report["headline"]["reason"] = "formal run lacks a complete evidence-valid exact attempt matrix or has technical failures"
        return report

    first = repetition_reports[0]
    game_ids = list(first["universe"]["game_ids"])
    series = {game_id: [] for game_id in game_ids}
    for repetition_report in repetition_reports:
        for game_id in game_ids:
            series[game_id].extend(repetition_report.get("series", {}).get(game_id, []))
    coverage_per_game = {}
    teacher_e_count = 0
    candidate_exact_count = 0
    overlap_count = 0
    partial_abs = 0
    signed_numerator = 0
    max_abs = None
    for game_id in game_ids:
        numerator = denominator = 0
        for repetition_report in repetition_reports:
            value = repetition_report["e_exact_coverage"]["per_game"][game_id]
            numerator += value["numerator"]
            denominator += value["denominator"]
        coverage_per_game[game_id] = {
            "numerator": numerator,
            "denominator": denominator,
            "fraction": numerator / denominator if denominator else None,
        }
    for repetition_report in repetition_reports:
        coverage = repetition_report["e_exact_coverage"]
        error = repetition_report["diagnostic_error"]
        teacher_e_count += coverage["teacher_e_count"]
        candidate_exact_count += coverage["candidate_exact_count"]
        overlap_count += error["overlap_count"]
        partial_abs += error["partial_numerator_abs_cp"]
        signed_numerator += (error["mean_signed_error_cp"] or 0) * error["overlap_count"]
        value = error["max_abs_error_cp"]
        if value is not None:
            max_abs = value if max_abs is None else max(max_abs, value)
    aggregate = {
        "schema_version": 1,
        "benchmark_id": first["benchmark_id"],
        "run_type": plan.get("run_type"),
        "repetitions": repetitions,
        "node_policy": _report_node_policy(plan, config),
        "selection_diagnostics": _selection_diagnostics(plan),
        "hashes": dict(first.get("hashes", {})),
        "cshogi": plan.get("cshogi"),
        "universe": dict(first["universe"]),
        "teacher_masks": None,
        "teacher_masks_by_repetition": [report["teacher_masks"] for report in repetition_reports],
        "headline": {
            "candidate": "sekirei",
            "mae_cp": None,
            "valid": False,
            "formal": False,
            "reason": "pilot is a repeatability diagnostic; no single repetition is a formal baseline headline",
            "defined_game_count": 0,
        },
        "per_game": None,
        "per_game_by_repetition": [report["per_game"] for report in repetition_reports],
        "e_exact_coverage": {
            "teacher_e_count": teacher_e_count,
            "candidate_exact_count": candidate_exact_count,
            "fraction": candidate_exact_count / teacher_e_count if teacher_e_count else None,
            "per_game": coverage_per_game,
        },
        "diagnostic_error": {
            "overlap_count": overlap_count,
            "micro_mae_cp": partial_abs / overlap_count if overlap_count else None,
            "mean_signed_error_cp": signed_numerator / overlap_count if overlap_count else None,
            "max_abs_error_cp": max_abs,
            "partial_numerator_abs_cp": partial_abs,
        },
        "accuracy_curve": None,
        "accuracy_curve_by_repetition": [report["accuracy_curve"] for report in repetition_reports],
        "bound_diagnostic": _aggregate_bound_diagnostics(repetition_reports),
        "mate_diagnostic": _aggregate_mate_diagnostics(repetition_reports),
        "position_type_diagnostics": _position_type_diagnostics(plan, teacher_rows, candidate_rows),
        "node_evidence": _node_evidence_diagnostics(
            teacher_rows,
            candidate_rows,
            requested_nodes=plan.get("requested_nodes"),
            policy_limit=plan.get("max_reported_nodes"),
        ),
        "all_u_cross_table": _sum_cross_tables(repetition_reports, "all_u_cross_table", CROSS_CATEGORIES),
        "all_u_operational_cross_table": _sum_cross_tables(
            repetition_reports, "all_u_operational_cross_table", OPERATIONAL_CATEGORIES
        ),
        "all_u_operational_type_cross_table": _sum_cross_tables(
            repetition_reports,
            "all_u_operational_type_cross_table",
            ("operational", "failure", "missing"),
            CROSS_CATEGORIES,
        ),
        "graph": {
            "panel_count": len(game_ids),
            "y_min": shared_y_domain({"series": series})[0],
            "y_max": shared_y_domain({"series": series})[1],
            "x": "after_move_ply",
            "y": "raw_sente_cp",
            "pilot_repetitions": "all repetitions retained; lines are per repetition",
        },
        "series": series,
        "repetition_reports": repetition_reports,
        "repeatability": repeatability,
        "pilot_repeatability": repeatability,
        "validity": validity,
        "run_validity": validity,
    }
    aggregate["attempt_count"] = repeatability["attempt_count"]
    return aggregate


def _category_counts(points):
    counts = {category: 0 for category in (*SUCCESS_TYPES, "no_score", "failure", "missing")}
    for point in points:
        status = point["status"]
        category = status if status in counts else "failure"
        counts[category] += 1
    return counts


def _nice_limit(value):
    if value <= 0:
        return 1.0
    # Keep a linear, symmetric domain while avoiding overly precise axes.
    exponent = math.floor(math.log10(value))
    base = 10 ** exponent
    for factor in (1, 2, 5, 10):
        candidate = factor * base
        if candidate >= value:
            return float(candidate)
    return float(10 * base)


def shared_y_domain(report):
    """Return the common symmetric raw-cp domain used by every panel."""
    values = []
    for points in report.get("series", {}).values():
        values.extend(
            point["cp_sente"]
            for point in points
            if point.get("cp_sente") is not None
        )
    limit = _nice_limit(max([abs(value) for value in values] or [1]))
    return -limit, limit


def render_svg(report, *, public=False):
    """Render five fixed panels with one shared symmetric raw-cp domain."""
    series = report.get("series", {})
    game_ids = list(report.get("universe", {}).get("game_ids", series.keys()))
    max_plies = {}
    for game_id in game_ids:
        max_plies[game_id] = 1
        for point in series.get(game_id, []):
            max_plies[game_id] = max(max_plies[game_id], int(point["ply"]))
    y_min, y_max = shared_y_domain(report)
    limit = y_max
    panel_width, panel_height = 260, 300
    margin_left, margin_right, margin_top, margin_bottom = 42, 12, 34, 30
    plot_width = panel_width - margin_left - margin_right
    plot_height = panel_height - margin_top - margin_bottom
    width = panel_width * max(1, len(game_ids))
    height = panel_height

    def x_pos(game_id, ply):
        maximum = max_plies[game_id]
        return margin_left + (ply - 1) * plot_width / max(1, maximum - 1)

    def y_pos(value):
        return margin_top + (limit - value) * plot_height / (2 * limit)

    def label(game_id, index):
        return f"Game {index + 1:02d}" if public else game_id

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" data-panel-count="{len(game_ids)}" data-y-min="{y_min:g}" data-y-max="{y_max:g}">',
        "<title>Development 1M-node evaluation comparison</title>",
        "<desc>Five after-move-ply panels, shared symmetric linear raw sente centipawn domain.</desc>",
        "<style>.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.teacher{fill:none;stroke:#075985;stroke-width:1.4}.candidate{fill:none;stroke:#b91c1c;stroke-width:1.4}.bound{stroke:#7c3aed;stroke-width:1.1}.gap{fill:#6b7280}.mate{fill:#111827}.text{font-family:monospace;font-size:9px;fill:#111827}</style>",
    ]
    for index, game_id in enumerate(game_ids):
        offset = index * panel_width
        out.append(f'<g transform="translate({offset},0)">')
        out.append(f'<text class="text" x="{margin_left}" y="14">{html.escape(label(game_id, index))}</text>')
        out.append(f'<text class="text" x="{margin_left}" y="25">y ±{limit:g} cp (sente)</text>')
        for fraction in (0, 0.5, 1):
            value = limit - 2 * limit * fraction
            y = margin_top + plot_height * fraction
            out.append(f'<line class="grid" x1="{margin_left}" y1="{y:.3f}" x2="{margin_left + plot_width}" y2="{y:.3f}"/>')
            out.append(f'<text class="text" x="2" y="{y + 3:.3f}">{value:g}</text>')
        out.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}"/>')
        out.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}"/>')
        game_points = series.get(game_id, [])
        for engine in ("teacher", "sekirei"):
            all_engine_points = [point for point in game_points if point["engine"] == engine]
            color_class = "teacher" if engine == "teacher" else "candidate"
            repetitions = sorted({point.get("repetition", 1) for point in all_engine_points}) or [1]
            for repetition in repetitions:
                points = [point for point in all_engine_points if point.get("repetition", 1) == repetition]
                run = []
                previous_ply = None
                for point in points:
                    if point["status"] == "exact_cp" and point.get("cp_sente") is not None:
                        if previous_ply is not None and point["ply"] != previous_ply + 1:
                            if len(run) >= 2:
                                coordinates = " ".join(f"{x_pos(game_id, p['ply']):.3f},{y_pos(p['cp_sente']):.3f}" for p in run)
                                out.append(f'<polyline class="{color_class}" data-repetition="{repetition}" points="{coordinates}"/>')
                            run = []
                        run.append(point)
                        previous_ply = point["ply"]
                    else:
                        if len(run) >= 2:
                            coordinates = " ".join(f"{x_pos(game_id, p['ply']):.3f},{y_pos(p['cp_sente']):.3f}" for p in run)
                            out.append(f'<polyline class="{color_class}" data-repetition="{repetition}" points="{coordinates}"/>')
                        run = []
                        previous_ply = None
                if len(run) >= 2:
                    coordinates = " ".join(f"{x_pos(game_id, p['ply']):.3f},{y_pos(p['cp_sente']):.3f}" for p in run)
                    out.append(f'<polyline class="{color_class}" data-repetition="{repetition}" points="{coordinates}"/>')
                # An exact observation is always visible, even when it is an
                # isolated point that cannot be connected by a line.
                for point in points:
                    if point["status"] == "exact_cp" and point.get("cp_sente") is not None:
                        x = x_pos(game_id, point["ply"])
                        y = y_pos(point["cp_sente"])
                        position_type = html.escape(str(point.get("position_type", "normal")), quote=True)
                        out.append(f'<circle class="exact" data-engine="{engine}" data-repetition="{repetition}" data-position-type="{position_type}" cx="{x:.3f}" cy="{y:.3f}" r="2"/>')
                for point in points:
                    x = x_pos(game_id, point["ply"])
                    status = point["status"]
                    position_type = html.escape(str(point.get("position_type", "normal")), quote=True)
                    if status == "bound_cp" and point.get("cp_sente") is not None:
                        y = y_pos(point["cp_sente"])
                        direction = -8 if point.get("bound_sente") == "lowerbound" else 8
                        out.append(f'<line class="bound" data-position-type="{position_type}" x1="{x:.3f}" y1="{y:.3f}" x2="{x:.3f}" y2="{y + direction:.3f}" marker-end="url(#arrow)"/>')
                    elif status == "mate":
                        winner = point.get("winner_sente")
                        y = y_pos(limit * 0.96 if winner == "black" else -limit * 0.96 if winner == "white" else 0)
                        out.append(f'<path class="mate" data-position-type="{position_type}" d="M {x - 3:.3f} {y + 3:.3f} L {x + 3:.3f} {y + 3:.3f} L {x:.3f} {y - 3:.3f} Z"/>')
                    elif status not in ("exact_cp", "bound_cp"):
                        # A gap marker is outside the data line and is not a fake
                        # zero score; missing values remain missing in the graph.
                        y = margin_top + plot_height + 12
                        out.append(f'<circle class="gap" data-position-type="{position_type}" cx="{x:.3f}" cy="{y:.3f}" r="1.8"/>')
            counts = _category_counts(all_engine_points)
            legend_y = panel_height - 16 if engine == "teacher" else panel_height - 6
            summary = ", ".join(f"{key}={value}" for key, value in counts.items())
            out.append(f'<text class="text" x="{margin_left}" y="{legend_y}">{engine}: {summary}</text>')
        out.append("</g>")
    headline = _public_headline(report.get("headline")) if public else report.get("headline", {})
    valid_text = "valid" if headline.get("valid") else "incomplete"
    out.insert(3, '<defs><marker id="arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 z" fill="#7c3aed"/></marker></defs>')
    out.append(f'<text class="text" x="4" y="{height - 1}">candidate headline MAE: {valid_text}</text>')
    out.append("</svg>\n")
    return "".join(out)


def _summary_markdown(report, *, public=False):
    headline = _public_headline(report.get("headline")) if public else report.get("headline", {})
    coverage = report.get("e_exact_coverage", {})
    error = report.get("diagnostic_error", {})
    bound = report.get("bound_diagnostic", {})
    mate = report.get("mate_diagnostic", {})
    table = report.get("all_u_cross_table", {})
    operational_cross = report.get("all_u_operational_cross_table", {})
    operational_table = report.get("all_u_operational_type_cross_table", {})
    validity = report.get("validity", {})
    repeatability = report.get("repeatability", {})
    position_counts = (report.get("position_type_diagnostics") or {}).get("counts", {})
    node_evidence = report.get("node_evidence") or repeatability.get("positive_node_evidence") or {}
    position_text = ", ".join(
        f"{position_type}={position_counts.get(position_type, 0)}"
        for position_type in POSITION_CLASSIFICATION_TYPES
    )
    selection_counts = (report.get("selection_diagnostics") or {}).get("counts", {})
    selection_text = ", ".join(
        f"{selection}={selection_counts.get(selection, 0)}"
        for selection in SELECTION_TYPES
    )
    node_text = "; ".join(
        f"{engine_id}: {value.get('positive_observation_count', 0)} positive observations / "
        f"{value.get('positive_attempt_count', 0)} attempts (has_positive={value.get('has_positive', False)})"
        for engine_id, value in (("teacher", node_evidence.get("teacher", {})), ("sekirei", node_evidence.get("sekirei", {})))
    )
    all_evidence_m = node_evidence.get("all_evidence_m", {})
    m_text = "; ".join(
        f"{engine_id}: evidence={value.get('evidence_count', 0)} "
        f"positive={value.get('positive_count', 0)} zero={value.get('zero_count', 0)} "
        f"missing={value.get('missing_count', 0)} invalid={value.get('invalid_count', 0)} "
        f"p50={value.get('p50')} p95={value.get('p95')} p99={value.get('p99')} max={value.get('max')} "
        f">N={value.get('count_above_requested_nodes')} >C={value.get('count_above_policy_limit')} "
        f"max_overrun={value.get('max_positive_overrun')} "
        f"rate={value.get('max_positive_overrun_rate')}"
        for engine_id, value in (("teacher", all_evidence_m.get("teacher", {})), ("sekirei", all_evidence_m.get("sekirei", {})))
    )
    lines = [
        "# Development 1M-node baseline validation",
        "",
        "This is a validation report for the fixed development split; it is not the final holdout.",
        "",
        f"- run type: `{report.get('run_type')}`",
        f"- games / occurrences: {report.get('universe', {}).get('game_count')} / {report.get('universe', {}).get('total_occurrences')}",
        f"- cshogi conversion check: {_public_cshogi(report.get('cshogi')) if public else report.get('cshogi')}",
        f"- teacher-E exact points: {coverage.get('teacher_e_count')}",
        f"- candidate exact coverage on E: {coverage.get('candidate_exact_count')} / {coverage.get('teacher_e_count')} ({coverage.get('fraction')})",
        f"- candidate headline MAE: {headline.get('mae_cp')} cp ({'valid' if headline.get('valid') else 'incomplete'})",
        f"- formal headline eligible: {headline.get('formal')} / evidence-valid run: {validity.get('complete_evidence_valid')}",
        f"- exact attempt matrix: {validity.get('exact_attempt_matrix', {}).get('observed_count')} / {validity.get('exact_attempt_matrix', {}).get('expected_count')} (missing {validity.get('missing_count')}, technical failures {validity.get('technical_failure_count')})",
        f"- retained repetitions / attempts: {repeatability.get('expected_repetitions')} / {repeatability.get('attempt_count')}; observed max node evidence: {repeatability.get('observed_max_reported_nodes')}",
        f"- position types: {position_text}",
        f"- position selections: {selection_text}",
        f"- positive node evidence: {node_text}",
        f"- all-evidence M distribution: {m_text}",
        f"- M quantiles: {NODE_QUANTILE_DEFINITION}",
        f"- diagnostic overlap micro MAE: {error.get('micro_mae_cp')} cp; mean signed error: {error.get('mean_signed_error_cp')} cp; max absolute error: {error.get('max_abs_error_cp')} cp",
        "",
        "## Fixed masks and diagnostics",
        "",
        "The headline is defined only when every teacher-E point in all five games has a candidate exact cp result. Bounds and mates are retained as diagnostics and are not converted to cp.",
        "",
        f"- bound interval checks: {bound.get('teacher_bound_count')} teacher-B points; lower violations {bound.get('lower_violation_count')}; upper violations {bound.get('upper_violation_count')}",
        f"- mate winner counts: {mate.get('winner_counts')}; known distance pairs {mate.get('distance_known_pairs')}",
        "",
        "## All-U type cross-table",
        "",
        "| teacher \\ candidate | exact_cp | bound_cp | mate | no_score | failure | missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for teacher in CROSS_CATEGORIES:
        row = table.get(teacher, {})
        lines.append("| " + teacher + " | " + " | ".join(str(row.get(candidate, 0)) for candidate in CROSS_CATEGORIES) + " |")
    lines.extend([
        "",
        "## All-U operational/type cross-table",
        "",
        "| teacher operational state \\ candidate type | exact_cp | bound_cp | mate | no_score | failure | missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for teacher in ("operational", "failure", "missing"):
        row = operational_table.get(teacher, {})
        lines.append("| " + teacher + " | " + " | ".join(str(row.get(candidate, 0)) for candidate in CROSS_CATEGORIES) + " |")
    lines.extend([
        "",
        "| teacher operational state \\ candidate operational state | operational | failure | missing |",
        "| --- | ---: | ---: | ---: |",
    ])
    for teacher in OPERATIONAL_CATEGORIES:
        row = operational_cross.get(teacher, {})
        lines.append("| " + teacher + " | " + " | ".join(str(row.get(candidate, 0)) for candidate in OPERATIONAL_CATEGORIES) + " |")
    if report.get("hashes"):
        lines.extend(["", "## Reproducibility hashes", ""])
        for key, value in sorted(report["hashes"].items()):
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def write_local_reports(output_dir, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "report.json", report)
    atomic_write(output_dir / "summary.md", _summary_markdown(report))
    atomic_write(output_dir / "evaluation.svg", render_svg(report))
    with (output_dir / "observations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "game_id", "ply", "repetition", "selection", "position_type",
            "teacher_status", "teacher_cp_sente",
            "teacher_reported_nodes_at_score", "teacher_last_reported_nodes",
            "teacher_max_reported_nodes_evidence",
            "candidate_status", "candidate_cp_sente",
            "candidate_reported_nodes_at_score", "candidate_last_reported_nodes",
            "candidate_max_reported_nodes_evidence", "signed_error_cp",
        ))
        for game_id in report.get("universe", {}).get("game_ids", []):
            by_ply = {}
            for point in report.get("series", {}).get(game_id, []):
                key = (point["ply"], point.get("repetition", 1))
                by_ply.setdefault(key, {})[point["engine"]] = point
            for ply, repetition in sorted(by_ply):
                teacher = by_ply[(ply, repetition)].get("teacher", {})
                candidate = by_ply[(ply, repetition)].get("sekirei", {})
                teacher_cp = teacher.get("cp_sente")
                candidate_cp = candidate.get("cp_sente")
                signed = candidate_cp - teacher_cp if (
                    teacher.get("status") == "exact_cp"
                    and candidate.get("status") == "exact_cp"
                    and teacher_cp is not None
                    and candidate_cp is not None
                ) else None
                writer.writerow((
                    game_id,
                    ply,
                    repetition,
                    teacher.get("selection") or candidate.get("selection") or "formal",
                    teacher.get("position_type") or candidate.get("position_type") or "normal",
                    teacher.get("status", "missing"),
                    teacher_cp,
                    teacher.get("reported_nodes_at_score"),
                    teacher.get("last_reported_nodes"),
                    teacher.get("max_reported_nodes_evidence"),
                    candidate.get("status", "missing"),
                    candidate_cp,
                    candidate.get("reported_nodes_at_score"),
                    candidate.get("last_reported_nodes"),
                    candidate.get("max_reported_nodes_evidence"),
                    signed,
                ))
    return output_dir


def _public_report(report):
    """Strip per-position data and identifiers before export."""
    public = copy.deepcopy({
        "schema_version": 1,
        "report": "development-baseline-aggregate-validation",
        "run_type": report.get("run_type"),
        "node_policy": report.get("node_policy"),
        "selection_diagnostics": report.get("selection_diagnostics"),
        "game_count": report.get("universe", {}).get("game_count"),
        "occurrence_count": report.get("universe", {}).get("total_occurrences"),
        "headline": _public_headline(report.get("headline")),
        "e_exact_coverage": report.get("e_exact_coverage"),
        "diagnostic_error": report.get("diagnostic_error"),
        "accuracy_curve": report.get("accuracy_curve"),
        "bound_diagnostic": report.get("bound_diagnostic"),
        "mate_diagnostic": report.get("mate_diagnostic"),
        "position_type_diagnostics": report.get("position_type_diagnostics"),
        "node_evidence": report.get("node_evidence"),
        "validity": _public_validity(report.get("validity")),
        "repeatability": _public_repeatability(report.get("repeatability")),
        "all_u_cross_table": report.get("all_u_cross_table"),
        "all_u_operational_cross_table": report.get("all_u_operational_cross_table"),
        "all_u_operational_type_cross_table": report.get("all_u_operational_type_cross_table"),
        "hashes": report.get("hashes"),
        "cshogi": _public_cshogi(report.get("cshogi")),
    })
    # Replace internal game keys in per-game coverage with ordinal labels.
    coverage = public.get("e_exact_coverage", {})
    if isinstance(coverage, dict) and isinstance(coverage.get("per_game"), dict):
        coverage["per_game"] = {
            f"game-{index + 1:02d}": value
            for index, value in enumerate(coverage["per_game"].values())
        }
    curve = public.get("accuracy_curve") or []
    for item in curve:
        if isinstance(item.get("per_game"), dict):
            item["per_game"] = {
                f"game-{index + 1:02d}": value
                for index, value in enumerate(item["per_game"].values())
            }
    return public


def _public_cshogi(value):
    """Keep only the typed, aggregate cshogi fields in public JSON."""
    if not isinstance(value, dict):
        return None
    result = {}
    if isinstance(value.get("available"), bool):
        result["available"] = value["available"]
    if isinstance(value.get("version"), str):
        result["version"] = value["version"]
    for key in ("checked_legal_moves", "checked_occurrences"):
        if isinstance(value.get(key), int) and not isinstance(value.get(key), bool) and value[key] >= 0:
            result[key] = value[key]
    counts = value.get("classification_counts")
    if isinstance(counts, dict):
        result["classification_counts"] = {
            position_type: counts.get(position_type, 0)
            for position_type in POSITION_CLASSIFICATION_TYPES
            if isinstance(counts.get(position_type, 0), int)
            and not isinstance(counts.get(position_type, 0), bool)
            and counts.get(position_type, 0) >= 0
        }
    if isinstance(value.get("matched"), bool):
        result["matched"] = value["matched"]
    return result or None


def _public_headline(value):
    if not isinstance(value, dict):
        return None
    return {
        key: value.get(key)
        for key in ("candidate", "mae_cp", "valid", "formal", "defined_game_count")
        if key in value
    }


def _public_validity(value):
    if not isinstance(value, dict):
        return None
    result = {
        key: value.get(key)
        for key in (
            "evidence_validator_passed",
            "technical_failure_count",
            "missing_count",
            "complete_evidence_valid",
            "formal_run_valid",
            "positive_node_evidence",
        )
        if key in value
    }
    status_counts = value.get("status_counts")
    if isinstance(status_counts, dict):
        result["status_counts"] = {
            engine_id: {
                status: counts.get(status, 0)
                for status in KNOWN_STATUSES
            }
            for engine_id, counts in status_counts.items()
            if engine_id in ("teacher", "sekirei") and isinstance(counts, dict)
        }
    teacher_categories = value.get("teacher_reference_categories")
    if isinstance(teacher_categories, dict):
        result["teacher_reference_categories"] = {
            category: teacher_categories.get(category, 0)
            for category in CROSS_CATEGORIES
        }
    matrix = value.get("exact_attempt_matrix")
    if isinstance(matrix, dict):
        result["exact_attempt_matrix"] = {
            key: matrix.get(key)
            for key in ("expected_count", "observed_count", "unique_count", "valid", "missing_count", "extra_count", "duplicate_count")
            if key in matrix
        }
    return result


def _public_repeatability(value):
    if not isinstance(value, dict):
        return None
    result = {
        key: value.get(key)
        for key in (
            "expected_repetitions",
            "expected_positions",
            "expected_attempts",
            "attempt_count",
            "observed_max_reported_nodes",
            "all_evidence_m",
            "units",
        )
        if value.get(key) is not None
    }
    engines = {}
    for engine_id, engine in (value.get("engines") or {}).items():
        if not isinstance(engine, dict):
            continue
        safe_engine = {
            key: engine.get(key)
            for key in (
                "expected_attempts",
                "attempt_count",
                "status_counts",
                "stable_position_count",
                "unstable_position_count",
                "position_count",
                "distributions",
                "max_reported_nodes_evidence",
                "positive_node_evidence",
                "all_evidence_m",
            )
            if engine.get(key) is not None
        }
        if engine_id in ("teacher", "sekirei"):
            engines[engine_id] = safe_engine
    result["engines"] = engines
    return result


def _known_source_game_ids():
    """Load every development source ID from tracked provenance metadata."""
    entries = development_manifest_entries()
    return tuple(sorted(
        item["source_game_id"]
        for item in entries.values()
        if isinstance(item.get("source_game_id"), str) and item["source_game_id"]
    ))


def review_public_text(text, *, source_game_ids=()):
    """Fail closed if a candidate public export contains local identifiers."""
    forbidden = ["/home/", "position startpos moves", "EvalDir", "model_path", "raw_log", "source_game_id"]
    username = os.environ.get("USER")
    if username:
        forbidden.append(username)
    forbidden.extend(str(value) for value in source_game_ids if value)
    for value in forbidden:
        if value and value in text:
            raise ValueError(f"public export contains forbidden text: {value}")
    if re.search(r"(?:^|[\s=\"'])/[A-Za-z0-9_.-]+(?:/|$)", text):
        raise ValueError("public export contains an absolute path")
    return True


def export_public(output_dir, report, *, source_game_ids=None):
    """Write redacted Markdown/SVG plus content hashes for later review."""
    if report.get("run") and report["run"].get("status") != "complete":
        raise ValueError("refusing to export an incomplete run")
    if report.get("run") and not report.get("validity", {}).get("evidence_validator_passed"):
        raise ValueError("refusing to export a run that did not pass strict evidence validation")
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"public export output is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError("public export output directory must be empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    source_game_ids = _known_source_game_ids() if source_game_ids is None else tuple(source_game_ids)
    public_report = _public_report(report)
    markdown = _summary_markdown(report, public=True)
    svg = render_svg(report, public=True)
    public_json = json.dumps(public_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    review_public_text(markdown, source_game_ids=source_game_ids)
    review_public_text(svg, source_game_ids=source_game_ids)
    review_public_text(public_json, source_game_ids=source_game_ids)
    atomic_write(output_dir / "validation.md", markdown)
    atomic_write(output_dir / "reviewed.svg", svg)
    atomic_write(output_dir / "validation.json", public_json)
    hashes = {
        "schema_version": 1,
        "files": {
            "validation.md": digest_bytes(markdown.encode("utf-8")),
            "reviewed.svg": digest_bytes(svg.encode("utf-8")),
            "validation.json": digest_bytes(public_json.encode("utf-8")),
        },
    }
    hash_text = json.dumps(hashes, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    review_public_text(hash_text, source_game_ids=source_game_ids)
    atomic_write(output_dir / "manifest.json", hash_text)
    return hashes


def report_from_run(run_dir, output_dir=None, *, accuracy_thresholds=None):
    manifest, plan, attempts = load_run(run_dir)
    teacher_rows = [row for row in attempts if row.get("engine_id") == "teacher"]
    candidate_rows = [row for row in attempts if row.get("engine_id") == "sekirei"]
    report = score_observations(
        plan,
        teacher_rows,
        candidate_rows,
        accuracy_thresholds=accuracy_thresholds,
    )
    validity = report.get("validity")
    if isinstance(validity, dict):
        validity["evidence_validator_passed"] = True
        matrix = validity.get("exact_attempt_matrix", {})
        validity["complete_evidence_valid"] = bool(
            matrix.get("valid")
            and validity.get("technical_failure_count") == 0
            and validity.get("missing_count") == 0
        )
        validity["formal_run_valid"] = bool(
            validity["complete_evidence_valid"]
            and plan.get("run_type") == "formal"
            and plan.get("repetitions") == 1
        )
        if plan.get("run_type") == "formal" and not validity["complete_evidence_valid"]:
            report["headline"]["valid"] = False
            report["headline"]["mae_cp"] = None
            report["headline"]["reason"] = "formal run contains a technical failure or lacks an exact evidence-valid attempt matrix"
    report["run"] = {
        "run_id": manifest.get("run_id"),
        "fingerprint": manifest.get("fingerprint"),
        "status": manifest.get("status"),
    }
    if output_dir is not None:
        write_local_reports(output_dir, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("report", "export"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds")
    args = parser.parse_args(argv)
    if args.action == "export":
        # Export is an aggregate-only boundary: report_from_run validates and
        # scores in memory, and no local/per-position tree is created.
        report = report_from_run(args.run_dir, None, accuracy_thresholds=args.thresholds)
        hashes = export_public(args.output, report)
        print(json.dumps(hashes, indent=2, ensure_ascii=False))
    else:
        report = report_from_run(args.run_dir, args.output, accuracy_thresholds=args.thresholds)
        print(json.dumps({key: report[key] for key in ("headline", "e_exact_coverage", "bound_diagnostic", "mate_diagnostic")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
