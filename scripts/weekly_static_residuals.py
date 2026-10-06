#!/usr/bin/env python3
"""Aggregate already-proven static STM residuals; never launch a model or search.

Only a complete weekly technical proof may enter the physical-file path. Its
dedicated consumer is repeated under the recorded locks before any aggregation.
Both native_core_cp and score_cp are side-to-move cp: fixed Board passes STM to
Accumulator::evaluate_with, and the pack decoder preserves teacher_eval_cp_stm.
The signed error is candidate minus teacher; positive means higher for STM.
Old-training absence means nonoverlap with that prior dataset, not untrained.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import weekly_nonlinear_proof as proof
from benchmark import nonblocking_lock
from train_cpu import termination_guard
from weekly_nonlinear_post_training import PhysicalBytes, require
from weekly_nonlinear_preflight import validate_source_binding


PLY_BANDS = ((16, 32), (32, 64), (64, 96), (96, 128), (128, 192), (192, None))
TEACHER_BANDS = ((0, 300), (300, 1000), (1000, 3000), (3000, 10000), (10000, 30000))
TOP_PERCENT = (1, 5, 10, 20)
POLICY = {
    "schema": "sekirei.weekly-static-residual-policy.v1",
    "error": "native_core_cp_stm - teacher_score_cp_stm",
    "positive_error": "candidate evaluates side to move higher than teacher",
    "ply_bands_left_closed_right_open": [list(x) for x in PLY_BANDS],
    "absolute_teacher_cp_bands_left_closed_right_open": [list(x) for x in TEACHER_BANDS],
    "previous_train_board_key": "first three SFEN fields: board, STM, hands; excludes ply",
    "previous_train_absence": "nonoverlap with previous train; not a claim of being untrained",
    "top_train_squared_error_percent": list(TOP_PERCENT),
    "top_row_count": "ceil(train_count * integer_percent / 100)",
    "top_order": "descending integer squared error; ascending original row index for ties",
    "zero_total_squared_error_share": None,
    "scope": "static native forward on pack train/fixed holdout; no development/final or search",
}


def band(value, bands):
    for low, high in bands:
        if low <= value and (high is None or value < high):
            return f"[{low},{'infinity' if high is None else high})"
    raise ValueError("value outside fixed band domain")


def board_key(sfen, gate):
    gate.material(sfen)  # Strict SFEN and fixed material interpretation.
    return " ".join(sfen.split(" ")[:3])


def previous_train_boards(raw, count, gate):
    rows = gate.rows(raw)
    require(len(rows) == count and count > 0, "previous train row count differs")
    keys = [board_key(row["sfen"], gate) for row in rows]
    require(len(set(keys)) == count, "duplicate previous train board")
    return set(keys)


def totals(errors, denominator):
    count = len(errors)
    require(type(denominator) is int and denominator > 0 and count <= denominator,
            "valid complete split denominator required")
    absolute, signed, squared = sum(abs(e) for e in errors), sum(errors), sum(e * e for e in errors)
    return {"count": count, "split_denominator": denominator, "row_fraction": count / denominator,
            "absolute_error_sum_cp": absolute, "signed_error_sum_cp": signed,
            "squared_error_sum_cp2": squared,
            "mae_cp": absolute / count if count else None,
            "mean_signed_error_cp": signed / count if count else None,
            "mean_squared_error_cp2": squared / count if count else None}


def concentration(errors):
    require(errors and all(type(e) is int for e in errors), "integer nonempty train errors required")
    ranked = sorted(enumerate(errors), key=lambda item: (-item[1] * item[1], item[0]))
    denominator = sum(e * e for e in errors)
    result = []
    for percent in TOP_PERCENT:
        count = (len(errors) * percent + 99) // 100
        chosen = ranked[:count]
        squared = sum(e * e for _, e in chosen)
        result.append({"top_percent": percent, "count": count, "train_denominator": len(errors),
            "squared_error_sum_cp2": squared, "train_squared_error_sum_cp2": denominator,
            "squared_error_share": squared / denominator if denominator else None,
            "cutoff_squared_error_cp2": chosen[-1][1] * chosen[-1][1]})
    return result


def analyze_split(manifest, position_raw, label_raw, core_raw, old_boards, split, count, gate):
    """Pure bounded fixture-compatible analysis; production supplies fixed counts."""
    require(split in {"train", "holdout"} and type(count) is int and count > 0, "explicit split/count required")
    positions, labels = gate.rows(position_raw), gate.rows(label_raw)
    raw_core = core_raw.splitlines()
    require(len(positions) == len(labels) == len(raw_core) == count, "complete position/label/core counts differ")
    games = {}
    packs = set()
    for game in manifest["games"]:
        gid, pack = game["game_id"], game["pack_sha256"]
        gate.sha(gid); gate.sha(pack)
        require(gid not in games and game["split"] in {"train", "holdout"}, "duplicate or invalid manifest game")
        games[gid] = game
        packs.add(pack)
    targets = {}
    for label in labels:
        gate.obj(label, {"sfen", "score_cp", "label_depth", "teacher_identity"})
        gate.fixed(label, {"label_depth": 0, "teacher_identity": gate.TEACHER})
        require(type(label["score_cp"]) is int and abs(label["score_cp"]) < 30000,
                "ordinary integer STM teacher required")
        board_key(label["sfen"], gate)
        require(label["sfen"] not in targets, "duplicate teacher label")
        targets[label["sfen"]] = label["score_cp"]
    grouped = {"pack": {p: [] for p in sorted(packs)},
        "ply": {band(low, PLY_BANDS): [] for low, _ in PLY_BANDS},
        "absolute_teacher_cp": {band(low, TEACHER_BANDS): [] for low, _ in TEACHER_BANDS},
        "previous_train_board_overlap": {"present": [], "absent": []},
        "side_to_move": {"black": [], "white": []}}
    errors, sfens, boards, locations = [], set(), set(), set()
    for index, (position, raw) in enumerate(zip(positions, raw_core)):
        gate.obj(position, {"schema_version", "sfen", "source", "tags"})
        gate.fixed(position, {"schema_version": 1})
        gate.obj(position["source"], {"kind", "path", "ply"})
        gate.fixed(position["source"], {"kind": "gensfen-pack"})
        sfen, source = position["sfen"], position["source"]
        key = board_key(sfen, gate)
        require(sfen not in sfens and key not in boards and sfen in targets, "duplicate board or missing exact label")
        sfens.add(sfen); boards.add(key)
        require(type(source["path"]) is str and source["path"] in games
                and games[source["path"]]["split"] == split, "position game/split differs")
        ply = source["ply"]
        require(type(ply) is int and ply >= 16 and sfen.split(" ")[3] == str(ply), "position/SFEN ply differs")
        location = (source["path"], ply)
        require(location not in locations, "duplicate source location")
        locations.add(location)
        side = "black" if sfen.split(" ")[1] == "b" else "white"
        gate.obj(position["tags"], {"side_to_move", "phase"})
        gate.fixed(position["tags"], {"side_to_move": side, "phase": "middlegame"})
        row = gate.probe.core_row(raw, index, gate.material(sfen))
        teacher = targets[sfen]
        error = row["native_core_cp"] - teacher  # Both STM; no black-view sign conversion.
        errors.append(error)
        keys = {"pack": games[source["path"]]["pack_sha256"], "ply": band(ply, PLY_BANDS),
            "absolute_teacher_cp": band(abs(teacher), TEACHER_BANDS),
            "previous_train_board_overlap": "present" if key in old_boards else "absent",
            "side_to_move": side}
        for dimension, group in keys.items():
            grouped[dimension][group].append(error)
    require(sfens == set(targets), "label set differs from complete positions")
    if split == "holdout":
        require(not grouped["previous_train_board_overlap"]["present"], "fixed holdout overlaps previous train")
    summary = {"count": count, "overall": totals(errors, count),
        "groups": {dimension: {key: totals(values, count) for key, values in groups.items()}
                   for dimension, groups in grouped.items()}}
    for groups in summary["groups"].values():
        require(sum(v["count"] for v in groups.values()) == count
                and sum(v["squared_error_sum_cp2"] for v in groups.values()) == summary["overall"]["squared_error_sum_cp2"],
                "group partition lost rows or squared error")
    if split == "train":
        summary["top_squared_error_contribution"] = concentration(errors)
    return summary


def public_projection(mode, summaries):
    """Allow only aggregate fields; physical provenance stays in the private receipt."""
    require(type(mode) is str and mode == "white-view-diverse-games-seed42-e3-v1", "weekly mode required")
    require(set(summaries) == {"train", "holdout"}, "only train/fixed holdout aggregates allowed")
    projected = {}
    for split in ("train", "holdout"):
        fields = {"count", "overall", "groups"}
        if split == "train":
            fields.add("top_squared_error_contribution")
        require(set(summaries[split]) == fields, "unexpected nonaggregate summary fields")
        projected[split] = deepcopy(summaries[split])
    return {"schema": "sekirei.weekly-static-residual-summary.v1", "status": "aggregated-static-diagnostics",
        "mode": mode, "policy": deepcopy(POLICY), "splits": projected,
        "proof_complete_revalidated": True, "model_inference_started": False, "search_started": False,
        "development_used": False, "final_used": False, "adoption_claimed": False}


def fresh_private(path, expected, protected=()):
    require(path.is_absolute() and not path.exists() and path.parent.resolve(strict=True) == path.parent,
            "fresh canonical output directory required")
    require(not path.is_relative_to(proof.REPO), "private diagnostic output required")
    result = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "--is-inside-work-tree"],
                            capture_output=True, text=True)
    require(result.returncode != 0, "output is inside a Git worktree")
    for name in expected:
        require(not path.is_relative_to(Path(name)) and not Path(name).is_relative_to(path),
                "output overlaps a preserved input file")
    for directory in protected:
        require(not path.is_relative_to(directory) and not directory.is_relative_to(path),
                "output overlaps a preserved run or dataset directory")


def run(args, state):
    proof_ref = proof.physical_ref(args.proof)
    require(proof_ref["sha256"] == args.expected_proof_sha256, "external technical proof SHA differs")
    value = proof.strict_json(proof.read_ref(proof_ref))
    gate, pure_sources = proof.load_numeric_gate(args.numeric_build_contract)
    gate.fixed(value, {"schema": "sekirei.weekly-nonlinear-model-technical-proof.v1", "status": "complete",
                       "model_adopted": False, "adoption_claimed": False, "final_used": False})
    expected = dict(value["inputs_before"])
    proof.collect_refs(expected, [proof_ref, value])
    proof.merge(expected, pure_sources)
    proof.merge(expected, {str(Path(__file__).resolve()): proof.info(Path(__file__).resolve())})
    fresh_private(args.output, expected)
    lock_paths = {args.output.parent / ".heavy.lock"}
    require(type(value["lock_records"]) is list and len(value["lock_records"]) >= 5, "proof lock records required")
    for record in value["lock_records"]:
        gate.obj(record, {"path", "exclusive", "nonblocking", "acquired"})
        gate.fixed(record, {"exclusive": True, "nonblocking": True, "acquired": True})
        path = Path(record["path"])
        require(path.is_absolute() and path.resolve() == path, "canonical proof lock required")
        lock_paths.add(path)
    with ExitStack() as locks:
        for path in sorted(lock_paths):
            locks.enter_context(nonblocking_lock(path, exclusive=True))
        bound = validate_source_binding(value["recipe"])
        fresh_private(args.output, expected, (args.proof.parent,
            Path(value["native03"]["path"]).parent, Path(bound["build"]["source_root"]).parent,
            Path(bound["recipe"]["manifest"]["path"]).parent))
        allowlist = proof.strict_json(proof.read_ref(bound["parent"]["process_evidence"]["allowlist"]))
        scans_before = proof.require_clear_processes(allowlist)
        before = proof.verify_map(expected)
        reader = PhysicalBytes(expected, gate=gate, physical_ref=proof.physical_ref)
        completed = reader.json(value["training_completion"])
        manifest = reader.json(value["engine_manifest"])
        proof.validate_technical_proof(value, reader, bound, completed, value["training_completion"],
                                       manifest, value["engine_identity"], gate)
        origin = reader.json(bound["parent"]["dataset_origin_proof"])
        old_ref = origin["frozen_dataset_inputs"]["train.positions.jsonl"]
        old_boards = previous_train_boards(reader.read(old_ref), proof.COUNTS["train"], gate)
        dataset = reader.json(bound["recipe"]["manifest"])
        summaries = {}
        for split in ("train", "holdout"):
            inputs = bound["source_binding"]["dataset_inputs"]
            summaries[split] = analyze_split(dataset,
                reader.read(inputs[split + ".positions.jsonl"]), reader.read(inputs[split + ".labels.jsonl"]),
                reader.read(value["core"][split]["stdout"]), old_boards, split, proof.COUNTS[split], gate)
        public = public_projection(value["mode"], summaries)
        after = proof.verify_map(before)
        scans_after = proof.require_clear_processes(allowlist)
        require(state["signal"] is None, "cancelled before diagnostic publication")
        state["cleaning"] = True
        try:
            args.output.mkdir(mode=0o700)
            state["owned_output_directory"] = True
            require(state["signal"] is None, "cancelled before diagnostic publication")
            public_ref = proof.write_owned_json(args.output / "public-summary.json", public, state,
                                               "owned_public", args.output / "public-summary.json")
            private = {"schema": "sekirei.weekly-static-residual-receipt.v1", "status": "complete",
                "proof": proof_ref, "producer_source": proof.physical_ref(Path(__file__).resolve()),
                "previous_train_positions": old_ref, "public_summary": public_ref, "policy": deepcopy(POLICY),
                "inputs_before": before, "inputs_after": after,
                "process_scan_records_before": scans_before, "process_scan_records_after": scans_after,
                "proof_complete_revalidated": True, "model_inference_started": False, "search_started": False,
                "development_used": False, "final_used": False, "adoption_claimed": False}
            proof.verify_map(before)
            saved = proof.write_owned_json(args.output / "diagnostic-receipt.json", private, state,
                                          "owned_receipt", args.output / "diagnostic-receipt.json")
        finally:
            state["cleaning"] = False
        require(state["signal"] is None, "cancelled during diagnostic publication")
        proof.verify_map(before)
        print(json.dumps({"status": "static-diagnostics-verified", "receipt": saved,
                          "public_summary": public_ref, "model_adopted": False}), flush=True)


def main():
    import os
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--expected-proof-sha256", required=True)
    parser.add_argument("--numeric-build-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with termination_guard() as state:
        state.update(owned_public=None, owned_receipt=None, owned_output_directory=False)
        try:
            run(args, state)
            require(state["signal"] is None, "cancelled at diagnostic parent exit")
        except BaseException as error:
            state["cleaning"] = True
            try:
                for key in ("owned_public", "owned_receipt"):
                    if state[key] is not None and state[key].exists():
                        state[key].rename(args.output / ("rejected-" + state[key].name))
                if state["owned_output_directory"] and args.output.exists():
                    proof.write_new(args.output / "failure.json", {"status": "failed", "error": str(error),
                        "error_type": type(error).__name__, "signal": state["signal"], "adoption_claimed": False})
            finally:
                state["cleaning"] = False
            raise


if __name__ == "__main__":
    main()
