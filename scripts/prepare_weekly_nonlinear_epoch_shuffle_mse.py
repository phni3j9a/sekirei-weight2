#!/usr/bin/env python3
"""Create a fresh source checkout with compile-bound weekly input identities.

This prepares source only. It neither compiles nor starts an engine or trainer.
The old published snapshot and all previous runtime directories remain inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from weekly_nonlinear_epoch_shuffle_mse_post_training import require


REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "preparations/white-view-paired-nonlinear-rust-v1/compiled-source"
REPAIR = REPO / "preparations/white-view-paired-nonlinear-runtime-v2"
UPSTREAM = "f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9"
TEACHER = "external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d"
FROZEN_MANIFEST_SHA = "ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6"
DATASET_FILES = ("manifest.json", "train.positions.jsonl", "train.labels.jsonl",
                 "holdout.positions.jsonl", "holdout.labels.jsonl")


SHUFFLE_MODE = 'white-view-diverse-games-shuffle-seed42-e3-v1'
SHUFFLE_PLAN_PIN = {'bytes': 4948, 'sha256': 'cc4fcdcfa3d78b8571e3ba4e073e761dc4890fad62b990787e51f5fcfbccdd23'}
SHUFFLE_MANIFEST_PIN = {'bytes': 2999, 'sha256': 'f786e09b609bb0ba92b6e524d199fa8f8b490e21080ffb0c4802846d0a0f1dd7'}
SHUFFLE = REPO / 'preparations/white-view-paired-nonlinear-epoch-shuffle-mse-v1'
SHUFFLE_CHANGED_FILES = ['crates/sekirei-train/src/paired_nonlinear_cli.rs', 'crates/sekirei-train/src/paired_nonlinear_parent_binding.rs', 'crates/sekirei-train/src/paired_nonlinear_positions.rs']
SHUFFLE_RUST_TEST_SUFFIXES = ['shuffle_stable_integer_vectors', 'shuffle_full_permutation_bijection_and_epoch_pins', 'shuffle_rejects_seed_epoch_missing_duplicate_and_order', 'shuffle_consumption_success_counts_and_exact_updates', 'shuffle_consumption_rejects_wrong_step_order_duplicate_and_incomplete', 'shuffle_declaration_has_exact_separate_init_seed_and_holdout_order']
from weekly_nonlinear_epoch_shuffle_mse_order import declaration
ROW_ORDER = declaration()


def raw_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def digest(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return actual.keys() == expected.keys() and all(typed_equal(actual[k], v) for k, v in expected.items())
    if type(actual) is list:
        return len(actual) == len(expected) and all(typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_plan(plan):
    keys = {"schema", "status", "mode", "hypothesis", "created_utc", "architecture", "training",
            "dataset_inputs", "teacher_identity", "selection_profile", "resources", "adoption", "incumbent_model"}
    require(type(plan) is dict and set(plan) == keys, "weekly exact plan keys required")
    require(plan["schema"] == "sekirei.weekly-nonlinear-epoch-shuffle-mse-selected-plan.v1"
            and plan["status"] == "frozen-before-build" and plan["mode"] == SHUFFLE_MODE,
            "exact dedicated fixed-shuffle MSE plan required")
    for key in ("hypothesis", "created_utc"):
        require(type(plan[key]) is str and plan[key], "nonempty weekly plan text required")
    architecture = {"feature_schema": "flat_white_view_aux_tied_v1", "dimensions": [2420, 256, 32],
                    "native_magic": "SEKIRW03", "protected_material_units": [0, 1, 2, 3],
                    "all_ft_bias_fixed_q": 64, "hand_ties": [[0, 3], [1, 2]], "aux_ft_channels": 254, "paired_heads": 14}
    training = {"seed": 42, "epochs": 3, "selected_epoch": 3, "train_count": 112681, "holdout_count": 5895,
                "objective": "absolute-cp-mse", "optimizer": "fresh-adam-tied-masters", "learning_rate_schedule": "constant",
                "learning_rate_f32_bits": "3a83126f", "head_init_width_f32_bits": "3b800000",
                "head_bias_init_f32_bits": "40800000", "output_native_l1_budget_f32_bits": "47000000",
                "shuffle_seed": 20261006, "row_order": ROW_ORDER, "resume_allowed": False, "ft_saved_q_max": 797, "ft_bias_q": 64}
    adoption = {"development_games": 5, "go_nodes": 1000000, "mae_multipv": 1, "top3_multipv": 3,
                "rule": "strict MAE decrease and no Top3 decrease against latest incumbent",
                "final_used_for_daily_selection": False, "learning_loss_is_adoption_criterion": False}
    require(typed_equal(plan["architecture"], architecture) and typed_equal(plan["training"], training)
            and typed_equal(plan["adoption"], adoption), "fixed weekly numeric/architecture/adoption condition differs")
    budget = plan["resources"]
    budget_keys = {"heavy_serial", "build_jobs", "analysis_jobs", "threads", "training_wall_limit_seconds",
                   "added_storage_budget_bytes", "minimum_ssd_remaining_bytes"}
    require(type(budget) is dict and set(budget) == budget_keys, "weekly exact resource keys required")
    for key, value in (("heavy_serial", True), ("build_jobs", 2), ("analysis_jobs", 1), ("threads", 1)):
        require(typed_equal(budget[key], value), "fixed resource option differs")
    for key in ("training_wall_limit_seconds", "added_storage_budget_bytes", "minimum_ssd_remaining_bytes"):
        require(type(budget[key]) is int and budget[key] > 0, "positive typed resource budget required")
    require(budget["training_wall_limit_seconds"] <= 86400, "training attempt exceeds one-day limit")


def ref(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
            and path.is_file() and not path.is_symlink(), "canonical regular file required")
    return {"path": str(path), **digest(path.read_bytes())}


def pinned_json(path, expected):
    raw = path.read_bytes()
    require(digest(raw)["sha256"] == expected, "externally pinned JSON SHA differs")
    return raw_json(raw), digest(raw)


def verify_base_snapshot():
    manifest = raw_json((BASE / "public-source-manifest.json").read_bytes())
    require(manifest["upstream_commit"] == UPSTREAM, "snapshot upstream changed")
    for name, expected in manifest["files"].items():
        require(digest((BASE / name).read_bytes()) == expected, "published snapshot changed")


def profile_source(plan, plan_info, manifest, manifest_info, profile, profile_info):
    """Generate only compiled identities; numeric training code stays unchanged."""
    require(type(plan) is dict and plan.get("schema") == "sekirei.weekly-nonlinear-epoch-shuffle-mse-selected-plan.v1"
            and plan.get("status") == "frozen-before-build", "weekly frozen plan required")
    require(typed_equal(plan_info, SHUFFLE_PLAN_PIN), "frozen fixed-shuffle MSE raw plan pin differs")
    mode = plan.get("mode")
    require(mode == SHUFFLE_MODE, "dedicated fixed-shuffle MSE mode required")
    require(type(mode) is str and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", mode),
            "canonical distinct weekly candidate mode required")
    require(mode != "white-view-paired-nonlinear-adam-e3-v1", "new candidate identity required")
    require(plan.get("teacher_identity") == TEACHER
            and manifest.get("teacher_identity") == TEACHER, "original teacher identity required")
    dataset = plan.get("dataset_inputs")
    require(type(dataset) is dict and set(dataset) == set(DATASET_FILES), "five dataset refs required")
    require(dataset["manifest.json"].get("bytes") == manifest_info["bytes"]
            and dataset["manifest.json"].get("sha256") == manifest_info["sha256"],
            "plan dataset manifest identity differs")
    require(plan.get("selection_profile", {}).get("sha256") == profile_info["sha256"]
            and plan["selection_profile"].get("bytes") == profile_info["bytes"],
            "plan profile identity differs")
    derivation = manifest.get("derivation", {})
    require(profile.get("kind") == "whole-pack-hash-ranked-frozen-holdout-profile-v2"
            and profile.get("sampling") == "whole-pack SHA256(seed:pack-sha:index) rank; explicit per-pack game counts; ply>=16/every4/max32"
            and derivation.get("kind") == "whole-pack-hash-ranked-frozen-holdout-v2",
            "new explicit per-pack selection profile/derivation required")
    require(derivation.get("profile_sha256") == profile_info["sha256"]
            and typed_equal(derivation.get("profile"), profile), "manifest selection profile differs")
    require(profile.get("frozen_dataset_manifest_sha256") == FROZEN_MANIFEST_SHA,
            "frozen holdout source manifest differs")
    require(typed_equal(manifest.get("positions"), {"train": 112681, "holdout": 5895}),
            "full fixed train/holdout counts required")
    require(profile.get("train_budget") == 112681, "fixed train budget required")
    seed = profile.get("selection_seed")
    require(type(seed) is str and re.fullmatch(r"[A-Za-z0-9:_-]+", seed),
            "canonical ASCII selection seed required")
    games = profile.get("games_per_pack")
    require(type(games) is int and 1 <= games <= 1000, "selection cap in 1..1000 required")
    packs = profile.get("packs")
    require(type(packs) is list and len(packs) == 13, "the fixed 13-pack source required")
    total_selected = 0
    for pack in packs:
        require(type(pack) is dict and set(pack) == {"sha256", "bytes", "games", "selected_games"},
                "complete per-pack preregistration fields required")
        require(type(pack["games"]) is int and pack["games"] > 0
                and type(pack["selected_games"]) is int
                and 1 <= pack["selected_games"] <= min(games, pack["games"]),
                "per-pack selected count differs from declared census and cap")
        total_selected += pack["selected_games"]
    corpus = profile.get("corpus_manifest_sha256")
    require(TEACHER == "external:suisho11beta-1m-pack:" + str(corpus), "pack corpus SHA differs")
    values = {"MODE": mode, "PLAN_SHA": plan_info["sha256"], "MANIFEST_SHA": manifest_info["sha256"],
              "PROFILE_SHA": profile_info["sha256"], "CORPUS_SHA": corpus,
              "FROZEN_MANIFEST_SHA": FROZEN_MANIFEST_SHA, "SELECTION_SEED": seed}
    lines = ["//! Generated frozen weekly identities; never modify after source binding."]
    lines.extend(f"pub const {name}:&str={json.dumps(value, ensure_ascii=True)};"
                 for name, value in values.items())
    lines.extend([f"pub const PLAN_BYTES:u64={plan_info['bytes']};",
                  f"pub const PACK_COUNT:u64={len(packs)};",
                  f"pub const GAMES_PER_PACK:u64={games};",
                  f"pub const TOTAL_SELECTED_GAMES:u64={total_selected};"])
    return ("\n".join(lines) + "\n").encode()



def shuffle_variant_sources():
    path = SHUFFLE / "public-source-manifest.json"
    raw = path.read_bytes()
    require(digest(raw) == SHUFFLE_MANIFEST_PIN, "fixed Shuffle source-variant manifest differs")
    body = raw_json(raw)
    require(body["mode"] == SHUFFLE_MODE and body["objective"] == "absolute-cp-mse"
            and body["shuffle_seed"] == 20261006 and body["row_order"] == ROW_ORDER, "Shuffle variant identity differs")
    require(set(body["changed_files"]) == set(SHUFFLE_CHANGED_FILES), "Shuffle changed-file membership differs")
    patch = SHUFFLE / "candidate.patch"
    require(digest(patch.read_bytes()) == body["candidate_patch"], "Shuffle candidate patch differs")
    postimages = {}
    for name, info in body["changed_files"].items():
        target = SHUFFLE / "source" / name
        require(digest(target.read_bytes()) == info["after"], "Shuffle postimage differs")
        postimages[name] = ref(target)
    return {"manifest": ref(path), "patch": ref(patch), "postimages": postimages}


def require_shuffle_tests(tests):
    require(type(tests) is dict and type(tests.get("required_test_names")) is list,
            "actual typed Rust test names required")
    for suffix in SHUFFLE_RUST_TEST_SUFFIXES:
        require(any(type(name) is str and name.endswith(suffix) for name in tests["required_test_names"]),
                "required Shuffle/control test absent from actual bound log: " + suffix)


def apply_shuffle_source_variant(source):
    sources = shuffle_variant_sources()
    manifest = raw_json(Path(sources["manifest"]["path"]).read_bytes())
    for name, info in manifest["changed_files"].items():
        require(digest((source / name).read_bytes()) == info["before"], "Shuffle patch preimage differs")
    run(["git", "apply", "--check", sources["patch"]["path"]], source)
    run(["git", "apply", sources["patch"]["path"]], source)
    for name, info in manifest["changed_files"].items():
        require(digest((source / name).read_bytes()) == info["after"], "Shuffle patch postimage differs")
    return sources


def run(argv, cwd=None):
    return subprocess.run(argv, cwd=cwd, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def prepare(args):
    verify_base_snapshot()
    plan_path, profile_path = args.plan.resolve(strict=True), args.profile.resolve(strict=True)
    plan, plan_info = pinned_json(plan_path, args.expected_plan_sha256)
    validate_plan(plan)
    profile, profile_info = pinned_json(profile_path, args.expected_profile_sha256)
    manifest_path = Path(plan["dataset_inputs"]["manifest.json"]["path"])
    manifest_ref = ref(manifest_path)
    manifest, manifest_info = pinned_json(manifest_path, manifest_ref["sha256"])
    require(plan["dataset_inputs"]["manifest.json"] == manifest_ref, "plan manifest fullref differs")
    require(plan["selection_profile"] == ref(profile_path), "plan profile fullref differs")
    for name in DATASET_FILES:
        declared = plan["dataset_inputs"][name]
        require(Path(declared["path"]) == manifest_path.parent / name
                and ref(Path(declared["path"])) == declared, "dataset sibling fullref differs")
    profile_bytes = profile_source(plan, plan_info, manifest, manifest_info, profile, profile_info)
    output = args.output.expanduser().absolute()
    require(not output.exists() and output.parent.resolve(strict=True) == output.parent,
            "fresh output under a canonical existing parent required")
    require(not output.is_relative_to(REPO) and not REPO.is_relative_to(output),
            "private source checkout must be outside the repository")
    upstream = args.upstream.expanduser().resolve(strict=True)
    require(run(["git", "-C", str(upstream), "rev-parse", UPSTREAM]).decode().strip() == UPSTREAM,
            "complete fixed upstream commit required")
    output.mkdir(mode=0o700)
    source = output / "source"
    run(["git", "clone", "--no-hardlinks", "--no-checkout", str(upstream), str(source)])
    run(["git", "checkout", "--detach", UPSTREAM], source)
    require(not run(["git", "status", "--porcelain", "--untracked-files=all"], source),
            "fresh upstream checkout not clean")
    patch = BASE / "nonlinear-complete.patch"
    run(["git", "apply", "--check", str(patch)], source)
    run(["git", "apply", str(patch)], source)
    published = raw_json((BASE / "public-source-manifest.json").read_bytes())["files"]
    for name, expected in published.items():
        if name.startswith("source/"):
            require(digest((source / name[len("source/"):]).read_bytes()) == expected,
                    "complete patch postimage differs from compiled snapshot")
    train_source = source / "crates/sekirei-train/src"
    for name in ("paired_nonlinear_actual.rs", "paired_nonlinear_parent_binding.rs",
                 "weekly_nonlinear_manifest_tests.rs"):
        (train_source / name).write_bytes((REPAIR / name).read_bytes())
    main = train_source / "main.rs"
    content = main.read_text()
    needle = '#[cfg(feature = "nnue_white_view_aux_tied")]\nmod paired_nonlinear_sha256;'
    require(content.count(needle) == 1, "snapshot main module registration changed")
    main.write_text(content.replace(needle, '#[cfg(feature = "nnue_white_view_aux_tied")]\nmod weekly_nonlinear_profile;\n' + needle))
    cli = train_source / "paired_nonlinear_cli.rs"
    content = cli.read_text()
    needle = 'pub const MODE:&str="white-view-paired-nonlinear-adam-e3-v1";'
    require(content.count(needle) == 1, "snapshot CLI mode declaration changed")
    cli.write_text(content.replace(needle, "pub const MODE:&str=crate::weekly_nonlinear_profile::MODE;"))
    (train_source / "weekly_nonlinear_profile.rs").write_bytes(profile_bytes)
    row_order_variant = apply_shuffle_source_variant(source)
    files = run(["git", "ls-files", "-z"], source).decode().split("\x00")[:-1]
    files += run(["git", "ls-files", "--others", "--exclude-standard", "-z"], source).decode().split("\x00")[:-1]
    identities = {str(source / name): {k: v for k, v in ref(source / name).items() if k != "path"}
                  for name in sorted(set(files))}
    receipt = {"schema": "sekirei.weekly-nonlinear-epoch-shuffle-mse-source-preparation.v1", "status": "source-only",
               "upstream_commit": UPSTREAM, "source_root": str(source), "plan": ref(plan_path),
               "manifest": manifest_ref, "selection_profile": ref(profile_path),
               "base_patch": ref(patch), "source_files": identities,
               "numeric_update_export_modules_preserved": False, "row_order_variant": row_order_variant, "compiled": False,
               "training_started": False, "model_adoption_claimed": False, "final_used": False}
    for path, before in ((plan_path, plan_info), (profile_path, profile_info), (manifest_path, manifest_info)):
        require(digest(path.read_bytes()) == before, "frozen preparation input changed")
    with (output / "source-preparation.json").open("xb") as stream:
        stream.write((json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": "source-only", "source_root": str(source),
                      "receipt": ref(output / "source-preparation.json")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
