"""Pure functional-anchor preparation; no filesystem adapter or launch readiness.

All inputs are supplied bytes. The caller pins the original manifest SHA before
building the canonical spec, and pins the spec SHA before validating a train
view. Only train score_cp/teacher_identity change; other label fields remain
original source metadata, and both original file orders are preserved.

The returned recipe always says real_generation_ready=False. Source checkout,
initializer, exclusion receipts, and complete native-core proofs still need a
separate real-data preflight. SFEN parsing checks shape/inventory, not legality.
There is deliberately no CLI, file writer, subprocess, or diagnostic wrapper.
"""
import copy
import hashlib
import json
import re

import material_init


FILES = frozenset(f"{split}.{kind}.jsonl" for split in ("train", "holdout")
                  for kind in ("positions", "labels"))
SPEC_SCHEMA = "sekirei.functional-anchor-spec.v1"
VIEW_SCHEMA = "sekirei.functional-anchor-train-view.v1"
IDENTITY_PREFIX = "external:sekirei-functional-anchor-v1:"
# Match the existing pack/frozen-dataset contract without importing its I/O
# modules (prepare.py reads a repository lock at import time).
SPLIT_RECIPE = "SHA256(seed:complete-game-identity) modulo 10; holdout=0"
SAMPLING = "first bounded games of each hash-ordered pack; ply>=16/every4/max32"
EXCLUSION_POLICY = "exclude entire acquired pool; no final split files opened"
MATERIAL_IMPLEMENTATION_SHA256 = "d7943d5d9e01946f17300a38c143eccb0017af37f36a70554dd394a70f6bfbad"
MATERIAL_INIT_SHA256 = "bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40"
FORMULA = {"teacher_numerator": 1, "material_numerator": 1, "denominator": 2,
           "rounding": "nearest-even-signed-integer", "perspective": "side-to-move",
           "label_depth": 0, "cp_abs_limit_exclusive": 30000, "split": "train",
           "nnue_output": "absolute"}
MATERIAL = {"kind": "fixed-sekirei-v0.3.39-material-fallback",
            "commit": material_init.COMMIT,
            "core_source_sha256": dict(material_init.SOURCE_HASHES),
            "implementation_sha256": MATERIAL_IMPLEMENTATION_SHA256,
            "initializer_sha256": MATERIAL_INIT_SHA256,
            "piece_values": list(material_init.VALUES)}
MANDATORY_PREFLIGHT = [
    "original source packs, frozen/reserved games and whole-pool exclusion receipts",
    "fixed checkout/core source and material_init implementation content hashes",
    "fixed absolute material initializer SHA/FNV and full source/build provenance",
    "all train rows: fixed material initializer native-core equality and prefix proof",
    "all holdout rows: reused or new core proof with exact order/hash binding",
    "input/source immutability, private new output, capacity and serial execution locks",
    "preregistration after prior valid candidates fail adoption; epoch3 fixed",
]


def sha256_bytes(data):
    if type(data) is not bytes:
        raise ValueError("input must be bytes")
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value):
    """Canonical UTF-8 JSON; reject nonfinite metadata, including nested values."""
    def json_types(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                json_types(child)
        elif type(item) is list:
            for child in item:
                json_types(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("expected JSON value types")
    json_types(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("invalid canonical JSON") from error


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON object key")
        result[name] = value
    return result


def _reject_constant(_):
    raise ValueError("nonfinite JSON value")


def _json(data):
    sha256_bytes(data)
    try:
        result = json.loads(data, object_pairs_hook=_object, parse_constant=_reject_constant)
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid input JSON") from error
    canonical_json_bytes(result)
    return result


def _rows(data):
    sha256_bytes(data)
    result = [_json(line) for line in data.splitlines()]
    if not result or any(type(row) is not dict for row in result):
        raise ValueError("JSONL requires nonempty object rows")
    return result


def _sha(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA-256")
    return value


def _integer(value, name, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer in range")
    return value


def _cp(value, name):
    _integer(value, name)
    if abs(value) >= 30000:
        raise ValueError(f"{name} must have abs(cp) < 30000")
    return value


def blend_target(teacher_cp, material_cp):
    """Validate T first, then round (T + M)/2 with symmetric ties-to-even."""
    _cp(teacher_cp, "original teacher cp")
    _integer(material_cp, "fixed material cp")
    quotient, remainder = divmod(teacher_cp + material_cp, 2)
    return _cp(quotient + (1 if remainder and quotient % 2 else 0), "target cp")


def _parsed_position(sfen):
    if type(sfen) is not str:
        raise ValueError("SFEN must be a string")
    return material_init.parse_sfen(sfen)


def fixed_material_cp(sfen):
    """Pinned fallback STM values; no fitted piece values can be supplied."""
    return material_init.material(_parsed_position(sfen))


def _split_for_game(game_id, seed):
    digest = hashlib.sha256(f"{seed}:{game_id}".encode()).hexdigest()
    return "holdout" if int(digest[:16], 16) % 10 == 0 else "train"


def _source_provenance(manifest):
    source = _sha(manifest.get("source_corpus_manifest_sha256"))
    if manifest.get("teacher_identity") != f"external:suisho11beta-1m-pack:{source}":
        raise ValueError("original teacher/source pack identity mismatch")
    if (type(manifest.get("seed")) is not str or not manifest["seed"]
            or manifest.get("split") != SPLIT_RECIPE or manifest.get("sampling") != SAMPLING):
        raise ValueError("unrecognized source split/sampling provenance")
    cap = _integer(manifest.get("games_per_pack"), "games_per_pack", 1)
    if cap > 1000 or type(manifest.get("dependencies")) is not dict:
        raise ValueError("invalid source cap/dependencies")
    exclusion = manifest.get("independent_exclusions")
    if (type(exclusion) is not dict or type(exclusion.get("games")) is not int
            or exclusion["games"] != 1000 or exclusion.get("policy") != EXCLUSION_POLICY):
        raise ValueError("missing whole-pool exclusion provenance")
    _integer(exclusion.get("unique_positions"), "excluded unique positions", 1)
    for field in ("corpus_canonical_sha256", "source_manifest_sha256"):
        _sha(exclusion.get(field))
    derivation = manifest.get("derivation")
    if (type(derivation) is not dict
            or derivation.get("kind") != "expanded-train-frozen-holdout-v1"
            or type(derivation.get("input_sha256")) is not dict
            or not derivation["input_sha256"]):
        raise ValueError("missing frozen-holdout derivation provenance")
    for name, digest in derivation["input_sha256"].items():
        if type(name) is not str or not name:
            raise ValueError("invalid original input provenance name")
        _sha(digest)
    games, locations = {}, set()
    if type(manifest.get("games")) is not list:
        raise ValueError("missing source game inventory")
    for game in manifest["games"]:
        if type(game) is not dict:
            raise ValueError("invalid source game")
        gid, pack = _sha(game.get("game_id")), _sha(game.get("pack_sha256"))
        index = _integer(game.get("game_index"), "game_index", 0)
        if (index >= cap or gid in games or (pack, index) in locations
                or game.get("split") != _split_for_game(gid, manifest["seed"])):
            raise ValueError("duplicate/misassigned source game")
        games[gid] = game
        locations.add((pack, index))
    if not games:
        raise ValueError("empty source game inventory")
    return games


def validate_original(manifest_bytes, files, *, expected_manifest_sha256):
    """Validate pinned original bytes and complete caches, without I/O/probes."""
    if sha256_bytes(manifest_bytes) != _sha(expected_manifest_sha256):
        raise ValueError("original manifest hash mismatch")
    manifest = _json(manifest_bytes)
    if (type(manifest) is not dict or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or type(files) is not dict
            or set(files) != FILES or type(manifest.get("files")) is not dict
            or set(manifest["files"]) != FILES
            or type(manifest.get("positions")) is not dict
            or set(manifest["positions"]) != {"train", "holdout"}
            or "functional_anchor" in manifest or "split_teacher_identities" in manifest):
        raise ValueError("expected original four-file dataset")
    games = _source_provenance(manifest)
    parsed = {}
    for name in sorted(FILES):
        info = manifest["files"][name]
        if type(info) is not dict:
            raise ValueError("invalid file metadata")
        size = _integer(info.get("bytes"), "file bytes", 1)
        if sha256_bytes(files[name]) != _sha(info.get("sha256")) or len(files[name]) != size:
            raise ValueError(f"original file hash/size mismatch: {name}")
        parsed[name] = _rows(files[name])
        if "count" in info and _integer(info["count"], "file count", 1) != len(parsed[name]):
            raise ValueError("file row count mismatch")
    keys, split_games = {}, {}
    for split in ("train", "holdout"):
        positions, labels = parsed[f"{split}.positions.jsonl"], parsed[f"{split}.labels.jsonl"]
        if len(positions) != _integer(manifest["positions"][split], "split count", 1):
            raise ValueError("position count mismatch")
        sfens, boards, gids = set(), set(), set()
        for row in positions:
            sfen = row.get("sfen")
            position = _parsed_position(sfen)
            # Semantic keys catch alternate SFEN spellings and ignore ply.
            board = (tuple(sorted(position["pieces"])),
                     tuple(tuple(hand) for hand in position["hands"]), position["stm"])
            source = row.get("source")
            if type(source) is not dict:
                raise ValueError("missing position source")
            gid = source.get("path")
            if (sfen in sfens or board in boards or gid not in games
                    or games[gid]["split"] != split or source.get("kind") != "gensfen-pack"
                    or type(row.get("schema_version")) is not int or row["schema_version"] != 1):
                raise ValueError("duplicate board or wrong source game")
            sfens.add(sfen); boards.add(board); gids.add(gid)
        cache = {}
        for row in labels:
            sfen = row.get("sfen")
            _cp(row.get("score_cp"), "original teacher cp")
            if (type(sfen) is not str or sfen in cache
                    or row.get("teacher_identity") != manifest["teacher_identity"]
                    or type(row.get("label_depth")) is not int or row["label_depth"] != 0):
                raise ValueError("invalid/duplicate original teacher label")
            cache[sfen] = row
        if set(cache) != sfens:
            raise ValueError("original cache must equal complete SFEN set")
        keys[split], split_games[split] = boards, gids
    if keys["train"] & keys["holdout"] or split_games["train"] & split_games["holdout"]:
        raise ValueError("original train/holdout overlap")
    return manifest, parsed


def build_spec(manifest_bytes, files, *, expected_manifest_sha256):
    """Hash inputs/formula only. Derived outputs never enter the spec hash."""
    manifest, _ = validate_original(manifest_bytes, files,
                                    expected_manifest_sha256=expected_manifest_sha256)
    return {"schema": SPEC_SCHEMA, "original_manifest_sha256": expected_manifest_sha256,
            "original_teacher_identity": manifest["teacher_identity"],
            "original_files": copy.deepcopy(manifest["files"]),
            "positions": copy.deepcopy(manifest["positions"]),
            "source_provenance": copy.deepcopy({k: v for k, v in manifest.items()
                if k not in ("files", "positions", "teacher_identity")}),
            "formula": copy.deepcopy(FORMULA), "material": copy.deepcopy(MATERIAL)}


def spec_sha256(spec):
    if (type(spec) is not dict or set(spec) != {"schema", "original_manifest_sha256",
            "original_teacher_identity", "original_files", "positions", "source_provenance",
            "formula", "material"} or spec.get("schema") != SPEC_SCHEMA
            or canonical_json_bytes(spec.get("formula")) != canonical_json_bytes(FORMULA)
            or canonical_json_bytes(spec.get("material")) != canonical_json_bytes(MATERIAL)):
        raise ValueError("invalid canonical derivation spec")
    _sha(spec["original_manifest_sha256"])
    if (type(spec["source_provenance"]) is not dict
            or {"files", "positions", "teacher_identity", "functional_anchor", "split_teacher_identities"}
            & set(spec["source_provenance"])
            or type(spec["positions"]) is not dict or set(spec["positions"]) != {"train", "holdout"}
            or type(spec["original_files"]) is not dict or set(spec["original_files"]) != FILES):
        raise ValueError("invalid spec source structure")
    original = copy.deepcopy(spec["source_provenance"])
    if type(original.get("schema_version")) is not int or original["schema_version"] != 1:
        raise ValueError("invalid spec source schema")
    original["teacher_identity"] = spec["original_teacher_identity"]
    _source_provenance(original)
    for split in ("train", "holdout"):
        _integer(spec["positions"][split], "spec split count", 1)
    for name, info in spec["original_files"].items():
        if type(info) is not dict:
            raise ValueError("invalid spec file metadata")
        _sha(info.get("sha256"))
        _integer(info.get("bytes"), "spec file bytes", 1)
        if "count" in info and _integer(info["count"], "spec file count", 1) != spec["positions"][name.split(".")[0]]:
            raise ValueError("invalid spec file count")
    return sha256_bytes(canonical_json_bytes(spec))


def derived_identity(spec):
    return IDENTITY_PREFIX + spec_sha256(spec)


def derive_train_view(manifest_bytes, files, spec, *, expected_manifest_sha256):
    """Return byte-preserving input view and recipe, always awaiting preflight."""
    expected = build_spec(manifest_bytes, files, expected_manifest_sha256=expected_manifest_sha256)
    digest = spec_sha256(spec)
    if canonical_json_bytes(spec) != canonical_json_bytes(expected):
        raise ValueError("spec is not bound to the pinned original inputs")
    original, parsed = validate_original(manifest_bytes, files,
                                         expected_manifest_sha256=expected_manifest_sha256)
    identity = IDENTITY_PREFIX + digest
    output_files = dict(files)
    labels, errors_twice = [], []
    for source in parsed["train.labels.jsonl"]:
        material_cp = fixed_material_cp(source["sfen"])
        target = blend_target(source["score_cp"], material_cp)
        row = copy.deepcopy(source)
        row.update(score_cp=target, teacher_identity=identity)
        labels.append(row)
        errors_twice.append(2 * target - source["score_cp"] - material_cp)
    output_files["train.labels.jsonl"] = b"".join(canonical_json_bytes(r) + b"\n" for r in labels)
    output_info = copy.deepcopy(original["files"])
    for name in FILES:
        output_info[name].update(sha256=sha256_bytes(output_files[name]), bytes=len(output_files[name]))
    manifest = copy.deepcopy(original)
    manifest.update(teacher_identity=identity, files=output_info,
                    split_teacher_identities={"train": identity, "holdout": original["teacher_identity"]},
                    functional_anchor={"schema": VIEW_SCHEMA, "spec_sha256": digest,
                        "original_manifest_sha256": expected_manifest_sha256,
                        "original_teacher_identity": original["teacher_identity"]})
    recipe = {"schema": VIEW_SCHEMA, "scope": "pure-transformation-only",
              "spec_sha256": digest, "target_identity": identity,
              "original_teacher_identity": original["teacher_identity"],
              "original_manifest_sha256": expected_manifest_sha256,
              "original_files": copy.deepcopy(original["files"]),
              "output_files": copy.deepcopy(output_info),
              "output_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest) + b"\n"),
              "preserved_bytes": ["train.positions.jsonl", "holdout.positions.jsonl", "holdout.labels.jsonl"],
              "label_metadata_policy": "only score_cp/teacher_identity replaced; all other fields are original source metadata",
              "rounding": {"count": len(labels), "half_integer_count": sum(e != 0 for e in errors_twice),
                  "max_abs_error_twice_cp": max(map(abs, errors_twice)),
                  "sum_abs_error_twice_cp": sum(map(abs, errors_twice)),
                  "sum_signed_error_twice_cp": sum(errors_twice)},
              "real_generation_ready": False,
              "mandatory_preflight": {"status": "not-verified", "required": list(MANDATORY_PREFLIGHT)}}
    return {"spec": copy.deepcopy(spec), "manifest": manifest, "files": output_files, "recipe": recipe}


def validate_train_view(view, manifest_bytes, files, *, expected_manifest_sha256,
                        expected_spec_sha256):
    """Reject changed spec/identity/hash/rows and preserve existing split guards.

    Success is only a pure transformation check, never a completed real-data
    preflight, core proof, training run, diagnosis, or model adoption.
    """
    if type(view) is not dict or set(view) != {"spec", "manifest", "files", "recipe"}:
        raise ValueError("invalid train view")
    if spec_sha256(view["spec"]) != _sha(expected_spec_sha256):
        raise ValueError("preregistered spec hash mismatch")
    expected = derive_train_view(manifest_bytes, files, view["spec"],
                                expected_manifest_sha256=expected_manifest_sha256)
    if type(view["files"]) is not dict or set(view["files"]) != FILES:
        raise ValueError("unexpected train-view files")
    for name in FILES:
        sha256_bytes(view["files"][name])
        if view["files"][name] != expected["files"][name]:
            raise ValueError(f"train-view bytes/order changed: {name}")
    for name in ("spec", "manifest", "recipe"):
        if canonical_json_bytes(view[name]) != canonical_json_bytes(expected[name]):
            raise ValueError(f"train-view {name} binding mismatch")
    return {"scope": "pure-transformation-only", "spec_sha256": expected_spec_sha256,
            "target_identity": derived_identity(view["spec"]), "real_generation_ready": False}
