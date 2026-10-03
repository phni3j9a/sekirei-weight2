#!/usr/bin/env python3
"""Construct an exact material initializer for pinned Sekirei v0.3.39 (stdlib only).

This generates weights and performs a Python arithmetic check, not an engine
validation or a training run. See docs/WEIGHT_IMPROVEMENT.md for the proof and its domain.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

COMMIT = "f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9"
SOURCE_HASHES = {
    "crates/sekirei-core/src/nnue.rs": "a467e8b1b75f6b82629f369a1c804476b1c610354a561822365f980e6e428561",
    "crates/sekirei-core/src/eval.rs": "69fce9b3522a96c68a7f6dce15a342ba34c90f0addd0fa13b7478bef4d25e839",
    "crates/sekirei-core/src/piece.rs": "2ccf9248daf5a5b5eba2651ce03a5f85918608b4c5651ab5c728ddf6ba0a50a3",
    "crates/sekirei-core/src/square.rs": "60ee89de778fc18e30bbe0897f7380e4dfbf2d80f27afbe6d2b4e405c92e68b2",
}
INPUT, L1, L2 = 2420, 256, 32
BOARD_INPUT, HAND_THRESHOLDS = 2268, 38
HAND_OFFSETS = (0, 18, 22, 26, 30, 34, 36)
HAND_MAX = (18, 4, 4, 4, 4, 2, 2)
VALUES = (100, 430, 470, 640, 680, 890, 1040, 0,
          600, 600, 600, 640, 1150, 1300)
BASE = (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 5, 6)
SFEN_KIND = {name: index for index, name in enumerate(
    ("P", "L", "N", "S", "G", "B", "R", "K", "+P", "+L", "+N", "+S", "+B", "+R"))}
EXPECTED_BYTES = 8 + INPUT * L1 * 2 + L1 * 2 + 2 * L1 * L2 * 4 + L2 * 8 + 4
MASK64 = (1 << 64) - 1


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fnv1a(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & MASK64
    return f"{value:016x}"


def verify_source(source):
    """Pin both commit and relevant file contents; a trainer-only patch is OK."""
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise ValueError("source is not the pinned Sekirei v0.3.39 commit")
    for name, expected in SOURCE_HASHES.items():
        if sha256((source / name).read_bytes()) != expected:
            raise ValueError(f"pinned source changed: {name}")
    nnue = (source / "crates/sekirei-core/src/nnue.rs").read_text()
    evaluation = (source / "crates/sekirei-core/src/eval.rs").read_text()
    if not all(f"pub const {name}: usize = {value};" in nnue
               for name, value in (("L1", L1), ("L2", L2), ("HAND_THRESHOLDS", HAND_THRESHOLDS))):
        raise ValueError("NNUE dimension declarations disagree with generator")
    block = re.search(r"pub const PIECE_VALUE:.*?=\s*\[(.*?)\];", evaluation, re.S)
    if block is None:
        raise ValueError("missing PIECE_VALUE")
    uncommented = re.sub(r"//[^\n]*", "", block.group(1))
    if tuple(map(int, re.findall(r"\d+", uncommented))) != VALUES:
        raise ValueError("PIECE_VALUE disagrees with generator")
    return {"commit": head, "sha256": SOURCE_HASHES,
            "architecture": "default flat SEKIRW01; not king_relative_b_small"}


class Lcg:
    """Explicit u64 LCG; one high-bit sign per draw, independent of Python RNG."""
    def __init__(self, seed):
        if type(seed) is not int or not 0 <= seed <= MASK64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        self.state = seed

    def sign(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & MASK64
        return 1 if self.state >> 63 else -1


def group(kind):
    if kind == 7:
        return None
    return 0 if kind in (0, 8) else 1


def build_weights(seed):
    rng = Lcg(seed)
    ft = array("h", [0]) * (INPUT * L1)
    for feature in range(INPUT):
        offset = feature * L1
        # All auxiliary features, including king features, participate.
        for channel in range(2, L1):
            ft[offset + channel] = rng.sign()
        if feature < BOARD_INPUT:
            kind, opponent = (feature % 28) // 2, feature % 2
            target = group(kind)
            if opponent == 0 and target is not None:
                ft[offset + target] = VALUES[kind] // 2
        else:
            bank, threshold = divmod(feature - BOARD_INPUT, HAND_THRESHOLDS)
            hand_color, perspective = divmod(bank, 2)
            kind = next(k for k in range(7)
                        if HAND_OFFSETS[k] <= threshold < HAND_OFFSETS[k] + HAND_MAX[k])
            if hand_color == perspective:
                ft[offset + group(kind)] = VALUES[kind] // 2
    ft_bias = array("h", [64]) * L1
    l2 = array("f", [0.0]) * (2 * L1 * L2)
    l2[0 * L2 + 0] = 1.0
    l2[1 * L2 + 1] = 1.0
    l2[L1 * L2 + 2] = 1.0
    l2[(L1 + 1) * L2 + 3] = 1.0
    # No auxiliary connection into the material L2 units. Residual output is
    # zero initially, but auxiliary neurons are inside both ReLU gates.
    for row in range(2 * L1):
        if row % L1 < 2:
            continue
        for column in range(4, L2):
            l2[row * L2 + column] = rng.sign() / 256.0
    return {"ft": ft, "ft_bias": ft_bias, "l2": l2,
            "l2_bias": array("f", [0.0] * 4 + [4.0] * (L2 - 4)),
            "out": array("f", [8192.0, 8192.0, -8192.0, -8192.0] + [0.0] * (L2 - 4)),
            "out_bias": 0.0}


def little_bytes(values, size):
    if values.itemsize != size:
        raise RuntimeError("unsupported Python array element size")
    copy = array(values.typecode, values)
    if sys.byteorder != "little":
        copy.byteswap()
    return copy.tobytes()


def encode(weights):
    parts = [b"SEKIRW01"]
    for key in ("ft", "ft_bias", "l2", "l2_bias", "out"):
        parts.append(little_bytes(weights[key], 2 if key.startswith("ft") else 4))
    parts.append(struct.pack("<f", weights["out_bias"]))
    result = b"".join(parts)
    if len(result) != EXPECTED_BYTES:
        raise ValueError("unexpected binary shape")
    return result


def decode(data):
    if len(data) != EXPECTED_BYTES or data[:8] != b"SEKIRW01":
        raise ValueError("invalid SEKIRW01 size or magic")
    result, cursor = {}, 8
    for key, code, count in (("ft", "h", INPUT * L1), ("ft_bias", "h", L1),
                             ("l2", "f", 2 * L1 * L2), ("l2_bias", "f", L2), ("out", "f", L2)):
        size = 2 if code == "h" else 4
        value = array(code)
        value.frombytes(data[cursor:cursor + size * count])
        if sys.byteorder != "little":
            value.byteswap()
        result[key] = value
        cursor += size * count
    result["out_bias"] = struct.unpack_from("<f", data, cursor)[0]
    return result


def parse_sfen(sfen):
    """Inventory/shape validation only. This is NOT a shogi legality checker."""
    board, side, hand, ply = sfen.split()
    if side not in ("b", "w") or not ply.isascii() or not ply.isdigit() or int(ply) < 1:
        raise ValueError("invalid side or ply")
    ranks = board.split("/")
    if len(ranks) != 9:
        raise ValueError("expected nine SFEN ranks")
    pieces = []
    for rank, text in enumerate(ranks):
        file, promoted = 0, False
        for char in text:
            if char in "123456789":
                if promoted:
                    raise ValueError("promotion marker before a gap")
                file += int(char)
            elif char == "+":
                if promoted:
                    raise ValueError("repeated promotion marker")
                promoted = True
            else:
                token = ("+" if promoted else "") + char.upper()
                if token not in SFEN_KIND or file >= 9:
                    raise ValueError("invalid SFEN piece")
                pieces.append((file * 9 + rank, SFEN_KIND[token], 0 if char.isupper() else 1))
                file += 1
                promoted = False
        if file != 9 or promoted:
            raise ValueError("invalid SFEN rank width")
    hands = [[0] * 7 for _ in range(2)]
    if hand != "-":
        tokens = re.findall(r"([1-9][0-9]*)?([PLNSGBRplnsgbr])", hand)
        if "".join(count + char for count, char in tokens) != hand:
            raise ValueError("invalid hand syntax")
        for count, char in tokens:
            color, kind = 0 if char.isupper() else 1, SFEN_KIND[char.upper()]
            if hands[color][kind]:
                raise ValueError("repeated hand piece")
            hands[color][kind] = int(count or "1")
    totals = [sum(h[k] for h in hands) for k in range(7)]
    kings = [0, 0]
    for _, kind, color in pieces:
        if kind == 7:
            kings[color] += 1
        else:
            totals[BASE[kind]] += 1
    if kings != [1, 1] or any(n > limit for n, limit in zip(totals, HAND_MAX)):
        raise ValueError("position exceeds the standard piece inventory or lacks kings")
    return {"pieces": pieces, "hands": hands, "stm": 0 if side == "b" else 1}


def active_features(position, perspective):
    # Same square-major accumulation order as NnueAcc::refresh_with.
    features = [square * 28 + kind * 2 + (color != perspective)
                for square, kind, color in sorted(position["pieces"])]
    for color in range(2):
        for kind in range(7):
            for count in range(1, position["hands"][color][kind] + 1):
                features.append(BOARD_INPUT + (color * 2 + perspective) * HAND_THRESHOLDS
                                + HAND_OFFSETS[kind] + count - 1)
    return features


def material(position):
    scores = [0, 0]
    for _, kind, color in position["pieces"]:
        scores[color] += VALUES[kind]
    for color in range(2):
        scores[color] += sum(n * VALUES[k] for k, n in enumerate(position["hands"][color]))
    side = position["stm"]
    return scores[side] - scores[1 - side]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reference_forward(weights, position):
    """Scalar replica with explicit f32 steps; independent of engine execution."""
    accumulators = []
    for perspective in range(2):
        accum = list(weights["ft_bias"])
        for feature in active_features(position, perspective):
            offset = feature * L1
            for channel in range(L1):
                accum[channel] = max(-32768, min(32767, accum[channel] + weights["ft"][offset + channel]))
        accumulators.append(accum)
    l2 = list(weights["l2_bias"])
    us = position["stm"]
    for channel in range(L1):
        a = max(0, min(8128, accumulators[us][channel])) / 64.0
        b = max(0, min(8128, accumulators[1 - us][channel])) / 64.0
        for output in range(L2):
            l2[output] = f32(l2[output] + f32(a * weights["l2"][channel * L2 + output]))
            l2[output] = f32(l2[output] + f32(b * weights["l2"][(L1 + channel) * L2 + output]))
    output = weights["out_bias"]
    for channel in range(L2):
        output = f32(output + f32(max(0.0, min(127.0, l2[channel])) * weights["out"][channel]))
    return {"score_cp": int(f32(output / 64.0)), "float_cp": f32(output / 64.0),
            "ft_min_raw": min(map(min, accumulators)), "ft_max_raw": max(map(max, accumulators)),
            "aux_l2_min": min(l2[4:]), "aux_l2_max": max(l2[4:])}


def verify_fixtures(weights, fixtures):
    results = []
    for fixture in fixtures:
        position = parse_sfen(fixture["sfen"])
        expected = material(position)
        actual = reference_forward(weights, position)
        if expected != fixture["expected_cp_stm"] or actual["float_cp"] != expected or actual["score_cp"] != expected:
            raise ValueError(f"material mismatch in fixture {fixture['name']}")
        if not 0 < actual["aux_l2_min"] <= actual["aux_l2_max"] < 127:
            raise ValueError("auxiliary L2 units are not trainable at initialization")
        results.append(dict(fixture, **actual))
    return {"status": "python_reference_pass", "engine_verified": False, "fixture_count": len(results),
            "results": results, "scope": "standard-inventory reachable opening fixtures; no search or training"}


def generate(source, output, seed, fixture_path):
    source_info = verify_source(source)
    if output.exists():
        raise ValueError("output exists; keep prior artifacts and use a new directory")
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    inside_git = subprocess.run(["git", "-C", str(ancestor), "rev-parse", "--is-inside-work-tree"],
                                capture_output=True, text=True)
    if inside_git.returncode == 0 and inside_git.stdout.strip() == "true":
        raise ValueError("generated weights must remain outside every Git worktree")
    fixture_bytes = fixture_path.read_bytes()
    fixtures = json.loads(fixture_bytes)
    weights = build_weights(seed)
    data = encode(weights)
    checks = verify_fixtures(decode(data), fixtures)
    recipe = {"schema": "sekirei.material-init.v1", "source": source_info,
              "seed": seed, "rng": "u64 LCG: state=(state*6364136223846793005+1442695040888963407) mod 2^64; sign=+1 iff high bit=1",
              "rng_order": "FT feature 0..2419 / channel 2..255; then L2 row 0..511 skipping row%256<2 / column 4..31",
              "dimensions": {"input": INPUT, "l1": L1, "l2": L2}, "piece_values": VALUES,
              "material": {"ft_channels": [0, 1], "groups": ["pawn and promoted pawn", "all other non-king pieces"],
                           "own_feature_i16": "piece_value/2", "ft_bias_i16": 64,
                           "group_max_cp": [10800, 14980], "max_ft_accumulator_i16": 7554,
                           "l2_units": [0, 1, 2, 3], "l2_mapping": [0, 1, 256, 257],
                           "l2_copy_weight": 1, "l2_bias": 0, "out": [8192, 8192, -8192, -8192]},
              "auxiliary": {"ft_i16": [-1, 1], "ft_bias_i16": 64, "l2_weights": [-1/256, 1/256],
                            "l2_bias": 4, "out": 0, "initial_output_cp": 0},
              "nnue_output": "absolute", "baseline": "exact built-in material function at initialization",
              "weight_sha256": sha256(data), "weight_bytes": len(data), "checkpoint_hash": fnv1a(data),
              "generator_sha256": sha256(Path(__file__).read_bytes()), "fixtures_sha256": sha256(fixture_bytes),
              "limits": ["standard piece inventory, two kings; extra-piece SFEN excluded",
                         "Python SFEN parser verifies shape/inventory, not complete legality",
                         "engine refresh/incremental validation is a separate required check",
                         "training changes weights; exact material equivalence only at initialization"]}
    sidecar = {"format": "sekirei-nnue-output-v1", "nnue_output": "absolute",
               "checkpoint_hash": fnv1a(data), "baseline": None}
    output.mkdir(parents=True)
    (output / "material-init.bin").write_bytes(data)
    for name, value in (("material-init.meta.json", sidecar), ("material-init.recipe.json", recipe), ("verification.json", checks)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "weight_sha256": sha256(data),
                      "bytes": len(data), "fixture_count": len(fixtures), "engine_verified": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/material_init.json")
    args = parser.parse_args()
    generate(args.source.resolve(), args.output.resolve(), args.seed, args.fixtures.resolve())
