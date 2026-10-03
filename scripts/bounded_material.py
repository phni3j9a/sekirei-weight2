"""Pure byte contract for one conditional bounded-material candidate.

Technical preparation only: no preregistration, trainer/mask/optimizer test,
model export, core/engine execution, real-data preflight, or adoption evidence.
The fixed proposal uses material-init seed42, frozen FT/material coordinates,
and a 99 cp real-arithmetic auxiliary coefficient budget. A 100 cp integer
core difference remains UNVERIFIED. No CLI or file I/O is provided.
"""
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
import struct

import material_init


SEED = 42
REAL_RESIDUAL_CAP_CP = 99
PROPOSED_INTEGER_CAP_CP = 100
MATERIAL_SEED42_SHA256 = "bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40"
INPUT, L1, L2 = 2420, 256, 32
MATERIAL_ROWS = (0, 1, 256, 257)
MATERIAL_COLUMNS = (0, 1, 2, 3)
AUXILIARY_COLUMNS = tuple(range(4, L2))
FT_OFFSET = 8
FT_END = FT_OFFSET + INPUT * L1 * 2
L2_OFFSET = FT_END + L1 * 2
L2_BIAS_OFFSET = L2_OFFSET + 2 * L1 * L2 * 4
OUT_OFFSET = L2_BIAS_OFFSET + L2 * 4
OUT_BIAS_OFFSET = OUT_OFFSET + L2 * 4
WEIGHT_BYTES = OUT_BIAS_OFFSET + 4
POSITIVE_ZERO = b"\x00\x00\x00\x00"


class BoundedMaterialError(ValueError):
    """The byte artifact violates this fixed candidate contract."""


def _require(condition, message):
    if not condition:
        raise BoundedMaterialError(message)


def _shape(data):
    _require(type(data) is bytes, "weights must be immutable bytes")
    _require(len(data) == WEIGHT_BYTES, "wrong SEKIRW01 byte length")
    _require(data[:FT_OFFSET] == b"SEKIRW01", "wrong SEKIRW01 magic")


@lru_cache(maxsize=1)
def _reference():
    # Deterministic in-memory reference only; no source checkout, file or model
    # artifact is created. Bind the entire known initializer, not a new seed.
    data = material_init.encode(material_init.build_weights(SEED))
    _shape(data)
    _require(hashlib.sha256(data).hexdigest() == MATERIAL_SEED42_SHA256,
             "material-init seed42 implementation no longer matches the contract")
    return data


def _frozen_ft(data, reference):
    _require(data[FT_OFFSET:FT_END] == reference[FT_OFFSET:FT_END],
             "frozen FT bytes changed")
    _require(data[FT_END:L2_OFFSET] == reference[FT_END:L2_OFFSET],
             "frozen FT bias bytes changed")


@lru_cache(maxsize=1)
def _reference_native_roundtrip():
    prefix = _reference()[FT_OFFSET:L2_OFFSET]
    restored = bytearray()
    for (stored,) in struct.iter_unpack("<h", prefix):
        # Matches trainer.rs:526,529 and :570,575. Both operations are f32;
        # division/multiplication by 64 exactly preserve these initializer ints.
        raw = struct.unpack("<f", struct.pack("<f", stored / 64.0))[0]
        scaled = struct.unpack("<f", struct.pack("<f", raw * 64.0))[0]
        native = int(max(-32767.0, min(32767.0, scaled)))
        restored.extend(struct.pack("<h", native))
    _require(bytes(restored) == prefix, "frozen FT native roundtrip changed bytes")
    return hashlib.sha256(prefix).hexdigest()


def validate_native_ft_roundtrip(data):
    """Check frozen FT identity and its native f32 /64 -> *64 -> i16 orbit."""
    _shape(data)
    _frozen_ft(data, _reference())
    return {"status": "pure_arithmetic_pass", "ft_and_bias_values": INPUT * L1 + L1,
            "ft_and_bias_sha256": _reference_native_roundtrip(),
            "native_ft_bytes_equal": True,
            "reference_native_roundtrip_has_no_nearest_difference": True,
            "raw_checkpoint_ft_verified": False,
            "engine_verified": False, "training_verified": False}


def _fraction_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def validate_weights(data):
    """Validate inference bytes; no optimizer, activation or core claim is made.

    Free auxiliary L2/bias/out values may change, including signed zero. All
    frozen f32 coordinates are compared as bytes; required zeros are +0 bits.
    The bound uses exact rationals of the actual decoded IEEE binary32 values,
    never a decimal JSON approximation, signed sum, or tolerance.
    """
    _shape(data)
    floats = struct.unpack_from(f"<{(WEIGHT_BYTES - L2_OFFSET) // 4}f", data, L2_OFFSET)
    _require(all(math.isfinite(value) for value in floats), "nonfinite float weight")
    reference = _reference()
    _frozen_ft(data, reference)
    for row in range(2 * L1):
        start = L2_OFFSET + row * L2 * 4
        _require(data[start:start + 16] == reference[start:start + 16],
                 f"frozen material L2 columns changed at row {row}")
    for name, offset in (("L2 bias", L2_BIAS_OFFSET), ("out", OUT_OFFSET)):
        _require(data[offset:offset + 16] == reference[offset:offset + 16],
                 f"frozen material {name} bytes changed")
    _require(data[OUT_BIAS_OFFSET:] == POSITIVE_ZERO, "out bias must stay positive zero")
    for row in MATERIAL_ROWS:
        start = L2_OFFSET + (row * L2 + 4) * 4
        _require(data[start:start + len(AUXILIARY_COLUMNS) * 4]
                 == POSITIVE_ZERO * len(AUXILIARY_COLUMNS),
                 f"material input row {row} must have positive-zero auxiliary connections")
    auxiliary_out = struct.unpack_from("<28f", data, OUT_OFFSET + 4 * 4)
    norm = sum((abs(Fraction.from_float(value)) for value in auxiliary_out), Fraction())
    bound = norm * Fraction(127, 64)
    _require(bound <= REAL_RESIDUAL_CAP_CP, "auxiliary real residual budget exceeds 99 cp")
    roundtrip = validate_native_ft_roundtrip(data)
    return {"status": "pure_weight_bytes_pass", "weight_sha256": hashlib.sha256(data).hexdigest(),
            "weight_bytes": len(data), "material_initializer_sha256": MATERIAL_SEED42_SHA256,
            "seed": SEED, "frozen_coordinate_bytes_equal": True,
            "auxiliary_out_l1": _fraction_record(norm),
            "auxiliary_real_residual_bound_cp": _fraction_record(bound),
            "real_residual_cap_cp": REAL_RESIDUAL_CAP_CP,
            "native_ft_roundtrip": roundtrip,
            "engine_verified": False, "training_verified": False,
            "core_integer_difference_verified": False, "adoption_verified": False}


def candidate_recipe(data):
    """JSON-compatible preparation record with explicit unverified stages."""
    validation = validate_weights(data)
    return {"schema": "sekirei.bounded-material-preparation.v1",
            "status": "technical_preparation_only",
            "architecture": {"format": "SEKIRW01", "input": INPUT, "l1": L1, "l2": L2,
                             "nnue_output": "absolute"},
            "candidate_contract": {"seed": SEED, "real_residual_cap_cp": REAL_RESIDUAL_CAP_CP,
                "proposed_integer_cap_cp": PROPOSED_INTEGER_CAP_CP,
                "frozen": ["all FT and FT bias", "all 512 L2 rows, columns 0..3",
                           "L2 bias/out coordinates 0..3", "out bias positive zero",
                           "L2 rows 0,1,256,257 to columns 4..31 positive zero"],
                "mutable": ["other L2 columns 4..31", "L2 bias/out coordinates 4..31"],
                "bound": "exact Fraction: 127/64 * sum(abs(decoded binary32 auxiliary out)) <= 99"},
            "validation": validation,
            "preregistered": False, "real_data_preflight_verified": False,
            "trainer_mask_verified": False, "optimizer_verified": False,
            "core_integer_difference_verified": False, "engine_verified": False,
            "training_verified": False, "adoption_verified": False,
            "remaining": ["dedicated trainer source/build, masks and protected Adam moments",
                          "checkpoint/Adam/native-byte reproduction and provenance",
                          "finite activations, overflow and outward f32 error bound",
                          "core material-path, full residual limit and incremental/undo checks",
                          "candidate-owned pilot, formal MAE/Top3 and strict adoption comparison"]}
