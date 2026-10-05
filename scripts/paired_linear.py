"""Pure native serialization and dedicated absolute-output sidecar contract.

The native serializer is memory-only. Dedicated sidecar validation binds
externally frozen fit/control identities without executing their proof. No
fitter, checkpoint, Adam, trainer, engine, or adoption evidence is produced. The source-algebra bound is scoped
to unchanged seed42 FT/material inference and standard <=40 active features.
The conditional fit plan is frozen externally; its solver/projection, actual
activation, fitter and runtime proof are not implemented by this module.
"""
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
import struct

import material_init

PROTOTYPE_ONLY = True
MODE = 'paired-linear-constrained-ridge1-l1-39p5-v1'
CONDITIONAL_PLAN_SHA256 = 'dabad54e419237335fd1f370063a0c6e82ac7b58ac27f0d20fd91ca21d9dd900'
SEED = 42
INPUT, L1, L2 = 2420, 256, 32
CHANNELS = tuple(range(2, L1))
PAIR_COLUMNS = (4, 5)
MATERIAL_ROWS = (0, 1, 256, 257)
MATERIAL_COLUMNS = (0, 1, 2, 3)
FT_OFFSET = 8
FT_END = FT_OFFSET + INPUT * L1 * 2
L2_OFFSET = FT_END + L1 * 2
L2_BIAS_OFFSET = L2_OFFSET + 2 * L1 * L2 * 4
OUT_OFFSET = L2_BIAS_OFFSET + L2 * 4
OUT_BIAS_OFFSET = OUT_OFFSET + L2 * 4
WEIGHT_BYTES = OUT_BIAS_OFFSET + 4
SIGN_BIT = 0x80000000
POSITIVE_ZERO = b'\0\0\0\0'
MATERIAL_SEED42_SHA256 = 'bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40'
EXPECTED_SOURCE_HASHES = {
    'crates/sekirei-core/src/nnue.rs': 'a467e8b1b75f6b82629f369a1c804476b1c610354a561822365f980e6e428561',
    'crates/sekirei-core/src/eval.rs': '69fce9b3522a96c68a7f6dce15a342ba34c90f0addd0fa13b7478bef4d25e839',
    'crates/sekirei-core/src/piece.rs': '2ccf9248daf5a5b5eba2651ce03a5f85918608b4c5651ab5c728ddf6ba0a50a3',
    'crates/sekirei-core/src/square.rs': '60ee89de778fc18e30bbe0897f7380e4dfbf2d80f27afbe6d2b4e405c92e68b2'}
L1_CAP = Fraction(79, 2)
ACTIVE_FEATURE_CAP = 40
DIFFERENCE_CAP = Fraction(5, 4)
REAL_RESIDUAL_CAP = Fraction(395, 4)
SOURCE_CONDITIONAL_FLOAT_MARGIN = Fraction(3, 100)
SOURCE_CONDITIONAL_INTEGER_CAP = 99
OBSERVED_INTEGER_CORE_GUARD_CP = 100


class PairedLinearError(ValueError):
    """Native inference bytes violate this candidate-specific prototype."""


def require(condition, message):
    if not condition:
        raise PairedLinearError(message)


def fraction_record(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(exact(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def shape(data):
    require(type(data) is bytes, 'immutable native bytes required')
    require(len(data) == WEIGHT_BYTES and data[:FT_OFFSET] == b'SEKIRW01', 'exact default SEKIRW01 shape/magic required')


@lru_cache(maxsize=1)
def reference():
    # Public deterministic initializer in memory. No candidate artifact is made.
    require(material_init.SOURCE_HASHES == EXPECTED_SOURCE_HASHES, 'existing source guards changed')
    require((material_init.INPUT, material_init.L1, material_init.L2) == (INPUT, L1, L2), 'default dimensions changed')
    data = material_init.encode(material_init.build_weights(SEED))
    shape(data)
    require(hashlib.sha256(data).hexdigest() == MATERIAL_SEED42_SHA256, 'seed42 reference implementation changed')
    return data


def l2_cell(row, column):
    return L2_OFFSET + (row * L2 + column) * 4


def bits(data, offset):
    return struct.unpack_from('<I', data, offset)[0]


def _validate_coordinates(data):
    """Validate exact stored IEEE f32 coordinates and their functional cap.

    The independent coefficient is row(j), column4. Its zero must be +0. Every
    coefficient has exact raw-bit copies (+u,-u,-u,+u), including (+0,-0,-0,+0).
    All other unused auxiliary coordinates must be +0. Bounds use exact Fractions
    of the stored binary32 values, never tolerances, rounded JSON or signed sums.
    """
    shape(data)
    require(material_init.SOURCE_HASHES == EXPECTED_SOURCE_HASHES, 'existing source guards changed')
    values = struct.unpack_from(f'<{(WEIGHT_BYTES-L2_OFFSET)//4}f', data, L2_OFFSET)
    require(all(math.isfinite(value) for value in values), 'all stored f32 weights must be finite')
    base = reference()
    require(data[FT_OFFSET:L2_OFFSET] == base[FT_OFFSET:L2_OFFSET], 'all FT/FT bias bytes must equal original seed42')
    for row in range(2 * L1):
        offset = l2_cell(row, 0)
        require(data[offset:offset + 16] == base[offset:offset + 16], 'frozen material L2 columns changed')
        zero_start = l2_cell(row, 6)
        require(data[zero_start:zero_start + 26 * 4] == POSITIVE_ZERO * 26, 'unused L2 columns6..31 must be +0')
    for row in MATERIAL_ROWS:
        offset = l2_cell(row, 4)
        require(data[offset:offset + 8] == POSITIVE_ZERO * 2, 'material input rows must not enter paired head')
    coefficient_bytes, norm = bytearray(), Fraction()
    for channel in CHANNELS:
        origin = bits(data, l2_cell(channel, 4))
        require(origin != SIGN_BIT, 'independent zero coefficient must be canonical positive zero')
        for row, column, expected in ((channel, 4, origin), (channel, 5, origin ^ SIGN_BIT),
                (L1 + channel, 4, origin ^ SIGN_BIT), (L1 + channel, 5, origin)):
            require(bits(data, l2_cell(row, column)) == expected, 'shared coefficient signed-mirror bits differ')
        native = struct.unpack('<f', struct.pack('<I', origin))[0]
        coefficient_bytes.extend(struct.pack('<I', origin))
        norm += abs(Fraction.from_float(native))
    require(norm <= L1_CAP, 'exact stored f32 coefficient L1 exceeds39.5')
    for offset, name in ((L2_BIAS_OFFSET, 'L2 bias'), (OUT_OFFSET, 'out')):
        require(data[offset:offset + 16] == base[offset:offset + 16], 'frozen material ' + name + ' changed')
        require(data[offset + 6 * 4:offset + L2 * 4] == POSITIVE_ZERO * 26, 'unused auxiliary ' + name + ' must be +0')
    require(data[L2_BIAS_OFFSET + 4 * 4:L2_BIAS_OFFSET + 6 * 4] == struct.pack('<2f', 64., 64.), 'paired biases must be64/64')
    require(data[OUT_OFFSET + 4 * 4:OUT_OFFSET + 6 * 4] == struct.pack('<2f', 64., -64.), 'paired outputs must be+64/-64')
    require(data[OUT_BIAS_OFFSET:] == POSITIVE_ZERO, 'out bias must be positive zero')
    residual = 2 * DIFFERENCE_CAP * norm
    excursion = DIFFERENCE_CAP * norm
    return {'schema': 'sekirei.paired-linear-pure-native-validation.v1', 'status': 'pure_native_bytes_pass',
        'mode': MODE, 'conditional_plan_sha256': CONDITIONAL_PLAN_SHA256,
        'weight_sha256': hashlib.sha256(data).hexdigest(), 'weight_bytes': len(data),
        'material_initializer_sha256': MATERIAL_SEED42_SHA256, 'seed': SEED,
        'frozen_ft_and_material_bytes_equal': True, 'shared_coefficients': len(CHANNELS),
        'signed_mirror_bits_verified': True, 'independent_zero_rule': '+0; exactsignbit XOR mirrors',
        'unused_auxiliary_positive_zero_verified': True,
        'coefficient_bits_sha256': hashlib.sha256(coefficient_bytes).hexdigest(),
        'saved_f32_coefficient_l1': fraction_record(norm), 'coefficient_l1_cap': fraction_record(L1_CAP),
        'source_conditional_real_residual_bound_cp': fraction_record(residual),
        'source_conditional_preactivation_interval': [fraction_record(64-excursion), fraction_record(64+excursion)],
        'standard_active_feature_cap': ACTIVE_FEATURE_CAP,
        'source_conditional_float_margin_cp': fraction_record(SOURCE_CONDITIONAL_FLOAT_MARGIN),
        'source_conditional_integer_material_cap_cp': SOURCE_CONDITIONAL_INTEGER_CAP,
        'required_observed_integer_core_guard_cp': OBSERVED_INTEGER_CORE_GUARD_CP,
        'real_algebra_stm_antisymmetry_conditional': True,
        'native_bit_exact_stm_antisymmetry_verified': False, 'actual_core_bound_verified': False,
        'legal_dataset_inventory_verified': False, 'actual_fitter_verified': False,
        'preregistered': False, 'activation_verified': False, 'adoption_verified': False}


COEFFICIENT_BYTES = 254 * 4
from pathlib import Path
import json

C = Path('/home/server/.local/share/sekirei-weight2/campaign-17-autonomous-v1')
Q = C.parent / 'training-17-v1/paired-linear-constrained-ridge1-v1/run-v1'
W = Q / 'weights.bin'
COEFFICIENTS = Q / 'coefficients.f32.bin'
FIT_RECEIPT = Q / 'run.json'
PREREGISTRATION = C / 'paired-linear-preregistration-v1.json'
ACTIVATION = C / 'paired-linear-activation-v1.json'


def validate_coefficients(coefficients_f32):
    require(type(coefficients_f32) is bytes and len(coefficients_f32) == COEFFICIENT_BYTES,
            'exact immutable 254 little-endian f32 coefficients required')
    norm = Fraction()
    for (raw,) in struct.iter_unpack('<I', coefficients_f32):
        require(raw != SIGN_BIT, 'independent zero coefficient must be canonical positive zero')
        value = struct.unpack('<f', struct.pack('<I', raw))[0]
        require(math.isfinite(value), 'all coefficient f32 values must be finite')
        norm += abs(Fraction.from_float(value))
    require(norm <= L1_CAP, 'exact stored f32 coefficient L1 exceeds39.5')
    return norm


def serialize_coefficients(coefficients_f32, initializer=None):
    """Pure memory serializer; never accepts a filesystem path or writes a file.

    The seed42 initializer, if supplied, must match every byte of the pinned
    public initializer. No FT conversion, nearest export, checkpoint or Adam is
    involved. Signed mirror coordinates are written by raw-bit XOR.
    """
    validate_coefficients(coefficients_f32)
    require(material_init.SOURCE_HASHES == EXPECTED_SOURCE_HASHES, 'existing source guards changed')
    base = reference()
    if initializer is not None:
        require(type(initializer) is bytes and initializer == base,
                'initializer must equal all original seed42 native bytes')
    data = bytearray(base)
    for row in range(512):
        start = l2_cell(row, 4)
        data[start:start + 28 * 4] = POSITIVE_ZERO * 28
    for offset in (L2_BIAS_OFFSET, OUT_OFFSET):
        data[offset + 16:offset + L2 * 4] = POSITIVE_ZERO * 28
    struct.pack_into('<2f', data, L2_BIAS_OFFSET + 16, 64.0, 64.0)
    struct.pack_into('<2f', data, OUT_OFFSET + 16, 64.0, -64.0)
    data[OUT_BIAS_OFFSET:] = POSITIVE_ZERO
    for channel, (raw,) in zip(CHANNELS, struct.iter_unpack('<I', coefficients_f32)):
        for row, column, stored in ((channel, 4, raw), (channel, 5, raw ^ SIGN_BIT),
                                    (256 + channel, 4, raw ^ SIGN_BIT), (256 + channel, 5, raw)):
            struct.pack_into('<I', data, l2_cell(row, column), stored)
    return bytes(data)


def validate_native(data, coefficients_f32, initializer=None):
    """Exact full-byte reconstruction, plus conditional functional cap."""
    result = _validate_coordinates(data)
    require(data == serialize_coefficients(coefficients_f32, initializer),
            'native all-byte rebuild from externally bound coefficients differs')
    require(result['coefficient_bits_sha256'] == hashlib.sha256(coefficients_f32).hexdigest(),
            'coefficient byte identity differs')
    return {**result, 'canonical_full_native_rebuild_equal': True,
            'coefficient_bytes': COEFFICIENT_BYTES,
            'quantization_reexport_claimed': False, 'adam_used': False, 'epochs': 0}


def validate_weights(data):
    """Byte contract only; callers that bind coefficient artifact use validate_native."""
    result = _validate_coordinates(data)
    coefficients = b''.join(data[l2_cell(j, 4):l2_cell(j, 4)+4] for j in CHANNELS)
    require(data == serialize_coefficients(coefficients), 'native canonical full-byte rebuild differs')
    return {**result, 'canonical_full_native_rebuild_equal': True,
            'quantization_reexport_claimed': False, 'adam_used': False, 'epochs': 0}


def sha(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            'invalid typed SHA256')
    return value


def file_reference(value, path=None):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
            and type(value['path']) is str and Path(value['path']).is_absolute()
            and Path(value['path']).as_posix() == value['path'] and '..' not in Path(value['path']).parts
            and type(value['bytes']) is int and value['bytes'] >= 0,
            'invalid typed canonical file reference')
    if path is not None:
        require(value['path'] == str(path), 'wrong fixed file reference path')
    sha(value['sha256'])
    return value


def strict_json(data):
    require(type(data) is bytes, 'immutable JSON bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'nonfinite JSON constant'))


def validate_fit_binding(binding):
    """This expected binding must come from the outer SHA-frozen workflow.

    Hash equality provides identity, not solver/technical proof. Dedicated
    driver must independently verify its certificates and lifecycle before
    freezing this binding. A run cannot supply its own expected binding.
    """
    require(type(binding) is dict and set(binding) == {'plan_sha256', 'preregistration_sha256',
            'activation_sha256', 'fit_receipt', 'solver_certificate', 'source_helpers_sha256'},
            'complete externally frozen fit binding required')
    require(binding['plan_sha256'] == CONDITIONAL_PLAN_SHA256, 'wrong fixed fit plan')
    for key in ('plan_sha256', 'preregistration_sha256', 'activation_sha256'):
        sha(binding[key])
    file_reference(binding['fit_receipt'], FIT_RECEIPT)
    file_reference(binding['solver_certificate'])
    require(Path(binding['solver_certificate']['path']).parent == Q,
            'solver certificate must belong to dedicated run')
    helpers = binding['source_helpers_sha256']
    require(type(helpers) is dict and {'paired_linear.py', 'paired_linear_activation.py'} <= set(helpers),
            'dedicated helper identities required')
    for key, value in helpers.items():
        require(type(key) is str and key == Path(key).name and key.endswith('.py'), 'invalid helper filename')
        sha(value)
    return binding


def validate_fit_run(fit_run_bytes, data, coefficients_f32, binding):
    validate_fit_binding(binding)
    ref = binding['fit_receipt']
    require(type(fit_run_bytes) is bytes and len(fit_run_bytes) == ref['bytes']
            and hashlib.sha256(fit_run_bytes).hexdigest() == ref['sha256'],
            'fit run bytes differ from externally frozen receipt')
    run = strict_json(fit_run_bytes)
    require(type(run) is dict and run.get('schema') == 'sekirei.paired-linear-fit-run.v1'
            and run.get('status') == 'complete' and run.get('candidate') == MODE,
            'dedicated completed fit receipt required')
    fields = {'mode': MODE, 'seed': 42, 'nnue_output': 'absolute', 'optimizer': 'projected-FISTA',
              'adam_used': False, 'epochs': 0, 'resume_used': False, 'trainer_used': False,
              'inputs_unchanged': True, 'source_unchanged': True, 'cleanup_verified': True,
              'final_used': False, 'adoption_claimed': False,
              'plan_sha256': CONDITIONAL_PLAN_SHA256,
              'preregistration_sha256': binding['preregistration_sha256'],
              'activation_sha256': binding['activation_sha256']}
    for key, expected in fields.items():
        require(exact(run.get(key), expected), 'fit contract/flag/type mismatch: ' + key)
    require(not any(key in run for key in ('selected_epoch', 'epoch', 'adam_checkpoint', 'resume_fingerprint')),
            'Adam or epoch checkpoint cannot represent a dedicated fit')
    require(exact(run.get('source_helpers_sha256'), binding['source_helpers_sha256'])
            and exact(run.get('solver_certificate'), binding['solver_certificate']),
            'fit source or external solver certificate binding differs')
    outputs = run.get('outputs')
    require(type(outputs) is dict and {'weights', 'coefficients_f32'} <= set(outputs),
            'fit native and f32 output identities missing')
    for value in outputs.values():
        file_reference(value)
        require(Path(value['path']).parent == Q, 'fit output outside dedicated run')
        require(value['path'] not in (str(FIT_RECEIPT), str(W.with_suffix('.meta.json'))),
                'run cannot hash itself or its dependent sidecar')
    require(exact(outputs['weights'], {'path': str(W), 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()})
            and exact(outputs['coefficients_f32'], {'path': str(COEFFICIENTS), 'bytes': len(coefficients_f32),
                        'sha256': hashlib.sha256(coefficients_f32).hexdigest()}),
            'fit native/coefficient artifact identity differs')
    validate_native(data, coefficients_f32)
    return run


def sidecar_for_fit(data, coefficients_f32, fit_run_bytes, binding):
    """Pure construction after strict fit identity validation; no proof execution."""
    validate_fit_run(fit_run_bytes, data, coefficients_f32, binding)
    validation = validate_native(data, coefficients_f32)
    return {'format': 'sekirei-nnue-output-v1', 'nnue_output': 'absolute',
            'checkpoint_hash': material_init.fnv1a(data),
            'paired_linear_constrained_ridge_v1': {
                'schema': 'sekirei.paired-linear-fit-metadata.v1', 'mode': MODE,
                'plan_sha256': CONDITIONAL_PLAN_SHA256, 'seed': 42,
                'material_initializer_sha256': MATERIAL_SEED42_SHA256,
                'source_hashes': dict(EXPECTED_SOURCE_HASHES),
                'weight_sha256': hashlib.sha256(data).hexdigest(), 'weight_bytes': len(data),
                'coefficients_f32': {'path': str(COEFFICIENTS), 'bytes': len(coefficients_f32),
                                    'sha256': hashlib.sha256(coefficients_f32).hexdigest()},
                'saved_f32_coefficient_l1': validation['saved_f32_coefficient_l1'],
                'coefficient_l1_cap': fraction_record(L1_CAP),
                'required_observed_integer_core_guard_cp': 100,
                'optimizer': 'projected-FISTA', 'adam_used': False, 'epochs': 0,
                'resume_used': False, 'trainer_used': False,
                'fit_receipt': dict(binding['fit_receipt']),
                'solver_certificate': dict(binding['solver_certificate']),
                'preregistration_sha256': binding['preregistration_sha256'],
                'activation_sha256': binding['activation_sha256'],
                'source_helpers_sha256': dict(binding['source_helpers_sha256']),
                'canonical_full_native_rebuild_equal': True,
                'quantization_reexport_claimed': False,
                'native_bit_exact_stm_antisymmetry_verified': False,
                'actual_core_bound_verified': False, 'adoption_verified': False}}


def validate_sidecar(metadata, data, coefficients_f32, fit_run_bytes, expected_binding):
    require(exact(metadata, sidecar_for_fit(data, coefficients_f32, fit_run_bytes, expected_binding)),
            'typed dedicated absolute sidecar/fit/native/FNV binding differs')
    return {'schema': 'sekirei.paired-linear-sidecar-validation.v1', 'status': 'pure-byte-provenance-pass',
            'mode': MODE, 'canonical_full_native_rebuild_equal': True,
            'fit_receipt_sha256': expected_binding['fit_receipt']['sha256'],
            'actual_core_bound_verified': False, 'solver_certificate_verified_here': False,
            'quantization_reexport_claimed': False, 'adoption_verified': False}


def entry(*_args, **_kwargs):
    require(not PROTOTYPE_ONLY, 'PROTOTYPE_ONLY: no actual fitter entry or activation is permitted')
    raise NotImplementedError('Dedicated outer activation/PR/source/lifecycle fitter integration is not implemented')
