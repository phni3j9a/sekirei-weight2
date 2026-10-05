"""Gated new03 white-view paired-linear fitter for the supervised parent.

Pure extraction accepts caller-verified original bytes, never reads a dataset.
The dedicated gated CLI requires pinned activation, original-source preflight,
and build identities. Its parent holds the six fixed locks and two shared build
locks, and supervises the entire process group with a 1200-second wall deadline.
Only that parent may claim child cleanup and create run.json / metadata.
The disabled preparation source is preserved separately. Numeric modules keep
their blocked generic runtime entries.
"""
import os
import time
_PROCESS_STARTED = time.monotonic()
_NUMPY_PREIMPORTED = 'numpy' in __import__('sys').modules
THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
for _name in THREAD_ENV:
    os.environ[_name] = '1'

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import stat
import struct
import sys

PROTOTYPE_ONLY = False
TRAIN_COUNT = 112681
HOLDOUT_COUNT = 5895
DIMENSION = 254
WALL_SECONDS = 1200.0
INITIALIZER_SHA256 = 'bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40'
MANIFEST_SHA256 = 'ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6'
TEACHER = 'external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d'
NUMERIC_HASHES = {
    'verified_design_gram.py': 'ca65fc08248cb4327b1fb0ec733a65af303673010217e952ea49c878498a3b68',
    'paired_linear_solver_verified_design.py': '7884f324b73d38bbd2202a6d7518dab81963f6d5453cd0f50fea3c2fc35511d1',
    'paired_linear_solver.py': 'ba9c86b37e3ed442db930332f16f3390ff5cd84a98431fcceea84cc4dd9b066b',
    'material_init.py': 'd7943d5d9e01946f17300a38c143eccb0017af37f36a70554dd394a70f6bfbad',
    'functional_anchor.py': 'e704e153e6752abe1f3a6b686ac5e9eddea688216cd28626fffb598beef8130b',
    'paired_linear.py': 'b2df03bd42444958eb99bce7f856f36912d8f117e8d96b3ff88e84d5cf478272',
    'white_view_paired_linear.py': '79845db6f8ceec2e0c73488b705db526813da52bcc601ab6b10eb3f52f0ce6eb',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    require(type(data) is bytes, 'immutable bytes required')
    return hashlib.sha256(data).hexdigest()


def strict_sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'strict SHA256 required')
    return value


def exact(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(exact(a[k], b[k]) for k in a)
    if type(a) in (tuple, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def module(name):
    return importlib.import_module(name)


def numpy():
    np = module('numpy')
    require(np.__version__ == '1.26.4', 'fixed NumPy 1.26.4 required')
    return np


def deadline_check():
    require(time.monotonic() - _PROCESS_STARTED < WALL_SECONDS, 'whole child wall deadline exhausted')


def rational_json(value):
    """Canonical exact finite float/Fraction records; all solver keys retained."""
    if type(value) is Fraction or type(value) is float:
        require(type(value) is Fraction or math.isfinite(value), 'nonfinite certificate value')
        q = Fraction(value)
        return {'numerator': q.numerator, 'denominator': q.denominator}
    if type(value) in (int, bool, str) or value is None:
        return value
    if type(value) in (tuple, list):
        return [rational_json(item) for item in value]
    if type(value) is dict:
        require(all(type(k) is str for k in value), 'JSON keys must be strings')
        return {k: rational_json(v) for k, v in value.items()}
    raise ValueError('unsupported certificate type')


def json_bytes(value):
    return (json.dumps(rational_json(value), sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


@dataclass(frozen=True)
class TrainDesign:
    z_bytes: bytes
    d_bytes: bytes
    count: int
    dimension: int
    provenance: dict


def _manifest_train(manifest_bytes, position_bytes, label_bytes, expected_manifest_sha256):
    anchor = module('functional_anchor')
    require(sha(manifest_bytes) == strict_sha(expected_manifest_sha256), 'original manifest SHA mismatch')
    manifest = anchor._json(manifest_bytes)
    require(type(manifest) is dict and type(manifest.get('schema_version')) is int
            and manifest['schema_version'] == 1 and type(manifest.get('files')) is dict
            and set(manifest['files']) == anchor.FILES
            and type(manifest.get('positions')) is dict
            and set(manifest['positions']) == {'train', 'holdout'}
            and 'functional_anchor' not in manifest and 'split_teacher_identities' not in manifest,
            'original four-file manifest required')
    games = anchor._source_provenance(manifest)
    for split in ('train', 'holdout'):
        anchor._integer(manifest['positions'][split], 'split count', 1)
    for name, info in manifest['files'].items():
        require(type(info) is dict, 'invalid declared original file metadata')
        anchor._sha(info.get('sha256'))
        anchor._integer(info.get('bytes'), 'file bytes', 1)
        if 'count' in info:
            anchor._integer(info['count'], 'file count', 1)
    rows = {}
    for name, data in (('train.positions.jsonl', position_bytes), ('train.labels.jsonl', label_bytes)):
        info = manifest['files'][name]
        require(type(data) is bytes and sha(data) == info['sha256'] and len(data) == info['bytes'],
                'TRAIN input hash/size mismatch')
        rows[name] = anchor._rows(data)
        require(len(rows[name]) == manifest['positions']['train']
                and ('count' not in info or len(rows[name]) == info['count']), 'TRAIN count mismatch')
    return manifest, games, rows


def build_train_design(manifest_bytes, position_bytes, label_bytes, initializer_bytes, *,
                       expected_manifest_sha256, feature_binding, check_deadline=lambda: None):
    """Pure original TRAIN SFEN-keyed extraction; position order is authoritative.

    Only train input bytes are parsed. Declared holdout identities/provenance are
    retained; whole-dataset legal replay, exclusions and non-overlap come from
    the externally frozen source preflight and are NOT proved by this function.
    Synthetic manifests can use a synthetic teacher identity; the actual CLI
    additionally pins the fixed O teacher / manifest / 112681 TRAIN rows.
    Original label extra metadata remains in the unchanged original bytes;
    derived design provenance binds their full SHA, no metadata is discarded
    from or rewritten to the original cache.
    """
    require(callable(check_deadline), 'deadline callback required')
    check_deadline()
    manifest, games, rows = _manifest_train(manifest_bytes, position_bytes, label_bytes,
                                          expected_manifest_sha256)
    material = module('material_init')
    require(type(initializer_bytes) is bytes and len(initializer_bytes) == material.EXPECTED_BYTES
            and sha(initializer_bytes) == INITIALIZER_SHA256, 'fixed seed42 initializer bytes required')
    white = module('white_view_paired_linear')
    white.validate_binding(feature_binding)
    original01 = initializer_bytes
    initializer_bytes = white.transform_initializer(original01, binding=feature_binding)
    np = numpy()
    require(initializer_bytes[:8] == b'SEKIRW03', 'new native initializer magic mismatch')
    ft = np.frombuffer(initializer_bytes, dtype='<i2', offset=8,
                       count=material.INPUT * material.L1).reshape(material.INPUT, material.L1)
    bias = np.frombuffer(initializer_bytes, dtype='<i2', offset=8 + material.INPUT * material.L1 * 2,
                         count=material.L1)
    aux = ft[:, 2:256]
    require(bool(((aux == -1) | (aux == 1)).all()) and bool((bias[2:256] == 64).all()),
            'fixed auxiliary FT +/-1 and bias64 required')
    labels = rows['train.labels.jsonl']
    cache = {}
    for row in labels:
        sfen = row.get('sfen')
        anchor = module('functional_anchor')
        anchor._cp(row.get('score_cp'), 'original teacher score')
        require(type(sfen) is str and sfen not in cache
                and row.get('teacher_identity') == manifest['teacher_identity']
                and type(row.get('label_depth')) is int and row['label_depth'] == 0,
                'invalid/duplicate original label')
        cache[sfen] = row['score_cp']
    positions = rows['train.positions.jsonl']
    require(set(cache) == {row.get('sfen') for row in positions}, 'missing/extra SFEN-keyed TRAIN label')
    n = len(positions)
    require(type(n) is int and 1 <= n <= 1_000_000, 'TRAIN builder resource count exceeded')
    Z = np.empty((n, DIMENSION), dtype=np.int8, order='C')
    d = np.empty(n, dtype='<i4')
    seen_sfens, seen_boards = set(), set()
    ordered = hashlib.sha256()
    raw_min, raw_max, feature_min, feature_max, dmax = 104, 24, 40, 0, 0
    for index, row in enumerate(positions):
        if index % 256 == 0:
            check_deadline()
        sfen = row.get('sfen')
        require(type(sfen) is str and sfen and '\n' not in sfen and '\r' not in sfen,
                'canonical single-line SFEN required')
        position = material.parse_sfen(sfen)
        board = (tuple(sorted(position['pieces'])), tuple(tuple(x) for x in position['hands']), position['stm'])
        source = row.get('source')
        require(type(source) is dict and source.get('kind') == 'gensfen-pack'
                and source.get('path') in games and games[source['path']]['split'] == 'train'
                and type(row.get('schema_version')) is int and row['schema_version'] == 1
                and sfen not in seen_sfens and board not in seen_boards,
                'duplicate semantic position or non-TRAIN source')
        seen_sfens.add(sfen); seen_boards.add(board)
        us = white.active_features(position, position['stm'])
        them = white.active_features(position, 1 - position['stm'])
        require(2 <= len(us) == len(them) <= 40 and len(set(us)) == len(us)
                and len(set(them)) == len(them), 'standard40/equal feature count required')
        rawus = aux[us, :].sum(axis=0, dtype=np.int16) + np.int16(64)
        rawthem = aux[them, :].sum(axis=0, dtype=np.int16) + np.int16(64)
        difference = rawus - rawthem
        require(bool(((rawus >= 24) & (rawus <= 104)).all())
                and bool(((rawthem >= 24) & (rawthem <= 104)).all())
                and bool((difference % 2 == 0).all())
                and bool((np.abs(difference) <= 80).all()), 'FT raw range/parity proof failed')
        Z[index] = difference // 2
        M = material.material(position)
        require(type(M) is int and abs(M) <= 25780, 'fixed material absolute STM bound failed')
        target = cache[sfen] - M
        require(type(target) is int and abs(target) <= 55779, 'original T-M target bound failed')
        d[index] = target
        raw_min = min(raw_min, int(rawus.min()), int(rawthem.min()))
        raw_max = max(raw_max, int(rawus.max()), int(rawthem.max()))
        feature_min, feature_max = min(feature_min, len(us)), max(feature_max, len(us))
        dmax = max(dmax, abs(target))
        ordered.update(sfen.encode('utf-8') + b'\n')
    check_deadline()
    zb, db = Z.tobytes(order='C'), d.tobytes(order='C')
    provenance = {'schema': 'sekirei.white-view-paired-linear-train-design.v1', 'count': n, 'dimension': DIMENSION,
        'teacher_identity': manifest['teacher_identity'], 'original_manifest_sha256': sha(manifest_bytes),
        'original_files': manifest['files'], 'source_provenance': {k: manifest[k] for k in
            ('source_corpus_manifest_sha256', 'seed', 'split', 'sampling', 'games_per_pack',
             'dependencies', 'independent_exclusions', 'derivation', 'games')},
        'train_positions_sha256': sha(position_bytes), 'train_labels_sha256': sha(label_bytes),
        'material_initializer_sha256': sha(original01),
        'transformed_initializer_sha256': sha(initializer_bytes), 'native_magic': 'SEKIRW03',
        'feature_schema': white.FEATURE_SCHEMA, 'feature_binding': feature_binding,
        'feature_index_implementation': 'fixed white_view_paired_linear.active_features',
        'single_numpy_integer_adapter': True, 'actual_build_verified_by_builder': False,
        'ft_prefix_sha256': sha(initializer_bytes[:8 + material.INPUT * material.L1 * 2 + material.L1 * 2]),
        'ordered_sfens_sha256': ordered.hexdigest(), 'position_order_preserved': True,
        'labels_joined_by_exact_sfen': True, 'labels_extra_metadata_bound_in_original_sha': True,
        'z_dtype': 'signed-int8-C-row-major', 'd_dtype': 'signed-int32-little-endian',
        'z_definition': '(raw_us-raw_them)/2; channels 2:256', 'x_definition': 'Z/16',
        'd_definition': 'original absolute STM teacher T minus fixed material M',
        'z_sha256': sha(zb), 'd_sha256': sha(db), 'ft_raw_min': raw_min, 'ft_raw_max': raw_max,
        'active_feature_min': feature_min, 'active_feature_max': feature_max,
        'maximum_absolute_d': dmax, 'holdout_rows_read': False, 'development_used': False,
        'final_used': False, 'whole_dataset_replay_proof_claimed': False}
    return TrainDesign(zb, db, n, DIMENSION, provenance)


class OpenBLASOneThread:
    """Runtime evidence from NumPy's already loaded, externally hashed library.

    Environment variables alone cannot prove an existing pool's thread count.
    Linux RTLD_NOLOAD prevents loading an unrelated backend for a fake receipt.
    Fail closed if the exact bundled ILP64 symbols/library are unavailable.
    """
    def __init__(self, expected_runtime_sources):
        np = numpy()
        root = Path(np.__file__).resolve().parent.parent / 'numpy.libs'
        paths = list(root.glob('libopenblas64_p-*.so'))
        require(len(paths) == 1, 'exact one NumPy bundled OpenBLAS required')
        self.path = paths[0].resolve()
        require(str(self.path) in expected_runtime_sources, 'bundled BLAS absent from frozen runtime sources')
        self.expected = expected_runtime_sources[str(self.path)]
        self.identity = file_identity(self.path)
        require(exact(self.identity, self.expected), 'bundled BLAS bytes differ')
        self.lib = ctypes.CDLL(str(self.path), mode=os.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_NOLOAD)
        self.getter = self.lib.openblas_get_num_threads64_
        self.getter.argtypes = []; self.getter.restype = ctypes.c_int
        self.setter = self.lib.openblas_set_num_threads64_
        self.setter.argtypes = [ctypes.c_int]; self.setter.restype = None
        self.config = self.lib.openblas_get_config64_
        self.config.argtypes = []; self.config.restype = ctypes.c_char_p
        self.before = self.getter()
        self.setter(1)
        require(self.getter() == 1, 'OpenBLAS setter/getter failed to enforce one thread')
        self.configuration = self.config().decode('ascii')
        require('USE64BITINT' in self.configuration, 'NumPy ILP64 backend identity differs')
    def verify(self):
        require(self.getter() == 1 and exact(file_identity(self.path), self.identity),
                'OpenBLAS thread count/library changed')
        return {'schema': 'sekirei.numpy-openblas-thread-proof.v1', 'numpy_version': '1.26.4',
            'library': {'path': str(self.path), **self.identity},
            'get_symbol': 'openblas_get_num_threads64_', 'set_symbol': 'openblas_set_num_threads64_',
            'config_symbol': 'openblas_get_config64_', 'config': self.configuration,
            'threads_before': self.before, 'threads_during': 1, 'threads_after': self.getter(),
            'numpy_preimported_when_driver_loaded': _NUMPY_PREIMPORTED,
            'environment': {name: os.environ[name] for name in THREAD_ENV},
            'rtld_noload': True, 'threadpool_verified': True}


def file_identity(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and stat.S_ISREG(path.stat().st_mode),
            'canonical regular input path required')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def fullref(path, data):
    return {'path': str(path), 'bytes': len(data), 'sha256': sha(data)}


def write_new(path, data):
    require(type(data) is bytes, 'bytes artifact required')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    return fullref(path, data)


def sole_gram(result, producer_sha256):
    producer = module('verified_design_gram')
    require(len(producer._gram_origins) == 1, 'fresh child must issue exactly one Gram')
    artifact = next(iter(producer._gram_origins.values()))[0]
    binding = result['verified_design_gram']
    problem = producer.consume_gram(artifact, expected_design_sha256=binding['design_sha256'],
        expected_gram_sha256=binding['gram_sha256'], expected_producer_source_sha256=producer_sha256)
    return problem, artifact


def coefficient_bytes(result):
    f64, f32 = result['coefficient_f64'], result['coefficient_f32']
    require(len(f64) == len(f32) == DIMENSION, '254 coefficients required')
    for value in (*f64, *f32):
        require(type(value) is float and math.isfinite(value), 'finite float coefficients required')
    # Predetermined numeric export already canonicalizes either signed zero.
    packed64 = b''.join(struct.pack('<d', value) for value in f64)
    packed32 = b''.join(struct.pack('<f', 0.0 if value == 0.0 else value) for value in f32)
    require(tuple(struct.unpack('<254f', packed32)) == tuple(f32), 'f32 saved values changed')
    require(all(value != 0.0 or packed32[4*i:4*i+4] == b'\0'*4 for i, value in enumerate(f32)),
            'independent f32 zero must be canonical positive zero')
    for i, value in enumerate(f64):
        converted = struct.unpack('<f', struct.pack('<f', value))[0]
        expected = struct.pack('<f', 0.0 if converted == 0.0 else converted)
        require(expected == packed32[4*i:4*i+4], 'f32 is not canonical nearest cast of f64')
    return packed64, packed32


def make_run_template(context, outputs, solverref):
    return module('white_view_fit_contract').make_run_template(context, outputs, solverref)


def construct_fit_run(context, outputs, solverref, lifecycle):
    """Parent-only pure construction; child cannot prove its own reaping."""
    return module('white_view_fit_contract').construct_fit_run(context, outputs, solverref, lifecycle)


def _read_pinned(path, expected):
    data = Path(path).read_bytes()
    rec = file_identity(path)
    if type(expected) is str:
        require(rec['sha256'] == strict_sha(expected), 'pinned input SHA mismatch')
    else:
        require(exact(rec, expected), 'pinned input identity mismatch')
    require(sha(data) == rec['sha256'] and len(data) == rec['bytes'], 'input changed while read')
    return data


BUILD_CONTRACT_SHA256 = '305d4a2148feecfc0b25ca7e6834c108a635e250cf5b42124511ab3c976b8144'


def _load_runtime_validator(ref, runtime_sources):
    """Pinned real adjacent module; temporary cache binding is serial only.

    A same-named R module is never retagged or passed to the builder. Its cache
    entry is restored even if executing the already pinned worker bytes fails.
    """
    import importlib.util
    contract = module('white_view_fit_contract')
    contract.fullref(ref); contract.identity_map(runtime_sources)
    data = _read_pinned(ref['path'], {'bytes':ref['bytes'], 'sha256':ref['sha256']})
    adjacent_path = Path(ref['path']).with_name('white_view_build_contract.py')
    require(str(adjacent_path) in runtime_sources, 'real adjacent contract absent from frozen runtime sources')
    adjacent_info = runtime_sources[str(adjacent_path)]
    require(adjacent_info['sha256'] == BUILD_CONTRACT_SHA256, 'fixed adjacent contract SHA differs')
    adjacent_data = _read_pinned(adjacent_path, adjacent_info)
    name = '_white_view_actual_build_contract_' + sha(str(adjacent_path).encode())
    require(name not in sys.modules, 'fresh actual adjacent contract import required')
    spec = importlib.util.spec_from_file_location(name, str(adjacent_path))
    require(spec is not None and spec.loader is not None, 'adjacent contract source loader required')
    adjacent = importlib.util.module_from_spec(spec); sys.modules[name] = adjacent
    try:
        exec(compile(adjacent_data, str(adjacent_path), 'exec'), adjacent.__dict__)
        require(Path(adjacent.__file__) == adjacent_path and adjacent_path.resolve() == adjacent_path,
                'adjacent actual contract path required')
        worker_name = '_white_view_actual_build_validator'
        require(worker_name not in sys.modules, 'fresh actual build validator import required')
        spec = importlib.util.spec_from_file_location(worker_name, ref['path'])
        require(spec is not None and spec.loader is not None, 'build validator source loader required')
        obj = importlib.util.module_from_spec(spec); sys.modules[worker_name] = obj
        marker = object(); previous = sys.modules.get('white_view_build_contract', marker)
        try:
            sys.modules['white_view_build_contract'] = adjacent
            # Only the pinned raw worker is executed, with its real adjacent import.
            exec(compile(data, ref['path'], 'exec'), obj.__dict__)
        except BaseException:
            if sys.modules.get(worker_name) is obj: del sys.modules[worker_name]
            raise
        finally:
            if previous is marker: sys.modules.pop('white_view_build_contract', None)
            else: sys.modules['white_view_build_contract'] = previous
        require(getattr(obj, 'c', None) is adjacent and callable(getattr(obj, 'verify_runtime', None)),
                'dedicated actual adjacent verify_runtime API missing')
        require(exact(file_identity(ref['path']), {'bytes':ref['bytes'], 'sha256':ref['sha256']})
                and exact(file_identity(adjacent_path), adjacent_info), 'build worker/contract changed at import')
        return obj
    except BaseException:
        if sys.modules.get(name) is adjacent: del sys.modules[name]
        raise


def verify_new_runtime(prereg, args, validator):
    contract = module('white_view_fit_contract')
    binding = prereg['new_build_binding']
    require(binding['manifest']['sha256'] == strict_sha(args.expected_build_manifest_sha256)
            and binding['identity']['sha256'] == strict_sha(args.expected_build_identity_sha256)
            and binding['validator_source']['sha256'] == strict_sha(args.expected_build_validator_sha256),
            'externally expected new build bindings differ')
    verified = validator.verify_runtime(Path(binding['runtime']), args.expected_build_manifest_sha256,
                                        args.expected_build_identity_sha256)
    require(type(verified) is dict and set(verified) == {'manifest', 'identity'}, 'actual new build bundle required')
    for key in ('manifest', 'identity'):
        ref = binding[key]
        raw = _read_pinned(ref['path'], {'bytes':ref['bytes'], 'sha256':ref['sha256']})
        require(exact(contract.strict_json(raw), verified[key]), 'verified new build raw receipt differs')
    require(verified['manifest'].get('schema') == 'sekirei.white-view-runtime-build.v1'
            and verified['manifest'].get('status') == 'complete', 'actual new build schema differs')
    feature = validator.c.feature_binding_for_runtime(verified, contract.HELPER_SHA256)
    contract.feature_binding(feature)
    require(exact(feature, prereg['feature_binding'])
            and feature['core_binary_sha256'] == file_identity(Path(binding['runtime']) / 'bin/sekirei')['sha256'],
            'actual new source/core/helper binding differs')
    require(exact(file_identity(binding['validator_source']['path']),
                  {'bytes':binding['validator_source']['bytes'], 'sha256':binding['validator_source']['sha256']}),
            'build validator changed during verification')
    return feature


def _load_actual_context(args):
    contract = module('white_view_fit_contract')
    script = Path(__file__).resolve()
    require(script == Path(__file__) and script.name == 'fit_white_view_paired_linear.py', 'canonical new driver required')
    require(file_identity(script)['sha256'] == strict_sha(args.expected_driver_sha256), 'driver SHA differs')
    prereg = contract.strict_json(_read_pinned(contract.PREREGISTRATION, args.expected_preregistration_sha256))
    contract.validate_preregistration(prereg)
    require(exact(prereg['fit_driver'], {'path':str(script), **file_identity(script)}), 'PR driver binding differs')
    helpers = prereg['source_helpers']
    require(set(NUMERIC_HASHES) <= set(helpers), 'required frozen original/numeric/new feature helpers missing')
    for name, expected in NUMERIC_HASHES.items():require(helpers[name] == expected, 'fixed numeric/material/helper changed')
    require(helpers.get('fit_white_view_paired_linear.py') == args.expected_driver_sha256, 'driver helper binding differs')
    for name, digest in helpers.items():
        path = script.parent / name
        require(file_identity(path)['sha256'] == strict_sha(digest), 'public helper bytes changed')
        imported = sys.modules.get(name[:-3])
        if imported is not None:require(Path(imported.__file__).resolve() == path, 'preimported helper from another path')
    for path, rec in prereg['runtime_sources'].items():
        deadline_check(); require(exact(file_identity(path), rec), 'runtime file differs')
    activation = contract.strict_json(_read_pinned(contract.ACTIVATION, args.expected_activation_sha256))
    require(prereg['trigger']['sha256'] == args.expected_activation_sha256, 'activation external/PR SHA differs')
    contract.validate_activation(activation, prereg)
    pf = contract.strict_json(_read_pinned(contract.SOURCE_PREFLIGHT, args.expected_source_preflight_sha256))
    contract.validate_source_preflight(pf, prereg, args.expected_preregistration_sha256)
    before = dict(pf['inputs_before'])
    for path, rec in before.items():
        deadline_check(); require(exact(file_identity(path), rec), 'preflight bound input changed')
    additions = {str(contract.PREREGISTRATION):file_identity(contract.PREREGISTRATION),
        str(contract.SOURCE_PREFLIGHT):file_identity(contract.SOURCE_PREFLIGHT), str(script):file_identity(script)}
    for ref in prereg['new_build_binding'].values():
        if type(ref) is dict:additions[ref['path']] = {'bytes':ref['bytes'], 'sha256':ref['sha256']}
    for path,rec in additions.items():
        require(path not in before or exact(before[path], rec), 'input map rebinding forbidden');before[path] = rec
    validator = _load_runtime_validator(prereg['new_build_binding']['validator_source'], prereg['runtime_sources'])
    feature = verify_new_runtime(prereg, args, validator)
    require(not os.path.lexists(contract.Q), 'fresh dedicated white output directory required')
    return contract, prereg, before, contract.DATASET, prereg['initial_weights'], feature, validator


def actual_main(args):
    """Only root's later enabled/frozen public copy may execute this entry."""
    require(not PROTOTYPE_ONLY, 'source prototype only; actual fit entry is blocked')
    os.umask(0o077)
    deadline_check()
    paired, prereg, before, dataset, initial, feature, validator = _load_actual_context(args)
    pool = OpenBLASOneThread(prereg['runtime_sources'])
    deadline_check()
    manifest = _read_pinned(dataset / 'manifest.json', MANIFEST_SHA256)
    positions = _read_pinned(dataset / 'train.positions.jsonl', before[str(dataset / 'train.positions.jsonl')])
    labels = _read_pinned(dataset / 'train.labels.jsonl', before[str(dataset / 'train.labels.jsonl')])
    initializer = _read_pinned(initial['path'], {'bytes': initial['bytes'], 'sha256': initial['sha256']})
    design = build_train_design(manifest, positions, labels, initializer,
                               expected_manifest_sha256=MANIFEST_SHA256, feature_binding=feature, check_deadline=deadline_check)
    require(design.count == TRAIN_COUNT and design.dimension == DIMENSION
            and design.provenance['teacher_identity'] == TEACHER, 'actual TRAIN design count/identity differs')
    producer = module('verified_design_gram')
    require(not producer._gram_origins and not producer._design_origins, 'fresh numeric origin registry required')
    solver = module('paired_linear_solver_verified_design')
    deadline_check()
    remaining = WALL_SECONDS - (time.monotonic() - _PROCESS_STARTED)
    require(0.0 < remaining <= WALL_SECONDS, 'remaining inner fit budget exhausted')
    fit_started = time.monotonic()
    result = solver.solve_verified_design_bytes(design.z_bytes, design.d_bytes, design.count, design.dimension,
        expected_producer_source_sha256=NUMERIC_HASHES['verified_design_gram.py'],
        expected_solver_source_sha256=NUMERIC_HASHES['paired_linear_solver_verified_design.py'],
        max_iterations=20000, deadline_seconds=remaining)
    deadline_check()
    problem, artifact = sole_gram(result, NUMERIC_HASHES['verified_design_gram.py'])
    require(artifact.design.z_bytes == design.z_bytes and artifact.design.d_bytes == design.d_bytes,
            'sole Gram immutable design differs from original TRAIN extraction')
    f64, f32 = coefficient_bytes(result)
    require(result['certificate_f64']['solver_threshold_met'] is True
            and result['certificate_f64']['feasible'] is True
            and result['certificate_f32_saved_radius']['feasible'] is True
            and result['certificate_f32_saved_radius']['objective_delta_vs_zero'] <= 0
            and result['fallback_used'] is False, 'solver exact certificates failed')
    white = module('white_view_paired_linear')
    native = white.serialize_coefficients(f32, initializer, binding=feature)
    white.validate_native(native, f32, initializer, binding=feature)
    require(exact(verify_new_runtime(prereg, args, validator), feature), 'new build/source/core changed during fit')
    thread_proof = pool.verify()
    after = {}
    for path in before:
        deadline_check()
        after[path] = file_identity(path)
    require(exact(before, after), 'source/input changed during fit')
    for path, rec in prereg['runtime_sources'].items():
        deadline_check()
        require(exact(file_identity(path), rec), 'runtime changed during fit')
    numeric_execution = {'schema': 'sekirei.white-view-paired-linear-numeric-execution.v1',
        'whole_child_budget_seconds': 1200, 'parent_process_group_deadline_required': True,
        'boundary_checks_cannot_interrupt_blas': True, 'remaining_inner_budget_seconds': remaining,
        'fit_wall_seconds': time.monotonic() - fit_started,
        'whole_child_elapsed_seconds': time.monotonic() - _PROCESS_STARTED,
        'iterations': result['iterations'], 'restart_count': result['restart_count'],
        'single_gram_produced': True, 'single_fixed_fit': True, 'openblas': thread_proof,
        'lambda': 1, 'max_iterations': 20000, 'loss_half_unnormalized_sum': True,
        'saved_f32_positive_zero': True, 'actual_cleanup_claimed': False}
    deadline_check()
    paired.Q.mkdir(mode=0o700, parents=False)
    outputs = {}
    for key, name, data in (
        ('design_z', 'design.z.i8', design.z_bytes), ('targets_d', 'targets.d.i32', design.d_bytes),
        ('gram_z', 'gram.z.i64', b''.join(struct.pack('<q', x) for row in problem.gram for x in row)),
        ('rhs_z', 'rhs.z.i64', b''.join(struct.pack('<q', x) for x in problem.hz)),
        ('coefficients_f64', 'coefficients.f64.bin', f64), ('coefficients_f32', 'coefficients.f32.bin', f32),
        ('weights', 'weights.bin', native), ('design_receipt', 'design-receipt.json', json_bytes(design.provenance)),
        ('solver_certificate', 'solver-certificate.json', json_bytes(result))):
        deadline_check()
        outputs[key] = write_new(paired.Q / name, data)
    context = {'preregistration_sha256': args.expected_preregistration_sha256,
        'activation_sha256': prereg['trigger']['sha256'], 'source_helpers_sha256': prereg['source_helpers'],
        'source_preflight_sha256': args.expected_source_preflight_sha256,
        'inputs_before': before, 'inputs_after': after, 'runtime_sources': prereg['runtime_sources'],
        'numeric_execution': numeric_execution, 'feature_binding': feature,
        'new_build_binding': prereg['new_build_binding'], 'new_build_verified': True}
    template = make_run_template(context, outputs, outputs['solver_certificate'])
    draft = {'schema': 'sekirei.white-view-paired-linear-fit-result.v1', 'status': 'numeric-stage-complete',
        'run_template': template, 'numeric_execution': numeric_execution,
        'cleanup_verified': False, 'final_used': False, 'adoption_claimed': False}
    deadline_check()
    write_new(paired.Q / 'fit-result.json', json_bytes(draft))
    print('numeric-stage-complete; parent must wait/reap, reverify artifacts/inputs, then create run and sidecar')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('preregistration', 'activation', 'source-preflight', 'driver',
                 'build-manifest', 'build-identity', 'build-validator'):
        p.add_argument('--expected-' + flag + '-sha256', required=True)
    return p


if __name__ == '__main__':
    # Block before argument parsing or any actual input/file access in this preparation.
    if PROTOTYPE_ONLY:
        raise RuntimeError('source prototype only; actual fit entry is blocked')
    actual_main(parser().parse_args())
