"""TRAIN-only paired-linear fitter bound to external PR and source preflight.

Pure extraction accepts caller-verified original bytes, never reads a dataset.
The dedicated gated CLI is intended for a parent holding the five fixed locks
and supervising the entire process group with a 1200-second wall deadline.
Only that parent may claim child cleanup and create run.json / metadata.
No actual TRAIN extraction, fit, engine or model read was performed to prepare
this source. Numeric modules keep their blocked generic runtime entries.
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
                       expected_manifest_sha256, check_deadline=lambda: None):
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
    np = numpy()
    require(initializer_bytes[:8] == b'SEKIRW01', 'native initializer magic mismatch')
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
        us = material.active_features(position, position['stm'])
        them = material.active_features(position, 1 - position['stm'])
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
    provenance = {'schema': 'sekirei.paired-linear-train-design.v1', 'count': n, 'dimension': DIMENSION,
        'teacher_identity': manifest['teacher_identity'], 'original_manifest_sha256': sha(manifest_bytes),
        'original_files': manifest['files'], 'source_provenance': {k: manifest[k] for k in
            ('source_corpus_manifest_sha256', 'seed', 'split', 'sampling', 'games_per_pack',
             'dependencies', 'independent_exclusions', 'derivation', 'games')},
        'train_positions_sha256': sha(position_bytes), 'train_labels_sha256': sha(label_bytes),
        'material_initializer_sha256': sha(initializer_bytes),
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
    paired = module('paired_linear')
    require(type(context) is dict and type(outputs) is dict, 'typed fit context/outputs required')
    fields = {'schema': 'sekirei.paired-linear-fit-run.v1', 'status': 'complete', 'candidate': paired.MODE,
        'mode': paired.MODE, 'created_at': datetime.now(timezone.utc).isoformat(), 'seed': 42,
        'nnue_output': 'absolute', 'optimizer': 'projected-FISTA', 'adam_used': False, 'epochs': 0,
        'resume_used': False, 'trainer_used': False, 'inputs_unchanged': True, 'source_unchanged': True,
        'cleanup_verified': False, 'final_used': False, 'adoption_claimed': False,
        'positions': TRAIN_COUNT, 'dimension': DIMENSION, 'teacher_identity': TEACHER,
        'original_manifest_sha256': MANIFEST_SHA256, 'material_initializer_sha256': INITIALIZER_SHA256,
        'fit_holdout_used': False, 'fit_development_used': False, 'fit_final_used': False,
        'plan_sha256': paired.CONDITIONAL_PLAN_SHA256, 'outputs': outputs, 'solver_certificate': solverref}
    required = {'preregistration_sha256', 'activation_sha256', 'source_helpers_sha256',
                'source_preflight_sha256', 'inputs_before', 'inputs_after', 'runtime_sources',
                'numeric_execution'}
    require(set(context) == required and exact(context['inputs_before'], context['inputs_after']),
            'complete fixed run context and unchanged input map required')
    fields.update(context)
    for name in ('preregistration_sha256', 'activation_sha256', 'source_preflight_sha256'):
        strict_sha(fields[name])
    require({'weights', 'coefficients_f32', 'coefficients_f64', 'gram_z', 'rhs_z'} <= set(outputs),
            'required native/coefficient/full Gram output identities missing')
    for ref in outputs.values():
        paired.file_reference(ref)
        require(Path(ref['path']).parent == paired.Q and ref['path'] not in
                (str(paired.FIT_RECEIPT), str(paired.W.with_suffix('.meta.json')),
                 str(paired.Q / 'fit-result.json')), 'invalid output reference')
    paired.file_reference(solverref, paired.Q / 'solver-certificate.json')
    return fields


def construct_fit_run(context, outputs, solverref, lifecycle):
    """Parent-only pure final construction after real supervised child wait/reap.

    This checks the supplied lifecycle record, not the process table. The parent
    must obtain it from its own wait/reap/PG-cleanup; the child never calls this.
    """
    expected = {'returncode': 0, 'timed_out': False, 'waited': True, 'reaped': True,
                'process_group_stopped': True}
    require(exact(lifecycle, expected), 'actual successful wait/reap/PG-stop lifecycle required')
    run = make_run_template(context, outputs, solverref)
    run['cleanup_verified'] = True
    run['lifecycle'] = dict(lifecycle)
    return run


def _read_pinned(path, expected):
    data = Path(path).read_bytes()
    rec = file_identity(path)
    if type(expected) is str:
        require(rec['sha256'] == strict_sha(expected), 'pinned input SHA mismatch')
    else:
        require(exact(rec, expected), 'pinned input identity mismatch')
    require(sha(data) == rec['sha256'] and len(data) == rec['bytes'], 'input changed while read')
    return data


def _load_actual_context(args):
    paired = module('paired_linear')
    anchor = module('functional_anchor')
    script = Path(__file__).resolve()
    require(script == Path(__file__) and script.name == 'fit_paired_linear.py', 'canonical driver required')
    require(file_identity(script)['sha256'] == strict_sha(args.expected_driver_sha256), 'driver SHA differs')
    PR = paired.PREREGISTRATION
    PF = paired.C / 'paired-linear-original-source-preflight-v1.json'
    prereg = anchor._json(_read_pinned(PR, args.expected_preregistration_sha256))
    require(prereg.get('schema') == 'sekirei.paired-linear-preregistration.v1'
            and prereg.get('status') == 'frozen-before-fit'
            and prereg.get('candidate') == prereg.get('mode') == paired.MODE
            and prereg.get('plan_sha256') == paired.CONDITIONAL_PLAN_SHA256
            and prereg.get('fit_started') is False and prereg.get('final_used') is False
            and prereg.get('adoption_claimed') is False, 'frozen fixed preregistration required')
    require(exact(prereg.get('fit_driver'), {'path': str(script), **file_identity(script)}), 'PR driver binding differs')
    require(prereg.get('teacher_identity') == TEACHER
            and exact(prereg.get('counts'), {'train': TRAIN_COUNT, 'holdout': HOLDOUT_COUNT})
            and prereg.get('dataset_manifest_sha256') == MANIFEST_SHA256
            and prereg.get('dataset') == str(paired.C / 'source-input-recovery-v1/dataset')
            and prereg.get('output') == str(paired.Q), 'fixed dataset/count/output binding differs')
    helpers = prereg['source_helpers']
    require(type(helpers) is dict and set(NUMERIC_HASHES) <= set(helpers), 'required frozen helpers missing')
    for name, expected in NUMERIC_HASHES.items():
        require(helpers[name] == expected, 'fixed numeric/material source changed')
    require(helpers.get('fit_paired_linear.py') == args.expected_driver_sha256, 'driver helper binding differs')
    for name, digest in helpers.items():
        require(type(name) is str and name == Path(name).name and name.endswith('.py'), 'bad helper filename')
        path = script.parent / name
        require(file_identity(path)['sha256'] == strict_sha(digest), 'public helper bytes changed')
        imported = sys.modules.get(name[:-3])
        if imported is not None:
            require(Path(imported.__file__).resolve() == path, 'preimported helper from another path')
    runtimes = prereg['runtime_sources']
    require(type(runtimes) is dict and runtimes, 'frozen runtime identity map required')
    for path, rec in runtimes.items():
        deadline_check()
        require(type(rec) is dict and set(rec) == {'bytes', 'sha256'} and type(rec['bytes']) is int
                and rec['bytes'] >= 0 and strict_sha(rec['sha256']), 'typed runtime identity required')
        require(exact(file_identity(path), rec), 'runtime file differs')
    pf = anchor._json(_read_pinned(PF, args.expected_source_preflight_sha256))
    require(pf.get('schema') == 'sekirei.paired-linear-original-source-preflight.v1'
            and pf.get('status') == 'complete' and pf.get('candidate') == paired.MODE
            and pf.get('plan_sha256') == paired.CONDITIONAL_PLAN_SHA256
            and pf.get('preregistration_sha256') == args.expected_preregistration_sha256
            and pf.get('activation_sha256') == prereg['trigger']['sha256']
            and exact(pf.get('source_helpers_sha256'), helpers)
            and exact(pf.get('counts'), {'train': TRAIN_COUNT, 'holdout': HOLDOUT_COUNT})
            and pf.get('teacher_identity') == TEACHER and pf.get('original_manifest_sha256') == MANIFEST_SHA256,
            'completed original source preflight binding differs')
    for key, expected in {'original_labels_reparsed': True, 'raw1000_membership_reverified': True,
            'raw_replay_rerun': False, 'source_unchanged': True, 'inputs_unchanged': True,
            'final_used': False, 'fit_started': False}.items():
        require(exact(pf.get(key), expected), 'source preflight flag differs: ' + key)
    require(exact(pf.get('inputs_before'), pf.get('inputs_after')) and type(pf['inputs_before']) is dict,
            'source preflight immutable input map required')
    before = dict(pf['inputs_before'])
    for path, rec in before.items():
        deadline_check()
        require(exact(file_identity(path), rec), 'preflight bound input changed')
    additions = {str(PR): file_identity(PR), str(PF): file_identity(PF), str(script): file_identity(script)}
    for path, rec in additions.items():
        require(path not in before or exact(before[path], rec), 'input map rebinding forbidden')
        before[path] = rec
    dataset = Path(prereg['dataset'])
    manifest_path = dataset / 'manifest.json'
    require(before[str(manifest_path)]['sha256'] == MANIFEST_SHA256, 'manifest absent from source input map')
    initial = prereg['initial_weights']
    require(type(initial) is dict and set(initial) == {'path', 'bytes', 'sha256'}
            and initial['sha256'] == INITIALIZER_SHA256 and initial['bytes'] == 1305356
            and initial['path'] == str(paired.C.parent / 'training-15-v1/material-init-seed42/material-init.bin')
            and exact(before[initial['path']], {'bytes': initial['bytes'], 'sha256': initial['sha256']}),
            'fixed initializer input binding differs')
    require(prereg['trigger']['path'] == str(paired.ACTIVATION)
            and file_identity(paired.ACTIVATION)['sha256'] == prereg['trigger']['sha256'], 'activation changed')
    require(not os.path.lexists(paired.Q), 'fresh dedicated output directory required')
    return paired, prereg, before, dataset, initial


def actual_main(args):
    """Only root's later enabled/frozen public copy may execute this entry."""
    require(not PROTOTYPE_ONLY, 'source prototype only; actual fit entry is blocked')
    os.umask(0o077)
    deadline_check()
    paired, prereg, before, dataset, initial = _load_actual_context(args)
    pool = OpenBLASOneThread(prereg['runtime_sources'])
    deadline_check()
    manifest = _read_pinned(dataset / 'manifest.json', MANIFEST_SHA256)
    positions = _read_pinned(dataset / 'train.positions.jsonl', before[str(dataset / 'train.positions.jsonl')])
    labels = _read_pinned(dataset / 'train.labels.jsonl', before[str(dataset / 'train.labels.jsonl')])
    initializer = _read_pinned(initial['path'], {'bytes': initial['bytes'], 'sha256': initial['sha256']})
    design = build_train_design(manifest, positions, labels, initializer,
                               expected_manifest_sha256=MANIFEST_SHA256, check_deadline=deadline_check)
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
    native = paired.serialize_coefficients(f32, initializer=initializer)
    paired.validate_native(native, f32, initializer=initializer)
    thread_proof = pool.verify()
    after = {}
    for path in before:
        deadline_check()
        after[path] = file_identity(path)
    require(exact(before, after), 'source/input changed during fit')
    for path, rec in prereg['runtime_sources'].items():
        deadline_check()
        require(exact(file_identity(path), rec), 'runtime changed during fit')
    numeric_execution = {'schema': 'sekirei.paired-linear-numeric-execution.v1',
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
        'numeric_execution': numeric_execution}
    template = make_run_template(context, outputs, outputs['solver_certificate'])
    draft = {'schema': 'sekirei.paired-linear-fit-result.v1', 'status': 'numeric-stage-complete',
        'run_template': template, 'numeric_execution': numeric_execution,
        'cleanup_verified': False, 'final_used': False, 'adoption_claimed': False}
    deadline_check()
    write_new(paired.Q / 'fit-result.json', json_bytes(draft))
    print('numeric-stage-complete; parent must wait/reap, reverify artifacts/inputs, then create run and sidecar')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('preregistration', 'source-preflight', 'driver'):
        p.add_argument('--expected-' + flag + '-sha256', required=True)
    return p


if __name__ == '__main__':
    # Block before argument parsing or any actual input/file access in this preparation.
    if PROTOTYPE_ONLY:
        raise RuntimeError('source prototype only; actual fit entry is blocked')
    actual_main(parser().parse_args())
