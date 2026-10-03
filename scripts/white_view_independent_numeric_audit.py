"""Independent exact numerical proof source; actual artifact entry blocked.

No solver/NumPy import, Z.T@Z computation, fit, PSD LDL or engine. The saved
Gram's origin is a separate source-pinned producer receipt, not reproved here.
The independent O(p^2) arithmetic checks three certificates from stored IEEE
coefficient bytes. Preparation used tiny public synthetic fixtures only.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
from contextlib import ExitStack
import importlib.util

PROTOTYPE_ONLY = True
N_ACTUAL = 112681
DIM_ACTUAL = 254
RHO = Fraction(161791, 4096)
SAVED_RADIUS = Fraction(79, 2)
GAP_PER_SAMPLE = Fraction(1, 1_000_000)
MODE = 'white-view-paired-linear-constrained-ridge1-l1-39p5-v1'
PLAN_SHA = '1f274b187a96b16674814c51f77cb402ca8a80e6a7e5d738a5caf9354127e836'
PRODUCER_SHA = 'ca65fc08248cb4327b1fb0ec733a65af303673010217e952ea49c878498a3b68'
SOLVER_SHA = '7884f324b73d38bbd2202a6d7518dab81963f6d5453cd0f50fea3c2fc35511d1'
OLD_SOLVER_SHA = 'ba9c86b37e3ed442db930332f16f3390ff5cd84a98431fcceea84cc4dd9b066b'
GRAM_SCHEMA = 'sekirei.verified-design-algebraic-gram.source-prototype.v1'
GRAM_PRODUCER = 'numpy-float64-exact-integer-gram-v1'
R = Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
C = Path('/home/server/.local/share/sekirei-weight2/campaign-17-autonomous-v1')
Q = C.parent / 'training-17-v1/white-view-paired-linear-constrained-ridge1-v1/run-v1'
RUN = Q / 'run.json'
SELF = C / 'white-view-paired-linear-independent-numeric-audit-worker-v1.py'
OUT = C / 'white-view-paired-linear-independent-numeric-audit-v1.json'
ORIGINAL_MANIFEST_SHA = 'ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6'
INITIALIZER_SHA = 'bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40'
TEACHER = 'external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d'


class InvalidProof(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidProof(message)


def strict_sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'strict SHA256 required')
    return value


def sha(data):
    require(type(data) is bytes, 'immutable bytes required')
    return hashlib.sha256(data).hexdigest()


def exact(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(exact(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def strict_json(data):
    require(type(data) is bytes, 'JSON bytes required')
    def pairs(values):
        result = {}
        for k, v in values:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    def nonfinite(_):
        raise InvalidProof('nonfinite JSON constant')
    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), 'nonfinite JSON number')
        return result
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float)
    except (TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidProof('invalid JSON bytes') from error


def record(value):
    q = Fraction(value)
    return {'numerator': q.numerator, 'denominator': q.denominator}


def fraction_record(value):
    require(type(value) is dict and set(value) == {'numerator', 'denominator'}
            and type(value['numerator']) is int and type(value['denominator']) is int
            and value['denominator'] > 0, 'canonical strict rational record required')
    q = Fraction(value['numerator'], value['denominator'])
    require(exact(record(q), value), 'fraction not reduced/canonical')
    return q


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def parse_matrix(gram_bytes, rhs_bytes, N, dimension):
    require(type(N) is int and 1 <= N <= 1_000_000 and type(dimension) is int
            and 1 <= dimension <= DIM_ACTUAL, 'strict bounded count/dimension, never bool')
    require(type(gram_bytes) is bytes and len(gram_bytes) == dimension * dimension * 8
            and type(rhs_bytes) is bytes and len(rhs_bytes) == dimension * 8,
            'canonical signed64LE Gram/rhs shape required')
    flat = tuple(x[0] for x in struct.iter_unpack('<q', gram_bytes))
    G = tuple(flat[i*dimension:(i+1)*dimension] for i in range(dimension))
    h = tuple(x[0] for x in struct.iter_unpack('<q', rhs_bytes))
    require(all(abs(x) <= N*1600 for x in flat) and all(abs(x) <= N*40*55779 for x in h),
            'saved Gram/rhs integer bounds exceeded')
    require(all(G[i][j] == G[j][i] for i in range(dimension) for j in range(i))
            and all(G[i][i] >= 0 for i in range(dimension)), 'saved Gram symmetry/diagonal invalid')
    K = tuple(tuple(G[i][j] + (256 if i == j else 0) for j in range(dimension))
              for i in range(dimension))
    return G, K, h


def parse_coefficients(data, dimension, dtype):
    size = 8 if dtype == 'd' else 4 if dtype == 'f' else 0
    require(size and type(data) is bytes and len(data) == dimension * size,
            'canonical IEEE little-endian coefficient bytes required')
    values = tuple(x[0] for x in struct.iter_unpack('<' + dtype, data))
    require(all(type(x) is float and math.isfinite(x) for x in values), 'nonfinite saved coefficient')
    if dtype == 'f':
        require(all(x != 0.0 or data[i*4:(i+1)*4] == b'\0'*4 for i, x in enumerate(values)),
                'independent f32 zero is not canonical positive zero')
    return values


def independent_certificate(K, h, N, values, radius):
    """O(p^2) integer products with one shared power-of-two denominator.

    No Fraction matrix elimination, no approximate gradient. Fractions are
    constructed only for the final O(1) scalar results / coefficient records.
    """
    require(type(values) is tuple and len(values) == len(K) == len(h)
            and all(type(x) is float and math.isfinite(x) for x in values), 'strict finite vector required')
    ratios = tuple(x.as_integer_ratio() for x in values)
    Qden = max(den for _, den in ratios)
    require(Qden > 0 and Qden & (Qden - 1) == 0, 'coefficient denominator is not dyadic')
    U = tuple(num * (Qden // den) for num, den in ratios)
    KU = tuple(sum(k*u for k, u in zip(row, U)) for row in K)
    gn = tuple(ku - 16*Qden*hv for ku, hv in zip(KU, h))
    gd = 256 * Qden
    dot_n = sum(g*u for g, u in zip(gn, U))
    dot_d = 256 * Qden * Qden
    norm = Fraction(sum(abs(u) for u in U), Qden)
    max_gradient = Fraction(max(abs(g) for g in gn), gd)
    gap = Fraction(dot_n, dot_d) + radius * max_gradient
    delta = Fraction(sum(u*v for u, v in zip(U, KU)), 512 * Qden * Qden) - Fraction(sum(hv*u for hv, u in zip(h, U)), 16*Qden)
    feasible = norm <= radius
    require(not feasible or gap >= 0, 'negative exact Frank-Wolfe gap at feasible coefficients')
    per_sample = gap / N
    return {'norm': record(norm), 'radius': record(radius), 'feasible': feasible,
        'fw_gap': record(gap), 'fw_gap_per_sample': record(per_sample),
        'solver_threshold_met': feasible and 0 <= per_sample <= GAP_PER_SAMPLE,
        'objective_delta_vs_zero': record(delta), 'gradient_numerators': list(gn),
        'gradient_denominator': gd, 'coefficient_values': [record(x) for x in values]}


def _backend_metadata(value):
    require(type(value) is list and len(value) == 7
            and all(type(row) is list and len(row) == 2 and type(row[0]) is str for row in value),
            'canonical producer metadata rows required')
    first = value[4][1]
    require(type(first) is bool, 'producer import state must be bool')
    expected = [['numpy_version', '1.26.4'], ['dtype', 'float64'], ['design_order', 'C'],
        ['transpose', 'Z.T@Z;Z.T@d'], ['first_numpy_preimported', first],
        ['environment_set_before_first_numpy_import', not first],
        ['existing_import_threadpool_evidence_not_claimed', True]]
    require(exact(value, expected), 'producer backend/order/transpose metadata changed')
    return tuple(tuple(row) for row in value)


def producer_gram_digest(gram_bytes, rhs_bytes, N, dimension, design_sha256, metadata, producer_sha256):
    """Rehash saved content binding; does not execute/reprove Z.T@Z."""
    parts = [GRAM_SCHEMA.encode(), b'\x00gram\x00', GRAM_PRODUCER.encode(), b'\x00',
        bytes.fromhex(strict_sha(producer_sha256)), repr(metadata).encode('ascii'), b'\x00',
        bytes.fromhex(strict_sha(design_sha256)), struct.pack('<QQ', N, dimension), gram_bytes, rhs_bytes]
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def audit_numeric_bytes(gram_bytes, rhs_bytes, coefficient_f64_bytes, coefficient_f32_bytes,
                        solver_bytes, design_receipt_bytes, *, N, dimension,
                        design_z_sha256, targets_d_sha256, design_sha256,
                        expected_producer_sha256=PRODUCER_SHA, expected_solver_sha256=SOLVER_SHA):
    G, K, h = parse_matrix(gram_bytes, rhs_bytes, N, dimension)
    f64 = parse_coefficients(coefficient_f64_bytes, dimension, 'd')
    f32 = parse_coefficients(coefficient_f32_bytes, dimension, 'f')
    require(sum(abs(Fraction(x)) for x in f64) <= RHO, 'f64 solver radius infeasible')
    require(sum(abs(Fraction(x)) for x in f32) <= SAVED_RADIUS, 'f32 saved radius infeasible')
    for i, x in enumerate(f64):
        cast = struct.unpack('<f', struct.pack('<f', x))[0]
        expected = struct.pack('<f', 0.0 if cast == 0.0 else cast)
        require(coefficient_f32_bytes[i*4:(i+1)*4] == expected, 'f32 differs from canonical nearest f64 cast')
    cert = strict_json(solver_bytes)
    keys = {'status', 'iterations', 'restart_count', 'coefficient_f64', 'coefficient_f32',
        'certificate_f64', 'certificate_f32_solver_radius', 'certificate_f32_saved_radius',
        'saved_f32_gap_is_solver_stopping_criterion', 'lipschitz_exact', 'lipschitz_numeric_upper',
        'fallback_used', 'one_if_needed_f64_correction_used', 'one_if_needed_f64_correction_factor',
        'lambda', 'loss_is_half_unnormalized_sum', 'verified_design_gram'}
    require(type(cert) is dict and set(cert) == keys and cert['status'] == 'synthetic-certified',
            'full unchanged solver-result key contract required')
    require(type(cert['iterations']) is int and 1 <= cert['iterations'] <= 20000
            and type(cert['restart_count']) is int and 0 <= cert['restart_count'] <= cert['iterations']
            and cert['fallback_used'] is False and cert['saved_f32_gap_is_solver_stopping_criterion'] is False
            and type(cert['lambda']) is int and cert['lambda'] == 1
            and cert['loss_is_half_unnormalized_sum'] is True
            and type(cert['one_if_needed_f64_correction_used']) is bool
            and fraction_record(cert['one_if_needed_f64_correction_factor']) == Fraction(1)-Fraction(1,2**40),
            'fixed solver recipe/correction/iteration metadata differs')
    require(exact(cert['coefficient_f64'], [record(x) for x in f64])
            and exact(cert['coefficient_f32'], [record(x) for x in f32]), 'certificate vectors differ from IEEE artifacts')
    recomputed = {'certificate_f64': independent_certificate(K, h, N, f64, RHO),
        'certificate_f32_solver_radius': independent_certificate(K, h, N, f32, RHO),
        'certificate_f32_saved_radius': independent_certificate(K, h, N, f32, SAVED_RADIUS)}
    for name, value in recomputed.items():
        require(exact(cert[name], value), 'independent integer/dyadic certificate mismatch: '+name)
    require(recomputed['certificate_f64']['solver_threshold_met'] is True
            and recomputed['certificate_f32_saved_radius']['feasible'] is True
            and fraction_record(recomputed['certificate_f32_saved_radius']['objective_delta_vs_zero']) <= 0,
            'mandatory f64 FW / saved-f32 objective gate failed')
    lipschitz = Fraction(max(sum(abs(x) for x in row) for row in K), 256)
    upper = math.nextafter(float(lipschitz), math.inf)
    require(fraction_record(cert['lipschitz_exact']) == lipschitz
            and fraction_record(cert['lipschitz_numeric_upper']) == Fraction(upper),
            'exact Gershgorin / upward binary64 bound mismatch')
    proof = cert['verified_design_gram']
    require(type(proof) is dict and set(proof) == {'schema', 'producer', 'design_sha256', 'gram_sha256',
            'producer_source_sha256', 'solver_source_sha256', 'backend_metadata', 'sample_count', 'dimension',
            'algebraic_psd_proven_from_issued_design', 'runtime_threadpool_proof_claimed',
            'actual_dataset_provenance_claimed', 'source_prototype_only'}, 'full producer binding required')
    require(proof['schema'] == GRAM_SCHEMA and proof['producer'] == GRAM_PRODUCER
            and type(proof['sample_count']) is int and proof['sample_count'] == N
            and type(proof['dimension']) is int and proof['dimension'] == dimension
            and proof['design_sha256'] == strict_sha(design_sha256)
            and proof['producer_source_sha256'] == strict_sha(expected_producer_sha256)
            and proof['solver_source_sha256'] == strict_sha(expected_solver_sha256)
            and proof['algebraic_psd_proven_from_issued_design'] is True
            and proof['runtime_threadpool_proof_claimed'] is False
            and proof['actual_dataset_provenance_claimed'] is False and proof['source_prototype_only'] is True,
            'producer source/design/scope binding differs')
    metadata = _backend_metadata(proof['backend_metadata'])
    require(proof['gram_sha256'] == producer_gram_digest(gram_bytes, rhs_bytes, N, dimension,
            design_sha256, metadata, expected_producer_sha256), 'saved Gram/rhs full producer digest mismatch')
    design = strict_json(design_receipt_bytes)
    require(type(design) is dict and design.get('schema') == 'sekirei.white-view-paired-linear-train-design.v1'
            and type(design.get('count')) is int and design['count'] == N
            and type(design.get('dimension')) is int and design['dimension'] == dimension
            and design.get('z_sha256') == strict_sha(design_z_sha256)
            and design.get('d_sha256') == strict_sha(targets_d_sha256), 'design receipt/output SHA binding mismatch')
    return {'schema': 'sekirei.white-view-paired-linear-independent-numeric-math.v1', 'status': 'complete',
        'sample_count': N, 'dimension': dimension, 'recomputed_certificates': recomputed,
        'all_three_certificates_exactly_recomputed': True, 'saved_f32_nearest_canonical_cast_verified': True,
        'gershgorin_exact': record(lipschitz), 'gershgorin_numeric_upper': record(upper),
        'gram_sha256': proof['gram_sha256'], 'design_sha256': design_sha256,
        'gram_bytes_sha256': sha(gram_bytes), 'rhs_bytes_sha256': sha(rhs_bytes),
        'coefficient_f64_bytes_sha256': sha(coefficient_f64_bytes),
        'coefficient_f32_bytes_sha256': sha(coefficient_f32_bytes),
        'solver_certificate_sha256': sha(solver_bytes), 'design_receipt_sha256': sha(design_receipt_bytes),
        'design_z_sha256': design_z_sha256, 'targets_d_sha256': targets_d_sha256,
        'producer_source_sha256': expected_producer_sha256, 'solver_source_sha256': expected_solver_sha256,
        'producer_backend_metadata': proof['backend_metadata'], 'stored_gradient_matches_Hu_minus_b': True,
        'objective_definition': 'half unnormalized SSE plus half coefficient L2 squared; delta vs u=0',
        'raw_design_extraction_replayed': False, 'gram_product_recomputed': False,
        'gram_psd_independently_reproved': False, 'producer_origin_acknowledged_separately': True,
        'solver_trajectory_replayed': False, 'second_fit_run': False, 'native_core_proof_claimed': False,
        'final_used': False, 'adoption_claimed': False}


def file_identity(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
            and stat.S_ISREG(path.lstat().st_mode), 'canonical regular file required')
    def fingerprint(s):
        return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    before = path.lstat()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        require(fingerprint(before) == fingerprint(os.fstat(f.fileno())), 'file replaced before hash')
        digest = hashlib.file_digest(f, 'sha256').hexdigest(); after = os.fstat(f.fileno())
    require(fingerprint(before) == fingerprint(after) == fingerprint(path.lstat()), 'file changed during hash')
    return {'bytes': before.st_size, 'sha256': digest}


def read_ref(ref, fixed_path=None):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'}
            and type(ref['path']) is str and type(ref['bytes']) is int and ref['bytes'] >= 0,
            'external fullref required')
    strict_sha(ref['sha256'])
    path = Path(ref['path'])
    if fixed_path is not None:
        require(path == fixed_path, 'wrong fixed artifact path')
    require(exact(file_identity(path), {'bytes': ref['bytes'], 'sha256': ref['sha256']}), 'artifact fullref mismatch')
    data = path.read_bytes()
    require(len(data) == ref['bytes'] and sha(data) == ref['sha256']
            and exact(file_identity(path), {'bytes': ref['bytes'], 'sha256': ref['sha256']}),
            'artifact changed while read')
    return data


def design_digest_files(N, dimension, zref, dref):
    require(zref['bytes'] == N*dimension and dref['bytes'] == N*4, 'derived design output byte shapes differ')
    digest = hashlib.sha256(GRAM_SCHEMA.encode()+b'\x00design\x00'+struct.pack('<QQ', N, dimension))
    for ref in (zref, dref):
        require(exact(file_identity(ref['path']), {'bytes': ref['bytes'], 'sha256': ref['sha256']}),
                'derived design fullref mismatch')
        # Hash immutable derived files only, never multiply/parse/rebuild Gram.
        with Path(ref['path']).open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


REFERENCE03 = C.parent / 'training-17-v1/white-view-material-init-seed42-v1/reference03.bin'
PR = C / 'white-view-paired-linear-preregistration-v1.json'
ACT = C / 'white-view-paired-linear-activation-v1.json'
PF = C / 'white-view-paired-linear-original-source-preflight-v1.json'
BUILD_CONTRACT_SHA = '305d4a2148feecfc0b25ca7e6834c108a635e250cf5b42124511ab3c976b8144'
CONTRACT_API_NAMES = ('validate_preregistration','validate_activation','validate_source_preflight',
                      'validate_fit_run','validate_sidecar')


def fullref(path, expected_sha=None):
    rec = file_identity(path)
    if expected_sha is not None:
        require(rec['sha256'] == strict_sha(expected_sha), 'external SHA differs: ' + str(path))
    return {'path':str(path), **rec}


def identity_map(value):
    require(type(value) is dict and value, 'nonempty immutable identity map required')
    for path, rec in value.items():
        require(type(path) is str and Path(path).is_absolute() and type(rec) is dict
                and set(rec) == {'bytes','sha256'} and type(rec['bytes']) is int and rec['bytes'] >= 0,
                'strict canonical identity-map fields required')
        strict_sha(rec['sha256'])
    return value


class Reader:
    """Full raw byte binding and before/after file identity; no raw data parsing."""
    def __init__(self): self.files = {}
    def pin(self, path, expected=None):
        key = str(path); rec = file_identity(path)
        if type(expected) is str:
            require(rec['sha256'] == strict_sha(expected), 'external input SHA differs: ' + key)
        elif expected is not None:
            identity_map({key:expected}); require(exact(expected,rec), 'external identity differs: ' + key)
        require(key not in self.files or exact(self.files[key],rec), 'immutable input rebinding forbidden')
        self.files[key] = rec
        return {'path':key, **rec}
    def read(self, ref, path=None):
        rec = self.pin(ref['path'], {k:ref[k] for k in ('bytes','sha256')})
        return read_ref(rec,path)
    def document(self, ref, path=None): return strict_json(self.read(ref,path))
    def pin_map(self, value):
        for path, rec in identity_map(value).items(): self.pin(path,rec)
    def verify(self):
        current = {path:file_identity(path) for path in self.files}
        require(exact(current,self.files), 'independent audit inputs/source changed')
        return current


def load_public_source(reader, name, expected_sha):
    """Source SHA before/after import, canonical public path, no .pyc reuse."""
    path = R / 'scripts' / (name + '.py')
    ref = reader.pin(path,expected_sha); raw = reader.read(ref,path)
    if name in sys.modules:
        loaded = sys.modules[name]
        require(Path(loaded.__file__) == path, 'public helper is shadowed by another path')
        require(exact(file_identity(path), {k:ref[k] for k in ('bytes','sha256')}), 'cached helper bytes changed')
        return loaded
    spec = importlib.util.spec_from_file_location(name,str(path))
    require(spec is not None and spec.loader is not None, 'public source loader required')
    loaded = importlib.util.module_from_spec(spec); sys.modules[name] = loaded
    try:
        exec(compile(raw,str(path),'exec'),loaded.__dict__)
        reader.pin(path,{k:ref[k] for k in ('bytes','sha256')})
        return loaded
    except BaseException:
        if sys.modules.get(name) is loaded: del sys.modules[name]
        raise


def verify_control_binding(reader, args):
    """Dedicated new ABI full schemas; original replay is inherited, not rerun."""
    preref = reader.pin(PR,args.expected_preregistration_sha256)
    pr = reader.document(preref,PR)
    helpers = pr.get('source_helpers')
    require(type(helpers) is dict and helpers, 'frozen source helpers required')
    expected = {'fit_white_view_paired_linear.py':strict_sha(args.expected_driver_sha256),
        'white_view_fit_contract.py':strict_sha(args.expected_contract_sha256),
        'white_view_paired_linear.py':'79845db6f8ceec2e0c73488b705db526813da52bcc601ab6b10eb3f52f0ce6eb',
        'verified_design_gram.py':PRODUCER_SHA,'paired_linear_solver_verified_design.py':SOLVER_SHA,
        'paired_linear_solver.py':OLD_SOLVER_SHA}
    require(all(helpers.get(k) == v for k,v in expected.items()), 'new driver/contract/fixed numeric sources differ')
    for name, digest in helpers.items():
        require(type(name) is str and Path(name).name == name and name.endswith('.py'), 'strict helper basename required')
        reader.pin(R/'scripts'/name,strict_sha(digest))
    # First validate dependent import file identities. No arbitrary cached source
    # object is accepted as the actual dedicated driver or contract.
    for name in ('material_init','functional_anchor','paired_linear','white_view_paired_linear'):
        require(name+'.py' in helpers, 'original/new ABI dependency helper required')
        load_public_source(reader,name,helpers[name+'.py'])
    contract = load_public_source(reader,'white_view_fit_contract',args.expected_contract_sha256)
    require(all(callable(getattr(contract,name,None)) for name in CONTRACT_API_NAMES), 'new fit contract APIs missing')
    contract.validate_preregistration(pr)
    reader.pin_map(pr['runtime_sources'])
    actref = reader.pin(ACT,args.expected_activation_sha256)
    act = reader.document(actref,ACT); contract.validate_activation(act,pr)
    pfref = reader.pin(PF,args.expected_source_preflight_sha256)
    pf = reader.document(pfref,PF); contract.validate_source_preflight(pf,pr,args.expected_preregistration_sha256)
    reader.pin_map(act['inputs_before']); reader.pin_map(pf['inputs_before'])
    driver = load_public_source(reader,'fit_white_view_paired_linear',args.expected_driver_sha256)
    require(exact(pr['fit_driver'],reader.pin(R/'scripts/fit_white_view_paired_linear.py',args.expected_driver_sha256)),
            'preregistered actual driver fullref differs')
    for name, digest in driver.NUMERIC_HASHES.items():
        require(helpers.get(name) == digest, 'frozen numeric/original helper differs')
    build = pr['new_build_binding']
    for key, expected_sha in (('manifest',args.expected_build_manifest_sha256),
                             ('identity',args.expected_build_identity_sha256),
                             ('validator_source',args.expected_build_validator_sha256)):
        require(build[key]['sha256'] == strict_sha(expected_sha), 'external new build binding differs')
        reader.pin(build[key]['path'],{k:build[key][k] for k in ('bytes','sha256')})
    adjacent = Path(build['validator_source']['path']).with_name('white_view_build_contract.py')
    require(str(adjacent) in pr['runtime_sources'] and
            pr['runtime_sources'][str(adjacent)]['sha256'] == BUILD_CONTRACT_SHA, 'actual adjacent build contract binding differs')
    validator = driver._load_runtime_validator(build['validator_source'],pr['runtime_sources'])
    feature = driver.verify_new_runtime(pr,args,validator)
    verified = validator.verify_runtime(Path(build['runtime']),args.expected_build_manifest_sha256,args.expected_build_identity_sha256)
    require(type(verified) is dict and set(verified) == {'manifest','identity'}, 'actual new runtime bundle required')
    for key in ('manifest','identity'):
        require(exact(verified[key],reader.document(build[key])), 'actual raw build bundle differs')
    immutable = validator.c.immutable_inputs_for_runtime(verified,build['manifest'])
    reader.pin_map(immutable)
    require(exact(feature,pr['feature_binding']), 'feature/source/core binding differs')
    return contract, driver, pr, preref, actref, pfref


def forward_artifact_binding(native, coef, stock01, reference03, run_bytes, meta_bytes, binding, contract):
    """Full byte reconstruction only; actual core forward belongs to next proof."""
    run = contract.validate_fit_run(run_bytes,native,coef,stock01,binding)
    contract.validate_sidecar(meta_bytes,native,coef,stock01,run_bytes,binding)
    import white_view_paired_linear as white
    expected_reference = white.transform_initializer(stock01,binding=binding['feature_binding'])
    require(type(reference03) is bytes and reference03 == expected_reference, 'reference03 full transform bytes differ')
    return {'native03_full_reconstruction_verified':True,'sidecar_complete_run_binding_verified':True,
        'reference03_full_transform_verified':True,'reference03_sha256':sha(reference03),
        'native_magic':'SEKIRW03','feature_schema':'flat_white_view_aux_tied_v1',
        'coefficient_nearest_cast_core_forward_claimed':False,'actual_native_core_forward_verified':False,
        'actual_native_core_covariance_verified':False,'actual_incremental_verified':False,
        'forward_proof_requires_separate_new_runtime_core_execution':True}, run


def validate_design_provenance(design, manifest, before, feature, reference03, *, N=N_ACTUAL, dimension=DIM_ACTUAL):
    """Metadata/hash consistency only. No SFEN, teacher cache or replay parsing."""
    require(type(design) is dict and type(manifest) is dict, 'original manifest/design metadata required')
    fixed={'schema':'sekirei.white-view-paired-linear-train-design.v1','count':N,'dimension':dimension,
        'teacher_identity':TEACHER,'original_manifest_sha256':ORIGINAL_MANIFEST_SHA,
        'material_initializer_sha256':INITIALIZER_SHA,'transformed_initializer_sha256':sha(reference03),
        'native_magic':'SEKIRW03','feature_schema':'flat_white_view_aux_tied_v1','feature_binding':feature,
        'feature_index_implementation':'fixed white_view_paired_linear.active_features',
        'single_numpy_integer_adapter':True,'actual_build_verified_by_builder':False,
        'position_order_preserved':True,'labels_joined_by_exact_sfen':True,
        'labels_extra_metadata_bound_in_original_sha':True,'z_dtype':'signed-int8-C-row-major',
        'd_dtype':'signed-int32-little-endian','z_definition':'(raw_us-raw_them)/2; channels 2:256',
        'x_definition':'Z/16','d_definition':'original absolute STM teacher T minus fixed material M',
        'holdout_rows_read':False,'development_used':False,'final_used':False,
        'whole_dataset_replay_proof_claimed':False}
    require(all(exact(design.get(k),v) for k,v in fixed.items()), 'new original teacher/material/design provenance differs')
    identity_map(before)
    dataset=C/'source-input-recovery-v1/dataset'
    require(before.get(str(dataset/'manifest.json'),{}).get('sha256')==ORIGINAL_MANIFEST_SHA,
            'original manifest identity absent from completed fit')
    require(exact(design.get('original_files'),manifest.get('files')), 'original four file metadata differ')
    provenance_keys=('source_corpus_manifest_sha256','seed','split','sampling','games_per_pack',
                     'dependencies','independent_exclusions','derivation','games')
    require(all(k in manifest for k in provenance_keys) and
            exact(design.get('source_provenance'),{k:manifest[k] for k in provenance_keys}),
            'original corpus/game/exclusion provenance differs')
    for split in ('train','holdout'):
        for kind in ('positions','labels'):
            name=split+'.'+kind+'.jsonl';meta=manifest['files'].get(name)
            require(type(meta) is dict and type(meta.get('bytes')) is int and meta['bytes']>=0
                    and strict_sha(meta.get('sha256'))==before.get(str(dataset/name),{}).get('sha256')
                    and meta['bytes']==before.get(str(dataset/name),{}).get('bytes'), 'original file identities differ')
    for kind in ('positions','labels'):
        require(design.get('train_'+kind+'_sha256')==before[str(dataset/('train.'+kind+'.jsonl'))]['sha256'],
                'original TRAIN byte provenance differs')
    require(type(design.get('ft_raw_min')) is int and type(design.get('ft_raw_max')) is int and
            24<=design['ft_raw_min']<=design['ft_raw_max']<=104 and
            type(design.get('active_feature_min')) is int and type(design.get('active_feature_max')) is int and
            2<=design['active_feature_min']<=design['active_feature_max']<=40 and
            type(design.get('maximum_absolute_d')) is int and 0<=design['maximum_absolute_d']<=55779,
            'reported design bounded integer extrema invalid')
    strict_sha(design.get('ordered_sfens_sha256'));strict_sha(design.get('ft_prefix_sha256'))
    return {'original_four_file_identities_verified':True,'original_source_metadata_equal':True,
        'train_position_label_byte_hashes_equal':True,'extraction_replayed':False,'legal_replay_rerun':False}


def actual_main(args):
    require(not PROTOTYPE_ONLY, 'source prototype only; actual numerical audit entry blocked')
    os.umask(0o077); sys.dont_write_bytecode = True
    require(Path(__file__) == SELF and file_identity(SELF)['sha256'] == strict_sha(args.expected_worker_sha256),
            'canonical externally pinned audit worker required')
    require(not os.path.lexists(OUT) and C.resolve() == C and not C.stat().st_mode & 0o077,
            'fresh private audit receipt required')
    sys.path.insert(0,str(R/'scripts'))
    # Pin the bootstrap control/source bytes before importing lock helpers; all
    # of them are checked again inside the fixed locks by the same reader.
    reader=Reader(); reader.pin(SELF,args.expected_worker_sha256)
    bootstrap_ref=reader.pin(PR,args.expected_preregistration_sha256)
    bootstrap=reader.document(bootstrap_ref,PR)
    helpers=bootstrap.get('source_helpers')
    require(type(helpers) is dict and helpers, 'bootstrap externally bound helpers required')
    for name,digest in helpers.items():
        require(type(name) is str and Path(name).name==name and name.endswith('.py'), 'strict bootstrap helper required')
        reader.pin(R/'scripts'/name,strict_sha(digest))
    lock_module=load_public_source(reader,'benchmark',helpers.get('benchmark.py'))
    lifecycle_module=load_public_source(reader,'diagnose_anchor',helpers.get('diagnose_anchor.py'))
    nonblocking_lock=lock_module.nonblocking_lock; termination_guard=lifecycle_module.termination_guard
    # Fixed lock namespace. The reader launches no engine or fitting process.
    # The runtime validator invokes read-only compiler/source version commands.
    T=C.parent/'training-15-v1'; B=C.parent/'suisho11beta-sekirei-v0.3.39-v1'
    F=C.parent/'training-17-v1/bounded-material-fanin509-trainer-v1'
    N=C.parent/'suisho11beta-sekirei-v0.3.39-white-view-v1'
    exclusive=[T/'.training.lock',B/'.prepare.lock',B/'.benchmark.lock',F/'.build.lock',F/'.training.lock',Q.parent/'.fit.lock']
    shared=[N/'.build.lock',N/'.prepare.lock']
    with termination_guard(), ExitStack() as stack:
        for path in exclusive: stack.enter_context(nonblocking_lock(path,exclusive=True))
        for path in shared: stack.enter_context(nonblocking_lock(path,exclusive=False))
        reader.pin(SELF,args.expected_worker_sha256)
        contract,driver,pr,preref,actref,pfref = verify_control_binding(reader,args)
        runref=reader.pin(RUN,args.expected_run_sha256); run_bytes=reader.read(runref,RUN)
        run=strict_json(run_bytes)
        # validate_fit_run later closes the entire exact schema and byte ABI.
        require(type(run) is dict and exact(run.get('inputs_before'),run.get('inputs_after')), 'completed fit unchanged maps required')
        reader.pin_map(run['inputs_before'])
        require(exact(run.get('source_helpers_sha256'),pr['source_helpers']) and
                exact(run.get('runtime_sources'),pr['runtime_sources']) and exact(run.get('feature_binding'),pr['feature_binding'])
                and exact(run.get('new_build_binding'),pr['new_build_binding']), 'new fit/source/build context differs from PR')
        outputs=run.get('outputs')
        require(type(outputs) is dict and set(outputs) == set(contract.OUTPUT_NAMES), 'exact new Q nine outputs required')
        raw={}
        for key,name in contract.OUTPUT_NAMES.items():
            ref=outputs[key]; contract.fullref(ref,Q/name)
            require(ref['sha256'] == strict_sha(getattr(args,'expected_'+key+'_sha256')), 'external numeric/native artifact SHA differs: '+key)
            reader.pin(ref['path'],{k:ref[k] for k in ('bytes','sha256')})
            if key not in ('design_z','targets_d'): raw[key]=reader.read(ref,Q/name)
        require(exact(run.get('solver_certificate'),outputs['solver_certificate']), 'sole certificate fullref differs')
        metaref=reader.pin(Q/'weights.meta.json',args.expected_metadata_sha256)
        meta_bytes=reader.read(metaref,Q/'weights.meta.json')
        initialref=pr['initial_weights']; stock01=reader.read(initialref,contract.ORIGINAL_INITIALIZER)
        ref03=reader.pin(REFERENCE03,args.expected_reference03_sha256); reference03=reader.read(ref03,REFERENCE03)
        fitbind={'plan_sha256':PLAN_SHA,'preregistration_sha256':args.expected_preregistration_sha256,
            'activation_sha256':args.expected_activation_sha256,'source_preflight_sha256':args.expected_source_preflight_sha256,
            'fit_receipt':runref,'solver_certificate':outputs['solver_certificate'],
            'source_helpers_sha256':pr['source_helpers'],'feature_binding':pr['feature_binding'],'new_build_binding':pr['new_build_binding']}
        forward,run=forward_artifact_binding(raw['weights'],raw['coefficients_f32'],stock01,reference03,
                                             run_bytes,meta_bytes,fitbind,contract)
        design_sha=design_digest_files(N_ACTUAL,DIM_ACTUAL,outputs['design_z'],outputs['targets_d'])
        design=strict_json(raw['design_receipt'])
        original_manifest_ref=reader.pin(contract.DATASET/'manifest.json',ORIGINAL_MANIFEST_SHA)
        original_manifest=reader.document(original_manifest_ref,contract.DATASET/'manifest.json')
        provenance=validate_design_provenance(design,original_manifest,run['inputs_before'],pr['feature_binding'],reference03)
        result=audit_numeric_bytes(raw['gram_z'],raw['rhs_z'],raw['coefficients_f64'],raw['coefficients_f32'],
            raw['solver_certificate'],raw['design_receipt'],N=N_ACTUAL,dimension=DIM_ACTUAL,
            design_z_sha256=outputs['design_z']['sha256'],targets_d_sha256=outputs['targets_d']['sha256'],design_sha256=design_sha)
        before=dict(reader.files); after=reader.verify()
        receipt={'schema':'sekirei.white-view-paired-linear-independent-numeric-audit.v1','status':'complete',
            'created_at':datetime.now(timezone.utc).isoformat(),'candidate':MODE,'mode':MODE,'plan_sha256':PLAN_SHA,
            'fit_receipt':runref,'native':outputs['weights'],'metadata':metaref,'reference03':ref03,
            'coefficients_f64':outputs['coefficients_f64'],'coefficients_f32':outputs['coefficients_f32'],
            'solver_certificate':outputs['solver_certificate'],'design_receipt':outputs['design_receipt'],
            'preregistration':preref,'activation':actref,'source_preflight':pfref,
            'worker_sha256':args.expected_worker_sha256,'fit_driver_sha256':args.expected_driver_sha256,
            'fit_contract_sha256':args.expected_contract_sha256,'source_helpers_sha256':pr['source_helpers'],
            'feature_binding':pr['feature_binding'],'new_build_binding':pr['new_build_binding'],
            'new_build_verified':True,'numerical_math':result,'forward_artifact_binding':forward,
            'design_provenance_binding':provenance,
            'inputs_before':before,'inputs_after':after,'inputs_unchanged':True,'source_unchanged':True,
            'exclusive_locks':list(map(str,exclusive)),'architecture_shared_locks':list(map(str,shared)),
            'no_child_process_started':False,'no_engine_or_fit_process_started':True,
            'runtime_validator_may_invoke_readonly_source_compiler_checks':True,
            'actual_fit_repeated':False,'parent_math_is_not_independent_receipt':True,
            'gram_psd_and_design_origin_from_external_fit_producer':True,'original_dataset_replay_rerun':False,
            'audit_input_scope':'completed new ABI run, all frozen input/source/build identities and numeric artifacts; original replay inherited',
            'native_core_proof_claimed':False,'final_used':False,'adoption_claimed':False}
        with OUT.open('xb') as f:
            f.write(canonical_json(receipt)); f.flush(); os.fsync(f.fileno())
        print(json.dumps({'status':'complete','receipt':fullref(OUT)}))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    names=('worker','run','driver','contract','preregistration','activation','source-preflight',
        'build-manifest','build-identity','build-validator','metadata','reference03',
        'weights','gram-z','rhs-z','coefficients-f64','coefficients-f32','solver-certificate',
        'design-receipt','design-z','targets-d')
    for name in names:p.add_argument('--expected-'+name+'-sha256',required=True)
    return p


if __name__ == '__main__':
    if PROTOTYPE_ONLY:
        raise RuntimeError('source prototype only; actual numerical audit entry blocked')
    actual_main(parser().parse_args())
