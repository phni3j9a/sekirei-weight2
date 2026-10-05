"""SHA-bound authorization evidence for the fixed conditional FanIn509 route.

This verifier starts no process and makes no adoption decision. Runtime calls
read the completed control receipts and the stopped input snapshot; pure
validate_bundle accepts synthetic objects without filesystem access.
"""
import hashlib
import json
from pathlib import Path
import stat

from prepare import REPO

C = Path('/home/server/.local/share/sekirei-weight2/campaign-17-autonomous-v1')
B = C.parent / 'suisho11beta-sekirei-v0.3.39-v1'
T = C.parent / 'training-15-v1'
F = C.parent / 'training-17-v1/bounded-material-trainer-v3'
E = C / 'eval-bounded-material-residual-100cp-e3-v1'
W = F / 'run-e3-v1/weights.bin'
ACTIVATION = C / 'bounded-material-fanin509-activation-v1.json'
PLAN = C / 'bounded-material-fanin509-plan-v1.json'
PLAN_SHA256 = '359cf9daa1a5e57be6c7a2d56b2ecfef5c9e24cfc7648e45f7930544b264ae6e'
PRIOR = 'bounded-material-residual-100cp-e3-v1'
NEXT = 'bounded-material-fanin509-100cp-e3-v1'
MODEL_SHA256 = 'edd072bc04f1f018245d90d563576f1a9df77884c5e0370277b3e498f111927a'
META_SHA256 = '3a18c8b911e49e23b42738ce68ba9036d5f71331134e5a278c22a0aae045ab2a'
PREREG_SHA256 = '3f164c4d062cf02f1072e933f86fffd569c9190db7a925159e425d516d74b7d2'
PREFLIGHT_SHA256 = '094d83606986e8db758caa3a2560e1d72c129aecd24a7dd69d75017acb68e631'
PREFIX = 'development-17-bounded-material-e3-v1'
PATHS = {'comparison': C / 'comparison-bounded-material-residual-100cp-e3-v1.json',
         'independent_review': C / 'bounded-independent-formal-review-v1.json',
         'stop_receipt': C / 'bounded-formal-stopped-v1.json',
         'operational': C / 'bounded-formal-operational-verification-v1.json',
         'terminal': C / 'bounded-formal-executor-terminal-observation-v1.json',
         'evaluation': E / 'evaluation.json'}
STOP_WORKER = C / 'bounded-formal-stop-worker-v1.py'
WORKERS = sorted([str(C / 'bounded-formal-evaluation-worker-v1.py'), str(E / 'evaluation-script.py')])
EXECUTABLES = sorted([str(B / 'build/sekirei/release/sekirei'),
                      str(B / 'sources/yaneuraou/source/YaneuraOu-by-gcc')])
LOCKS = [str(p) for p in (T / '.training.lock', B / '.prepare.lock', B / '.benchmark.lock',
                         F / '.build.lock', F / '.training.lock')]
STAGES = {'mae-pilot': 102, 'mae': 1140, 'top3-pilot': 36, 'top3': 551}
BASE_RUNS = {'mae-pilot': 'development-pilot-20261002-v039-v1',
             'mae': 'development-baseline-20261002-v039-v1',
             'top3-pilot': 'development-top3-baseline-pilot-20261002-v1',
             'top3': 'development-top3-baseline-20261002-v1'}
IDENTITIES = {'execution_except_model', 'occurrences', 'teacher_results', 'teacher_E_cp',
              'top3_occurrences', 'top3_legal_moves_sha256', 'top3_denominators', 'top3_teacher_bestmoves'}
SCOPE = {'split': 'development', 'games': 5, 'teacher_E': 266, 'top3_denominator': 551,
         'candidate_attempts': 1829, 'baseline_attempts': 1829, 'own_pilot_attempts': 138,
         'requested_nodes': 1000000, 'maximum_reported_nodes': 1010000}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(exact(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def sha(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            'invalid exact SHA256')
    return value


def validate_reference(value, expected_path):
    require(type(value) is dict and set(value) == {'path', 'sha256'}
            and type(value['path']) is str and value['path'] == str(expected_path),
            'wrong fixed activation/reference path or fields')
    sha(value['sha256'])
    return value


def validate_trigger_reference(value):
    return validate_reference(value, ACTIVATION)


def validate_build_binding(build, reference, verifier_sha256, proof):
    validate_trigger_reference(reference)
    sha(verifier_sha256)
    require(type(build) is dict and exact(build.get('trigger'), reference)
            and build.get('activation_helper_sha256') == verifier_sha256
            and exact(build.get('activation_validation'), proof),
            'build/preregistration activation or verifier source mismatch')


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(_value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def file_bytes(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink()
            and stat.S_ISREG(path.stat().st_mode), 'noncanonical or nonregular activation input')
    return path.read_bytes()


def file_info(path):
    # Actual stopped model bytes are hashed at future runtime only, never in
    # source-only preparation or pure fixtures.
    data = file_bytes(path)
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def verify_public_module(expected_sha256):
    expected = REPO / 'scripts/fanin509_activation.py'
    require(Path(__file__) == expected and Path(__file__).resolve() == expected,
            'use the canonical public activation verifier; alternate import rejected')
    require(file_info(expected)['sha256'] == sha(expected_sha256), 'activation verifier source changed')
    return file_info(expected)


def validate_bundle(activation, docs, actual):
    """Pure: actual is canonical path -> actual {bytes, sha256} file identities."""
    require(type(activation) is dict and activation.get('schema') == 'sekirei.bounded-material-fanin509-activation.v1'
            and activation.get('status') == 'verified-after-valid-nonadopt-stopped'
            and activation.get('prior_candidate') == PRIOR and activation.get('next_candidate') == NEXT
            and activation.get('plan_sha256') == PLAN_SHA256, 'wrong conditional activation contract')
    for name, expected in {'formal_valid': True, 'adopted': False, 'all_groups_stopped': True, 'final_used': False}.items():
        require(exact(activation.get(name), expected), 'activation flag mismatch: ' + name)
    require(type(docs) is dict and set(docs) == set(PATHS), 'six actual control receipts required')
    require(type(actual) is dict, 'actual SHA inventory required')
    for path, value in actual.items():
        require(type(path) is str and Path(path).is_absolute() and Path(path).as_posix() == path
                and '..' not in Path(path).parts and type(value) is dict and set(value) == {'bytes', 'sha256'}
                and type(value['bytes']) is int and value['bytes'] >= 0, 'invalid actual file identity')
        sha(value['sha256'])
    def identity(path):
        require(str(path) in actual, 'actual reference SHA missing: ' + str(path))
        return actual[str(path)]
    def bound(value, path, full=False):
        require(type(value) is dict and type(value.get('path')) is str and value['path'] == str(path)
                and value.get('sha256') == identity(path)['sha256'], 'receipt dependency path/SHA mismatch')
        if full:
            require(exact(value, {'path': str(path), **identity(path)}), 'receipt byte count/fields mismatch')
    for name in ('comparison', 'independent_review', 'stop_receipt'):
        validate_reference(activation.get(name), PATHS[name]); bound(activation[name], PATHS[name])
    cp, au, stop = (docs[name] for name in ('comparison', 'independent_review', 'stop_receipt'))
    op, term, evaluation = (docs[name] for name in ('operational', 'terminal', 'evaluation'))
    for value, schema in ((cp, 'sekirei.formal-comparison.v1'),
                          (au, 'sekirei.bounded-independent-formal-review.v1')):
        require(type(value) is dict and value.get('schema') == schema and value.get('status') == 'complete',
                'incomplete comparison/independent audit')
        for key, expected in {'comparison_valid': True, 'adopt': False, 'inputs_unchanged': True, 'final_used': False}.items():
            require(exact(value.get(key), expected), 'invalid completed formal flag: ' + key)
        require(value.get('decision') in ('hold', 'reject') and type(value.get('mae_improved')) is bool
                and type(value.get('top3_preserved')) is bool
                and value['adopt'] == (value['mae_improved'] and value['top3_preserved']), 'formal decision inconsistent')
        checks = value.get('checks')
        require(type(checks) is dict and set(checks) == IDENTITIES, 'eight formal identities required')
        for item in checks.values():
            require(type(item) is dict and item.get('equal') is True
                    and sha(item.get('baseline_sha256')) == sha(item.get('candidate_sha256')),
                    'formal identity not equal')
            if 'mismatch_count' in item:
                require(exact(item['mismatch_count'], 0), 'formal identity mismatch count')
    require(exact(cp.get('requested_nodes'), 1000000), 'formal node condition changed')
    for name in ('decision', 'adopt', 'mae_improved', 'top3_preserved', 'checks'):
        require(exact(cp.get(name), au.get(name)), 'comparison/independent decision or identity conflict')
    require(au.get('goal_achieved') is False and au.get('preregistration_sha256') == PREREG_SHA256
            and au.get('launch_preflight_sha256') == PREFLIGHT_SHA256, 'wrong independently audited candidate provenance')
    bound(au.get('comparison_receipt'), PATHS['comparison'])
    bound(au.get('operational_receipt'), PATHS['operational'])
    scope = au.get('scope')
    require(type(scope) is dict and all(exact(scope.get(k), v) for k, v in SCOPE.items()),
            'wrong full independent audit scope/type')
    sources = {'candidate': {'config': str(E / 'candidate.json'), 'mae': str(B / 'runs' / (PREFIX + '-mae')),
                            'top3': str(B / 'runs' / (PREFIX + '-top3')), 'runtime': str(B), 'evaluation': str(PATHS['evaluation'])},
               'baseline': {'config': str(REPO / 'config/development-benchmark.json'),
                            'mae': str(B / 'runs' / BASE_RUNS['mae']), 'top3': str(B / 'runs' / BASE_RUNS['top3']),
                            'runtime': str(B), 'evaluation': 'None'}}
    require(exact(cp.get('sources'), sources), 'wrong comparison subjects/config/stage paths')
    model = {'path': str(W), 'bytes': 1305356, 'sha256': MODEL_SHA256}
    require(exact(identity(W), {k: model[k] for k in ('bytes', 'sha256')})
            and identity(W.with_suffix('.meta.json'))['sha256'] == META_SHA256, 'stopped model/metadata changed')
    require(exact(cp.get('candidate', {}).get('model_identity'), {'kind': 'nnue', **model})
            and exact(cp.get('baseline', {}).get('model_identity'),
                      {'kind': 'material_fallback', 'path': None, 'sha256': None, 'bytes': None}), 'wrong model/baseline identity')
    require(type(au.get('candidate_model')) is dict and all(exact(au['candidate_model'].get(k), v) for k, v in model.items())
            and au['candidate_model'].get('metadata_sha256') == META_SHA256, 'independent candidate model binding changed')
    audits = au.get('run_audits')
    require(type(audits) is dict and set(audits) == {'candidate', 'baseline'}, 'full candidate/baseline audits required')
    for subject, stages in audits.items():
        require(type(stages) is dict and set(stages) == set(STAGES), 'four audited stages required')
        for stage, count in STAGES.items():
            record = stages[stage]
            path = B / 'runs' / ((PREFIX + '-' + stage) if subject == 'candidate' else BASE_RUNS[stage])
            require(type(record) is dict and record.get('path') == str(path), 'audited stage path changed')
            for key, expected in {'attempts': count, 'cleanup_ok': count, 'runner_exit_zero': count,
                                  'raw_sha256_and_lifecycle_verified': count, 'technical_failures': 0, 'timeout_failures': 0}.items():
                require(exact(record.get(key), expected), 'audited stage count/cleanup/type mismatch: ' + key)
    require(type(op) is dict and op.get('schema') == 'sekirei.bounded-formal-operational-verification.v1'
            and op.get('status') == 'complete' and op.get('inputs_unchanged') is True and op.get('build_unchanged') is True
            and op.get('adoption_claimed') is False and op.get('final_used') is False
            and exact(op.get('selected_epoch'), 3) and op.get('preregistration_sha256') == PREREG_SHA256,
            'wrong operational completion/provenance')
    inputs = op.get('inputs_before')
    require(type(inputs) is dict and len(inputs) >= 1204 and exact(inputs, op.get('inputs_after')),
            'operational full input snapshot changed/incomplete')
    for path, value in inputs.items():
        require(type(path) is str and Path(path).is_absolute() and Path(path).as_posix() == path
                and '..' not in Path(path).parts and type(value) is dict and set(value) == {'bytes', 'sha256'}
                and type(value['bytes']) is int and value['bytes'] >= 0, 'operational input identity type/path invalid')
        sha(value['sha256'])
    for path, expected in ((W, MODEL_SHA256), (W.with_suffix('.meta.json'), META_SHA256),
                           (C / 'bounded-material-preregistration-v2.json', PREREG_SHA256),
                           (C / 'launch-preflight-bounded-material-residual-e3-v1.json', PREFLIGHT_SHA256)):
        require(str(path) in inputs and inputs[str(path)]['sha256'] == expected,
                'operational fixed model/preregistration/preflight binding missing')
    require(exact(au['operational_receipt'].get('preflight_input_count'), 1204)
            and exact(au['operational_receipt'].get('bound_input_count'), len(inputs)), 'operational audit inventory count mismatch')
    require(op.get('evaluation_sha256') == identity(PATHS['evaluation'])['sha256'], 'operational/evaluation reference mismatch')
    require(type(evaluation) is dict and evaluation.get('status') == 'complete' and evaluation.get('weight_sha256') == MODEL_SHA256
            and exact(evaluation.get('run_ids'), {s: PREFIX + '-' + s for s in STAGES}), 'four completed candidate stages required')
    require(type(term) is dict and term.get('schema') == 'sekirei.executor-terminal-observation.v1'
            and exact(term.get('session_id'), 28371) and exact(term.get('exit_code'), 0)
            and term.get('session_closed') is True, 'executor was not reaped successfully')
    require(type(stop) is dict and stop.get('schema') == 'sekirei.bounded-formal-stopped.v1'
            and stop.get('status') == 'verified-stopped-under-locks' and stop.get('candidate') == PRIOR,
            'wrong fixed stop receipt')
    for key, expected in {'all_groups_stopped': True, 'comparison_valid': True, 'adopt': False, 'final_used': False,
                          'inputs_unchanged': True, 'executor_session_id': 28371, 'executor_exit_code': 0,
                          'recorded_cleanup_verified_candidate_attempts': 1829,
                          'recorded_cleanup_verified_baseline_attempts': 1829}.items():
        require(exact(stop.get(key), expected), 'stop flag/scope/type mismatch: ' + key)
    require(exact(stop.get('exclusive_locks'), LOCKS) and exact(stop.get('related_worker_paths'), WORKERS)
            and exact(stop.get('related_executables'), EXECUTABLES),
            'wrong stopped exclusion locks/workers/executables')
    scans = stop.get('process_scans')
    require(type(scans) is list and len(scans) == 2, 'two stopped process scans required')
    for scan in scans:
        require(type(scan) is dict and set(scan) == {'related_processes', 'owned_processes_scanned'}
                and exact(scan['related_processes'], []) and type(scan['owned_processes_scanned']) is int
                and scan['owned_processes_scanned'] >= 0, 'related process remained or invalid scan type')
    for field, name in {'comparison_receipt': 'comparison', 'independent_review_receipt': 'independent_review',
                        'operational_receipt': 'operational', 'terminal_observation': 'terminal',
                        'evaluation_receipt': 'evaluation'}.items():
        bound(stop.get(field), PATHS[name], full=True)
    bound(stop.get('candidate_model'), W, full=True)
    snapshot_paths = {STOP_WORKER, PATHS['terminal'], PATHS['operational'], PATHS['comparison'],
                      PATHS['independent_review'], PATHS['evaluation'], W, W.with_suffix('.meta.json')}
    snapshot = {str(p): identity(p) for p in snapshot_paths}
    require(exact(stop.get('inputs_before'), snapshot) and exact(stop.get('inputs_after'), snapshot),
            'stop snapshot actual file hashes changed/incomplete')
    return {'schema': 'sekirei.fanin509-activation-validation.v1', 'status': 'verified',
            'prior_candidate': PRIOR, 'next_candidate': NEXT, 'plan_sha256': PLAN_SHA256,
            'formal_valid': True, 'adopted': False, 'all_groups_stopped': True, 'final_used': False,
            'receipt_and_stopped_input_identities': actual,
            'limitation': 'Stopped state is an observation under cooperating locks; no process replay or new adoption decision.'}


def verify_activation(reference):
    """Future runtime: actual SHA-bound receipt parsing and stopped snapshot rehash."""
    validate_trigger_reference(reference)
    actual = {}
    def read(path):
        data = file_bytes(path)
        actual[str(path)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        return strict_json(data)
    activation = read(ACTIVATION)
    require(actual[str(ACTIVATION)]['sha256'] == reference['sha256'], 'externally pinned activation SHA changed')
    docs = {name: read(path) for name, path in PATHS.items()}
    for path in (STOP_WORKER, W, W.with_suffix('.meta.json')):
        actual[str(path)] = file_info(path)
    actual[str(PLAN)] = file_info(PLAN)
    require(actual[str(PLAN)]['sha256'] == PLAN_SHA256, 'fixed next plan changed')
    proof = validate_bundle(activation, docs, actual)
    require(all(exact(file_info(path), value) for path, value in actual.items()), 'activation inputs changed during verification')
    return proof
