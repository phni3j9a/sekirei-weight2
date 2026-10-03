"""Synthetic JSON-only activation fixtures; no private file, model or process."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fanin509_activation as guard
import prepare_fanin509 as builder
import train_fanin509 as training
import diagnose_fanin509 as diagnosis


def synthetic_bundle():
    actual = {str(guard.W): {'bytes': 1305356, 'sha256': guard.MODEL_SHA256},
              str(guard.W.with_suffix('.meta.json')): {'bytes': 101, 'sha256': guard.META_SHA256},
              str(guard.STOP_WORKER): {'bytes': 22, 'sha256': '1' * 64}}
    docs, raw = {}, {}
    def register(name, value):
        data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        path = str(guard.PATHS[name]); raw[path] = data; docs[name] = value
        actual[path] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    def ref(name):
        path = str(guard.PATHS[name]); return {'path': path, **actual[path]}
    checks = {name: {'equal': True, 'baseline_sha256': '2' * 64, 'candidate_sha256': '2' * 64}
              for name in guard.IDENTITIES}
    sources = {
        'candidate': {'config': str(guard.E / 'candidate.json'), 'mae': str(guard.B / 'runs' / (guard.PREFIX + '-mae')),
                      'top3': str(guard.B / 'runs' / (guard.PREFIX + '-top3')), 'runtime': str(guard.B),
                      'evaluation': str(guard.PATHS['evaluation'])},
        'baseline': {'config': str(guard.REPO / 'config/development-benchmark.json'),
                     'mae': str(guard.B / 'runs' / guard.BASE_RUNS['mae']),
                     'top3': str(guard.B / 'runs' / guard.BASE_RUNS['top3']), 'runtime': str(guard.B), 'evaluation': 'None'}}
    common = {'status': 'complete', 'comparison_valid': True, 'inputs_unchanged': True,
              'final_used': False, 'adopt': False, 'decision': 'reject', 'mae_improved': False,
              'top3_preserved': False, 'checks': checks}
    register('evaluation', {'status': 'complete', 'weight_sha256': guard.MODEL_SHA256,
                            'run_ids': {s: guard.PREFIX + '-' + s for s in guard.STAGES}})
    register('terminal', {'schema': 'sekirei.executor-terminal-observation.v1',
                          'session_id': 28371, 'exit_code': 0, 'session_closed': True})
    inputs = {f'/tmp/synthetic-private-input-{i}': {'bytes': i, 'sha256': '3' * 64} for i in range(1201)}
    inputs.update({str(guard.W): actual[str(guard.W)], str(guard.W.with_suffix('.meta.json')): actual[str(guard.W.with_suffix('.meta.json'))],
                   str(guard.C / 'bounded-material-preregistration-v2.json'): {'bytes': 101, 'sha256': guard.PREREG_SHA256},
                   str(guard.C / 'launch-preflight-bounded-material-residual-e3-v1.json'): {'bytes': 101, 'sha256': guard.PREFLIGHT_SHA256}})
    register('operational', {'schema': 'sekirei.bounded-formal-operational-verification.v1',
        'status': 'complete', 'inputs_unchanged': True, 'build_unchanged': True, 'adoption_claimed': False,
        'final_used': False, 'selected_epoch': 3, 'preregistration_sha256': guard.PREREG_SHA256,
        'inputs_before': inputs, 'inputs_after': copy.deepcopy(inputs),
        'evaluation_sha256': actual[str(guard.PATHS['evaluation'])]['sha256']})
    model = {'path': str(guard.W), **actual[str(guard.W)]}
    register('comparison', {'schema': 'sekirei.formal-comparison.v1', **copy.deepcopy(common),
        'requested_nodes': 1000000, 'sources': sources, 'candidate': {'model_identity': {'kind': 'nnue', **model}},
        'baseline': {'model_identity': {'kind': 'material_fallback', 'path': None, 'bytes': None, 'sha256': None}}})
    audits = {subject: {stage: {'path': str(guard.B / 'runs' / ((guard.PREFIX + '-' + stage)
                               if subject == 'candidate' else guard.BASE_RUNS[stage])),
        'attempts': n, 'cleanup_ok': n, 'runner_exit_zero': n, 'raw_sha256_and_lifecycle_verified': n,
        'technical_failures': 0, 'timeout_failures': 0} for stage, n in guard.STAGES.items()}
        for subject in ('candidate', 'baseline')}
    register('independent_review', {'schema': 'sekirei.bounded-independent-formal-review.v1', **copy.deepcopy(common),
        'goal_achieved': False, 'preregistration_sha256': guard.PREREG_SHA256,
        'launch_preflight_sha256': guard.PREFLIGHT_SHA256, 'scope': copy.deepcopy(guard.SCOPE),
        'candidate_model': {**model, 'metadata_sha256': guard.META_SHA256}, 'run_audits': audits,
        'comparison_receipt': ref('comparison'), 'operational_receipt': {**ref('operational'),
                                                 'bound_input_count': len(inputs), 'preflight_input_count': 1204}})
    snapshot_paths = {guard.STOP_WORKER, guard.W, guard.W.with_suffix('.meta.json'),
                      *(guard.PATHS[n] for n in ('terminal', 'operational', 'comparison', 'independent_review', 'evaluation'))}
    snapshot = {str(p): actual[str(p)] for p in snapshot_paths}
    register('stop_receipt', {'schema': 'sekirei.bounded-formal-stopped.v1', 'status': 'verified-stopped-under-locks',
        'candidate': guard.PRIOR, 'all_groups_stopped': True, 'comparison_valid': True, 'adopt': False,
        'final_used': False, 'inputs_unchanged': True, 'executor_session_id': 28371, 'executor_exit_code': 0,
        'recorded_cleanup_verified_candidate_attempts': 1829, 'recorded_cleanup_verified_baseline_attempts': 1829,
        'exclusive_locks': list(guard.LOCKS), 'related_worker_paths': list(guard.WORKERS),
        'related_executables': list(guard.EXECUTABLES),
        'process_scans': [{'related_processes': [], 'owned_processes_scanned': 20} for _ in range(2)],
        **{k: ref(n) for k, n in {'comparison_receipt': 'comparison', 'independent_review_receipt': 'independent_review',
                                 'operational_receipt': 'operational', 'terminal_observation': 'terminal',
                                 'evaluation_receipt': 'evaluation'}.items()}, 'candidate_model': model,
        'inputs_before': snapshot, 'inputs_after': copy.deepcopy(snapshot)})
    activation = {'schema': 'sekirei.bounded-material-fanin509-activation.v1',
        'status': 'verified-after-valid-nonadopt-stopped', 'prior_candidate': guard.PRIOR, 'next_candidate': guard.NEXT,
        'plan_sha256': guard.PLAN_SHA256, 'formal_valid': True, 'adopted': False,
        'all_groups_stopped': True, 'final_used': False, 'created_at': 'synthetic',
        **{n: {'path': str(guard.PATHS[n]), 'sha256': actual[str(guard.PATHS[n])]['sha256']}
           for n in ('comparison', 'independent_review', 'stop_receipt')}}
    data = json.dumps(activation, sort_keys=True).encode(); raw[str(guard.ACTIVATION)] = data
    actual[str(guard.ACTIVATION)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    actual[str(guard.PLAN)] = {'bytes': 10, 'sha256': guard.PLAN_SHA256}
    return activation, docs, actual, raw


class ActivationTests(unittest.TestCase):
    def test_valid_full_bundle_and_runtime_actual_receipt_bytes(self):
        activation, docs, actual, raw = synthetic_bundle()
        proof = guard.validate_bundle(activation, docs, actual)
        self.assertIs(proof['formal_valid'], True); self.assertIs(proof['adopted'], False)
        # Fake only new verifier file IO. No legacy guard or private file is read.
        with patch.object(guard, 'file_bytes', side_effect=lambda p: raw[str(p)]), \
             patch.object(guard, 'file_info', side_effect=lambda p: actual[str(p)]):
            result = guard.verify_activation({'path': str(guard.ACTIVATION), 'sha256': actual[str(guard.ACTIVATION)]['sha256']})
        self.assertEqual(result, proof)

    def test_actual_sha_binding_and_late_mutation_fail_closed(self):
        _, _, actual, raw = synthetic_bundle()
        ref = {'path': str(guard.ACTIVATION), 'sha256': actual[str(guard.ACTIVATION)]['sha256']}
        with patch.object(guard, 'file_bytes', side_effect=lambda p: raw[str(p)]), \
             patch.object(guard, 'file_info', side_effect=lambda p: actual[str(p)]):
            with self.assertRaisesRegex(ValueError, 'externally pinned'):
                guard.verify_activation(dict(ref, sha256='0' * 64))
            # Changed actual comparison bytes cannot satisfy its preregistered ref.
            raw[str(guard.PATHS['comparison'])] += b' '
            with self.assertRaisesRegex(ValueError, 'dependency path/SHA'):
                guard.verify_activation(ref)
        _, _, actual, raw = synthetic_bundle()
        def changed(path):
            result = dict(actual[str(path)])
            if Path(path) == guard.PATHS['terminal']: result['sha256'] = '0' * 64
            return result
        with patch.object(guard, 'file_bytes', side_effect=lambda p: raw[str(p)]), \
             patch.object(guard, 'file_info', side_effect=changed), self.assertRaisesRegex(ValueError, 'changed during'):
            guard.verify_activation(ref)

    def test_incomplete_adopted_type_and_conflicting_decisions(self):
        for doc, field, value in [('comparison', 'status', 'running'), ('independent_review', 'adopt', True),
             ('stop_receipt', 'adopt', 0), ('comparison', 'requested_nodes', 1000000.0),
             ('independent_review', 'mae_improved', True), ('terminal', 'session_closed', 1)]:
            a, d, actual, _ = synthetic_bundle(); d[doc][field] = value
            with self.subTest(doc=doc, field=field), self.assertRaises(ValueError): guard.validate_bundle(a, d, actual)
        a, d, actual, _ = synthetic_bundle(); a['adopted'] = 0
        with self.assertRaises(ValueError): guard.validate_bundle(a, d, actual)

    def test_scope_cleanup_identity_and_operational_count_types(self):
        mutations = [lambda d: d['independent_review']['scope'].update(candidate_attempts=1829.0),
            lambda d: d['independent_review']['run_audits']['candidate']['mae'].update(cleanup_ok=1139),
            lambda d: d['independent_review']['run_audits']['baseline']['top3'].update(timeout_failures=False),
            lambda d: d['comparison']['checks']['occurrences'].update(equal=1),
            lambda d: d['independent_review']['operational_receipt'].update(bound_input_count=True),
            lambda d: d['operational'].update(inputs_after={}),
            lambda d: d['evaluation']['run_ids'].update(top3='wrong-stage')]
        for mutate in mutations:
            a, d, actual, _ = synthetic_bundle(); mutate(d)
            with self.assertRaises(ValueError): guard.validate_bundle(a, d, actual)

    def test_stop_locks_scans_snapshot_and_input_bytes(self):
        mutations = [lambda d: d['stop_receipt']['exclusive_locks'].pop(),
            lambda d: d['stop_receipt']['process_scans'].pop(),
            lambda d: d['stop_receipt']['related_executables'].pop(),
            lambda d: d['stop_receipt']['process_scans'][0].update(related_processes=[{'pid': 20}]),
            lambda d: d['stop_receipt']['process_scans'][1].update(owned_processes_scanned=True),
            lambda d: d['stop_receipt']['inputs_before'][str(guard.W)].update(bytes=True),
            lambda d: d['stop_receipt']['comparison_receipt'].update(bytes=1),
            lambda d: d['stop_receipt'].update(recorded_cleanup_verified_baseline_attempts=1828)]
        for mutate in mutations:
            a, d, actual, _ = synthetic_bundle(); mutate(d)
            with self.assertRaises(ValueError): guard.validate_bundle(a, d, actual)

    def test_wrong_model_metadata_references_and_path_aliases(self):
        for where, field, value in [('candidate_model', 'sha256', 'f' * 64),
                                    ('candidate_model', 'metadata_sha256', 'f' * 64),
                                    ('comparison_receipt', 'path', '/tmp/comparison.json')]:
            a, d, actual, _ = synthetic_bundle(); d['independent_review'][where][field] = value
            with self.assertRaises(ValueError): guard.validate_bundle(a, d, actual)
        for ref in [{'path': '/tmp/activation.json', 'sha256': 'a' * 64},
                    {'path': str(guard.C / '../campaign-17-autonomous-v1' / guard.ACTIVATION.name), 'sha256': 'a' * 64},
                    {'path': str(guard.ACTIVATION), 'sha256': True}]:
            with self.assertRaises(ValueError): guard.validate_trigger_reference(ref)

    def test_build_identity_prereg_trigger_verifier_and_proof_match(self):
        a, d, actual, _ = synthetic_bundle(); proof = guard.validate_bundle(a, d, actual)
        ref = {'path': str(guard.ACTIVATION), 'sha256': actual[str(guard.ACTIVATION)]['sha256']}
        build = {'trigger': dict(ref), 'activation_helper_sha256': '4' * 64, 'activation_validation': copy.deepcopy(proof)}
        guard.validate_build_binding(build, ref, '4' * 64, proof)
        for key, value in [('trigger', dict(ref, sha256='5' * 64)),
                           ('activation_helper_sha256', '6' * 64), ('activation_validation', {})]:
            bad = dict(build, **{key: value})
            with self.assertRaises(ValueError): guard.validate_build_binding(bad, ref, '4' * 64, proof)

    def test_strict_json_duplicate_nonfinite_and_prototype_barriers(self):
        for raw in (b'{"adopt":false,"adopt":false}', b'{"value":NaN}', b'{"value":Infinity}'):
            with self.assertRaises(ValueError): guard.strict_json(raw)
        # The disabled branch stays testable after public activation enables
        # the builder; restore its actual setting when this fixture exits.
        with patch.object(builder, 'PROTOTYPE_ONLY', True):
            for call, args in [(builder.prepare, (None, None, None)), (builder.verify_build, (None, None)),
                               (training.train, (None,)), (diagnosis.diagnose, (None,)), (diagnosis._diagnose, (None, None, None))]:
                with self.assertRaisesRegex(ValueError, 'prepared prototype only'): call(*args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
