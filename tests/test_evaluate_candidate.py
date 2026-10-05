"""Untrained tiny weights and synthetic manifests; never execute engines."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_candidate as e


def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


class EvaluationFixture:
    def __init__(self, root, mismatch=None, mutate=None):
        self.root, self.mismatch, self.mutate = root, mismatch, mutate
        self.weight = root / 'untrained-public-fixture.bin'
        self.weight.write_bytes(b'untrained tiny public fixture')
        self.runtime = root / 'explicit-white-runtime'
        self.runtime.mkdir()
        self.base = e.b.load_config(e.b.CONFIG_PATH)
        self.base['candidate_model'] = {'kind': 'nnue', 'path': str(root / 'incumbent.bin')}
        self.base['engines']['sekirei']['options']['EvalFile'] = str(root / 'incumbent.bin')
        self.base_path = root / 'base.json'
        save(self.base_path, self.base)
        self.args = Namespace(weight=self.weight, expected_weight_sha256=e.sha256(self.weight),
            runtime=self.runtime, base_config=self.base_path, output=root / 'evaluation', prefix='public-candidate')
        self.calls = []
        self.snapshots = []

    def recorded_model(self, name, config):
        model = e.b._model_identity(config)
        if name == self.mismatch:
            model['sha256'] = 'a' * 64
        return model

    def execute(self, runtime, config, run_type, *, run_id):
        name = 'mae-pilot' if run_type == 'pilot' else 'mae'
        self.calls.append((name, runtime)); self.snapshots.append((name, deepcopy(config)))
        manifest = {'run_id': run_id, 'run_type': run_type, 'fingerprint': name + '-fingerprint',
                    'runtime_identity': {'candidate_model': self.recorded_model(name, config)}}
        save(runtime / 'runs' / run_id / 'manifest.json', manifest)
        if self.mutate == name: self.weight.write_bytes(b'physical weight changed inside stage')

    def top3(self, args):
        config = e.b.load_config(args.mae_config)
        name = 'top3-pilot' if args.action == 'pilot' else 'top3'
        self.calls.append((name, args.runtime)); self.snapshots.append((name, deepcopy(config)))
        expected_pilot = None if name == 'top3-pilot' else self.args.prefix + '-top3-pilot'
        if args.pilot_run_id != expected_pilot:
            raise AssertionError('Top3 must use the candidate own Top3 pilot')
        if args.mae_run != self.runtime / 'runs' / (self.args.prefix + '-mae'):
            raise AssertionError('Top3 must use the candidate own formal MAE')
        manifest = {'run_id': args.run_id, 'run_type': args.action,
            'payload': {'identity': {'base_execution_identity': {'runtime': {
                'candidate_model': self.recorded_model(name, config)}}}}}
        save(args.runtime / 'runs' / args.run_id / 'manifest.json', manifest)
        save(args.runtime / 'runs' / args.run_id / 'top3-report.json', {'valid': True, 'top3_rate': 0.5})
        if self.mutate == name: self.weight.write_bytes(b'physical weight changed inside stage')
        if self.mutate == 'old-manifest-at-end' and name == 'top3':
            path = self.runtime / 'runs' / (self.args.prefix + '-mae-pilot') / 'manifest.json'
            body = json.loads(path.read_bytes()); body['runtime_identity']['candidate_model']['sha256'] = 'b' * 64
            save(path, body)

    def report(self, run_dir, _output, *, config):
        if run_dir.name.endswith('mae-pilot'):
            if self.mutate == 'between-stages': self.weight.write_bytes(b'physical weight changed between stages')
            return {'validity': {'complete_evidence_valid': True}, 'repeatability': {'engines': {
                'teacher': {'unstable_position_count': 0, 'stable_position_count': 17},
                'sekirei': {'unstable_position_count': 0, 'stable_position_count': 17}}}}
        return {'validity': {'formal_run_valid': True}, 'headline': {'valid': True, 'mae_cp': 123.0},
                'per_game': {game['id']: {'mae_cp': 123.0} for game in config['games']}}

    def validate_artifacts(self, run_dir, _config, **_kwargs):
        return json.loads((run_dir / 'manifest.json').read_bytes()), None, None, [], None, None

    def formal_gate(self, runtime, config):
        if runtime != self.runtime:
            raise AssertionError('explicit runtime must reach the formal gate')
        evidence = config['formal']['pilot_evidence']
        if evidence['pilot_run_id'] != self.args.prefix + '-mae-pilot':
            raise AssertionError('candidate own MAE pilot is required')
        if evidence['pilot_fingerprint'] != 'mae-pilot-fingerprint':
            raise AssertionError('candidate actual pilot fingerprint is required')

    def run(self):
        with ExitStack() as stack:
            for owner, name, value in [(e.b, 'execute_run', self.execute), (e.top3, 'run', self.top3),
                (e.br, 'report_from_run', self.report), (e.br, 'export_public', lambda *_args: None),
                (e.b, 'validate_run_artifacts', self.validate_artifacts),
                (e.b, 'observed_max_reported_nodes', lambda *_args, **_kwargs: 1000000),
                (e.b, 'validate_formal_gate', self.formal_gate), (e.b, 'repo_identity', lambda: {'public_fixture': True})]:
                stack.enter_context(patch.object(owner, name, side_effect=value))
            with redirect_stdout(io.StringIO()): return e.evaluate(self.args)


class CandidateEvaluationTests(unittest.TestCase):
    def test_explicit_runtime_all_four_stages_use_same_model_and_own_pilots(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            state = fixture.run()
            self.assertEqual(state['status'], 'complete')
            names = ['mae-pilot', 'mae', 'top3-pilot', 'top3']
            self.assertEqual(fixture.calls, [(name, fixture.runtime) for name in names])
            self.assertIsNone(fixture.snapshots[0][1]['formal']['pilot_evidence'])
            self.assertEqual(fixture.snapshots[1][1]['formal']['pilot_evidence']['pilot_run_id'], 'public-candidate-mae-pilot')
            self.assertEqual(set(state['stage_model_identities']), set(names))
            for name, snapshot in fixture.snapshots:
                self.assertEqual(snapshot['candidate_model']['path'], str(fixture.weight))
                self.assertEqual(snapshot['engines']['sekirei']['options']['EvalFile'], str(fixture.weight))
                self.assertEqual(list(snapshot['engines']['sekirei']['options']),
                                 list(fixture.base['engines']['sekirei']['options']))
                for identity in state['stage_model_identities'][name].values():
                    self.assertEqual(identity, state['candidate_model'])

    def test_each_actual_stage_manifest_mismatch_stops_before_later_stages(self):
        stages = ['mae-pilot', 'mae', 'top3-pilot', 'top3']
        for name in stages:
            with self.subTest(stage=name), tempfile.TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory), mismatch=name)
                with self.assertRaisesRegex(ValueError, 'recorded .* candidate model'):
                    fixture.run()
                self.assertEqual([s for s, _ in fixture.calls], stages[:stages.index(name) + 1])
                state = json.loads((fixture.args.output / 'evaluation.json').read_bytes())
                self.assertEqual(state['status'], 'failed')

    def test_physical_change_within_each_stage_is_rejected_even_if_recorded_sha_matches(self):
        for name in ('mae-pilot', 'mae', 'top3-pilot', 'top3'):
            with self.subTest(stage=name), tempfile.TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory), mutate=name)
                with self.assertRaisesRegex(ValueError, 'physical candidate model'):
                    fixture.run()
                state = json.loads((fixture.args.output / 'evaluation.json').read_bytes())
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(state['stage'], name)

    def test_stage_before_pin_and_final_reinspection_detect_late_changes(self):
        for change, expected_calls in [('between-stages', 1), ('old-manifest-at-end', 4)]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory), mutate=change)
                with self.assertRaises(ValueError): fixture.run()
                self.assertEqual(len(fixture.calls), expected_calls)

    def test_wrong_explicit_pin_and_noncanonical_sha_rejected_before_outputs_or_engines(self):
        for digest in ('a' * 64, 'A' * 64, 'invalid', ''):
            with self.subTest(digest=digest), tempfile.TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory)); fixture.args.expected_weight_sha256 = digest
                with self.assertRaises((ValueError, e.b.ConfigurationError)): fixture.run()
                self.assertFalse(fixture.args.output.exists()); self.assertFalse(fixture.calls)

    def test_fixed_five_game_nodes_and_exact_policy_reject_changes_before_engines(self):
        for change in ('nodes', 'bool-nodes', 'policy', 'bool-policy', 'game', 'order', 'universe', 'hash', 'final'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory))
                if change == 'nodes': fixture.base['requested_nodes'] = 500000
                if change == 'bool-nodes': fixture.base['requested_nodes'] = True
                if change == 'policy': fixture.base['node_policy']['rate_numerator'] = 2
                if change == 'bool-policy': fixture.base['node_policy']['version'] = True
                if change == 'game': fixture.base['games'].pop()
                if change == 'order': fixture.base['games'].reverse()
                if change == 'universe': fixture.base['universe']['total_occurrences'] -= 1
                if change == 'hash': fixture.base['hashes']['universe_sha256'] = 'a' * 64
                if change == 'final': fixture.base['split'] = 'final'
                save(fixture.base_path, fixture.base)
                with self.assertRaises((ValueError, e.b.ConfigurationError)): fixture.run()
                self.assertFalse(fixture.args.output.exists()); self.assertFalse(fixture.calls)

    def test_run_identity_missing_wrong_type_duplicate_json_and_bool_size_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'candidate-mae'
            model = {'kind': 'nnue', 'path': '/public/fixture', 'bytes': 1, 'sha256': 'a' * 64}
            good = {'run_id': run.name, 'run_type': 'formal', 'runtime_identity': {'candidate_model': model}}
            for change in ('missing', 'run-type', 'bool-size', 'duplicate'):
                body = deepcopy(good)
                if change == 'missing': body['runtime_identity'] = {}
                if change == 'run-type': body['run_type'] = 'pilot'
                if change == 'bool-size': body['runtime_identity']['candidate_model']['bytes'] = True
                save(run / 'manifest.json', body)
                if change == 'duplicate':
                    (run / 'manifest.json').write_text('{"run_id":"one","run_id":"two"}')
                with self.subTest(change=change), self.assertRaises(ValueError):
                    e.verify_stage_model(run, 'mae', model)


if __name__ == '__main__':
    unittest.main()
