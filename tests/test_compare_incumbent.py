"""Public synthetic comparison contracts; no real positions, weights or engines."""
from contextlib import ExitStack
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compare_incumbent as ci


E_COUNTS = (42, 72, 66, 40, 46)
TOP_COUNTS = (84, 122, 127, 104, 114)


def fixture(error=100, hits=50, model_sha='a' * 64):
    """Construct the fixed-size semantic maps and equal-game integer metrics."""
    model = {'kind': 'nnue', 'path': '/tmp/public-' + model_sha[:1] + '.bin',
             'bytes': 16, 'sha256': model_sha}
    config = {'split': 'development', 'requested_nodes': 1000000,
              'universe': {'plies': dict(ci.GAMES)},
              'formal': {'pilot_evidence': {'pilot_run_id': 'synthetic-mae-pilot'}},
              'candidate_model': {'kind': 'nnue', 'path': model['path']}}
    occurrences, teachers, records, e_cp, eligible, per, top_per = [], {}, [], {}, [], {}, {}
    mae = Fraction(error)
    rate = Fraction(0)
    for i, (game, plies) in enumerate(ci.GAMES.items()):
        for ply in range(1, plies + 1):
            key = game + ':' + str(ply)
            p = {'game_id': game, 'ply': ply, 'occurrence_id': f'{game}:{ply:03d}', 'position': 'synthetic:' + key,
                 'side_to_move': 'b', 'position_type': 'normal', 'classification': {'type': 'normal'}}
            occurrences.append(p)
            is_e = ply <= E_COUNTS[i]
            semantics = {field: None for field in ci.cc.TEACHER_FIELDS}
            semantics.update(status='exact_cp' if is_e else 'bound_cp', score_cp_sente=ply,
                             bestmove='7g7f')
            teachers[key] = {'side_to_move': 'b', 'position_sha256': 'public:' + key, **semantics}
            records.append({'engine_id': 'teacher', 'game_id': game, 'ply': ply, 'status': semantics['status'],
                            'side_to_move': 'b', 'position_sha256': 'public:' + key, 'result': semantics})
            candidate = dict(semantics, status='exact_cp', score_cp_sente=ply + error)
            records.append({'engine_id': 'sekirei', 'game_id': game, 'ply': ply, 'status': 'exact_cp',
                            'side_to_move': 'b', 'position_sha256': 'public:' + key, 'result': candidate})
            if is_e:
                e_cp[key] = ply
            if ply <= TOP_COUNTS[i]:
                eligible.append(p)
        count = min(hits, TOP_COUNTS[i])
        top_per[game] = {'hits': count, 'denominator': TOP_COUNTS[i], 'observed': TOP_COUNTS[i],
                         'rate': count / TOP_COUNTS[i]}
        per[game] = {'mae': ci.cc.rational(mae), 'e_count': E_COUNTS[i], 'top3': top_per[game]}
        rate += Fraction(count, TOP_COUNTS[i]) / 5
    execution = {'runtime': {'binaries': {'sekirei': {'sha256': 'c' * 64},
                                         'yaneuraou': {'sha256': 'd' * 64}},
                              'build_manifest': {'sha256': 'e' * 64}},
                 'requested_nodes': 1000000, 'node_policy': dict(ci.b.NODE_POLICY),
                 'engines': {'sekirei': {'options': {'MultiPV': '1'}}}}
    comparable = {'execution_except_model': execution, 'occurrences': occurrences,
                  'teacher_results': teachers, 'teacher_E_cp': e_cp,
                  'top3_occurrences': [p['occurrence_id'] for p in eligible],
                  'top3_legal_moves_sha256': 'f' * 64,
                  'top3_denominators': dict(zip(ci.GAMES, TOP_COUNTS)),
                  'top3_teacher_bestmoves': {p['occurrence_id']: '7g7f' for p in eligible}}
    summary = {'model_identity': model, 'mae_report_valid': True, 'top3_report_valid': True,
               'teacher_count': 570, 'teacher_E_count': 266, 'top3_count': 551,
               'mae': ci.cc.rational(mae), 'top3': ci.cc.rational(rate), 'per_game': per}
    manifest_execution = copy.deepcopy(execution)
    manifest_execution.update(candidate_model=model,
                              resolved_options={'sekirei': {'EvalFile': model['path'], 'NnueOutput': 'absolute'}},
                              resolved_option_order={'sekirei': ['EvalFile', 'NnueOutput']})
    manifest_execution['runtime']['candidate_model'] = model
    manifest = {'fingerprint': '0' * 64, 'runtime_identity': {'candidate_model': model},
                'fingerprint_payload': {'execution_identity': manifest_execution}}
    report = {'validity': {'formal_run_valid': True}, 'headline': {'valid': True, 'mae_cp': float(mae)},
              'universe': {'total_occurrences': 570},
              'e_exact_coverage': {'teacher_e_count': 266, 'candidate_exact_count': 266},
              'per_game': {game: {'mae_cp': float(mae)} for game in ci.GAMES}}
    ranking = {'valid': True, 'run_type': 'formal', 'top3_rate': float(rate),
               'attempts': 551, 'eligible_positions': 551, 'per_game': top_per,
               'fingerprint': '1' * 64}
    teacher_reference = {(r['game_id'], r['ply']): r['result'] for r in records if r['engine_id'] == 'teacher'}
    top_identity = {'eligible_occurrences': comparable['top3_occurrences'], 'legal_moves_sha256': 'f' * 64}
    return {'model': model, 'config': config, 'verified': (comparable, summary, mae, rate),
            'manifest': manifest, 'plan': {'positions': occurrences}, 'records': records,
            'report': report, 'ranking': ranking,
            'context': (top_identity, eligible, teacher_reference, {}, [])}


class ComparisonTests(unittest.TestCase):
    def validated(self, data):
        return ci.validate_verified(data['verified'], data['model'])

    def test_two_nnue_models_improve_without_fallback(self):
        incumbent = fixture()
        candidate = fixture(error=99, hits=50, model_sha='b' * 64)
        result = ci.compare_verified(self.validated(incumbent), self.validated(candidate))
        self.assertTrue(result['adopt'])
        self.assertEqual(set(result['checks']), ci.CHECKS)
        self.assertEqual(result['incumbent']['model_identity']['sha256'], 'a' * 64)

    def test_canonical_occurrence_ids_bind_unpadded_teacher_keys(self):
        data = fixture()
        comparable = data['verified'][0]
        self.assertEqual(comparable['occurrences'][0]['occurrence_id'], 'development-01:001')
        self.assertIn('development-01:1', comparable['teacher_results'])
        self.assertNotIn('development-01:001', comparable['teacher_results'])
        self.validated(data)
        comparable['occurrences'][0]['occurrence_id'] = 'development-01:1'
        with self.assertRaisesRegex(ValueError, 'canonical'):
            self.validated(data)

    def test_equal_mae_is_not_improvement_and_top3_drop_is_hold(self):
        base = self.validated(fixture())
        self.assertFalse(ci.compare_verified(base, self.validated(fixture()))['adopt'])
        candidate = self.validated(fixture(error=99, hits=49))
        result = ci.compare_verified(base, candidate)
        self.assertTrue(result['mae_improved'])
        self.assertFalse(result['top3_preserved'])
        self.assertEqual(result['decision'], 'hold')

    def test_comparison_uses_fractions_not_rounded_float(self):
        base, candidate = fixture(), fixture(model_sha='b' * 64)
        # A tiny exact change remains significant even when both printed
        # headline floats round to the same display precision.
        candidate['verified'][1]['per_game']['development-01']['mae']['numerator'] = 4199
        candidate['verified'][1]['per_game']['development-01']['mae']['denominator'] = 42
        candidate['verified'] = (*candidate['verified'][:2], Fraction(100) - Fraction(1, 210),
                                  candidate['verified'][3])
        result = ci.compare_verified(self.validated(base), self.validated(candidate))
        self.assertTrue(result['adopt'])

    def test_each_identity_mismatch_rejects_comparison(self):
        base = fixture()
        for field in ci.CHECKS:
            candidate = fixture(error=99, model_sha='b' * 64)
            candidate['verified'][0][field] = {'changed': 'synthetic'}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'mismatch'):
                ci.compare_verified(base['verified'], candidate['verified'])

    def test_missing_cp_top3_and_teacher_set_are_rejected(self):
        for change in ('teacher_E', 'occurrence', 'top3', 'observed', 'model', 'integer'):
            data = fixture()
            c, s, _m, _t = data['verified']
            if change == 'teacher_E':
                c['teacher_E_cp'].pop(next(iter(c['teacher_E_cp'])))
            elif change == 'occurrence':
                c['teacher_results']['unexpected'] = c['teacher_results'].pop(next(iter(c['teacher_results'])))
            elif change == 'top3':
                c['top3_occurrences'].pop()
            elif change == 'observed':
                s['per_game']['development-01']['top3']['observed'] -= 1
            elif change == 'model':
                s['model_identity'] = dict(data['model'], sha256='b' * 64)
            elif change == 'integer':
                c['teacher_E_cp'][next(iter(c['teacher_E_cp']))] = 1.0
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validated(data)

    def test_explicit_pin_rejects_fallback_or_wrong_model_before_raw_reads(self):
        data = fixture()
        for identity in (dict(data['model'], sha256='b' * 64), {'kind': 'material_fallback', 'sha256': None}):
            with mock.patch.object(ci.b, '_model_identity', return_value=identity), \
                 mock.patch.object(ci.cc, 'verify_run') as reader, self.assertRaises(ValueError):
                ci.verify({}, data['config'], 'a' * 64)
            reader.assert_not_called()


class StrictReaderTests(unittest.TestCase):
    """Exercise the real shared metric reader with mocked parser boundaries.

    Raw parsing and process cleanup already have their own USI fixtures. These
    tests confirm this consumer invokes that reader and cannot rescue missing
    coverage, an unstable pilot, or a model activation mismatch.
    """
    def read(self, data, pilot_stable=True):
        desc = {'runtime': Path('/tmp/public-runtime'), 'mae': Path('/tmp/public-runtime/runs/mae'),
                'top3': Path('/tmp/public-runtime/runs/top3')}
        pilot = {'validity': {'complete_evidence_valid': True},
                 'repeatability': {'engines': {name: {'stable_position_count': 17 if pilot_stable else 16,
                                                     'unstable_position_count': 0 if pilot_stable else 1}
                                               for name in ('teacher', 'sekirei')}}}
        with ExitStack() as stack:
            patches = [mock.patch.object(ci.b, '_model_identity', return_value=data['model']),
                       mock.patch.object(ci.b, 'validate_runtime'), mock.patch.object(ci.b, 'validate_formal_gate'),
                       mock.patch.object(ci.b, 'validate_run_artifacts', return_value=(data['manifest'], data['plan'], None,
                                                                                     data['records'], None, None)),
                       mock.patch.object(ci.cc.br, 'report_from_run', side_effect=[pilot, data['report']]),
                       mock.patch.object(ci.cc.top3, 'context', return_value=data['context']),
                       mock.patch.object(ci.cc.top3, 'inspect', return_value=data['ranking']),
                       mock.patch.object(ci.cc, 'read_json', return_value=data['ranking'])]
            for patch in patches:
                stack.enter_context(patch)
            return ci.verify(desc, data['config'], data['model']['sha256'])

    def test_real_shared_reader_is_called_for_both_metrics(self):
        result = self.read(fixture())
        self.assertEqual(result[2], Fraction(100))
        self.assertEqual(result[1]['teacher_E_count'], 266)

    def test_missing_cp_cannot_reduce_mae_denominator(self):
        data = fixture()
        next(r for r in data['records'] if r['engine_id'] == 'sekirei')['result']['score_cp_sente'] = None
        with self.assertRaisesRegex(ValueError, 'integer'):
            self.read(data)

    def test_wrong_teacher_cp_set_is_rejected(self):
        data = fixture()
        next(r for r in data['records'] if r['engine_id'] == 'teacher')['status'] = 'bound_cp'
        with self.assertRaisesRegex(ValueError, '266'):
            self.read(data)

    def test_missing_top3_and_unstable_pilot_are_rejected(self):
        data = fixture()
        data['ranking']['per_game']['development-01']['observed'] -= 1
        with self.assertRaisesRegex(ValueError, 'denominator'):
            self.read(data)
        with self.assertRaisesRegex(ValueError, 'unstable'):
            self.read(fixture(), pilot_stable=False)

    def test_saved_model_hash_cannot_be_replaced(self):
        data = fixture()
        data['manifest']['runtime_identity']['candidate_model'] = dict(data['model'], sha256='b' * 64)
        with self.assertRaisesRegex(ValueError, 'model'):
            self.read(data)


class OutputTests(unittest.TestCase):
    def arguments(self, root, output):
        args = []
        for role in ('incumbent', 'candidate'):
            args += ['--' + role + '-config', str(root / role / 'candidate.json'),
                     '--' + role + '-mae-run', str(root / 'runtime/runs' / (role + '-mae')),
                     '--' + role + '-top3-run', str(root / 'runtime/runs' / (role + '-top3')),
                     '--' + role + '-model-sha256', 'a' * 64]
        return args + ['--output-json', str(output)]

    def test_invalid_missing_inputs_write_only_new_private_invalid_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'results/comparison.json'
            with mock.patch('sys.stdout'):
                code = ci.main(self.arguments(root, output))
            result = json.loads(output.read_text())
            self.assertEqual(code, 2)
            self.assertFalse(result['adopt'])
            self.assertFalse(result['comparison_valid'])
            self.assertEqual(result['decision'], 'invalid')
            with self.assertRaises(ValueError):
                ci.main(self.arguments(root, output))

    def test_unsafe_output_never_receives_even_invalid_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for output in (root / 'runtime/comparison.json', root / 'incumbent/comparison.json'):
                with self.subTest(output=output), self.assertRaises(ci.OutputError):
                    ci.main(self.arguments(root, output))
                self.assertFalse(output.exists())

    def test_final_run_is_rejected_before_manifest_read(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'runtime/runs/final-mae'
            run.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'final'):
                ci.cc.run_path(run)

    def test_actual_model_pin_hashes_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'public-weight.bin'
            path.write_bytes(b'public fixture')
            config = fixture()['config']
            config.update(candidate_model={'kind': 'nnue', 'path': str(path)},
                          engines={'sekirei': {'options': {}}})
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(ci.model_pin(config, expected)['sha256'], expected)
            path.write_bytes(b'changed fixture')
            with self.assertRaisesRegex(ValueError, 'pin'):
                ci.model_pin(config, expected)


if __name__ == '__main__':
    unittest.main()
