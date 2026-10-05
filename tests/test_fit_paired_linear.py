"""Public synthetic bytes only. No real dataset/model/fit/probe/engine."""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fit_paired_linear as fitter
import functional_anchor as anchor
import material_init as material
import paired_linear as paired


def encoded(rows):
    return ''.join(json.dumps(row, sort_keys=True) + ' \r\n' for row in rows).encode()


def fixture():
    seed = 'public-paired-driver-v1'
    gid = next(hashlib.sha256(('fixture-' + str(i)).encode()).hexdigest() for i in range(1000)
               if anchor._split_for_game(hashlib.sha256(('fixture-' + str(i)).encode()).hexdigest(), seed) == 'train')
    sfens = ['4k4/9/9/9/9/9/9/9/4K4 b P 16', '4k4/9/9/9/9/9/9/9/4K4 w P 16']
    positions = [{'schema_version': 1, 'sfen': sfen,
                  'source': {'kind': 'gensfen-pack', 'path': gid, 'ply': 16}} for sfen in sfens]
    teacher = 'external:suisho11beta-1m-pack:' + 'a'*64
    labels = [{'sfen': sfen, 'score_cp': cp, 'label_depth': 0, 'teacher_identity': teacher,
               'source_metadata': {'fixture': '合成'}} for sfen, cp in zip(sfens, (110, -130))]
    manifest = {'schema_version': 1, 'source_corpus_manifest_sha256': 'a'*64, 'teacher_identity': teacher,
        'seed': seed, 'split': anchor.SPLIT_RECIPE, 'sampling': anchor.SAMPLING,
        'dependencies': {'fixture': True}, 'games_per_pack': 1,
        'games': [{'game_id': gid, 'pack_sha256': 'b'*64, 'game_index': 0, 'split': 'train'}],
        'independent_exclusions': {'games': 1000, 'policy': anchor.EXCLUSION_POLICY,
            'unique_positions': 1, 'corpus_canonical_sha256': 'c'*64, 'source_manifest_sha256': 'd'*64},
        'derivation': {'kind': 'expanded-train-frozen-holdout-v1', 'input_sha256': {'fixture': 'e'*64}},
        'positions': {'train': 2, 'holdout': 1}}
    return manifest, positions, list(reversed(labels))


def raw_fixture(manifest, positions, labels):
    p, l = encoded(positions), encoded(labels)
    manifest = copy.deepcopy(manifest)
    manifest['files'] = {name: {'bytes': len(data), 'sha256': fitter.sha(data), 'count': count}
        for name, data, count in (('train.positions.jsonl', p, len(positions)),
                                 ('train.labels.jsonl', l, len(labels)),
                                 ('holdout.positions.jsonl', b'not read by TRAIN extractor\n', 1),
                                 ('holdout.labels.jsonl', b'not read by TRAIN extractor either\n', 1))}
    m = json.dumps(manifest, sort_keys=True).encode()
    return m, p, l, fitter.sha(m)


@unittest.skipUnless(importlib.util.find_spec('numpy') is not None,
                     'fixed NumPy runtime unavailable; synthetic extraction tests skipped')
class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = material.encode(material.build_weights(42))
        assert fitter.sha(cls.initial) == fitter.INITIALIZER_SHA256

    def build(self, fixture_data=None, initializer=None):
        m, p, l, h = raw_fixture(*(fixture_data or fixture()))
        return fitter.build_train_design(m, p, l, self.initial if initializer is None else initializer,
                                        expected_manifest_sha256=h)

    def test_position_order_sfen_join_stm_and_exact_scalar_features(self):
        design = self.build()
        self.assertEqual((design.count, design.dimension), (2, 254))
        self.assertEqual(list(struct.unpack('<2i', design.d_bytes)), [10, -30])
        weights = material.decode(self.initial)
        for i, row in enumerate(fixture()[1]):
            pos = material.parse_sfen(row['sfen'])
            u, t = (material.active_features(pos, side) for side in (pos['stm'], 1-pos['stm']))
            expected = [sum(weights['ft'][f*256+j] for f in u) -
                        sum(weights['ft'][f*256+j] for f in t) for j in range(2, 256)]
            expected = [x//2 for x in expected]
            actual = [x[0] for x in struct.iter_unpack('<b', design.z_bytes[i*254:(i+1)*254])]
            self.assertEqual(actual, expected)
        self.assertEqual([x[0] for x in struct.iter_unpack('<b', design.z_bytes[:254])],
                         [-x[0] for x in struct.iter_unpack('<b', design.z_bytes[254:])])
        self.assertFalse(design.provenance['holdout_rows_read'])
        self.assertTrue(design.provenance['labels_extra_metadata_bound_in_original_sha'])

    def test_cache_order_does_not_change_design(self):
        data = fixture()
        first = self.build(data)
        data[2].reverse()
        second = self.build(data)
        self.assertEqual((first.z_bytes, first.d_bytes), (second.z_bytes, second.d_bytes))
        self.assertNotEqual(first.provenance['train_labels_sha256'], second.provenance['train_labels_sha256'])

    def test_teacher_type_depth_identity_range_rejected(self):
        for field, value in [('score_cp', True), ('score_cp', 110.0), ('score_cp', 30000),
                             ('score_cp', float('nan')), ('label_depth', False),
                             ('label_depth', 1), ('teacher_identity', 'wrong')]:
            data = fixture(); data[2][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.build(data)

    def test_duplicate_missing_extra_label_rejected(self):
        for change in ('duplicate', 'missing', 'extra'):
            data = fixture()
            if change == 'duplicate':
                data[2][0] = copy.deepcopy(data[2][1])
            elif change == 'missing':
                data[2].pop()
            else:
                data[2][0]['sfen'] = '4k4/9/9/9/9/9/9/9/4K4 b 2P 16'
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(data)

    def test_duplicate_semantic_sfen_ignores_ply(self):
        data = fixture()
        data[1][1]['sfen'] = data[1][0]['sfen'].replace(' 16', ' 17')
        data[2][0]['sfen'] = data[1][1]['sfen']
        with self.assertRaises(ValueError):
            self.build(data)

    def test_nontrain_source_inventory_and_king_rejected(self):
        for change in ('source', 'schema_bool', 'inventory', 'king'):
            data = fixture()
            if change == 'source': data[1][0]['source']['path'] = 'f'*64
            if change == 'schema_bool': data[1][0]['schema_version'] = True
            if change == 'inventory': data[1][0]['sfen'] = data[1][0]['sfen'].replace(' b P ', ' b 19P ')
            if change == 'king': data[1][0]['sfen'] = data[1][0]['sfen'].replace('4K4', '9')
            data[2][1]['sfen'] = data[1][0]['sfen']
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(data)

    def test_manifest_train_hash_and_fixed_initializer_mutation_rejected(self):
        m, p, l, h = raw_fixture(*fixture())
        with self.assertRaises(ValueError):
            fitter.build_train_design(m, p+b' ', l, self.initial, expected_manifest_sha256=h)
        with self.assertRaises(ValueError):
            fitter.build_train_design(m, p, l, self.initial, expected_manifest_sha256='0'*64)
        mutated = bytearray(self.initial); mutated[100] ^= 1
        with self.assertRaises(ValueError): self.build(initializer=bytes(mutated))

    def test_deadline_callback_applies_before_preparation(self):
        def stopped(): raise TimeoutError('fixture deadline')
        m, p, l, h = raw_fixture(*fixture())
        with self.assertRaises(TimeoutError):
            fitter.build_train_design(m, p, l, self.initial, expected_manifest_sha256=h,
                                      check_deadline=stopped)

    def test_exact_dyadic_serialization_and_nonfinite_rejection(self):
        result = anchor._json(fitter.json_bytes({'float': .1, 'fraction': Fraction(-3, 8),
                                                'bool': True, 'integer': 1, 'tuple': (0.0,)}))
        q = Fraction(.1)
        self.assertEqual(result['float'], {'numerator': q.numerator, 'denominator': q.denominator})
        self.assertEqual(result['fraction'], {'numerator': -3, 'denominator': 8})
        self.assertIs(result['bool'], True)
        self.assertEqual(result['tuple'], [{'numerator': 0, 'denominator': 1}])
        for value in (float('nan'), float('inf'), object()):
            with self.assertRaises(ValueError): fitter.json_bytes({'bad': value})

    def test_canonical_f32_zero_and_nearest_cast(self):
        result = {'coefficient_f64': tuple([0.0]*254), 'coefficient_f32': tuple([-0.0]*254)}
        f64, f32 = fitter.coefficient_bytes(result)
        self.assertEqual(f32, b'\0'*1016)
        self.assertEqual(len(f64), 2032)
        result['coefficient_f64'] = tuple([-1e-50]+[0.0]*253)
        tiny64, tiny32 = fitter.coefficient_bytes(result)
        self.assertEqual(tiny32, b'\0'*1016)
        self.assertEqual(struct.unpack('<d', tiny64[:8])[0], -1e-50)
        result['coefficient_f64'] = tuple([1.0]+[0.0]*253)
        with self.assertRaises(ValueError): fitter.coefficient_bytes(result)

    def test_pure_template_requires_real_parent_lifecycle(self):
        source = {'paired_linear.py': 'a'*64, 'paired_linear_activation.py': 'b'*64}
        context = {'preregistration_sha256': 'c'*64, 'activation_sha256': 'd'*64,
            'source_helpers_sha256': source, 'source_preflight_sha256': 'e'*64,
            'inputs_before': {'fixture': {'bytes': 1, 'sha256': 'f'*64}},
            'inputs_after': {'fixture': {'bytes': 1, 'sha256': 'f'*64}},
            'runtime_sources': {}, 'numeric_execution': {'fixture': True}}
        outputs = {key: {'path': str(paired.Q/name), 'bytes': 1, 'sha256': 'a'*64} for key, name in
            [('weights','weights.bin'), ('coefficients_f32','coefficients.f32.bin'),
             ('coefficients_f64','coefficients.f64.bin'), ('gram_z','gram.z.i64'), ('rhs_z','rhs.z.i64')]}
        cert = {'path': str(paired.Q/'solver-certificate.json'), 'bytes': 1, 'sha256': 'b'*64}
        self.assertFalse(fitter.make_run_template(context, outputs, cert)['cleanup_verified'])
        lifecycle = {'returncode': 0, 'timed_out': False, 'waited': True, 'reaped': True,
                     'process_group_stopped': True}
        self.assertTrue(fitter.construct_fit_run(context, outputs, cert, lifecycle)['cleanup_verified'])
        for key in lifecycle:
            mutated = dict(lifecycle); mutated[key] = 1 if type(mutated[key]) is bool else False
            with self.subTest(key=key), self.assertRaises(ValueError):
                fitter.construct_fit_run(context, outputs, cert, mutated)
        outputs['self'] = {'path': str(paired.Q/'run.json'), 'bytes': 1, 'sha256': 'c'*64}
        with self.assertRaises(ValueError): fitter.make_run_template(context, outputs, cert)
        outputs['self']['path'] = str(paired.Q/'fit-result.json')
        with self.assertRaises(ValueError): fitter.make_run_template(context, outputs, cert)

    def test_runtime_main_is_blocked_before_input_access(self):
        result = subprocess.run([sys.executable, '-B', str(Path(fitter.__file__))],
                                capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('required', result.stderr)

    def test_actual_openblas_getter_setter_tiny_no_blas_fit(self):
        np = fitter.numpy()
        paths = list((Path(np.__file__).resolve().parent.parent/'numpy.libs').glob('libopenblas64_p-*.so'))
        self.assertEqual(len(paths), 1)
        path = paths[0].resolve()
        pool = fitter.OpenBLASOneThread({str(path): fitter.file_identity(path)})
        proof = pool.verify()
        self.assertTrue(proof['threadpool_verified'])
        self.assertEqual(proof['threads_during'], 1)
        self.assertEqual(proof['threads_after'], 1)
        self.assertTrue(proof['rtld_noload'])


if __name__ == '__main__': unittest.main()
