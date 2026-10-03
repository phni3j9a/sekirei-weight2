"""Pure public synthetic helper fixtures; no private input, process or compiler."""
import copy
import importlib.util
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'scripts/prepare_bounded.py').is_file():
    ROOT = Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
PROTOTYPE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(PROTOTYPE))
import prepare_fanin509 as build
import train_fanin509 as train
import diagnose_fanin509 as diagnosis
import prepare_bounded as old_build
import train_bounded as old_train
import diagnose_bounded as old_diagnosis
from export_nearest import SOURCE_HASHES, fnv1a
from train_cpu import float32


def mode_fixture():
    return {
        'mode': 'bounded-material-fanin509-v1', 'mask_version': 1, 'fixed_ft': 'all',
        'fixed_l2_columns': [0, 1, 2, 3], 'fixed_l2_rows': [0, 1, 256, 257],
        'fixed_bias_out': [0, 1, 2, 3], 'fixed_out_bias': '+0', 'auxiliary_units': 28,
        'residual_budget_cp': 99, 'shrink': 'uniform-aux-out-f32-margin-2^-20-max8',
        'shrink_moments': False, 'optimizer': 'fresh-adam', 'resume_supported': False,
        'init_seed': 42, 'initial_weights_fnv': '5ae69099a8bbcdad',
        'loaded_initial_binding': 'core-save-weights-exact-bytes', 'label_depth': 0,
        'target': 'original-absolute-external-cp', 'teacher_identity': train.TEACHER,
        'integer_core_bound_proven': False, 'l2_effective_lr_denominator': 509,
        'out_effective_lr_denominator': 1, 'saved_f32_l1_upper': 0.0,
        'saved_f32_budget_cp_upper': 0.0,
    }


def prereg_fixture():
    training = {'epochs': 3, 'selected_epoch': 3, 'learning_rate': 0.0001,
                'min_lr': 0.0, 'lr_schedule': 'step-half', 'lr_schedule_epochs': 3,
                'seed': 42, 'shuffle_seed': 42, 'timeout_seconds': 1200,
                'float_subnormal_policy': 'x86-ftz-daz', 'label_depth': 0,
                'teacher_score_cap': 30000, 'positions': {'train': 112681, 'holdout': 5895},
                'bounded_mode': 'bounded-material-fanin509-v1', 'real_residual_cap_cp': 99,
                'l2_effective_lr_denominator': 509, 'out_effective_lr_denominator': 1,
                'l2_effective_lr_rule': 'f32(epoch_lr)/f32(509)'}
    return {'schema': 'sekirei.bounded-material-fanin509-preregistration.v1',
            'status': 'frozen-before-training', 'candidate': 'bounded-material-fanin509-100cp-e3-v1',
            'plan_sha256': '359cf9daa1a5e57be6c7a2d56b2ecfef5c9e24cfc7648e45f7930544b264ae6e',
            'dataset_manifest_sha256': train.DATASET_SHA256, 'teacher_identity': train.TEACHER,
            'initial_weights': {'sha256': train.INITIAL_SHA256},
            'build_manifest_sha256': 'a' * 64, 'training': training,
            'trigger': {'path': str(build.activation.ACTIVATION), 'sha256': 'b' * 64},
            'source_helpers': {name: 'c' * 64 for name in train.HELPERS}}


def epoch_fixture(epoch=3):
    return {'epoch': epoch, 'epochs': 3, 'train_count': 112681, 'valid_count': 0,
            'cache_hits': 112681, 'cache_misses': 0, 'init_seed': 42, 'split_seed': 42,
            'shuffle_seed': 42, 'lr_schedule_epochs': 3, 'warmup_epochs': 0,
            'label_depth': 0, 'source_cap': 0, 'split_hash': 0, 'nnue_output': 'absolute',
            'teacher_eval': 'external', 'teacher_identity': train.TEACHER,
            'lr_schedule': 'StepHalf', 'float_subnormal_policy': 'x86-ftz-daz',
            'architecture': 'INPUT=2420 L1=256 L2=32', 'cache_only': True,
            'exclude_mate_labels': False, 'side_balance': False, 'games_dir': None,
            'teacher_weights': None, 'wdl_lambda': None, 'phase_weights': {},
            'validation_ratio': 0.0, 'teacher_score_cap': 30000.0,
            'search_target_weight': 1.0, 'lr': float32(0.0001), 'min_lr': float32(0.0),
            'positions': '/tmp/public-fanin509-run/positions.jsonl',
            'bounded_material_fanin509_v1': mode_fixture()}


class FanIn509Helpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the existing PUBLIC synthetic generator, not checkpoint files.
        spec = importlib.util.spec_from_file_location('synthetic_old_fixture', ROOT / 'tests/test_diagnose_bounded.py')
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        fixture.ActualAdamTests.setUpClass()
        cls.native, cls.adam = fixture.ActualAdamTests.native, fixture.ActualAdamTests.adam
        cls.edit = staticmethod(fixture.edit)
        cls.native_with_parameters = staticmethod(fixture.native_with_parameters)
        sys.path.insert(0, str(PROTOTYPE))

    def test_01_new_recipe_accepts_exact_fields_and_old_mode_guards_reject_it(self):
        prereg = prereg_fixture()
        self.assertEqual(train.validate_preregistration(prereg), diagnosis.TRAINING)
        self.assertIs(diagnosis.validate_preregistration(prereg, build.PLAN_SHA256), prereg)
        with self.assertRaises(ValueError):
            old_train.validate_preregistration(prereg)
        with self.assertRaises(ValueError):
            old_diagnosis.validate_preregistration(prereg, build.PLAN_SHA256)
        self.assertEqual(old_build.BOUNDED_PATCH_SHA, 'ee819bd80c41301ee78a0d3ce29efd976c3d762c1ddf07bf6cc259703d38933b')
        self.assertEqual(SOURCE_HASHES['crates/sekirei-train/src/trainer.rs'],
                         'c86b294c35383ef25c65f209735eb1484b58dc15e43777c3f7134b5985724f53')

    def test_02_recipe_scale_types_rules_counts_and_trigger_reference_fail_closed(self):
        for name, value in [('l2_effective_lr_denominator', 508),
                            ('l2_effective_lr_denominator', 509.0),
                            ('out_effective_lr_denominator', True),
                            ('l2_effective_lr_rule', 'gradient/509')]:
            prereg = prereg_fixture(); prereg['training'][name] = value
            with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                train.validate_preregistration(prereg)
            with self.assertRaises(ValueError):
                diagnosis.validate_preregistration(prereg, build.PLAN_SHA256)
        prereg = prereg_fixture(); prereg['training']['positions']['train'] = 112681.0
        with self.assertRaises(ValueError): train.validate_preregistration(prereg)
        for value in [None, {'path': '../trigger.json', 'sha256': 'a'*64},
                      {'path': '/tmp/trigger.json', 'sha256': True}]:
            with self.subTest(trigger=value), self.assertRaises(ValueError):
                train.validate_trigger_reference(value)

    def test_03_new_metadata_exact_denominators_cap_and_old_mode_collision(self):
        mode = mode_fixture(); diagnosis.validate_fanin509_metadata(mode)
        with self.assertRaises(ValueError): old_diagnosis.validate_bounded_metadata(mode)
        for name, value in [('mode', 'bounded-material-v1'), ('mask_version', 1.0),
                            ('l2_effective_lr_denominator', 509.0), ('out_effective_lr_denominator', True),
                            ('residual_budget_cp', True), ('saved_f32_budget_cp_upper', 99.0001),
                            ('saved_f32_budget_cp_upper', float('nan'))]:
            bad = dict(mode, **{name: value})
            with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                diagnosis.validate_fanin509_metadata(bad)

    def test_04_epoch_metadata_all_three_and_wrong_epoch_original_cache_or_mode(self):
        path = Path('/tmp/public-fanin509-run/positions.jsonl')
        for epoch in (1, 2, 3): diagnosis.validate_epoch_metadata(epoch_fixture(epoch), epoch, positions_path=path)
        for name, value in [('epoch', 2), ('cache_misses', 1), ('teacher_identity', 'external:wrong'),
                            ('positions', '/tmp/wrong-order/positions.jsonl')]:
            bad = epoch_fixture(); bad[name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                diagnosis.validate_epoch_metadata(bad, 3, positions_path=path)
        bad = epoch_fixture(); bad['bounded_material_v1'] = mode_fixture()
        with self.assertRaises(ValueError): diagnosis.validate_epoch_metadata(bad, 3, positions_path=path)
        with self.assertRaises(ValueError): diagnosis.validate_epoch_metadata(epoch_fixture(), True, positions_path=path)

    def test_05_final_native_FNV_new_mode_and_E3_match(self):
        epoch = epoch_fixture()
        sidecar = {'format': 'sekirei-nnue-output-v1', 'nnue_output': 'absolute', 'baseline': None,
                   'checkpoint_hash': fnv1a(self.native), 'bounded_material_fanin509_v1': mode_fixture()}
        train.validate_final_sidecar(self.native, sidecar, self.native, epoch)
        bad = dict(sidecar, bounded_material_v1=mode_fixture())
        with self.assertRaises(ValueError): diagnosis.validate_final_sidecar(self.native, bad, epoch)
        with self.assertRaises(ValueError): train.validate_final_sidecar(self.native, sidecar, self.native[:-1], epoch)
        bad = dict(sidecar, checkpoint_hash='0'*16)
        with self.assertRaises(ValueError): diagnosis.validate_final_sidecar(self.native, bad, epoch)

    def test_06_raw_Adam_shared_invariant_nested_without_false_LR_claim(self):
        proof = diagnosis.validate_fanin509_adam(self.adam, self.native, 338043)
        shared = proof['shared_parameter_invariant_proof']
        self.assertTrue(shared['actual_raw_ft_binary32_fixed'])
        self.assertTrue(shared['protected_moments_positive_zero'])
        self.assertTrue(shared['raw_native_reexport_all_bytes_equal'])
        self.assertTrue(shared['raw_nearest_export_all_bytes_equal'])
        self.assertEqual(shared['actual_step'], 338043)
        self.assertFalse(proof['learning_rate_scaling_proven_by_checkpoint_alone'])
        self.assertFalse(proof['adoption_claimed'])
        with self.assertRaises(ValueError): diagnosis.validate_fanin509_adam(self.adam, self.native, 112681)
        bad = self.edit(self.adam, {'out_m': {0: -0.0}})
        with self.assertRaises(ValueError): diagnosis.validate_fanin509_adam(bad, self.native, 338043)
        bits = struct.unpack('<I', struct.pack('<f', self.adam['ft'][0]))[0]
        drift = struct.unpack('<f', struct.pack('<I', bits + 1))[0]
        bad = self.edit(self.adam, {'ft': {0: drift}})
        with self.assertRaisesRegex(ValueError, 'actual raw FT/bias binary32 differs'):
            diagnosis.validate_fanin509_adam(bad, self.native, 338043)
        bad = self.edit(self.adam, {'out': {4: 32.0, 5: -32.0}})
        native = self.native_with_parameters(self.native, bad)
        with self.assertRaisesRegex(ValueError, 'auxiliary real residual budget exceeds 99 cp'):
            diagnosis.validate_fanin509_adam(bad, native, 338043)

    def test_07_only_new_mode_flag_same_fresh_original_cache_and_no_resume(self):
        root, data, initial = Path('/tmp/public-run'), Path('/tmp/public-original-O'), Path('/tmp/public-init.bin')
        argv = diagnosis.training_argv('/tmp/public-train', root, data, initial)
        self.assertEqual(argv, train.training_argv('/tmp/public-train', root, data, {'path': str(initial)}))
        self.assertEqual(argv[:-1], old_train.training_argv('/tmp/public-train', root, data,
                                                          {'path': str(initial)})[:-1])
        self.assertEqual(argv[-1], '--bounded-material-fanin509-v1')
        self.assertNotIn('--bounded-material-v1', argv)
        self.assertNotIn('--resume', argv)
        for flag, value in [('--label-depth', '0'), ('--teacher-score-cap', '30000'),
                            ('--epochs', '3'), ('--nnue-output', 'absolute')]:
            self.assertEqual(argv[argv.index(flag)+1], value)
        self.assertIn('--cache-only', argv); self.assertIn('--reuse-teacher-cache', argv)

    def test_08_pinned_new_build_identity_and_exact_sixteen_test_inventory(self):
        value = {'schema': 'sekirei.fanin509-trainer-build-identity.v1', 'upstream_commit': build.UPSTREAM,
                 'toolchain_lock_sha256': build.LOCK_SHA, 'external_patch_path': '/tmp/public-external.patch',
                 'external_patch_sha256': build.EXTERNAL_PATCH_SHA,
                 'fanin509_patch_path': '/tmp/public-fanin.patch', 'fanin509_patch_sha256': build.FANIN509_PATCH_SHA,
                 'expected_source_main_sha256': build.FANIN509_MAIN_SHA,
                 'expected_source_trainer_sha256': build.FANIN509_TRAINER_SHA,
                 'rustflags': build.RUSTFLAGS, 'rustc': build.RUSTC, 'cargo': 'cargo 1.96.0 public-fixture',
                 'rust_tests': sorted(build.REQUIRED_TESTS), 'recipe': copy.deepcopy(build.RECIPE),
                 'plan_sha256': build.PLAN_SHA256, 'trigger': prereg_fixture()['trigger'],
                 'activation_helper_sha256': 'd' * 64}
        self.assertIs(build.validate_identity_values(value), value)
        self.assertEqual(len(build.REQUIRED_TESTS), 16)
        for key, bad_value in [('fanin509_patch_sha256', old_build.BOUNDED_PATCH_SHA),
                               ('expected_source_trainer_sha256', old_build.BOUNDED_TRAINER_SHA),
                               ('rust_tests', sorted(old_build.REQUIRED_TESTS))]:
            bad = dict(value, **{key: bad_value})
            with self.subTest(field=key), self.assertRaises(ValueError): build.validate_identity_values(bad)
        bad = copy.deepcopy(value); bad['recipe']['out_effective_lr_denominator'] = True
        with self.assertRaises(ValueError): build.validate_identity_values(bad)

    def test_09_all_runtime_entries_disabled_before_any_private_IO(self):
        # Exercise the disabled branch even after root activates the public
        # builder, without weakening or invoking any real runtime guard.
        with patch.object(build, 'PROTOTYPE_ONLY', True):
            for function, args in [(build.prepare, (None, None, None)), (build.verify_build, (None, None)),
                                   (train.train, (None,)), (diagnosis.diagnose, (None,)),
                                   (diagnosis._diagnose, (None, None, None))]:
                with self.subTest(entry=function.__name__), self.assertRaisesRegex(ValueError, 'prepared prototype only'):
                    function(*args)
        for name in ('prepare_fanin509.py', 'train_fanin509.py', 'diagnose_fanin509.py'):
            self.assertEqual(build.verify_public_entry(name, build.REPO / 'scripts' / name),
                             build.REPO / 'scripts' / name)
            with self.assertRaisesRegex(ValueError, 'alternate entry copy rejected'):
                build.verify_public_entry(name, PROTOTYPE / name)
        with self.assertRaises(ValueError):
            train.verify_helpers(prereg_fixture())
        # Isolate the NEW self-source SHA branch without changing old guard
        # constants or reading any input file. A wrong real entry hash must
        # stop before the rest of the helper inventory is inspected.
        with patch.object(build, 'verify_public_entry', return_value=PROTOTYPE / 'train_fanin509.py'), \
             patch.object(train, 'sha256', return_value='f' * 64), \
             self.assertRaisesRegex(ValueError, 'actual training entry source changed'):
            train.verify_helpers(prereg_fixture())


if __name__ == '__main__':
    unittest.main(verbosity=2)
