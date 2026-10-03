#!/usr/bin/env python3
"""Pinned FanIn509 trainer builder with verified conditional activation.

The complete new patch applies after the unchanged external-label patch.
Original bounded guards are imported only for mode-independent filesystem,
compiler, source inventory, command supervision and test-log parsing. None
of their constants or validation functions is reassigned.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import shutil
import sys

from prepare import REPO
from prepare_bounded import (PRIVATE, COMPARISON, OLD_TRAINER, UPSTREAM, LOCK_SHA,
    EXTERNAL_PATCH_SHA, EXTERNAL_MAIN_SHA, TRAIN_CPU_SHA, BENCHMARK_SHA,
    RUSTFLAGS, RUSTC, TRACKED_COUNT, MAIN, TRAINER, CHANGED,
    require, digest, info, canonical, read_json, sha, save_new,
    git, paths_from_nul, source_identity, outside_git,
    compiler_versions, compiler_files, run_step, parse_tests)
from train_bounded import exact_value
import fanin509_activation as activation

PROTOTYPE_ONLY = False
PLAN_SHA256 = "359cf9daa1a5e57be6c7a2d56b2ecfef5c9e24cfc7648e45f7930544b264ae6e"
FANIN509_PATCH_SHA = "d5da76c8f316a6df4db3b23293feb47e9568bc75a7707a63fdce3c3b183d70c7"
FANIN509_MAIN_SHA = "21834567c71a2b6604422980a21f5b5577cc9af9ca839644e8e3abdc116955e2"
FANIN509_TRAINER_SHA = "0da66b6880785cb057f996c143bbab6347c23fbfc89f18196551b6a979db064b"
MODE = "bounded-material-fanin509-v1"

SHARED_HELPERS = {'prepare_bounded.py': '863ab0c22a65120737df93d91cbd55b12bd19d21e4a7b2d7e560a7d9a1703731', 'train_bounded.py': 'c8a9566bf25b86fd093718fd3cfb9355422f286551c6ecf64f0c7aadb513cc29'}


def verify_public_entry(name, actual_path):
    require(name in {"prepare_fanin509.py", "train_fanin509.py", "diagnose_fanin509.py"},
            "unexpected public helper name")
    expected = REPO / "scripts" / name
    require(Path(actual_path).is_absolute() and Path(actual_path) == expected
            and Path(actual_path).resolve() == expected,
            "execute the canonical preregistered public helper; alternate entry copy rejected")
    return expected


def verify_shared_helpers(expected_activation_sha256):
    verify_public_entry("prepare_fanin509.py", Path(__file__))
    import prepare_bounded
    import train_bounded
    for module in (prepare_bounded, train_bounded):
        path = REPO / "scripts" / (module.__name__ + ".py")
        require(Path(module.__file__).resolve() == path, "shared helper import path changed")
        require(info(path)["sha256"] == SHARED_HELPERS[path.name], "shared source helper changed")
    shared = {str(REPO / "scripts" / name): info(REPO / "scripts" / name) for name in SHARED_HELPERS}
    shared[str(REPO / "scripts/fanin509_activation.py")] = activation.verify_public_module(expected_activation_sha256)
    return shared


def require_runtime_ready():
    require(not PROTOTYPE_ONLY,
            "prepared prototype only: public source freeze/root activation review pending; build/train/diagnosis disabled")


RECIPE = {'mode': MODE, 'mask_version': 1, 'seed': 42,
          'shuffle_seed': 42, 'epochs': 3, 'selected_epoch': 3,
          'learning_rate': 0.0001, 'lr_schedule': 'step-half',
          'lr_schedule_epochs': 3, 'min_lr': 0.0, 'label_depth': 0,
          'residual_budget_cp': 99, 'resume_supported': False,
          'target': 'original-absolute-external-cp', 'nnue_output': 'absolute',
          'integer_core_bound_proven': False,
          'l2_effective_lr_denominator': 509, 'out_effective_lr_denominator': 1,
          'l2_effective_lr_rule': 'f32(epoch_lr)/f32(509)'}
REQUIRED_TESTS = {
    'tests::bounded_cli_accepts_only_fresh_original_absolute_recipe',
    'tests::bounded_cli_rejects_all_alternate_update_and_resume_flags',
    'tests::bounded_recipe_fingerprint_separates_normal_mode',
    'tests::bounded_loaded_initial_bytes_are_bound_to_actual_core_serialization',
    'tests::bounded_checkpoint_metadata_reads_saved_native_and_rejects_malformed_paths',
    'trainer::tests::bounded_adam_skips_protected_params_and_nonzero_moments',
    'trainer::tests::bounded_auxiliary_output_then_l2_receives_gradient',
    'trainer::tests::bounded_ft_native_roundtrip_and_saved_f32_values_are_exact',
    'trainer::tests::bounded_uniform_shrink_checks_saved_f32_endpoints_and_keeps_moments',
    'trainer::tests::bounded_guards_reject_nonfinite_stale_and_changed_fixed_state',
    'tests::bounded_fanin509_cli_shares_fresh_guards_and_rejects_mode_collisions',
    'tests::bounded_fanin509_recipe_fingerprint_separates_legacy_and_normal_modes',
    'tests::bounded_fanin509_checkpoint_and_final_sidecars_bind_saved_native_and_fixed_denominators',
    'trainer::tests::bounded_fanin509_scales_actual_l2_and_bias_steps_without_scaling_moments_or_out',
    'trainer::tests::bounded_fanin509_closed_mode_and_fresh_guards_reject_invalid_state',
    'trainer::tests::bounded_fanin509_native_and_raw_adam_full_save_preserve_fixed_bytes',
}
DOCUMENT_KEYS = {'schema', 'upstream_commit', 'toolchain_lock_sha256',
                 'external_patch_path', 'external_patch_sha256',
                 'fanin509_patch_path', 'fanin509_patch_sha256',
                 'expected_source_main_sha256', 'expected_source_trainer_sha256',
                 'rustflags', 'rustc', 'cargo', 'rust_tests', 'recipe', 'plan_sha256',
                 'trigger', 'activation_helper_sha256'}


def validate_identity_values(value):
    require(type(value) is dict and set(value) == DOCUMENT_KEYS and
            value['schema'] == 'sekirei.fanin509-trainer-build-identity.v1', 'unexpected identity schema/fields')
    require(value['upstream_commit'] == UPSTREAM and value['toolchain_lock_sha256'] == LOCK_SHA
            and value['external_patch_sha256'] == EXTERNAL_PATCH_SHA
            and value['fanin509_patch_sha256'] == FANIN509_PATCH_SHA
            and value['expected_source_main_sha256'] == FANIN509_MAIN_SHA
            and value['expected_source_trainer_sha256'] == FANIN509_TRAINER_SHA
            and value['rustflags'] == RUSTFLAGS and value['rustc'] == RUSTC
            and exact_value(value['recipe'], RECIPE) and value['plan_sha256'] == PLAN_SHA256, 'identity changes the fixed recipe/base/compiler')
    require(type(value['cargo']) is str and value['cargo'].startswith('cargo 1.96.0 '),
            'unexpected Cargo release')
    activation.validate_trigger_reference(value['trigger'])
    for name in ('fanin509_patch_sha256', 'expected_source_main_sha256', 'expected_source_trainer_sha256', 'activation_helper_sha256'):
        sha(value[name])
    tests = value['rust_tests']
    require(type(tests) is list and all(type(name) is str and
            re.fullmatch(r'(?:tests|trainer::tests)::bounded_[a-z0-9_]+', name) for name in tests)
            and len(tests) == len(set(tests)) == 16 and set(tests) == REQUIRED_TESTS,
            'exactly the sixteen meaningful v3/FanIn509 tests must be declared')
    return value


def load_identity(path, expected_sha256):
    path = canonical(Path(path), exists=True)
    require(info(path)['sha256'] == sha(expected_sha256), 'frozen identity document changed')
    value = validate_identity_values(read_json(path))
    require(canonical(Path(value['external_patch_path']), exists=True) ==
            REPO / 'patches/sekirei-train-external-labels.patch', 'unexpected base patch path')
    for prefix in ('external', 'fanin509'):
        require(info(canonical(Path(value[prefix + '_patch_path']), exists=True))['sha256'] ==
                value[prefix + '_patch_sha256'], 'patch bytes differ from frozen identity')
    return value

def verify_build(runtime, expected_manifest_sha256):
    """Read-only: verify externally frozen full manifest, source, deps, tools, binary.

    The caller must also bind this manifest/patch/post-source SHA to its frozen
    original hypothesis. No caller-selected replacement patch is accepted here.
    Returns the manifest dictionary, with source_files relative paths -> info.
    """
    require_runtime_ready()
    runtime = canonical(Path(runtime), exists=True)
    manifest_path = runtime / 'build-manifest.json'
    require(info(manifest_path)['sha256'] == sha(expected_manifest_sha256), 'build manifest changed')
    manifest = read_json(manifest_path)
    require(manifest.get('schema') == 'sekirei.fanin509-trainer-build.v1' and
            manifest.get('status') == 'complete' and manifest.get('bounded_mode') == 'bounded-material-fanin509-v1'
            and manifest.get('upstream_commit') == UPSTREAM and manifest.get('rustflags') == RUSTFLAGS
            and manifest.get('patch_sha256') == EXTERNAL_PATCH_SHA and exact_value(manifest.get('recipe'), RECIPE) and manifest.get('plan_sha256') == PLAN_SHA256,
            'invalid bounded build contract')
    shared_helpers = verify_shared_helpers(manifest.get("activation_helper_sha256"))
    require(manifest.get("shared_helpers") == shared_helpers and
            manifest.get("builder_path") == str(Path(__file__).resolve()) and
            manifest.get("builder_sha256") == info(Path(__file__).resolve())["sha256"],
            "actual new builder/shared helpers differ from frozen build")
    identity = load_identity(Path(manifest['identity_document_path']), manifest['identity_document_sha256'])
    require(exact_value(manifest.get('trigger'), identity['trigger'])
            and manifest.get('activation_helper_sha256') == identity['activation_helper_sha256'],
            'build trigger/verifier source differs from frozen identity')
    proof = activation.verify_activation(identity['trigger'])
    require(exact_value(manifest.get('activation_validation'), proof), 'build activation evidence changed')
    require(manifest['fanin509_patch_sha256'] == identity['fanin509_patch_sha256'] and
            manifest['source_main_sha256'] == identity['expected_source_main_sha256'] and
            manifest['source_trainer_sha256'] == identity['expected_source_trainer_sha256'],
            'build does not match frozen bounded patch/post-source identity')
    require(info(REPO / 'config/toolchain.lock.json')['sha256'] == LOCK_SHA and
            info(REPO / 'scripts/train_cpu.py')['sha256'] == TRAIN_CPU_SHA and
            info(REPO / 'scripts/benchmark.py')['sha256'] == BENCHMARK_SHA, 'fixed outer helper/lock changed')
    source_files, deps_files = source_identity(runtime / 'source', patched=True)
    require(source_files == manifest['source_files'] and deps_files == manifest['deps_files'],
            'tracked source or dependency bytes changed')
    require(source_files[MAIN]['sha256'] == manifest['source_main_sha256'] and
            source_files[TRAINER]['sha256'] == manifest['source_trainer_sha256'], 'source scalar/map mismatch')
    current = compiler_versions()
    require(current == manifest['compiler'] == {'rustc': identity['rustc'], 'cargo': identity['cargo']},
            'current compiler differs from the frozen build')
    require(compiler_files() == manifest['compiler_files'], 'actual Rust/Cargo toolchain executable bytes changed')
    binary = canonical(Path(manifest['binary']), exists=True)
    require(binary == runtime / 'build/release/train' and os.access(binary, os.X_OK)
            and info(binary)['sha256'] == manifest['binary_sha256'], 'dedicated trainer binary changed')
    require(manifest['rust_tests']['expected'] == identity['rust_tests'] and
            manifest['rust_tests']['passed'] == identity['rust_tests'] and
            type(manifest['rust_tests']['failed']) is int and manifest['rust_tests']['failed'] == 0 and
            type(manifest['rust_tests']['ignored']) is int and manifest['rust_tests']['ignored'] == 0,
            'bounded Rust test receipt mismatch')
    for rel, frozen in manifest['logs'].items():
        require(Path(rel).name == rel and info(runtime / rel) == frozen, 'build/test log changed')
    return manifest


def prepare(runtime, identity_path, expected_identity_sha256):
    require_runtime_ready()
    os.umask(0o077)
    verify_public_entry("prepare_fanin509.py", Path(__file__))
    runtime = canonical(Path(runtime))
    require(runtime.parent.is_dir() and runtime.is_relative_to(PRIVATE) and runtime != PRIVATE,
            'new runtime must have an existing private parent')
    protected = (REPO, Path('/home/server/projects/sekirei-weight2'), COMPARISON, OLD_TRAINER)
    require(not any(runtime.is_relative_to(p) or p.is_relative_to(runtime) for p in protected),
            'dedicated build overlaps a protected checkout/runtime')
    require(not runtime.exists(), 'fresh runtime only; never reuse/overwrite a build')
    outside_git(runtime.parent)
    identity_path = canonical(Path(identity_path), exists=True)
    identity = load_identity(identity_path, expected_identity_sha256)
    shared_helpers = verify_shared_helpers(identity["activation_helper_sha256"])
    activation_proof = activation.verify_activation(identity["trigger"])
    require(info(REPO / 'config/toolchain.lock.json')['sha256'] == LOCK_SHA and
            info(REPO / 'scripts/train_cpu.py')['sha256'] == TRAIN_CPU_SHA and
            info(REPO / 'scripts/benchmark.py')['sha256'] == BENCHMARK_SHA, 'fixed helper/lock changed')
    require(compiler_versions() == {'rustc': identity['rustc'], 'cargo': identity['cargo']},
            'unexpected compiler versions')
    compiler_before = compiler_files()
    require(shutil.disk_usage(runtime.parent).free >= 4 * 2**30, 'dedicated build needs 4 GiB free')
    pins = {str(identity_path): info(identity_path),
            str(REPO / 'config/toolchain.lock.json'): info(REPO / 'config/toolchain.lock.json'),
            str(Path(__file__).resolve()): info(Path(__file__).resolve()),
            str(REPO / 'scripts/train_cpu.py'): info(REPO / 'scripts/train_cpu.py'),
            str(REPO / 'scripts/benchmark.py'): info(REPO / 'scripts/benchmark.py')}
    pins.update(shared_helpers)
    pins.update(activation_proof["receipt_and_stopped_input_identities"])
    for prefix in ('external', 'fanin509'):
        path = Path(identity[prefix + '_patch_path']); pins[str(path)] = info(path)
    runtime.mkdir(mode=0o700)
    with ExitStack() as locks:
        sys.path.insert(0, str(REPO / 'scripts'))
        from benchmark import nonblocking_lock
        for lock in (COMPARISON / '.prepare.lock', COMPARISON / '.benchmark.lock', runtime / '.build.lock'):
            locks.enter_context(nonblocking_lock(lock, exclusive=True))
        env = dict(os.environ, LC_ALL='C', CARGO_TARGET_DIR=str(runtime / 'build'),
                   RUSTFLAGS=RUSTFLAGS, CARGO_BUILD_JOBS='2', RAYON_NUM_THREADS='1',
                   OMP_NUM_THREADS='1', SEKIREI_TRAIN_FTZ_DAZ='1')
        for name in ('RUSTC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'CARGO_ENCODED_RUSTFLAGS',
                     'CARGO_BUILD_RUSTFLAGS', 'CARGO_BUILD_TARGET'):
            env.pop(name, None)
        compiler_after_lock = compiler_files()
        require(compiler_after_lock == compiler_before,
                'compiler identity changed before build: ' + json.dumps(
                    {'before': compiler_before, 'after_lock': compiler_after_lock}, sort_keys=True))
        require(exact_value(activation.verify_activation(identity["trigger"]), activation_proof),
                "activation evidence changed before clone/build under locks")
        source = runtime / 'source'; logs = []
        def step(command, cwd, name, seconds=120):
            run_step(command, cwd, runtime / name, env, seconds);logs.append(name)
        step(['git', 'clone', '--no-hardlinks', '--no-checkout', str(COMPARISON / 'sources/sekirei'), str(source)], runtime, 'clone.log')
        step(['git', 'checkout', '--detach', UPSTREAM], source, 'checkout.log')
        baseline, deps_before = source_identity(source, patched=False)
        external = identity['external_patch_path']; bounded = identity['fanin509_patch_path']
        step(['git', 'apply', '--check', external], source, 'external-check.log')
        step(['git', 'apply', external], source, 'external-apply.log')
        require(set(paths_from_nul(git(source, 'diff', '--name-only', '-z', UPSTREAM))) == {MAIN}
                and info(source / MAIN)['sha256'] == EXTERNAL_MAIN_SHA,
                'base external-label patch changed the original single-main contract')
        step(['git', 'apply', '--check', bounded], source, 'fanin509-check.log')
        step(['git', 'apply', bounded], source, 'fanin509-apply.log')
        files, deps = source_identity(source, patched=True)
        require(files[MAIN]['sha256'] == identity['expected_source_main_sha256'] and
                files[TRAINER]['sha256'] == identity['expected_source_trainer_sha256'],
                'post-patch source differs from frozen FanIn509 identity')
        require(deps == deps_before and all(files[name] == baseline[name] for name in files if name not in CHANGED),
                'bounded patch changed a dependency or unrelated tracked source')
        test_command = ['cargo', 'test', '--release', '--locked', '-j', '2', '-p',
                        'sekirei-train', '--bin', 'train', 'bounded_', '--', '--test-threads=1']
        step(test_command, source, 'fanin509-tests.log', 1200)
        tests = parse_tests(runtime / 'fanin509-tests.log', identity['rust_tests'])
        build_command = ['cargo', 'build', '--release', '--locked', '-j', '2', '-p', 'sekirei-train', '--bin', 'train']
        step(build_command, source, 'build.log', 1200)
        after_files, after_deps = source_identity(source, patched=True)
        require(files == after_files and deps == after_deps and
                all(info(Path(path)) == frozen for path, frozen in pins.items()) and
                compiler_files() == compiler_before,
                'source/dependency/patch/identity inputs changed during preparation')
        require(exact_value(activation.verify_activation(identity["trigger"]), activation_proof),
                "activation evidence changed during build")
        binary = canonical(runtime / 'build/release/train', exists=True)
        require(os.access(binary, os.X_OK), 'built train binary is not executable')
        manifest = {'schema': 'sekirei.fanin509-trainer-build.v1', 'status': 'complete',
                    'binary': str(binary), 'binary_sha256': info(binary)['sha256'],
                    'upstream_commit': UPSTREAM, 'rustflags': RUSTFLAGS,
                    'patch_sha256': EXTERNAL_PATCH_SHA, 'fanin509_patch_sha256': identity['fanin509_patch_sha256'],
                    'source_main_sha256': files[MAIN]['sha256'], 'source_trainer_sha256': files[TRAINER]['sha256'],
                    'source_files': files, 'deps_files': deps, 'compiler': compiler_versions(),
                    'compiler_files': compiler_before,
                    'purpose': 'fresh seed42 FanIn509 material auxiliary28; inference/search core unchanged',
                    'bounded_mode': 'bounded-material-fanin509-v1', 'recipe': RECIPE, 'plan_sha256': PLAN_SHA256,
                    'trigger': identity['trigger'], 'activation_helper_sha256': identity['activation_helper_sha256'],
                    'activation_validation': activation_proof, 'identity_document_path': str(identity_path), 'identity_document_sha256': expected_identity_sha256,
                    'shared_helpers': shared_helpers, 'builder_path': str(Path(__file__).resolve()), 'builder_sha256': info(Path(__file__).resolve())['sha256'],
                    'test_command': test_command, 'build_command': build_command, 'rust_tests': tests,
                    'logs': {name: info(runtime / name) for name in logs},
                    'inputs_before': pins, 'inputs_unchanged': True, 'source_unchanged_during_tests_build': True,
                    'source_files_count': TRACKED_COUNT, 'engine_probe_or_training_started': False,
                    'integer_core_bound_proven': False}
        save_new(runtime / 'build-manifest.json', manifest)
        verify_build(runtime, info(runtime / 'build-manifest.json')['sha256'])
        return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity-document', type=Path, required=True)
    parser.add_argument('--expected-identity-sha256', required=True)
    args = parser.parse_args()
    built = prepare(args.output, args.identity_document, args.expected_identity_sha256)
    print(json.dumps({'status': built['status'], 'bounded_mode': built['bounded_mode'],
                      'manifest_sha256': info(args.output / 'build-manifest.json')['sha256'],
                      'binary_sha256': built['binary_sha256'],
                      'tests_passed': len(built['rust_tests']['passed'])}))
