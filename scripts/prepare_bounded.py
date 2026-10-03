#!/usr/bin/env python3
"""Build and verify a dedicated, pinned bounded-material trainer.

Fresh private runtime only. Old preparation/export/diagnosis guards stay intact.
The v2 patch identity document must be frozen externally before launch.
No data/model loading, training, engine probe, search, or adoption is performed.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
PRIVATE = Path.home() / '.local/share/sekirei-weight2'
COMPARISON = PRIVATE / 'suisho11beta-sekirei-v0.3.39-v1'
OLD_TRAINER = PRIVATE / 'training-15-v1'
UPSTREAM = 'f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9'
LOCK_SHA = '4343a043250cf300e4b029f3dc9f953f1ec047bd6bcb4024efeda0a3e9b5bca5'
EXTERNAL_PATCH_SHA = 'd41878e8f50a89a415e1c2dba05c09f0449d20a6e55aa9e8388b1709df7b12cf'
EXTERNAL_MAIN_SHA = '1a56e29f3111cd502aa0a5c7b2078b47f96725970f95492e491b1e5f80b82d0a'
BOUNDED_PATCH_SHA = '6602c12ea83b602cb1a93fa7f16624d1c67e4f23a05aa7bd4d253b21426d48c8'
BOUNDED_MAIN_SHA = '16b6728cdda9bc4f0bd7cd3294028fe3e110edc6b17f5cc034519674b8b04db4'
BOUNDED_TRAINER_SHA = 'b6447b10b0afbe7be4d0f275f5cbe6ab2b97384f8da5170966d2b6f71ff28a43'
TRAIN_CPU_SHA = 'e1e21e3c6ad251098c864345bd45af84d013144a403c7296af6f87c6780fdc07'
BENCHMARK_SHA = '108b3f8b973c843210cd058009667415f1140436b7743484af268c1cf26b7e98'
RUSTFLAGS = '-C target-cpu=x86-64-v3'
RUSTC = 'rustc 1.96.0 (ac68faa20 2026-05-25)'
TRACKED_COUNT = 526
MAIN = 'crates/sekirei-train/src/main.rs'
TRAINER = 'crates/sekirei-train/src/trainer.rs'
CHANGED = {MAIN, TRAINER}
RECIPE = {'mode': 'bounded-material-v1', 'mask_version': 1, 'seed': 42,
          'shuffle_seed': 42, 'epochs': 3, 'selected_epoch': 3,
          'learning_rate': 0.0001, 'lr_schedule': 'step-half',
          'lr_schedule_epochs': 3, 'min_lr': 0.0, 'label_depth': 0,
          'residual_budget_cp': 99, 'resume_supported': False,
          'target': 'original-absolute-external-cp', 'nnue_output': 'absolute',
          'integer_core_bound_proven': False}
REQUIRED_TESTS = {
    'tests::bounded_cli_accepts_only_fresh_original_absolute_recipe',
    'tests::bounded_cli_rejects_all_alternate_update_and_resume_flags',
    'tests::bounded_recipe_fingerprint_separates_normal_mode',
    'tests::bounded_loaded_initial_bytes_are_bound_to_actual_core_serialization',
    'trainer::tests::bounded_adam_skips_protected_params_and_nonzero_moments',
    'trainer::tests::bounded_auxiliary_output_then_l2_receives_gradient',
    'trainer::tests::bounded_ft_native_roundtrip_and_saved_f32_values_are_exact',
    'trainer::tests::bounded_uniform_shrink_checks_saved_f32_endpoints_and_keeps_moments',
    'trainer::tests::bounded_guards_reject_nonfinite_stale_and_changed_fixed_state',
}
DOCUMENT_KEYS = {'schema', 'upstream_commit', 'toolchain_lock_sha256',
                 'external_patch_path', 'external_patch_sha256',
                 'bounded_patch_path', 'bounded_patch_sha256',
                 'expected_source_main_sha256', 'expected_source_trainer_sha256',
                 'rustflags', 'rustc', 'cargo', 'rust_tests', 'recipe'}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def info(path):
    path = canonical(Path(path), exists=True)
    require(stat.S_ISREG(path.stat().st_mode), 'input must be a regular file')
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': digest(data)}


def canonical(path, *, exists=False):
    require(path.is_absolute() and path == path.resolve(strict=exists),
            'absolute canonical paths without symlink aliases are required')
    return path


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=strict_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'invalid SHA256')
    return value


def save_new(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_command(command, cwd=None):
    # Only bounded, read-only Git/compiler version commands use this route.
    result = subprocess.run(command, cwd=cwd, capture_output=True,
                            env=dict(os.environ, LC_ALL='C', GIT_OPTIONAL_LOCKS='0'), timeout=30)
    require(result.returncode == 0, 'read-only Git/compiler query failed')
    return result.stdout


def git(source, *args):
    return read_command(['git', '-C', str(source), *args])


def paths_from_nul(data):
    require(not data or data.endswith(b'\0'), 'malformed Git path list')
    values = [] if not data else data[:-1].decode('utf-8').split('\0')
    for value in values:
        require(value and not Path(value).is_absolute() and
                Path(value).as_posix() == value and '..' not in Path(value).parts,
                'unsafe tracked path')
    require(len(values) == len(set(values)), 'duplicate tracked path')
    return sorted(values)


def source_identity(source, *, patched):
    source = canonical(source, exists=True)
    require(git(source, 'rev-parse', 'HEAD').decode().strip() == UPSTREAM, 'source HEAD changed')
    require(git(source, 'rev-parse', '--is-bare-repository').decode().strip() == 'false',
            'source must be a checkout')
    # Detached state is checked by the existence of no symbolic HEAD. A return
    # code of one is expected for this one read-only command.
    symbolic = subprocess.run(['git', '-C', str(source), 'symbolic-ref', '-q', 'HEAD'],
                             capture_output=True, env=dict(os.environ, LC_ALL='C', GIT_OPTIONAL_LOCKS='0'), timeout=30)
    require(symbolic.returncode == 1 and not symbolic.stdout and not symbolic.stderr,
            'source must remain detached at the pinned commit')
    tracked = paths_from_nul(git(source, 'ls-files', '-z'))
    committed = paths_from_nul(git(source, 'ls-tree', '-r', '--name-only', '-z', UPSTREAM))
    require(tracked == committed and len(tracked) == TRACKED_COUNT, 'tracked set is not the pinned 526 files')
    require(not git(source, 'diff', '--cached', '--name-only', '-z'), 'staged source changes forbidden')
    changes = set(paths_from_nul(git(source, 'diff', '--name-only', '-z', UPSTREAM)))
    expected = CHANGED if patched else set()
    require(changes == expected, 'source changed-file set differs from contract')
    status = git(source, 'status', '--porcelain=v1', '-z', '--untracked-files=all')
    rows = [] if not status else status[:-1].decode('utf-8').split('\0')
    require(not status or status.endswith(b'\0'), 'malformed Git status')
    require(set(rows) == {' M ' + name for name in expected} and len(rows) == len(expected),
            'only the two unstaged modified source files are permitted; no untracked source')
    files = {name: info(source / name) for name in tracked}
    deps = {name: value for name, value in files.items()
            if Path(name).name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'rust-toolchain')}
    require('Cargo.toml' in deps and 'Cargo.lock' in deps, 'missing workspace dependency files')
    return files, deps


def outside_git(path):
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    result = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--is-inside-work-tree'],
                            capture_output=True, text=True,
                            env=dict(os.environ, LC_ALL='C', GIT_OPTIONAL_LOCKS='0'), timeout=30)
    require(result.returncode != 0, 'private outputs cannot be inside any Git repository')
    require(not result.stdout and result.stderr.startswith('fatal: not a git repository'),
            'Git containment query failed for an unexpected reason')


def load_identity(path, expected_sha256):
    path = canonical(Path(path), exists=True)
    require(info(path)['sha256'] == sha(expected_sha256), 'frozen identity document changed')
    value = read_json(path)
    require(type(value) is dict and set(value) == DOCUMENT_KEYS and
            value['schema'] == 'sekirei.bounded-trainer-build-identity.v1', 'unexpected identity schema/fields')
    require(value['upstream_commit'] == UPSTREAM and value['toolchain_lock_sha256'] == LOCK_SHA
            and value['external_patch_sha256'] == EXTERNAL_PATCH_SHA
            and value['bounded_patch_sha256'] == BOUNDED_PATCH_SHA
            and value['expected_source_main_sha256'] == BOUNDED_MAIN_SHA
            and value['expected_source_trainer_sha256'] == BOUNDED_TRAINER_SHA
            and value['rustflags'] == RUSTFLAGS and value['rustc'] == RUSTC
            and value['recipe'] == RECIPE, 'identity changes the fixed recipe/base/compiler')
    require(type(value['cargo']) is str and value['cargo'].startswith('cargo 1.96.0 '),
            'unexpected Cargo release')
    for name in ('bounded_patch_sha256', 'expected_source_main_sha256', 'expected_source_trainer_sha256'):
        sha(value[name])
    require(canonical(Path(value['external_patch_path']), exists=True) ==
            REPO / 'patches/sekirei-train-external-labels.patch', 'unexpected base patch path')
    for prefix in ('external', 'bounded'):
        require(info(canonical(Path(value[prefix + '_patch_path']), exists=True))['sha256'] ==
                value[prefix + '_patch_sha256'], 'patch bytes differ from frozen identity')
    tests = value['rust_tests']
    require(type(tests) is list and all(type(name) is str and
            re.fullmatch(r'(?:tests|trainer::tests)::bounded_[a-z0-9_]+', name) for name in tests)
            and len(tests) == len(set(tests)) == 9 and set(tests) == REQUIRED_TESTS,
            'exactly the nine final v2 meaningful bounded tests must be declared')
    return value


def compiler_versions():
    return {'rustc': read_command(['rustc', '--version']).decode().strip(),
            'cargo': read_command(['cargo', '--version']).decode().strip()}


def compiler_files():
    sysroot = Path(read_command(['rustc', '--print', 'sysroot']).decode().strip()).resolve(strict=True)
    return {str(sysroot / 'bin' / name): info(sysroot / 'bin' / name)
            for name in ('rustc', 'cargo')}


def verify_build(runtime, expected_manifest_sha256):
    """Read-only: verify externally frozen full manifest, source, deps, tools, binary.

    The caller must also bind this manifest/patch/post-source SHA to its frozen
    original hypothesis. No caller-selected replacement patch is accepted here.
    Returns the manifest dictionary, with source_files relative paths -> info.
    """
    runtime = canonical(Path(runtime), exists=True)
    manifest_path = runtime / 'build-manifest.json'
    require(info(manifest_path)['sha256'] == sha(expected_manifest_sha256), 'build manifest changed')
    manifest = read_json(manifest_path)
    require(manifest.get('schema') == 'sekirei.bounded-trainer-build.v1' and
            manifest.get('status') == 'complete' and manifest.get('bounded_mode') == 'bounded-material-v1'
            and manifest.get('upstream_commit') == UPSTREAM and manifest.get('rustflags') == RUSTFLAGS
            and manifest.get('patch_sha256') == EXTERNAL_PATCH_SHA and manifest.get('recipe') == RECIPE,
            'invalid bounded build contract')
    identity = load_identity(Path(manifest['identity_document_path']), manifest['identity_document_sha256'])
    require(manifest['bounded_patch_sha256'] == identity['bounded_patch_sha256'] and
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
            manifest['rust_tests']['failed'] == 0 and manifest['rust_tests']['ignored'] == 0,
            'bounded Rust test receipt mismatch')
    for rel, frozen in manifest['logs'].items():
        require(Path(rel).name == rel and info(runtime / rel) == frozen, 'build/test log changed')
    return manifest


def cleanup(process):
    if process is None:
        return
    def group_exists():
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
    if group_exists():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if group_exists():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)
    deadline = time.monotonic() + 5
    while group_exists() and time.monotonic() < deadline:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    require(not group_exists(), 'build process group remains; no completed build receipt')


def run_step(command, cwd, log_path, env, seconds):
    # Keep the child handle inside this caller across cancellation; no blocked
    # TERM/INT mask is inherited by Cargo, Rust tests, or Git mutations.
    sys.path.insert(0, str(REPO / 'scripts'))
    from train_cpu import termination_guard
    process = None
    with termination_guard() as state:
        with log_path.open('wb') as log:
            try:
                state['spawning'] = True
                try:
                    process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=True)
                finally:
                    state['spawning'] = False
                require(state['signal'] is None, 'cancelled during build spawn')
                returncode = process.wait(timeout=seconds)
                require(returncode == 0, 'build preparation/test command failed; inspect private log')
            finally:
                state['cleaning'] = True
                cleanup(process)
                state['cleaning'] = False
            require(state['signal'] is None, 'cancelled during build cleanup')


def parse_tests(path, expected):
    text = path.read_text(encoding='utf-8')
    rows = re.findall(r'^test ([a-zA-Z0-9_:]+) \.\.\. (ok|FAILED|ignored)$', text, re.MULTILINE)
    passed = [name for name, status in rows if status == 'ok']
    require(len(rows) == len(expected) and len(passed) == len(set(passed)) and
            set(passed) == set(expected) and all(status == 'ok' for _, status in rows),
            'actual bounded Rust test names/counts differ from frozen identity')
    summary = re.findall(r'^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', text, re.MULTILINE)
    require(len(summary) == 1 and tuple(map(int, summary[0])) == (len(expected), 0, 0),
            'Rust harness completion/counts not established')
    return {'expected': expected, 'passed': expected, 'failed': 0, 'ignored': 0}


def prepare(runtime, identity_path, expected_identity_sha256):
    os.umask(0o077)
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
    for prefix in ('external', 'bounded'):
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
        source = runtime / 'source'; logs = []
        def step(command, cwd, name, seconds=120):
            run_step(command, cwd, runtime / name, env, seconds);logs.append(name)
        step(['git', 'clone', '--no-hardlinks', '--no-checkout', str(COMPARISON / 'sources/sekirei'), str(source)], runtime, 'clone.log')
        step(['git', 'checkout', '--detach', UPSTREAM], source, 'checkout.log')
        baseline, deps_before = source_identity(source, patched=False)
        external = identity['external_patch_path']; bounded = identity['bounded_patch_path']
        step(['git', 'apply', '--check', external], source, 'external-check.log')
        step(['git', 'apply', external], source, 'external-apply.log')
        require(set(paths_from_nul(git(source, 'diff', '--name-only', '-z', UPSTREAM))) == {MAIN}
                and info(source / MAIN)['sha256'] == EXTERNAL_MAIN_SHA,
                'base external-label patch changed the original single-main contract')
        step(['git', 'apply', '--check', bounded], source, 'bounded-check.log')
        step(['git', 'apply', bounded], source, 'bounded-apply.log')
        files, deps = source_identity(source, patched=True)
        require(files[MAIN]['sha256'] == identity['expected_source_main_sha256'] and
                files[TRAINER]['sha256'] == identity['expected_source_trainer_sha256'],
                'post-patch source differs from frozen v2 identity')
        require(deps == deps_before and all(files[name] == baseline[name] for name in files if name not in CHANGED),
                'bounded patch changed a dependency or unrelated tracked source')
        test_command = ['cargo', 'test', '--release', '--locked', '-j', '2', '-p',
                        'sekirei-train', '--bin', 'train', 'bounded_', '--', '--test-threads=1']
        step(test_command, source, 'bounded-tests.log', 1200)
        tests = parse_tests(runtime / 'bounded-tests.log', identity['rust_tests'])
        build_command = ['cargo', 'build', '--release', '--locked', '-j', '2', '-p', 'sekirei-train', '--bin', 'train']
        step(build_command, source, 'build.log', 1200)
        after_files, after_deps = source_identity(source, patched=True)
        require(files == after_files and deps == after_deps and
                all(info(Path(path)) == frozen for path, frozen in pins.items()) and
                compiler_files() == compiler_before,
                'source/dependency/patch/identity inputs changed during preparation')
        binary = canonical(runtime / 'build/release/train', exists=True)
        require(os.access(binary, os.X_OK), 'built train binary is not executable')
        manifest = {'schema': 'sekirei.bounded-trainer-build.v1', 'status': 'complete',
                    'binary': str(binary), 'binary_sha256': info(binary)['sha256'],
                    'upstream_commit': UPSTREAM, 'rustflags': RUSTFLAGS,
                    'patch_sha256': EXTERNAL_PATCH_SHA, 'bounded_patch_sha256': identity['bounded_patch_sha256'],
                    'source_main_sha256': files[MAIN]['sha256'], 'source_trainer_sha256': files[TRAINER]['sha256'],
                    'source_files': files, 'deps_files': deps, 'compiler': compiler_versions(),
                    'compiler_files': compiler_before,
                    'purpose': 'fresh seed42 bounded material auxiliary28 training; inference/search core unchanged',
                    'bounded_mode': 'bounded-material-v1', 'recipe': RECIPE,
                    'identity_document_path': str(identity_path), 'identity_document_sha256': expected_identity_sha256,
                    'builder_path': str(Path(__file__).resolve()), 'builder_sha256': info(Path(__file__).resolve())['sha256'],
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
