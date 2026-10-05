#!/usr/bin/env python3
"""SOURCE ONLY new03 original source preflight; disabled before actual I/O.

Completed whole-pool legal replay is reused with every recorded byte pinned.
This worker performs no fit, native probe, teacher search, or adoption decision.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys

sys.dont_write_bytecode = True
R = Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
C = Path('/home/server/.local/share/sekirei-weight2/campaign-17-autonomous-v1')
T = C.parent / 'training-15-v1'
B = C.parent / 'suisho11beta-sekirei-v0.3.39-v1'
F = C.parent / 'training-17-v1/bounded-material-fanin509-trainer-v1'
sys.path.insert(0, str(R / 'scripts'))
import functional_anchor as original
import material_init
import diagnose_anchor as lifecycle
import white_view_fit_operational_common as common
import white_view_fit_contract as contract
activation = common
from benchmark import nonblocking_lock

PROTOTYPE_ONLY = True
PR = contract.PREREGISTRATION
OUT = contract.SOURCE_PREFLIGHT
OLD = C / 'bounded-material-original-source-preflight-v1.json'
OLD_SHA = '1540bdf1991a174bab774b99ee32c52ceee0eb1d6004557705917f477063dcdd'
PREV = C / 'bounded-material-fanin509-original-source-preflight-v1.json'
PREV_SHA = 'd0d936e7c7c729feadcbedd90944344edd2729b7d5f0523ceac752d732529d3e'
CORE = C / 'functional-anchor-core-proof-v1/snapshot-before.json'
CORE_SHA = '80ddbf5056cbf6778bf3dddc109e3b8957c498f2f511eb6be41735fa64377f1a'
SOURCE = T / 'material-fit-112k-v1/snapshot-before.json'
SOURCE_SHA = '15018dc3b380bb3e2b467dcacb15904e2c88b372db0f719b44c5d07b7c315f1f'
DATASET = C / 'source-input-recovery-v1/dataset'
DATA_SHA = 'ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6'
INITIAL = T / 'material-init-seed42/material-init.bin'
INITIAL_SHA = 'bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40'
TEACHER = 'external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d'
COUNTS = {'train': 112681, 'holdout': 5895}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
            and stat.S_ISREG(path.stat().st_mode), 'canonical regular input required')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


Reader = common.Reader


def fixed_git(core, source):
    git, names = lifecycle.git_source_identity(T / 'source')
    require(activation.exact(git, core['git'])
            and {str(p) for p in names} == set(source['fixed_source_tree']), 'fixed core checkout changed')
    require({str(p) for p in (T / 'build/release/deps').iterdir() if p.is_file()}
            == set(source['fixed_build_dependencies']), 'fixed dependency membership changed')
    return git


def main(args):
    require(PROTOTYPE_ONLY is False, 'SOURCE ONLY source preflight disabled before actual I/O')
    common.require_enabled()
    os.umask(0o077)
    worker = Path(__file__)
    require(worker == C / worker.name and worker.resolve() == worker, 'canonical worker copy required')
    require(not os.path.lexists(OUT) and C.resolve() == C and not C.stat().st_mode & 0o077,
            'new private output required')
    require(shutil.disk_usage(C).free >= 2 * 2**30, '2 GiB free required')
    reader = Reader()
    reader.pin(worker, args.expected_worker_sha256)
    reader.pin(Path(common.__file__), args.expected_common_sha256)
    with lifecycle.termination_guard(), ExitStack() as stack:
        locks = common.lock_paths()
        for path in locks:
            stack.enter_context(nonblocking_lock(path, exclusive=True))
        for path in common.architecture_shared_locks():
            stack.enter_context(nonblocking_lock(path, exclusive=False))
        common.bootstrap(reader,worker,args)
        prereg = original._json(reader.read(PR, args.expected_preregistration_sha256))
        contract.validate_preregistration(prereg)
        require(prereg['public_source_commit'] == args.expected_public_source_head, 'extern source HEAD differs')
        helpers = prereg['source_helpers']
        common.pin_helpers(reader, helpers)
        require(helpers['material_init.py'] == original.MATERIAL_IMPLEMENTATION_SHA256,
                'fixed material implementation changed')
        contract.identity_map(prereg['runtime_sources'])
        for path, rec in prereg['runtime_sources'].items():reader.pin(path, rec)
        # No legacy candidate6 activation or epoch guard is called here.
        proof = common.verify_activation(reader, prereg, args)
        reader.pin(contract.ACTIVATION, args.expected_activation_sha256)
        for path, rec in proof['receipt_and_stopped_input_identities'].items():reader.pin(path, rec)
        reader.pin(contract.PLAN, contract.PLAN_SHA256)
        git_before = common.verify_git(args.expected_public_source_head)
        inherited = original._json(reader.read(OLD, OLD_SHA))
        require(inherited.get('status') == 'complete' and inherited.get('inputs_unchanged') is True
                and inherited.get('final_used') is False and inherited.get('original_teacher_identity') == TEACHER,
                'completed original legal replay required')
        replay_inputs = contract.identity_map(inherited['source_inputs_before'])
        require(len(replay_inputs) == 1042 and activation.exact(replay_inputs, inherited['source_inputs_after']),
                'complete replay inventory changed')
        previous = original._json(reader.read(PREV, PREV_SHA))
        require(previous.get('status') == 'complete' and previous.get('source_unchanged') is True
                and activation.exact(previous['source_inputs_before'], replay_inputs)
                and activation.exact(previous['source_inputs_after'], replay_inputs), 'previous source evidence changed')
        for path, rec in replay_inputs.items():
            reader.pin(path, rec)
        rawroot = C.parent / 'shogiquest-human-v1/games'
        raw = {str(p.resolve()) for p in rawroot.glob('*.csa')}
        require(len(raw) == 1000 and raw == {p for p in replay_inputs if p.startswith(str(rawroot) + '/') and p.endswith('.csa')},
                'whole raw1000 membership changed')
        require(all(replay_inputs[p]['sha256'] == Path(p).stem for p in raw), 'raw filename SHA differs')
        files = {name: reader.read(DATASET / name) for name in sorted(original.FILES)}
        manifest, rows = original.validate_original(reader.read(DATASET / 'manifest.json', DATA_SHA), files,
                                                    expected_manifest_sha256=DATA_SHA)
        require(activation.exact(manifest['positions'], COUNTS) and manifest['teacher_identity'] == TEACHER,
                'original count/teacher differs')
        require(prereg['dataset'] == str(DATASET) and prereg['dataset_manifest_sha256'] == DATA_SHA,
                'fit input differs')
        reader.pin(INITIAL, INITIAL_SHA)
        core = original._json(reader.read(CORE, CORE_SHA))
        source = original._json(reader.read(SOURCE, SOURCE_SHA))
        require(len(core['files']) == 617 and len(source['fixed_source_tree']) == 526
                and len(source['fixed_build_dependencies']) == 63, 'fixed source membership differs')
        for path, rec in contract.identity_map(core['files']).items():
            reader.pin(path, rec)
        before_git = fixed_git(core, source)
        before = dict(reader.files)
        after = reader.verify()
        after_git = fixed_git(core, source)
        require(activation.exact(before, after) and activation.exact(before_git, after_git), 'source/input changed')
        receipt = {'schema': 'sekirei.white-view-paired-linear-original-source-preflight.v1', 'status': 'complete',
            'created_at': datetime.now(timezone.utc).isoformat(), 'candidate': contract.MODE,
            'plan_sha256': contract.PLAN_SHA256, 'activation_sha256': prereg['trigger']['sha256'],
            'activation_validation': proof, 'preregistration_sha256': args.expected_preregistration_sha256,
            'source_helpers_sha256': helpers, 'worker_sha256': args.expected_worker_sha256,
            'original_dataset': str(DATASET), 'original_manifest_sha256': DATA_SHA,
            'original_files': {name: before[str(DATASET / name)] for name in original.FILES},
            'counts': COUNTS, 'teacher_identity': TEACHER, 'original_labels_reparsed': True,
            'original_legal_replay_inherited': True, 'whole_pool_exclusions_reverified': True,
            'feature_binding': prereg['feature_binding'], 'new_build_binding': prereg['new_build_binding'],
            'new_build_verified': True,
            'raw1000_membership_reverified': True, 'raw_replay_rerun': False,
            'inherited_replay_receipt': {'path': str(OLD), **before[str(OLD)]},
            'inherited_source_preflight': {'path': str(PREV), **before[str(PREV)]},
            'source_inputs_before': replay_inputs, 'source_inputs_after': {p: after[p] for p in replay_inputs},
            'inputs_before': before, 'inputs_after': after, 'source_git_before': before_git, 'source_git_after': after_git,
            'source_unchanged': True, 'inputs_unchanged': True, 'fixed_core_reference_count': 617,
            'source_input_count': 1042, 'exclusive_locks': list(map(str, locks)), 'architecture_shared_locks':list(map(str,common.architecture_shared_locks())),
            'final_used': False, 'fit_started': False, 'adoption_claimed': False,
            'limitations': ['Completed pinned whole-pool legal replay reused; fresh original bytes and raw membership checked.',
                            'No fit, engine, probe or adoption performed by this worker.']}
        contract.validate_source_preflight(receipt, prereg, args.expected_preregistration_sha256)
        require(common.exact(git_before, common.verify_git(args.expected_public_source_head)), 'public source HEAD changed')
        with OUT.open('xb') as stream:
            stream.write(original.canonical_json_bytes(receipt) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(original.canonical_json_bytes({'status': 'complete', 'receipt': str(OUT), **identity(OUT)}).decode())


def parser():
    return common.parse_common_arguments(argparse.ArgumentParser(description=__doc__),preregistration=True,activation=True)


if __name__ == '__main__':
    if PROTOTYPE_ONLY:raise RuntimeError('SOURCE ONLY source preflight disabled before actual I/O')
    main(parser().parse_args())
