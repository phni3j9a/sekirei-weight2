"""Synthetic origin/DAG fixtures; no private corpus, Cargo, engine or fitting."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import weekly_nonlinear_preflight as w


def put(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body if type(body) is bytes else json.dumps(body, allow_nan=False).encode())
    return w.parent_run.physical_ref(path)


def gid(split):
    for n in range(10000):
        value = hashlib.sha256(f'public-game-{split}-{n}'.encode()).hexdigest()
        if w.split_for_game(value, 'public-seed') == split:
            return value
    raise AssertionError('synthetic split unavailable')


class OriginFixture:
    def __init__(self, root):
        self.root = root
        pack = b'\1' + struct.pack('<Hh', 257, 50) + b'\0\0\0'
        pack += b'\1' + struct.pack('<Hh', 385, 70) + b'\0\0\0'
        self.pack = put(root / 'packs/public.pack', pack)
        self.pack_ref = put(root / 'packs/manifest.json', {'unique_files': [
            {'path': 'public.pack', **w.small(self.pack)}]})
        self.teacher = 'external:suisho11beta-1m-pack:' + self.pack_ref['sha256']
        raw = b'V2.2\nN+public-black\nN-public-white\nPI\n+\n+7776FU\n%TORYO\n'
        rawsha = hashlib.sha256(raw).hexdigest()
        self.pool_raw = put(root / ('pool/games/' + rawsha + '.csa'), raw)
        parsed = w.parse_csa(raw.decode())
        aggregate = hashlib.sha256(parsed['canonical_sha256'].encode()).hexdigest()
        self.pool_ref = put(root / 'pool/manifest.json', {'accepted_games': 1,
                                                       'corpus_canonical_sha256': aggregate})
        excluded = {'games': 1, 'unique_positions': 2, 'corpus_canonical_sha256': aggregate,
                    'source_manifest_sha256': self.pool_ref['sha256'], 'policy': w.freeze.EXCLUSION_POLICY}
        self.train_gid, self.hold_gid = gid('train'), gid('holdout')
        games = [{'game_id': self.train_gid, 'split': 'train', 'pack_sha256': self.pack['sha256'], 'game_index': 0},
                 {'game_id': self.hold_gid, 'split': 'holdout', 'pack_sha256': self.pack['sha256'], 'game_index': 1}]
        self.old = {'schema_version': 1, 'teacher_identity': self.teacher,
                    'source_corpus_manifest_sha256': self.pack_ref['sha256'],
                    'dependencies': {'cshogi': '1.0.4', 'numpy': '1.26.4'}, 'seed': 'public-seed',
                    'split': 'public-whole-game-split', 'games': games, 'positions': {'train': 1, 'holdout': 1},
                    'independent_exclusions': excluded, 'label_depth': 'external sentinel'}
        self.frozen = self.dataset(root / 'frozen', self.old, 'old-public-board')
        args = Namespace(selection_seed='public-fixed-rank', games_per_pack=2, train_count=1,
                         expected_frozen_manifest_sha256=self.frozen['manifest.json']['sha256'])
        self.profile = w.diverse.selection_profile(args, {'manifest': self.old},
            {'unique_files': [{'path': 'public.pack', **w.small(self.pack)}]}, self.pack_ref['sha256'],
            {self.pack['sha256']: {'games': 2, 'selected_games': 2}})
        self.profile_ref = put(root / 'profile.json', self.profile)
        spans, census = w.diverse.select_spans(pack, self.pack['sha256'], args.selection_seed, 2)
        sources = {str(w.REPO / name): identity['sha256'] for name, identity in self.profile['producer_sources'].items()}
        for reference in [*self.frozen.values(), self.pack, self.pack_ref, self.pool_ref, self.pool_raw, self.profile_ref]:
            sources[reference['path']] = reference['sha256']
        new_games = deepcopy(games)
        new_games[0]['selection_rank'] = next(span['rank'] for span in spans if span['game_index'] == 0)
        self.manifest = {**deepcopy(self.old), 'games': new_games, 'sampling': self.profile['sampling'],
            'selection_seed': args.selection_seed, 'games_per_pack': 2, 'train_budget': 1, 'counts': {}, 'limitations': [],
            'derivation': {'kind': w.diverse.DERIVATION_KIND, 'profile': self.profile,
                'profile_sha256': self.profile_ref['sha256'], 'producer_sources': self.profile['producer_sources'],
                'input_sha256': sources, 'script_sha256': w.parent_run.info(Path(w.diverse.__file__))['sha256'],
                'ordering': self.profile['ordering'], 'indexes': [{'pack_sha256': self.pack['sha256'], 'census': census,
                    'selected_spans': spans, 'temporary_selected_bytes_sha256': hashlib.sha256(pack).hexdigest()}],
                'frozen_dataset_manifest_sha256': self.frozen['manifest.json']['sha256'],
                'train_histogram': {}, 'old_train_histogram': {}, 'old_train_board_overlap': 0,
                'input_unchanged': True, 'wall_seconds': 0.01}}
        self.dataset_refs = self.dataset(root / 'new', self.manifest, 'new-public-board')

    def dataset(self, root, manifest, board):
        files = {}
        for split, sfen, game in [('train', board + ' b - 16', self.train_gid),
                                  ('holdout', 'held-public-board w - 17', self.hold_gid)]:
            position = {'schema_version': 1, 'sfen': sfen, 'source': {'kind': 'gensfen-pack', 'path': game, 'ply': 16},
                        'tags': {'side_to_move': 'black' if split == 'train' else 'white', 'phase': 'middlegame'}}
            label = {'sfen': sfen, 'score_cp': 50, 'teacher_identity': self.teacher, 'label_depth': 0}
            for kind, row in [('positions', position), ('labels', label)]:
                name = split + '.' + kind + '.jsonl'
                files[name] = w.small(put(root / name, (json.dumps(row) + '\n').encode()))
        manifest['files'] = files
        put(root / 'manifest.json', manifest)
        return w.dataset_five(root)

    def limits(self):
        stack = ExitStack()
        for name, value in [('N', 1), ('H', 1), ('PACK_COUNT', 1), ('POOL_COUNT', 1)]:
            stack.enter_context(patch.object(w, name, value))
        stack.enter_context(patch.object(w.preparation, 'FROZEN_MANIFEST_SHA', self.frozen['manifest.json']['sha256']))
        stack.enter_context(patch.object(w.preparation, 'TEACHER', self.teacher))
        return stack

    def verify(self):
        return w.verify_origin_inputs(w.Reader(), self.dataset_refs, self.frozen, self.profile_ref, self.pack_ref, self.pool_ref)

    def save_new(self):
        self.dataset_refs['manifest.json'] = put(Path(self.dataset_refs['manifest.json']['path']), self.manifest)


class OriginTests(unittest.TestCase):
    def test_actual_raw_origin_join_index_pool_and_frozen_holdout_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = OriginFixture(Path(directory))
            with fixture.limits():
                _, _, _, hashes, before, counts = fixture.verify()
            self.assertEqual(counts, {'train': 1, 'holdout': 1, 'acquired_pool_games': 1, 'selected_pack_games': 2})
            self.assertIn(fixture.pool_raw['path'], before)
            self.assertTrue(set(hashes) <= set(before))
            self.assertTrue(not set(before) & {r['path'] for r in fixture.dataset_refs.values()})
            w.parent_run.verify_map(before)

    def test_unknown_schema_false_immutability_and_wrong_pack_count_are_rejected(self):
        for change in ('unknown', 'bool', 'rank', 'span', 'census'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fixture = OriginFixture(Path(directory))
                if change == 'unknown': fixture.manifest['unexpected'] = True
                if change == 'bool': fixture.manifest['derivation']['input_unchanged'] = 1
                if change == 'rank': fixture.manifest['games'][0]['selection_rank'] = 'f' * 64
                if change == 'span': fixture.manifest['derivation']['indexes'][0]['selected_spans'][0]['end'] += 1
                if change == 'census': fixture.manifest['derivation']['indexes'][0]['census']['games'] += 1
                fixture.save_new()
                with fixture.limits(), self.assertRaises(ValueError): fixture.verify()

    def test_changed_raw_teacher_labels_missing_join_and_holdout_change_are_rejected(self):
        for change in ('teacher', 'missing-label', 'holdout', 'pool-member', 'pack-raw'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fixture = OriginFixture(Path(directory))
                if change in ('teacher', 'missing-label'):
                    path = Path(fixture.dataset_refs['train.labels.jsonl']['path'])
                    label = json.loads(path.read_text())
                    label['teacher_identity'] = 'other-teacher'
                    put(path, b'' if change == 'missing-label' else (json.dumps(label) + '\n').encode())
                    fixture.dataset_refs['train.labels.jsonl'] = w.parent_run.physical_ref(path)
                    fixture.manifest['files']['train.labels.jsonl'] = w.small(fixture.dataset_refs['train.labels.jsonl'])
                    fixture.save_new()
                elif change == 'holdout':
                    fixture.dataset_refs = fixture.dataset(fixture.root / 'new', fixture.manifest, 'new-public-board')
                    path = Path(fixture.dataset_refs['holdout.positions.jsonl']['path'])
                    path.write_bytes(path.read_bytes() + b' ')
                elif change == 'pool-member':
                    put(fixture.root / 'pool/games/extra.csa', b'PI\n')
                else:
                    Path(fixture.pack['path']).write_bytes(b'changed raw pack')
                with fixture.limits(), self.assertRaises(ValueError): fixture.verify()

    def test_complete_independent_pool_including_initial_board_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = OriginFixture(Path(directory))
            with fixture.limits():
                keys = w.pool_exclusions(w.Reader(), fixture.pool_ref, fixture.manifest['independent_exclusions'])
            self.assertIn(w.tracker_key(w.benchmark.BoardTracker.standard()), keys)
            self.assertEqual(len(keys), 2)


class ProcessTests(unittest.TestCase):
    def allow(self, services=()):
        return {'schema': 'sekirei.weekly-nonlinear-heavy-process-allowlist.v1',
                'status': 'frozen-before-preflight', 'services': list(services)}

    def process(self, root, pid, comm='harmless', command=b'service\0', start=42):
        path = root / str(pid)
        put(path / 'comm', (comm + '\n').encode())
        put(path / 'cmdline', command)
        put(path / 'stat', (f'{pid} ({comm}) ' + ' '.join(['0'] * 19 + [str(start)])).encode())
        return {'pid': pid, 'uid': path.stat().st_uid, 'comm': comm, 'starttime_ticks': start,
                'cmdline_sha256': hashlib.sha256(command).hexdigest()}

    def test_real_proc_records_allowlist_pid_start_and_cmdline_identity(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(w.parent_run, 'scan_heavy_processes', return_value=[]):
            root = Path(directory); service = self.process(root, 1234)
            scan = w.observe_processes(self.allow([service]), root, own_pid=999)
            self.assertEqual(scan, {'conflicts': [], 'excluded_preexisting_services': [service]})
            for field, value in [('starttime_ticks', 43), ('cmdline_sha256', 'a' * 64), ('pid', True)]:
                changed = dict(service, **{field: value})
                with self.subTest(field=field), self.assertRaises(ValueError):
                    w.observe_processes(self.allow([changed]), root, own_pid=999)

    def test_heavy_comm_or_bound_scanner_conflicts_never_allowlisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); service = self.process(root, 1234, comm='train')
            with patch.object(w.parent_run, 'scan_heavy_processes', return_value=[]), self.assertRaises(ValueError):
                w.observe_processes(self.allow([service]), root, own_pid=999)
            (root / '1234/comm').write_text('harmless')
            with patch.object(w.parent_run, 'scan_heavy_processes', return_value=[{'pid': 2222}]), self.assertRaises(ValueError):
                w.observe_processes(self.allow(), root, own_pid=999)

    def test_unreadable_same_uid_command_requires_exact_null_allowlist(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(w.parent_run, 'scan_heavy_processes', return_value=[]):
            root = Path(directory); service = self.process(root, 1234)
            original = Path.read_bytes
            def denied(path):
                if path == root / '1234/cmdline': raise PermissionError('public-fixture')
                return original(path)
            with patch.object(Path, 'read_bytes', denied):
                with self.assertRaises(ValueError): w.observe_processes(self.allow(), root, own_pid=999)
                service['cmdline_sha256'] = None
                self.assertEqual(w.observe_processes(self.allow([service]), root, own_pid=999)
                                 ['excluded_preexisting_services'], [service])


def selected_plan(dataset, profile_ref, model_ref, teacher):
    return {'schema': 'sekirei.weekly-nonlinear-selected-plan.v1', 'status': 'frozen-before-build',
        'mode': 'weekly-public-fixture-e3-v1', 'hypothesis': 'public fixture', 'created_utc': '2026-10-05T00:00:00Z',
        'architecture': {'feature_schema': 'flat_white_view_aux_tied_v1', 'dimensions': [2420, 256, 32],
            'native_magic': 'SEKIRW03', 'protected_material_units': [0, 1, 2, 3], 'all_ft_bias_fixed_q': 64,
            'hand_ties': [[0, 3], [1, 2]], 'aux_ft_channels': 254, 'paired_heads': 14},
        'training': {'seed': 42, 'epochs': 3, 'selected_epoch': 3, 'train_count': 112681, 'holdout_count': 5895,
            'objective': 'absolute-cp-mse', 'optimizer': 'fresh-adam-tied-masters', 'learning_rate_schedule': 'constant',
            'learning_rate_f32_bits': '3a83126f', 'head_init_width_f32_bits': '3b800000',
            'head_bias_init_f32_bits': '40800000', 'output_native_l1_budget_f32_bits': '47000000',
            'shuffle_seed': None, 'resume_allowed': False, 'ft_saved_q_max': 797, 'ft_bias_q': 64},
        'dataset_inputs': dataset, 'teacher_identity': teacher, 'selection_profile': profile_ref,
        'resources': {'heavy_serial': True, 'build_jobs': 2, 'analysis_jobs': 1, 'threads': 1,
            'training_wall_limit_seconds': 21600, 'added_storage_budget_bytes': 8192, 'minimum_ssd_remaining_bytes': 8192},
        'adoption': {'development_games': 5, 'go_nodes': 1000000, 'mae_multipv': 1, 'top3_multipv': 3,
            'rule': 'strict MAE decrease and no Top3 decrease against latest incumbent',
            'final_used_for_daily_selection': False, 'learning_loss_is_adoption_criterion': False},
        'incumbent_model': model_ref}


class BindingTests(unittest.TestCase):
    def private_temp(self):
        # Some shared hosts intentionally mark /tmp as a Git workspace. The
        # private-output policy should still be exercised without bypassing it.
        return tempfile.TemporaryDirectory(dir=Path.home())

    def author(self, root, stack):
        fixture = OriginFixture(root)
        stack.enter_context(fixture.limits())
        plan = selected_plan(fixture.dataset_refs, fixture.profile_ref, put(root / 'incumbent.bin', b'incumbent'), fixture.teacher)
        plan_ref = put(root / 'plan.json', plan)
        reference = put(root / 'reference03.bin', b'public-reference-fullbytes')
        allowlist = put(root / 'allowlist.json', ProcessTests().allow())
        numeric = put(root / 'numeric/white_view_build_contract.py', b'public-numeric-fixture')
        hashes = list(w.parent_run.NUMERIC_SOURCE_SHA)
        hashes[2] = numeric['sha256']
        stack.enter_context(patch.object(w.parent_run, 'NUMERIC_SOURCE_SHA', tuple(hashes)))
        source = put(root / 'runtime/source/public.rs', b'public-source')
        build = {'source_root': str(Path(source['path']).parent), 'source_head': 'a' * 40,
                 'source_files': {source['path']: w.small(source)}, 'compiler_files': {numeric['path']: w.small(numeric)},
                 'engine_build': put(root / 'engine.json', {'public': 'engine'}),
                 'training_binary': put(root / 'runtime/build/release/train', b'public-executable')}
        build_ref = put(root / 'build.json', build)
        def read_build(reader, ref, *_args):
            actual = reader.json(ref)
            reader.map(actual['source_files']); reader.map(actual['compiler_files'])
            reader.pin(actual['engine_build']); reader.pin(actual['training_binary'])
            return actual
        stack.enter_context(patch.object(w, 'verify_build', side_effect=read_build))
        stack.enter_context(patch.object(w.parent_run, 'source_map', return_value=build['source_files']))
        stack.enter_context(patch.object(w, 'observe_processes', return_value={'conflicts': [], 'excluded_preexisting_services': []}))
        class PureGate:
            @staticmethod
            def encode_native(_value): return b'public-reference-fullbytes'
            @staticmethod
            def canonical_reference(): return object()
        stack.enter_context(patch.object(w.parent_run, 'load_numeric_gate', return_value=(PureGate, {numeric['path']: w.small(numeric)})))
        args = Namespace(plan=Path(plan_ref['path']), expected_plan_sha256=plan_ref['sha256'],
            training_build=Path(build_ref['path']), expected_training_build_sha256=build_ref['sha256'],
            reference03=Path(reference['path']), expected_reference03_sha256=reference['sha256'],
            pack_manifest=Path(fixture.pack_ref['path']), expected_pack_manifest_sha256=fixture.pack_ref['sha256'],
            pool_manifest=Path(fixture.pool_ref['path']), expected_pool_manifest_sha256=fixture.pool_ref['sha256'],
            frozen_dataset=Path(fixture.frozen['manifest.json']['path']).parent,
            expected_frozen_manifest_sha256=fixture.frozen['manifest.json']['sha256'],
            allowlist=Path(allowlist['path']), expected_allowlist_sha256=allowlist['sha256'],
            numeric_build_contract=Path(numeric['path']), lock=[root / '.heavy.lock'], output=root / 'preflight')
        with redirect_stdout(io.StringIO()): recipe_ref = w.create(args)
        return fixture, recipe_ref

    def rebind(self, recipe_ref, changed_parent=None, changed_binding=None, changed_recipe=None):
        recipe = json.loads(Path(recipe_ref['path']).read_bytes())
        sb = json.loads(Path(recipe['source_binding']['path']).read_bytes())
        parent = json.loads(Path(sb['parent_preflight']['path']).read_bytes())
        if changed_parent:
            changed_parent(parent)
            sb['parent_preflight'] = put(Path(sb['parent_preflight']['path']), parent)
        if changed_binding: changed_binding(sb)
        recipe['source_binding'] = put(Path(recipe['source_binding']['path']), sb)
        if changed_recipe: changed_recipe(recipe)
        return put(Path(recipe_ref['path']), recipe)

    def test_authors_fresh_four_artifact_dag_and_validator_never_relocks(self):
        with self.private_temp() as directory, ExitStack() as stack:
            _, ref = self.author(Path(directory), stack)
            with patch.object(w.benchmark, 'nonblocking_lock', side_effect=AssertionError('consumer must not lock')):
                verified = w.validate_source_binding(ref)
            self.assertEqual(set(verified), {'recipe', 'source_binding', 'parent', 'plan', 'build'})
            self.assertEqual(set(verified['recipe']), w.RECIPE_KEYS)
            parent = verified['parent']
            self.assertEqual(parent['source_inputs'], parent['inputs_before'])
            self.assertEqual(parent['source_inputs'], parent['inputs_after'])
            self.assertEqual(len(parent['process_evidence']['scan_records']), 2)
            origin = json.loads(Path(parent['dataset_origin_proof']['path']).read_bytes())
            self.assertFalse(origin['actual_replay_repeated_by_verifier'] or origin['final_used'])
            self.assertFalse(set(origin['inputs_before']) & {r['path'] for r in origin['dataset_inputs'].values()})

    def test_parent_sb_recipe_unknown_false_bool_mismatch_and_missing_dependency_are_rejected(self):
        changes = [('unknown', lambda p: p.update(unexpected=True), None, None),
            ('checks-bool', lambda p: p['checks'].update(no_conflicting_heavy_process=1), None, None),
            ('counts', lambda p: p['origin_counts'].update(selected_pack_games=99), None, None),
            ('missing-input', lambda p: p['source_inputs'].pop(next(k for k in p['source_inputs'] if k.endswith('/incumbent.bin'))), None, None),
            ('missing-lock', lambda p: p['process_evidence'].update(lock_records=[]), None, None),
            ('scan', lambda p: p['process_evidence']['scan_records'][0].update(conflicts=[{'pid': 123}]), None, None),
            ('model', None, lambda s: s.update(training_binary=s['reference03']), None),
            ('resume', None, None, lambda r: r.update(resume_allowed=True)),
            ('recipe-extra', None, None, lambda r: r.update(selected_plan={}))]
        for name, parent_change, sb_change, recipe_change in changes:
            with self.subTest(change=name), self.private_temp() as directory, ExitStack() as stack:
                _, ref = self.author(Path(directory), stack)
                ref = self.rebind(ref, parent_change, sb_change, recipe_change)
                with self.assertRaises(ValueError): w.validate_source_binding(ref)

    def test_rehashing_outer_dag_does_not_hide_changed_actual_source_or_reference(self):
        for name in ('source', 'reference'):
            with self.subTest(change=name), self.private_temp() as directory, ExitStack() as stack:
                _, ref = self.author(Path(directory), stack)
                target = Path(directory) / ('runtime/source/public.rs' if name == 'source' else 'reference03.bin')
                target.write_bytes(b'changed')
                with self.assertRaises(ValueError): w.validate_source_binding(ref)

    def test_strict_refs_and_private_fresh_output_reject_type_alias_and_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = put(root / 'raw.json', {'public': 1})
            for changed in (dict(ref, bytes=True), dict(ref, sha256='A' * 64), dict(ref, unexpected=0)):
                with self.assertRaises(ValueError): w.Reader().pin(changed)
            alias = root / 'alias.json'; alias.symlink_to(Path(ref['path']))
            with self.assertRaises(ValueError): w.Reader().pin(dict(ref, path=str(alias)))
            with self.assertRaises(ValueError): w.fresh_private_output(root / 'raw.json')
            (root / '.git').mkdir()
            with self.assertRaises(ValueError): w.fresh_private_output(root / 'new')


class BuildContractTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'runtime/source'
        published = json.loads((w.preparation.BASE / 'public-source-manifest.json').read_bytes())
        for name in published['files']:
            if name.startswith('source/'):
                put(source / name[len('source/'):], (w.preparation.BASE / name).read_bytes())
        train = source / 'crates/sekirei-train/src'
        for name in ('paired_nonlinear_actual.rs', 'paired_nonlinear_parent_binding.rs', 'weekly_nonlinear_manifest_tests.rs'):
            put(train / name, (w.preparation.REPAIR / name).read_bytes())
        main = train / 'main.rs'
        needle = '#[cfg(feature = "nnue_white_view_aux_tied")]\nmod paired_nonlinear_sha256;'
        main.write_text(main.read_text().replace(needle,
            '#[cfg(feature = "nnue_white_view_aux_tied")]\nmod weekly_nonlinear_profile;\n' + needle))
        cli = train / 'paired_nonlinear_cli.rs'
        cli.write_text(cli.read_text().replace('pub const MODE:&str="white-view-paired-nonlinear-adam-e3-v1";',
                                              'pub const MODE:&str=crate::weekly_nonlinear_profile::MODE;'))
        profile = {'kind': w.diverse.PROFILE_KIND, 'sampling': w.diverse.SAMPLING,
            'frozen_dataset_manifest_sha256': w.preparation.FROZEN_MANIFEST_SHA, 'train_budget': 112681,
            'selection_seed': 'public-fixture', 'games_per_pack': 500,
            'packs': [{'sha256': f'{i:064x}', 'bytes': 1000, 'games': 392 if i == 1 else 2000,
                       'selected_games': 392 if i == 1 else 500} for i in range(1, 14)],
            'corpus_manifest_sha256': w.preparation.TEACHER.split(':')[-1]}
        profile_ref = put(root / 'profile.json', profile)
        manifest = {'teacher_identity': w.preparation.TEACHER, 'positions': {'train': 112681, 'holdout': 5895},
                    'derivation': {'kind': w.diverse.DERIVATION_KIND, 'profile': profile,
                                   'profile_sha256': profile_ref['sha256']}}
        dataset = {name: put(root / 'data' / name, manifest if name == 'manifest.json' else b'public row\n')
                   for name in w.DATA_FILES}
        plan = selected_plan(dataset, profile_ref, put(root / 'incumbent.bin', b'public incumbent'), w.preparation.TEACHER)
        plan_ref = put(root / 'plan.json', plan)
        generated = w.preparation.profile_source(plan, w.small(plan_ref), manifest, w.small(dataset['manifest.json']),
                                                  profile, w.small(profile_ref))
        put(train / 'weekly_nonlinear_profile.rs', generated)
        sources = {str(p): w.parent_run.info(p) for p in source.rglob('*') if p.is_file()}
        compiler = {str(root / 'compiler' / name): w.small(put(root / 'compiler' / name, b'public compiler'))
                    for name in ('cargo', 'rustc')}
        dependency = put(root / '.cargo/registry/src/public-registry/public-package/lib.rs', b'public dependency')
        dependencies = {dependency['path']: w.small(dependency)}
        engine = {'compiler_files': compiler, 'source_files_after': {
            name: sources[str(source / name)] for name in ('crates/sekirei-core/Cargo.toml', 'crates/sekirei-core/src/nnue.rs')}}
        engine_ref = put(root / 'engine.json', engine)
        prep = {'schema': 'sekirei.weekly-nonlinear-source-preparation.v1', 'status': 'source-only',
                'upstream_commit': w.preparation.UPSTREAM, 'source_root': str(source), 'plan': plan_ref,
                'manifest': dataset['manifest.json'], 'selection_profile': profile_ref,
                'base_patch': w.parent_run.physical_ref(w.preparation.BASE / 'nonlinear-complete.patch'),
                'source_files': sources, 'numeric_update_export_modules_preserved': True, 'compiled': False,
                'training_started': False, 'model_adoption_claimed': False, 'final_used': False}
        prep_ref = put(source.parent / 'source-preparation.json', prep)
        names = ['paired_nonlinear_' + category for category in ('reader', 'sha', 'unique', 'native', 'adapter',
                                                                'float_policy', 'initialized', 'weekly')]
        names += [f'paired_nonlinear_public_{i}' for i in range(16)]
        text = '\n'.join('test ' + name + ' ... ok' for name in names)
        text += '\ntest result: ok. 24 passed; 0 failed; 0 ignored;\n'
        logs = [put(root / 'test.log', text.encode()), put(root / 'build.log', b'public build log')]
        cargo = str(root / 'compiler/cargo')
        common = ['--package', 'sekirei-train', '--bin', 'train', '--release', '--offline', '--locked',
                  '--jobs', '2', '--features', 'nnue_white_view_aux_tied']
        commands = [[cargo, 'test', *common, 'paired_nonlinear', '--', '--test-threads=1'], [cargo, 'build', *common]]
        outcomes = [{'command': command, 'pid': 1234 + i, 'pgid': 1234 + i, 'returncode': 0, 'waited': True,
            'reaped': True, 'timed_out': False, 'group_empty_scans': [True, True], 'wall_seconds': 0.1, 'log': logs[i]}
            for i, command in enumerate(commands)]
        binary = put(root / 'runtime/build/release/train', b'public executable fixture')
        Path(binary['path']).chmod(0o700)
        producer = w.parent_run.physical_ref(w.REPO / 'scripts/weekly_nonlinear_build.py')
        before = {**sources, **compiler, **dependencies,
                  **{reference['path']: w.small(reference) for reference in (prep_ref, engine_ref, plan_ref, producer)}}
        build = {'schema': 'sekirei.weekly-nonlinear-training-build.v1', 'status': 'complete', 'mode': plan['mode'],
                 'producer_source': producer, 'source_root': str(source), 'upstream_commit': w.preparation.UPSTREAM,
                 'source_head': 'a' * 40, 'source_files': sources, 'compiler_files': compiler, 'dependency_files': dependencies,
                 'engine_build': engine_ref, 'training_binary': binary, 'commands': commands,
                 'environment': {'CARGO_TARGET_DIR': str(root / 'runtime/build'), 'RUSTC': str(root / 'compiler/rustc'),
                                 'RUSTFLAGS': '-C target-cpu=x86-64-v3', 'CARGO_BUILD_JOBS': '2'},
                 'tests': w.builder.parse_test_log(Path(logs[0]['path'])), 'process_outcomes': outcomes,
                 'inputs_before': before, 'inputs_after': deepcopy(before), 'scalar_power_cache_used': False}
        return build, plan_ref, plan, engine, dependencies

    def run_build(self, root, build, plan_ref, plan, engine, dependencies):
        reference = put(root / 'training-build.json', build)
        def read_engine(reader, ref):
            reader.json(ref)
            return engine
        with patch.object(w, 'verify_engine', side_effect=read_engine), \
                patch.object(w.parent_run, 'source_map', return_value=build['source_files']), \
                patch.object(w.builder, 'registry_dependencies', return_value=dependencies), \
                patch.object(Path, 'home', return_value=root):
            return w.verify_build(w.Reader(), reference, plan_ref, plan)

    def test_complete_raw_build_uses_pinned_public_numerics_generated_profile_and_deps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); values = self.fixture(root)
            build = self.run_build(root, *values)
            self.assertEqual(build['tests']['passed'], 24)
            self.assertFalse(build['scalar_power_cache_used'])

    def test_typed_build_lifecycle_command_and_profile_changes_fail(self):
        for change in ('bool', 'unknown', 'group', 'command', 'test-name', 'source', 'dependency', 'prep-flags'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); build, plan_ref, plan, engine, deps = self.fixture(root)
                if change == 'bool': build['process_outcomes'][0]['returncode'] = False
                if change == 'unknown': build['unexpected'] = True
                if change == 'group': build['process_outcomes'][0]['group_empty_scans'] = [True, False]
                if change == 'command': build['commands'][1] += ['--jobs', '9']
                if change == 'test-name': build['tests']['required_test_names'].pop()
                if change == 'source':
                    path = Path(build['source_root']) / 'crates/sekirei-train/src/weekly_nonlinear_profile.rs'
                    path.write_bytes(b'changed compiled profile')
                if change == 'dependency':
                    put(Path(next(iter(deps))).parent / 'extra.rs', b'unbound package member')
                if change == 'prep-flags':
                    path = Path(build['source_root']).parent / 'source-preparation.json'
                    body = json.loads(path.read_bytes()); body['numeric_update_export_modules_preserved'] = 1
                    ref = put(path, body)
                    build['inputs_before'][ref['path']] = w.small(ref)
                    build['inputs_after'][ref['path']] = w.small(ref)
                with self.assertRaises(ValueError): self.run_build(root, build, plan_ref, plan, engine, deps)


if __name__ == '__main__':
    unittest.main()
