"""Only synthetic temporary bytes and mocked process/legacy parser objects."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import white_view_evaluation as w
import white_view_runtime_io as io
import white_view_production_ports as p
import white_view_operations as o
from test_white_view_evaluation import bundle, spec, draft, parent_docs, stage, config, UNLOADED

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parent/'white-view-source-fixtures-v1'


def new_ref(path, value):
    return io.write_new(path, io.json_bytes(value))


def terminal(role, model, ids):
    return {'schema': 'sekirei.white-view-evaluation-terminal.v1', 'status': 'observed-stopped',
        'exit_code': 0, 'session_closed': True, 'tool_observed_reaped': True,
        'all_related_groups_observed_stopped': True, 'role': role, 'model': model, 'run_ids': ids,
        'executor_session_id': 123}


class SourcePorts(unittest.TestCase):
    def test_v4_candidate_preflight_requires_v2_canonical_gate_worker(self):
        # All refs are fictitious typed declarations, never followed/read.
        s=spec('candidate')
        binding={'schema':'sekirei.white-view-candidate-gate-binding.v1',
            **{k:{'path':'/tmp/synthetic-'+k+'.json','bytes':1,'sha256':'a'*64} for k in w.GATE_BINDING_NAMES}}
        binding['gate']=s['candidate_gate']
        binding['gate_worker']={'path':str(p.C/'white-view-candidate-model-gate-worker-v2.py'),'bytes':1,'sha256':'a'*64}
        cfg={'path':str(p.R/'config/development-benchmark.json'),'bytes':1,'sha256':'a'*64}
        validator={'path':'/tmp/synthetic-build-validator.py','bytes':1,'sha256':'a'*64}
        for rec in [*binding.values(),cfg,validator]:
            if type(rec) is dict:s['inputs'][rec['path']]={k:rec[k] for k in ('bytes','sha256')}
        for name in p.PUBLIC_NAMES:
            path=str(p.R/'scripts'/(name+'.py'))
            s['source_helpers'][path]='a'*64;s['inputs'][path]={'bytes':1,'sha256':'a'*64}
        pf={'schema':'sekirei.white-view-evaluation-launch-preflight.v1','status':'frozen-before-first-write',
            'spec':s,'config_source':cfg,'new_build_binding':{'runtime':w.RUNTIME,
            'manifest':s['build_manifest'],'identity':s['build_identity'],'validator_source':validator},
            'repo_identity':{'commit':'a'*40,'dirty':False,'status_porcelain':''},
            'decoder_versions':{'numpy':'1.26.4','cshogi':'1.0.4'},'preregistration':binding['preregistration'],
            'candidate_binding':binding,'candidate_validator':binding['gate_worker']}
        self.assertIs(p.validate_launch_preflight(pf),pf)
        bad=copy.deepcopy(pf);bad['candidate_validator']['path']=str(p.C/'white-view-candidate-model-gate-worker-v1.py')
        bad['spec']['inputs'][bad['candidate_validator']['path']]={'bytes':1,'sha256':'a'*64}
        with self.assertRaisesRegex(ValueError,'dedicated new03 proof consumer'):
            p.validate_launch_preflight(bad)

    def test_actual_entry_stops_before_reads_and_processes(self):
        with (mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('read')),
             mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('process'))):
            for function, args in ((o.main, ()), (o.launch, (None,)), (o.audit, (None,)),
                (p.ProductionPorts, (None,)), (p.reparse_stage, (None,)*6), (p.audit_role, (None, None)),
                (p.read_audit_bundle, (None, None)), (p.read_verified_bridge, (None, None))):
                with self.subTest(function=function.__name__), self.assertRaisesRegex((ValueError, RuntimeError), 'SOURCE ONLY'):
                    function(*args)

    def test_reader_raw_sha_type_collision_and_symlink(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path = root/'toy.json'; ref = new_ref(path, {'typed': 1})
            reader = io.Reader(); self.assertEqual(reader.json_ref(ref), {'typed': 1})
            path.write_bytes(b'{"typed":true}')
            with self.assertRaises(ValueError): reader.json_ref(ref)
            bad = root/'link'; bad.symlink_to(path)
            with self.assertRaises(ValueError): io.info(bad)
            with self.assertRaises(ValueError): io.union_exact({'/tmp/synthetic': {'bytes': 1, 'sha256': 'a'*64}},
                {'/tmp/synthetic': {'bytes': True, 'sha256': 'a'*64}})

    def test_owned_immutable_outputs_and_atomic_state(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)/'own'; owner = io.OwnedOutput(root); owner.create()
            owner.write('evaluation.json', {'status': 'running'}, mutable=True)
            old = io.info(root/'evaluation.json')
            o.owned_state_replace(root/'evaluation.json', old, {'status': 'complete'})
            self.assertEqual(json.loads((root/'evaluation.json').read_bytes())['status'], 'complete')
            with self.assertRaises(ValueError): o.owned_state_replace(root/'evaluation.json', old, {})
            owner.write('immutable.json', {})
            with self.assertRaises(ValueError): owner.write('immutable.json', {}, mutable=True)
            with self.assertRaises(FileExistsError): io.write_new(root/'immutable.json', b'')

    def test_canonical_output_dual_overlap_and_unknown_git_failure(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); os.chmod(root, 0o700)
            with self.assertRaises(ValueError): io.private_output(root/'new', [root])
            git = mock.Mock(returncode=128, stderr=b'fatal: not a git repository')
            with mock.patch.object(io.subprocess, 'run', return_value=git), mock.patch('shutil.disk_usage', return_value=mock.Mock(free=2**40)):
                self.assertEqual(io.private_output(root/'new', []), root/'new')
            git.stderr = b'permission denied'
            with mock.patch.object(io.subprocess, 'run', return_value=git), self.assertRaises(ValueError):
                io.private_output(root/'new', [])

    def test_parent_spawn_cancel_retains_handle_then_reaps_no_signal_mask(self):
        child = mock.Mock(pid=9999, returncode=None)
        holder = {'process': None}; saved = []
        def spawn(*args, **kwargs):
            self.assertNotIn('preexec_fn', kwargs); self.assertEqual(kwargs['pass_fds'], (5,))
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return child
        def cleanup(process):
            self.assertIs(holder['process'], child); process.returncode = -15
            return {'waited': True, 'reaped': True, 'process_group_stopped': True}
        with self.assertRaises(io.Cancelled):
            io.supervise(['synthetic'], '/tmp', None, None, 1, holder, saved.append,
                popen=spawn, cleanup=cleanup, pass_fds=(5,))
        self.assertIsNone(holder['process']); self.assertEqual(len(saved), 1)
        self.assertTrue(saved[0]['reaped']); self.assertTrue(saved[0]['process_group_stopped'])

    def test_wait_failure_cleanup_failure_keeps_original_and_holder(self):
        child = mock.Mock(pid=9999, returncode=None)
        child.wait.side_effect = RuntimeError('original-wait')
        holder = {'process': None}; saved = []
        with self.assertRaisesRegex(RuntimeError, 'original-wait') as captured:
            io.supervise(['synthetic'], '/tmp', None, None, 1, holder, saved.append,
                popen=mock.Mock(return_value=child), cleanup=mock.Mock(side_effect=RuntimeError('cleanup')))
        self.assertIs(holder['process'], child); self.assertFalse(saved[0]['reaped'])
        self.assertFalse(saved[0]['process_group_stopped']); self.assertEqual(saved[0]['cleanup_error'], 'cleanup')
        self.assertEqual(captured.exception.parent_process_outcome['error'], 'original-wait')

    def test_failure_save_cannot_mask_primary_error(self):
        child = mock.Mock(pid=9999, returncode=1); child.wait.return_value = 1
        with self.assertRaisesRegex(ValueError, 'nonzero') as caught:
            io.supervise(['synthetic'], '/tmp', None, None, 1, {'process': None},
                mock.Mock(side_effect=OSError('save-error')), popen=mock.Mock(return_value=child),
                cleanup=lambda _: {'waited': True, 'reaped': True, 'process_group_stopped': True})
        self.assertEqual(caught.exception.failure_save_error, 'save-error')

    def test_success_wait_reap_and_bounded_cleanup(self):
        child = mock.Mock(pid=9999, returncode=0); child.wait.return_value = 0
        holder = {'process': None}
        value = io.supervise(['synthetic'], '/tmp', None, None, 1, holder,
            mock.Mock(side_effect=AssertionError('failure')), popen=mock.Mock(return_value=child),
            cleanup=lambda _: {'waited': True, 'reaped': True, 'process_group_stopped': True})
        self.assertIsNone(holder['process']); self.assertTrue(value['reaped'])
        self.assertFalse(value['timed_out']); self.assertEqual(value['returncode'], 0)
        with self.assertRaises(ValueError):
            io.bounded_cleanup(child, exists=lambda _: True, send=lambda *_: None,
                clock=mock.Mock(side_effect=(0.0, 6.0)), pause=mock.Mock())

    def test_reaped_leader_remaining_descendant_is_killed_then_observed_stopped(self):
        child = mock.Mock(pid=9999, returncode=0); child.wait.return_value = 0
        exists = mock.Mock(side_effect=(True, True, False)); sent = []
        result = io.bounded_cleanup(child, exists=exists, send=lambda pid, sig: sent.append((pid, sig)),
            clock=mock.Mock(return_value=0.0), pause=mock.Mock(side_effect=AssertionError('no real wait')))
        self.assertEqual(sent, [(9999, signal.SIGTERM), (9999, signal.SIGKILL)])
        self.assertEqual(result, {'waited': True, 'reaped': True, 'process_group_stopped': True})
        child.wait.assert_called_once_with(timeout=5)

    def test_kill_unverified_group_keeps_parent_holder(self):
        child = mock.Mock(pid=9999, returncode=0); child.wait.return_value = 0
        holder = {'process': None}; saved = []
        def cleanup(process):
            return io.bounded_cleanup(process, exists=lambda _: True, send=lambda *_: None,
                clock=mock.Mock(side_effect=(0.0, 6.0)), pause=mock.Mock())
        with self.assertRaisesRegex(ValueError, 'bounded KILL'):
            io.supervise(['synthetic'], '/tmp', None, None, 1, holder, saved.append,
                popen=mock.Mock(return_value=child), cleanup=cleanup)
        self.assertIs(holder['process'], child)
        self.assertFalse(saved[0]['process_group_stopped'])
        self.assertEqual(saved[0]['returncode'], 0)

    def test_candidate_common_cache_restored_and_exact_loaded_bytes_required(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);common=root/'proof_common.py';worker=root/'gate_worker.py'
            common.write_bytes(b'identity = "actual-source"\n')
            worker.write_bytes(b'import white_view_proof_common_v2 as p\n')
            cref={'path':str(common),**io.info(common)};wref={'path':str(worker),**io.info(worker)}
            old=types.SimpleNamespace(identity='unrelated-public-cache')
            names=('_white_view_gate_actual_common','_white_view_new03_candidate_gate')
            with mock.patch.dict(sys.modules, {'white_view_proof_common_v2':old}):
                for key in names:sys.modules.pop(key,None)
                reader=io.Reader()
                with p.candidate_validator(reader,{'common':cref},wref) as loaded:
                    self.assertEqual(loaded.p.identity,'actual-source')
                    self.assertIs(sys.modules['white_view_proof_common_v2'],old)
                    self.assertEqual(Path(loaded.p.__file__),common)
                self.assertIs(sys.modules['white_view_proof_common_v2'],old)
                wrong=types.SimpleNamespace(__file__=str(common))
                sys.modules[names[0]]=wrong
                with self.assertRaisesRegex(ValueError,'another source/bytes'):
                    with p.candidate_validator(io.Reader(),{'common':cref},wref):pass
                self.assertIs(sys.modules['white_view_proof_common_v2'],old)
                for key in names:sys.modules.pop(key,None)
                worker.write_bytes(b'import white_view_proof_common_v2 as p\nraise RuntimeError("synthetic-import")\n')
                failed={'path':str(worker),**io.info(worker)}
                with self.assertRaisesRegex(RuntimeError,'synthetic-import'):
                    with p.candidate_validator(io.Reader(),{'common':cref},failed):pass
                self.assertIs(sys.modules['white_view_proof_common_v2'],old)
                for key in names:sys.modules.pop(key,None)

    def test_legacy_public_transitive_cache_keeps_original_canonical_sha_guard(self):
        # Public top3 imports audit_pack transitively before its explicit loop
        # visit. It has no private-loader marker; old canonical+SHA rules still
        # apply. Private unique aliases above require their stronger marker.
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'legacy.py';path.write_bytes(b'synthetic = True\n')
            loaded=types.SimpleNamespace(__file__=str(path),synthetic=True)
            with mock.patch.dict(sys.modules,{'synthetic_public_legacy':loaded}):
                self.assertIs(p.load_source(io.Reader(),str(path),io.info(path),'synthetic_public_legacy'),loaded)
                bad=copy.deepcopy(io.info(path));bad['sha256']='0'*64
                with self.assertRaises(ValueError):p.load_source(io.Reader(),str(path),bad,'synthetic_public_legacy')

    def test_inherited_lock_is_same_open_description_and_fixed_contract(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path = root/'lock'; path.write_bytes(b'')
            with io.existing_lock(path, True) as stream, mock.patch.object(p, 'LOCKS', ((path, True),)):
                records = [{'path': str(path), 'exclusive': True, 'fd': stream.fileno()}]
                p.validate_inherited_locks(records)
                with self.assertRaises(BlockingIOError):
                    with io.existing_lock(path, True): pass
                records[0]['exclusive'] = 1
                with self.assertRaises(ValueError): p.validate_inherited_locks(records)

    def test_normal_parent_child_draft_wait_then_external_terminal_completion(self):
        """End-to-end boundary fixture; every process and runtime gate is fake."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); os.chmod(root, 0o700)
            cfgref = new_ref(root/'base-config.json', config())
            buildref = new_ref(root/'synthetic-build.json', {'synthetic': True})
            idref = new_ref(root/'synthetic-build-id.json', {'synthetic': True})
            inputs = {ref['path']: {k: ref[k] for k in ('bytes', 'sha256')} for ref in (cfgref, buildref, idref)}
            s = w.make_spec('fallback', 'development-synthetic-parent-v1', str(root/'evaluation'), UNLOADED,
                buildref, idref, {cfgref['path']: cfgref['sha256']}, inputs)
            pf = {'spec': s, 'config_source': cfgref}
            pfref = new_ref(root/'synthetic-preflight.json', pf)
            before = io.union_exact(inputs, {pfref['path']: {k: pfref[k] for k in ('bytes', 'sha256')}})
            locks = []
            for i in range(2):
                path = root/('serial-%d.lock' % i); path.write_bytes(b''); locks.append((path, True))
            observed = []
            class FakePorts:
                def __init__(inner, *_args, **_kwargs):
                    inner.reader = io.Reader()
                    for path, rec in before.items(): inner.reader.pin(path, rec)
                def coordinating_locks(inner, _spec):
                    from contextlib import nullcontext
                    return nullcontext()
                def preflight(inner, *_args):
                    for path, _ in locks:
                        with self.assertRaises(BlockingIOError):
                            with io.existing_lock(path, True): pass
                    observed.append('locked-preflight')
                def verify_sources(inner): observed.append('sources')
                def verify_new_runtime(inner, _config): observed.append('runtime')
            def fake_supervise(command, cwd, stdout, stderr, seconds, holder, save_failure, **kwargs):
                self.assertEqual(command[:2], [w.AUD, '-B']); self.assertEqual(len(kwargs['pass_fds']), 2)
                self.assertIn('child', command)
                # Pure generated draft is the result of the synthetic 4-stage
                # port, not a child claiming its own reaping.
                childdraft = w.four_stage_call_sequence(s, config(),
                    w.SyntheticPorts(inputs, {key: stage(key) for key in w.STAGES}))
                e = root/'evaluation'; e.mkdir(mode=0o700)
                io.write_new(e/'draft.json', io.json_bytes(childdraft))
                io.write_new(e/'evaluation.json', io.json_bytes({'status': 'measured-pending-parent-terminal'}))
                return {'status': 'finished', 'pid': 9999, 'pgid': 9999, 'returncode': 0,
                    'waited': True, 'reaped': True, 'process_group_stopped': True, 'timed_out': False}
            args = types.SimpleNamespace(preflight=pfref['path'], expected_preflight_sha256=pfref['sha256'],
                expected_worker_sha256='a'*64, parent_output=str(root/'parent'), timeout_seconds=1,
                expected_draft_sha256=None)
            def launch_reader(*_): return copy.deepcopy(pf), copy.deepcopy(pfref), io.Reader()
            with (mock.patch.object(o, 'PROTOTYPE_ONLY', False), mock.patch.object(w, 'PROTOTYPE_ONLY', False),
                 mock.patch.object(io, 'PROTOTYPE_ONLY', False), mock.patch.object(p, 'PROTOTYPE_ONLY', False),
                 mock.patch.object(o, 'read_launch', side_effect=launch_reader), mock.patch.object(p, 'LOCKS', tuple(locks)),
                 mock.patch.object(p, 'ProductionPorts', FakePorts), mock.patch.object(io, 'supervise', side_effect=fake_supervise),
                 mock.patch.object(io, 'private_output', side_effect=lambda path, _: Path(path))):
                parent = o.launch(args)
                self.assertFalse(parent['executor_tool_reaped']); self.assertTrue(parent['cleanup_verified'])
                self.assertEqual(observed[0], 'locked-preflight')
                pr = {'path': str(root/'parent/parent-outcome.json'), **io.info(root/'parent/parent-outcome.json')}
                termref = new_ref(root/'observed-terminal.json', terminal('fallback', UNLOADED, s['run_ids']))
                args.parent_outcome = pr['path']; args.expected_parent_outcome_sha256 = pr['sha256']
                args.terminal = termref['path']; args.expected_terminal_sha256 = termref['sha256']
                completed = o.complete(args)
                self.assertTrue(completed['executor_reaped']); self.assertTrue(completed['cleanup_verified'])
                self.assertEqual(completed['status'], 'complete')
                self.assertEqual(w.descriptor(completed, s['evaluation_path'], str(root/'evaluation/candidate.json'),
                    'fallback', UNLOADED)['runtime'], w.RUNTIME)
            for path, _ in locks:
                with io.existing_lock(path, True): pass

    def test_builder_full_normal_shape_compiler_and_cargo_endpoints(self):
        # These author-provided examples contain synthetic refs only. Never
        # follow /synthetic paths or actual build/model artifacts.
        identity = json.loads((BUILDER/'synthetic-build-identity.json').read_bytes())
        manifest = json.loads((BUILDER/'synthetic-build-manifest.json').read_bytes())
        self.assertTrue(w.validate_project_cargo_dependencies(manifest, identity))
        for kind in ('dependency_files_before', 'dependency_files_after'):
            bad = copy.deepcopy(manifest)
            key = next(iter(w.CARGO)); bad[kind][key]['sha256'] = '0'*64
            with self.assertRaises(ValueError): w.validate_project_cargo_dependencies(bad, identity)
        bad = copy.deepcopy(manifest); bad['compiler_files'][next(iter(bad['compiler_files']))]['sha256'] = '0'*64
        with self.assertRaises(ValueError): w.validate_project_cargo_dependencies(bad, identity)
        bad = copy.deepcopy(manifest); bad['dependency_files_after']['Cargo.lock']['sha256'] = '1'*64
        with self.assertRaises(ValueError): w.validate_project_cargo_dependencies(bad, identity)

    def test_adjacent_builder_module_cache_is_scoped_and_restored(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); contract = root/'white_view_build_contract.py'; worker = root/'builder.py'
            contract.write_bytes(b'VALUE=42\n')
            worker.write_bytes(b'import white_view_build_contract as c\ndef verify_runtime(*args): return c.VALUE\n')
            frozen = {str(path): io.info(path) for path in (contract, worker)}
            binding = {'validator_source': {'path': str(worker), **frozen[str(worker)]}}
            old = types.ModuleType('unrelated_public_contract')
            with mock.patch.dict(sys.modules, {'white_view_build_contract': old}):
                with p.runtime_validator(io.Reader(), binding, frozen, 'synthetic1') as (loaded, adjacent):
                    self.assertIs(loaded.c, adjacent); self.assertEqual(loaded.verify_runtime(), 42)
                    self.assertIs(sys.modules['white_view_build_contract'], old)
                self.assertIs(sys.modules['white_view_build_contract'], old)
            worker.write_bytes(b'import white_view_build_contract as c\nraise RuntimeError("toy-error")\n')
            frozen[str(worker)] = io.info(worker)
            with mock.patch.dict(sys.modules, {'white_view_build_contract': old}), self.assertRaisesRegex(RuntimeError, 'toy-error'):
                with p.runtime_validator(io.Reader(), binding, frozen, 'synthetic2'): pass
            self.assertNotIn('_white_view_actual_builder_synthetic2', sys.modules)
            for key in ('_white_view_adjacent_contract_synthetic1', '_white_view_actual_builder_synthetic1', '_white_view_adjacent_contract_synthetic2'):
                sys.modules.pop(key, None)

    def test_terminal_failure_or_unreaped_never_becomes_full_bundle(self):
        d, _ = draft(); good, _, _ = parent_docs(d)
        for key, value in (('exit_code', True), ('tool_observed_reaped', False), ('executor_session_id', True),
            ('session_closed', False), ('all_related_groups_observed_stopped', False)):
            bad = copy.deepcopy(good); bad[key] = value
            with self.assertRaises(ValueError): p.validate_audit_terminal(bad, d['role'], d['candidate_model'], d['run_ids'])

    def test_one_way_role_receipt_raw_membership_and_hash_are_rechecked(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); os.chmod(root, 0o700)
            runtime = root/'synthetic-runtime'; run_root = runtime/'runs'; run_root.mkdir(parents=True)
            value = bundle(runtime=str(runtime)); value.pop('audit_ref')
            ids = {s: 'synthetic-'+s for s in w.STAGES}
            dirs = []
            for runid in ids.values():
                directory = run_root/runid; directory.mkdir(); (directory/'raw.jsonl').write_bytes(b'synthetic events only\n'); dirs.append(directory)
            term = new_ref(root/'terminal.json', terminal('fallback', value['model'], ids))
            build = new_ref(runtime/'build-manifest.json', {'synthetic': True})
            inv = io.inventory(dirs, (term['path'], build['path']))
            value.update(run_ids=ids, terminal_ref=term, build_manifest=build, audit_inputs_before=inv['files'],
                audit_inputs_after=inv['files'], directory_inventory=inv['directories'])
            with (mock.patch.object(p, 'PROTOTYPE_ONLY', False), mock.patch.object(w, 'PROTOTYPE_ONLY', False),
                 mock.patch.object(io, 'PROTOTYPE_ONLY', False), mock.patch.object(io, 'private_output', return_value=root/'audit')):
                receipt = p.write_audit_receipt(root/'audit', value)
                self.assertNotIn('audit_ref', json.loads((root/'audit/reparsed-role-payload.json').read_bytes()))
                read = p.read_audit_bundle(io.Reader(), receipt)
                self.assertEqual(read['audit_ref'], receipt)
                (dirs[0]/'unexpected.json').write_bytes(b'{}')
                with self.assertRaises(ValueError): p.read_audit_bundle(io.Reader(), receipt)

    def test_all_prepared_sources_are_python_syntax_and_disabled(self):
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
        self.assertTrue(w.PROTOTYPE_ONLY and io.PROTOTYPE_ONLY and p.PROTOTYPE_ONLY and o.PROTOTYPE_ONLY)


if __name__ == '__main__': unittest.main()
