"""Synthetic memory, public tmp text and fake processes only; never actual NAS."""
import ast
import copy
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import white_view_archive_runtime_v1 as m
spec = importlib.util.spec_from_file_location('_white_archive_worker_fixture',HERE/'white-view-nas-archive-launch-worker-v1.py')
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
HELPER_SOURCE = HERE.parent/'white-view-source-fixtures-v1'/'archive-helper.py.txt'
# Historical generic helper is text-only. Compile only its original inventory
# functions and stdlib imports; its operational entry is never imported.
helper_tree = ast.parse(HELPER_SOURCE.read_bytes())
helper_nodes = [node for node in helper_tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                or isinstance(node, ast.FunctionDef) and node.name in
                {'fingerprint', 'inventory', 'combined_inventory'}]
h = SimpleNamespace()
exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(HELPER_SOURCE), 'exec'), h.__dict__)


def ident(size=4,digest='a'*64): return {'bytes':size,'sha256':digest}
def ref(path,size=4,digest='a'*64): return {'path':str(path),**ident(size,digest)}


def archive_fixture(adopt=False):
    c = m.c
    refs = {key:ref(path) for key,path in c.FIXED_REF_PATHS.items()}
    refs['factory_source'] = ref(c.C/'factory.py')
    gate_names = ('common','preregistration','activation','source_preflight','fit_run','native','metadata',
        'coefficients_f32','coefficients_f64','solver_certificate','reference03','numeric_audit','numeric_worker',
        'core','core_worker','incremental','incremental_worker','gate','gate_worker')
    binding = {'schema':'sekirei.white-view-candidate-gate-binding.v1'}
    binding.update({name:ref(c.C/('synthetic-'+name+'.bin')) for name in gate_names})
    binding['core'] = ref(c.C/'synthetic-core/receipt.json')
    binding['incremental'] = ref(c.C/'synthetic-incremental/receipt.json')
    model = {'kind':'nnue',**binding['native']}
    controls = {row['path']:row for row in [ref(c.OUT),ref(c.C/'stop-request.json'),*refs.values(),
                                           *[binding[k] for k in gate_names]]}
    inputs = {name:{k:row[k] for k in ('bytes','sha256')} for name,row in controls.items()}
    roots = []
    for role,runtime in (('old',c.B),('fallback',c.N),('candidate',c.N)):
        roots += [str(runtime/'runs'/('synthetic-'+role+'-'+stage)) for stage in ('mae-pilot','mae','top3-pilot','top3')]
    raw = {root+'/synthetic.txt':ident() for root in roots}; inputs.update(raw)
    stopped = {'schema':'sekirei.white-view-formal-stopped.v1','status':'verified-stopped-under-locks',
        'candidate':c.MODE,'source_head':c.HEAD,'model':model,'request':ref(c.C/'stop-request.json'),
        'refs':refs,'gate_binding':binding,'comparison_valid':True,'adopt':adopt,'all_groups_stopped':True,
        'inputs_unchanged':True,'inputs_before':copy.deepcopy(inputs),'inputs_after':copy.deepcopy(inputs),
        'source_originals_preserved':True,'final_used':False,'adoption_applied':False,'best_model_updated':False,
        'goal_complete':False,'audited_run_roots':roots,'audited_raw_files':raw}
    stop = ref(c.OUT); termref = ref(c.C/'synthetic-stop-terminal.json')
    terminal = {'schema':'sekirei.white-view-postformal-stop-terminal.v1','status':'observed-stopped',
        'stop_receipt':stop,'request':stopped['request'],'stop_session_id':123,'exit_code':0,'session_closed':True,
        'tool_observed_reaped':True,'all_related_groups_observed_stopped':True,'source_head':c.HEAD,
        'final_used':False,'goal_complete':False}
    mapping = {root:'runs/'+PurePosixPath(root).name for root in roots}
    mapping.update({str(c.Q):'fit/run',str(c.C/'synthetic-core'):'proofs/core',
        str(c.C/'synthetic-incremental'):'proofs/incremental',str(c.C/'synthetic-completion/control'):'completion/control'})
    source = {root:{'files':{'synthetic.txt':ident()},'directories':['.'],'symlinks':{}} for root in mapping}
    originals = {}; ledger = {}
    for i,row in enumerate(controls.values()):
        copied = ref(c.C/('synthetic-completion/control/%03d.bin'%i),row['bytes'],row['sha256'])
        ledger[row['path']] = copied
        source[str(c.C/'synthetic-completion/control')]['files']['%03d.bin'%i] = {k:row[k] for k in ('bytes','sha256')}
    combined = h.combined_inventory([(Path(root),PurePosixPath(target)) for root,target in sorted(mapping.items())],source)
    for root,target in mapping.items():
        for name in source[root]['files']: originals[str(PurePosixPath(target)/name)] = root+'/'+name
    selected = {'schema':'sekirei.white-view-selected-archive-inventory.v1','source_roots':mapping,
        'files':combined['files'],'directories':combined['directories'],'original_paths':originals,'original_copy_ledger':ledger}
    mapref=ref(c.C/'synthetic-map.json'); invref=ref(c.C/'synthetic-inventory.json')
    helper=ref(c.HELPER,16384,c.HELPER_SHA)
    for row in (termref,mapref,invref,helper): inputs[row['path']] = {k:row[k] for k in ('bytes','sha256')}
    for name,row in ledger.items(): inputs[row['path']] = {k:row[k] for k in ('bytes','sha256')}
    archived = set(originals.values())|set(ledger)
    deps = {name:row for name,row in inputs.items() if name not in archived}
    request = {'schema':'sekirei.white-view-archive-request.v1','status':'frozen-after-stopped-before-nas-copy',
        'candidate':c.MODE,'source_head':c.HEAD,'stop':stop,'stop_terminal':termref,'mapping':mapref,
        'closure_inventory':invref,'source_roots':mapping,'inputs':inputs,'required_archived_refs':list(controls.values()),
        'external_dependencies':deps,'destination':str(c.NAS/'archives/synthetic/new-white-archive'),
        'receipt_name':'synthetic-white-archive','helper':helper,
        'parent_exclusive_locks':list(map(str,c.ARCHIVE_OUTER_LOCKS)),
        'helper_exclusive_locks':list(map(str,c.ARCHIVE_HELPER_LOCKS)),
        'source_originals_preserved':True,'standalone_environment_restore':False,'final_used':False,
        'adoption_applied':False,'best_model_updated':False,'goal_complete':False}
    status = {'schema':'sekirei.private-archive.v1','status':'complete','phase':'verified',
        'destination':request['destination'],'mapping':mapping,'mapping_sha256':mapref['sha256'],
        'script_sha256':c.HELPER_SHA,'sources_deleted':False,'locks':list(map(str,c.ARCHIVE_HELPER_LOCKS)),
        'files':len(combined['files']),'bytes':sum(row['bytes'] for row in combined['files'].values()),
        'directories':len(combined['directories']),'symlinks':0}
    c.validate_archive_request(request,stopped,terminal,mapping,selected)
    return request,stopped,terminal,mapping,selected,source,combined,status


class FakeChild:
    pid = 777
    returncode = 0
    def __init__(self,fail=None): self.fail=fail; self.calls=[]
    def wait(self,timeout=None):
        self.calls.append(timeout)
        if self.fail: raise self.fail
        return self.returncode


class Tests(unittest.TestCase):
    def enabled(self):
        stack=ExitStack();stack.enter_context(mock.patch.object(m,'PROTOTYPE_ONLY',False))
        stack.enter_context(mock.patch.object(w,'PROTOTYPE_ONLY',False));return stack

    def run_launcher_fixture(self,fail=False,bad_status=False):
        q,st,t,mp,sel,src,full,status=archive_fixture()
        if bad_status:status['status']='running'
        local_sources=(HERE/'white-view-nas-archive-launch-worker-v1.py',Path(m.__file__),Path(m.c.__file__))
        fake_python=Path(str(m.PYTHON));fake_cfg=fake_python.parent.parent/'pyvenv.cfg'
        for pth in (*local_sources,fake_python,fake_cfg):
            q['inputs'][str(pth)]=ident(digest='b'*64)
            q['external_dependencies'][str(pth)]=ident(digest='b'*64)
        request_path=str(m.c.C/'synthetic-archive-request.json')
        qdocs={request_path:q,q['stop']['path']:st,q['stop_terminal']['path']:t,
            q['mapping']['path']:mp,q['closure_inventory']['path']:sel}
        receipt=Path(str(m.c.NAS))/'receipts'/q['receipt_name']
        for name,value in zip(('status.json','source-before.json','source-after.json','expected-destination.json','destination.json'),
                              (status,src,src,full,full)):qdocs[str(receipt/name)]=value
        args=SimpleNamespace(request=request_path,expected_request_bytes=4,expected_request_sha256='a'*64,
            expected_worker_sha256='b'*64,expected_runtime_source_sha256='b'*64,expected_contract_sha256='b'*64,
            expected_stop_session_id=123)
        held=[];events=[];saved={};child=FakeChild()
        @contextmanager
        def lock(pth):
            held.append(str(pth));events.append(('acquire',str(pth)))
            try:yield
            finally:events.append(('release',str(pth)));held.remove(str(pth))
        @contextmanager
        def no_signals():yield
        def info(pth):return q['inputs'].get(str(pth),ident())
        def save(pth,value):
            self.assertEqual(len(held),5);saved[str(pth)]=copy.deepcopy(value)
            return ref(pth)
        def run(cmd,holder,life):
            self.assertEqual(len(held),5);events.append(('run',len(held)));holder['child']=child
            life['child_spawned']=True
            if fail:raise RuntimeError('first wait failure')
            life.update(child_waited=True,child_reaped=True,helper_returncode=0,child_group_stopped=True,child_group_scans=[[],[]])
        def cleanup(holder,life):
            self.assertEqual(len(held),5);events.append(('cleanup',len(held)))
            if fail:
                life['cleanup_error']='RuntimeError: cleanup failure'
                raise RuntimeError('cleanup failure')
            holder['child']=None
        helper=SimpleNamespace(load_mapping=lambda _: [(Path(root),PurePosixPath(target)) for root,target in sorted(mp.items())],
                               inventory=lambda _:copy.deepcopy(full))
        patches=(mock.patch.object(m,'WORKER',local_sources[0]),mock.patch.object(m,'SOURCE',local_sources[1]),
            mock.patch.object(m,'CONTRACT',local_sources[2]),mock.patch.object(sys,'executable',str(fake_python)),
            mock.patch.object(sys,'prefix',str(fake_python.parent.parent)),mock.patch.object(Path,'resolve',lambda self,**kwargs:self),
            mock.patch.object(os.path,'lexists',return_value=False),mock.patch.object(m,'info',side_effect=info),
            mock.patch.object(m,'document',side_effect=lambda r:copy.deepcopy(qdocs[r['path']])),
            mock.patch.object(m,'read_ref',return_value=b'synthetic pinned bytes'),mock.patch.object(m,'verify_map',side_effect=lambda x:copy.deepcopy(x)),
            mock.patch.object(m,'exclusive_lock',side_effect=lock),mock.patch.object(m,'cancellation_handlers',side_effect=no_signals),
            mock.patch.object(m,'blocked_termination',side_effect=no_signals),mock.patch.object(m,'save_new',side_effect=save),
            mock.patch.object(m,'load_helper',return_value=helper),mock.patch.object(m,'inventory_sources',return_value=(src,full)),
            mock.patch.object(w,'run_helper',side_effect=run),mock.patch.object(w,'cleanup_held',side_effect=cleanup))
        caught=None;result=None
        with self.enabled(),ExitStack() as stack:
            for patch in patches:stack.enter_context(patch)
            try:result=w.launch(args)
            except BaseException as error:caught=error
        self.assertEqual(held,[])
        return result,caught,saved,events

    def test_real_launcher_body_keeps_all_five_locks_through_cleanup(self):
        result,error,saved,events=self.run_launcher_fixture()
        self.assertIsNone(error);self.assertEqual(result['status'],'complete-verified')
        self.assertTrue(result['child_reaped']);self.assertTrue(result['child_group_stopped'])
        self.assertEqual([n for n,v in events if n=='acquire'],['acquire']*5)
        self.assertLess(max(i for i,e in enumerate(events) if e[0]=='cleanup'),
                        min(i for i,e in enumerate(events) if e[0]=='release'))
        self.assertNotIn(str(m.FAILURE),saved)

    def test_real_launcher_body_preserves_original_failure_and_unconfirmed_cleanup(self):
        result,error,saved,events=self.run_launcher_fixture(fail=True)
        self.assertIsNone(result);self.assertEqual(str(error),'first wait failure')
        failed=saved[str(m.FAILURE)]
        self.assertEqual(failed['status'],'failed');self.assertIn('first wait failure',failed['error'])
        self.assertIn('cleanup failure',failed['cleanup_error']);self.assertTrue(failed['helper_handle_retained'])
        self.assertFalse(failed['child_reaped']);self.assertFalse(failed['child_group_stopped'])
        self.assertNotIn(str(m.RESULT),saved)
        self.assertLess(max(i for i,e in enumerate(events) if e[0]=='cleanup'),
                        min(i for i,e in enumerate(events) if e[0]=='release'))

    def test_real_launcher_body_never_promotes_running_helper_receipt(self):
        result,error,saved,events=self.run_launcher_fixture(bad_status=True)
        self.assertIsNone(result);self.assertIsInstance(error,ValueError)
        self.assertEqual(saved[str(m.FAILURE)]['status'],'failed');self.assertNotIn(str(m.RESULT),saved)

    def test_full_white_request_and_helper_receipts_positive_and_negative(self):
        for adopt in (False,True):
            q,st,t,mp,sel,src,full,status=archive_fixture(adopt)
            self.assertIs(m.c.validate_archive_request(q,st,t,mp,sel),q)
            stats=m.validate_helper_receipts(q,mp,sel,status,src,copy.deepcopy(src),full,copy.deepcopy(full))
            self.assertEqual(stats['files'],len(full['files']));self.assertIs(st['adopt'],adopt)

    def test_typed_terminal_and_old_candidate_schema_rejected(self):
        for key,value in (('exit_code',True),('exit_code',1),('tool_observed_reaped',False),('stop_session_id',True)):
            q,st,t,mp,sel,*_=archive_fixture();t[key]=value
            with self.assertRaises(ValueError):m.c.validate_archive_request(q,st,t,mp,sel)
        q,st,t,mp,sel,*_=archive_fixture();q['schema']='sekirei.paired-linear-archive-plan.v1'
        with self.assertRaises(ValueError):m.c.validate_archive_request(q,st,t,mp,sel)

    def test_full_copy_set_size_sha_directory_and_source_after_fail(self):
        q,st,t,mp,sel,src,full,status=archive_fixture()
        for change in ('hash','size','missing','extra','directory','symlink'):
            bad=copy.deepcopy(full)
            key=next(iter(bad['files']))
            if change=='hash':bad['files'][key]['sha256']='b'*64
            elif change=='size':bad['files'][key]['bytes']+=1
            elif change=='missing':bad['files'].pop(key)
            elif change=='extra':bad['files']['extra.txt']=ident()
            elif change=='directory':bad['directories'].append('extra')
            else:bad['symlinks']['unknown-link']='outside'
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.validate_helper_receipts(q,mp,sel,status,src,src,full,bad)
        after=copy.deepcopy(src);after[next(iter(after))]['files']['synthetic.txt']['bytes']+=1
        with self.assertRaises(ValueError):m.validate_helper_receipts(q,mp,sel,status,src,after,full,full)

    def test_symlink_in_source_is_never_accepted(self):
        q,st,t,mp,sel,src,full,status=archive_fixture()
        src[next(iter(src))]['symlinks']['link']='target'
        with self.assertRaises(ValueError):m.validate_source_inventory(mp,src,sel)

    def test_helper_status_flags_sha_mapping_locks_counts_fail_closed(self):
        q,st,t,mp,sel,src,full,status=archive_fixture()
        for key,value in (('status','failed'),('sources_deleted',True),('bytes',True),
            ('script_sha256','b'*64),('mapping_sha256','b'*64),('locks',status['locks']*2),('symlinks',1)):
            bad=copy.deepcopy(status);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_helper_receipts(q,mp,sel,bad,src,src,full,full)

    def test_all_actual_entries_stop_before_file_process_or_signal_io(self):
        callbacks=(w.main,lambda:w.launch(SimpleNamespace()),lambda:w.group_members(777),
            lambda:w.spawn_held([],{'child':None},w.lifecycle_new()),lambda:m.info('/never'),
            lambda:m.read_ref(ref('/never')),lambda:m.save_new(Path('/never'),{}),
            lambda:m.load_helper(ref('/never')))
        for fn in callbacks:
            with (mock.patch.object(Path,'read_bytes',side_effect=AssertionError('forbidden actual read')),
                 mock.patch.object(w.subprocess,'Popen',side_effect=AssertionError('forbidden actual process'))):
                with self.assertRaisesRegex(ValueError,'SOURCE ONLY'):fn()
        with mock.patch.object(signal,'pthread_sigmask',side_effect=AssertionError('forbidden actual signal')):
            with self.assertRaisesRegex(ValueError,'SOURCE ONLY'):
                with m.blocked_termination():pass

    def test_command_fixed_helper_interpreter_and_nonduplicated_lock_split(self):
        q,*_=archive_fixture();argv=m.command(q)
        self.assertEqual(argv[:3],[str(m.PYTHON),'-B',str(m.c.HELPER)])
        locks=[argv[i+1] for i,x in enumerate(argv) if x=='--lock']
        self.assertEqual(locks,list(map(str,m.c.ARCHIVE_HELPER_LOCKS)))
        self.assertTrue(set(locks).isdisjoint(map(str,m.c.ARCHIVE_OUTER_LOCKS)))
        self.assertEqual(len(set(locks)|set(map(str,m.c.ARCHIVE_OUTER_LOCKS))),9)

    def test_cancel_during_popen_records_handle_without_inherited_mask(self):
        holder={'child':None};life=w.lifecycle_new();handlers={};events=[];child=FakeChild()
        def set_handler(number,handler):handlers[number]=handler
        def mask(*args):events.append('mask');return set()
        def spawn(*args,**kwargs):
            self.assertEqual(events,[]);self.assertTrue(kwargs['start_new_session'])
            handlers[signal.SIGTERM](signal.SIGTERM,None)
            return child
        with (self.enabled(),mock.patch.object(signal,'getsignal',return_value=signal.SIG_DFL),
            mock.patch.object(signal,'signal',side_effect=set_handler),mock.patch.object(signal,'pthread_sigmask',side_effect=mask),
            mock.patch.object(w.subprocess,'Popen',side_effect=spawn)):
            with self.assertRaises(m.ArchiveCancelled):w.spawn_held(['synthetic'],holder,life)
        self.assertIs(holder['child'],child);self.assertTrue(life['child_spawned'])
        self.assertFalse(life['child_reaped']);self.assertEqual(events,['mask','mask'])

    def test_leader_reaped_but_descendants_get_kill_and_two_empty_scans(self):
        life=w.lifecycle_new();child=FakeChild()
        with (self.enabled(),mock.patch.object(w,'group_members',side_effect=[[777],[778],[],[],[]]),
            mock.patch.object(w.os,'killpg') as kill):
            w.stop_child_group(child,life)
        self.assertEqual([call.args[1] for call in kill.call_args_list],[signal.SIGTERM,signal.SIGKILL])
        self.assertTrue(life['child_waited']);self.assertTrue(life['child_reaped'])
        self.assertEqual(life['child_group_scans'],[[],[]]);self.assertTrue(life['child_group_stopped'])

    def test_cleanup_failure_keeps_handle_and_does_not_invent_reap(self):
        child=FakeChild(fail=RuntimeError('wait failed'));holder={'child':child};life=w.lifecycle_new()
        with (self.enabled(),mock.patch.object(w,'group_members',return_value=[777]),
            mock.patch.object(w.os,'killpg'),mock.patch.object(signal,'pthread_sigmask',return_value=set())):
            with self.assertRaisesRegex(RuntimeError,'wait failed'):w.cleanup_held(holder,life)
        self.assertIs(holder['child'],child);self.assertFalse(life['child_waited']);self.assertFalse(life['child_reaped'])
        self.assertFalse(life['child_group_stopped']);self.assertIn('wait failed',life['cleanup_error'])

    def test_scan_error_is_unconfirmed_and_handle_retained(self):
        child=FakeChild();holder={'child':child};life=w.lifecycle_new()
        with (self.enabled(),mock.patch.object(w,'group_members',side_effect=PermissionError('unknown scan')),
            mock.patch.object(signal,'pthread_sigmask',return_value=set())):
            with self.assertRaises(PermissionError):w.cleanup_held(holder,life)
        self.assertFalse(life['child_group_stopped']);self.assertEqual(life['child_group_scans'],[])
        self.assertIs(holder['child'],child)

    def test_wait_timeout_has_no_wait_reap_success(self):
        child=FakeChild(fail=subprocess.TimeoutExpired('synthetic',3600));holder={'child':None};life=w.lifecycle_new()
        def spawn(command,target,state):target['child']=child;state['child_spawned']=True
        with self.enabled(),mock.patch.object(w,'spawn_held',side_effect=spawn),mock.patch.object(w,'group_members') as scan:
            with self.assertRaises(subprocess.TimeoutExpired):w.run_helper([],holder,life)
        self.assertFalse(life['child_waited']);self.assertFalse(life['child_reaped']);scan.assert_not_called()

    def test_toy_helper_inventory_matches_selected_and_rejects_extra_directory(self):
        self.assertEqual(hashlib.sha256(HELPER_SOURCE.read_bytes()).hexdigest(),m.c.HELPER_SHA)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';root.mkdir();(root/'empty').mkdir();(root/'payload.txt').write_text('public fixture\n')
            src={str(root):h.inventory(root)};mapping={str(root):'nested/data'}
            selected=h.combined_inventory([(root,PurePosixPath('nested/data'))],src)
            with self.enabled():current,full=m.inventory_sources(h,mapping,selected)
            self.assertEqual(full,selected);self.assertEqual(current,src)
            (root/'unexpected').mkdir()
            with self.enabled(),self.assertRaises(ValueError):m.inventory_sources(h,mapping,selected)

    def test_output_controls_cannot_mutate_any_selected_root(self):
        q,st,t,mapping,*_=archive_fixture()
        for target in (Path(next(iter(mapping)))/'new-control.json',Path(str(m.c.Q)).parent):
            with self.assertRaises(ValueError):m.reject_output_overlap(target,mapping,q['inputs'])

    def test_toy_bound_reader_duplicate_json_symlink_and_new_writer(self):
        with tempfile.TemporaryDirectory() as directory,self.enabled():
            root=Path(directory);root.chmod(0o700);source=root/'source.json';raw=b'{"synthetic":true}\n';source.write_bytes(raw)
            bound=ref(source,len(raw),hashlib.sha256(raw).hexdigest());self.assertEqual(m.document(bound),{'synthetic':True})
            source.write_bytes(b'changed');
            with self.assertRaises(ValueError):m.read_ref(bound)
            link=root/'link';link.symlink_to(source)
            with self.assertRaises(ValueError):m.info(link)
            out=root/'new.json';value=m.save_new(out,{'synthetic':True});self.assertEqual(value['sha256'],m.info(out)['sha256'])
            with self.assertRaises(ValueError):m.save_new(out,{'synthetic':False})
        for raw in (b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":1e999}'):
            with self.assertRaises(ValueError):m.strict_json(raw)


if __name__=='__main__':unittest.main()
