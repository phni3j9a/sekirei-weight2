"""Tiny source-only fixtures. All file identities are synthetic declarations."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from unittest import mock
import white_view_build_contract as c

ROOT=Path(__file__).resolve().parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('white_view_disabled_builder',ROOT/'prepare_white_view_runtime.py')
w=importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


def ident(label,bytes_=12): return {'bytes':bytes_,'sha256':hashlib.sha256(label.encode()).hexdigest()}
def full(path,label=None): return {'path':path,**ident(label or path)}


def fixture():
    """Exact structural example; no actual file existence/build claims."""
    files={p:ident('source:'+p) for p in c.PROTECTED|set(c.CHANGED)|{'Cargo.toml','rust-toolchain.toml',
        'synthetic/a/Cargo.toml','synthetic/b/Cargo.toml','synthetic/c/Cargo.toml'}}
    files.update(copy.deepcopy(c.ORIGINAL))
    for n in range(526-len(files)): files['synthetic/file%03d.rs'%n]=ident(str(n))
    deps={p:files[p] for p in ('Cargo.toml','Cargo.lock','rust-toolchain.toml',*c.CHANGED[1:],
                             'synthetic/a/Cargo.toml','synthetic/b/Cargo.toml','synthetic/c/Cargo.toml')}
    identity={'schema':'sekirei.white-view-build-identity.v1','status':'frozen-before-build','runtime':c.RUNTIME,
        'base_commit':c.BASE,'plan_sha256':c.PLAN,'lock_sha256':c.LOCK,'rustflags':c.FLAGS,'build_jobs':2,'producer_commit':'a'*40,
        'architecture':copy.deepcopy(c.ARCH),'budget':copy.deepcopy(c.BUDGET),'base_source':'/synthetic/base/source','teacher_runtime':'/synthetic/teacher',
        'patch':{'path':'/synthetic/patch','bytes':16686,'sha256':c.PATCH},'source_files':files,'dependency_files':deps,
        'sources':copy.deepcopy(c.SOURCES),'compiler':{'rustc':'synthetic rustc','cargo':'synthetic cargo','target':'x86_64-unknown-linux-gnu'},
        'compiler_files':{'/synthetic/tool/rustc':ident('rustc'),'/synthetic/tool/cargo':ident('cargo')},
        'helper_files':{'/synthetic/helper.py':ident('helper')},'g++':'synthetic legacy teacher g++',
        'teacher_manifest':full('/synthetic/teacher/build-manifest.json'),
        'teacher_declared_binary_paths':{p:'/synthetic/teacher-declared/'+p for p in ('sekirei-train','shogiesa','yaneuraou')},
        'teacher_files':{p:full('/synthetic/teacher-resolved/'+p) for p in c.TEACHER_FILES},
        'probe_sources':{p:full('/synthetic/'+p+'.rs') for p in c.PROBES}}
    identity['teacher_files'][c.TEACHER_WEIGHT].update(c.WEIGHT)
    for name in c.PROBES: identity['probe_sources'][name]['sha256']=c.PROBE_SOURCE_SHA[name]
    iref=full(c.RUNTIME+'/white-view-build-identity.json')
    external={'path':'/synthetic/frozen-identity.json',**{k:iref[k] for k in ('bytes','sha256')}}
    inputs=c.declared_identity_inputs(identity); inputs[external['path']]={k:external[k] for k in ('bytes','sha256')}
    after=copy.deepcopy(files); after.update(copy.deepcopy(c.PATCHED)); da={p:after[p] for p in deps}
    copies={p:{'source':rec,'source_before':rec,'source_after':rec,
        'copy':{'path':c.RUNTIME+'/'+p,**{k:rec[k] for k in ('bytes','sha256')}},'regular_file':True,'different_inode':True}
        for p,rec in identity['teacher_files'].items()}
    bins={'sekirei':full(c.RUNTIME+'/bin/sekirei'),**{p:copies['bin/'+p]['copy'] for p in ('sekirei-train','shogiesa','yaneuraou')}}
    rlib=c.RUNTIME+'/build/white-view/release/deps/libsekirei_core-synthetic.rlib'
    link={'rlib':full(rlib),'core_fingerprint':full(c.RUNTIME+'/build/white-view/release/.fingerprint/core/lib-sekirei_core'),
        'core_fingerprint_json':full(c.RUNTIME+'/build/white-view/release/.fingerprint/core/lib-sekirei_core.json'),
        'usi_fingerprint_json':full(c.RUNTIME+'/build/white-view/release/.fingerprint/usi/bin-sekirei.json'),
        'actual_usi_dependency_binary':{'path':c.RUNTIME+'/build/white-view/release/deps/sekirei-synthetic',**{k:bins['sekirei'][k] for k in ('bytes','sha256')}},
        'core_dependency_fingerprint':123,'actual_usi_core_fingerprint_verified':True,'core_features':[c.FEATURE],'panic_strategy':'abort','profile':'release'}
    steps=[]
    for name,argv in c.expected_commands(identity,rlib).items():
        rec={'name':name,'argv':argv,'returncode':0,'reaped':True,'group_stopped':True,'timeout':False,
             'cleanup_status':'ok','log':full(c.RUNTIME+'/'+name+'.log'),'wall_seconds':0.01}
        if name=='conflict': rec.update(returncode=101,expected_failure=True,compile_error_verified=True)
        steps.append(rec)
    release_map={link[p]['path']:{k:link[p][k] for k in ('bytes','sha256')} for p in ('rlib','actual_usi_dependency_binary')}
    v={'schema':'sekirei.white-view-runtime-build.v1','status':'complete','runtime':c.RUNTIME,'base_commit':c.BASE,
        'plan_sha256':c.PLAN,'patch_sha256':c.PATCH,'lock_sha256':c.LOCK,'rustflags':c.FLAGS,'build_jobs':2,'producer_commit':identity['producer_commit'],
        'architecture':copy.deepcopy(c.ARCH),'sources':identity['sources'],'compiler':identity['compiler'],
        'rustc':identity['compiler']['rustc'],'cargo':identity['compiler']['cargo'],'g++':identity['g++'],'identity':iref,'external_identity_input':external,
        'source_files_before':files,'source_files_after':after,'changed_source_paths':list(c.CHANGED),
        'dependency_files_before':deps,'dependency_files_after':da,'build_source_files_before':after,'build_source_files_after':after,
        'build_dependency_files_before':da,'build_dependency_files_after':da,'base_source_files_before':files,'base_source_files_after':files,
        'base_source_unchanged':True,'compiler_files':identity['compiler_files'],'inputs_before':inputs,
        'inputs_after':copy.deepcopy(inputs),'teacher_copies':copies,'binaries':bins,
        'rust_tests':{key:{'expected':names,'passed':names,'failed':0,'ignored':0} for key,names in
            (('default',c.OLD_TESTS),('white_view',sorted(c.OLD_TESTS+c.NEW_TESTS)),('b_small',c.OLD_TESTS))},
        'feature_conflict':next(x for x in steps if x['name']=='conflict'),'core_link':link,'steps':steps,
        'probe_binaries':{p:full(c.RUNTIME+'/probes/'+p) for p in c.PROBES},
        'release_dependencies_before_probes':release_map,'release_dependencies_after_probes':copy.deepcopy(release_map),
        'source_unchanged':True,'inputs_unchanged':True,'compiler_unchanged':True,'dependency_unchanged_during_build':True,
        'engine_or_model_probe_started':False,'fit_started':False,'teacher_rebuilt':False,'teacher_copy_method':'regular-byte-copy',
        'teacher_legacy_declared_binary_paths':identity['teacher_declared_binary_paths'],'budget':copy.deepcopy(c.BUDGET),
        'actual_added_bytes':4096,'remaining_free_bytes_before_manifest':3*2**30}
    return identity,iref,copy.deepcopy(v)


class BuildContract(unittest.TestCase):
    def check(self,mutate):
        i,r,v=fixture(); mutate(i,v)
        with self.assertRaises(ValueError): c.validate_manifest(v,i,r)

    def test_normal_full_manifest_and_feature_binding(self):
        i,r,v=fixture(); self.assertIs(c.validate_manifest(v,i,r),v)
        b=c.feature_binding_for_runtime({'manifest':v,'identity':i},'1'*64)
        self.assertEqual(b['core_binary_sha256'],v['binaries']['sekirei']['sha256'])
        self.assertEqual(b['stock_initializer_sha256'],'bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40')
        frozen=c.immutable_inputs_for_runtime({'manifest':v,'identity':i},full(c.RUNTIME+'/build-manifest.json'))
        self.assertIn(i['base_source']+'/Cargo.lock',frozen); self.assertIn(c.RUNTIME+'/source/'+c.CHANGED[0],frozen)
        self.assertIn(i['teacher_manifest']['path'],frozen)

    def test_all_three_entries_block_before_any_io(self):
        with mock.patch.object(Path,'read_bytes',side_effect=AssertionError('I/O')),mock.patch.object(w,'load_helpers',side_effect=AssertionError('import')):
            for function,args in ((w.main,()),(w.prepare,(None,)),(w.verify_runtime,('invalid','a','b'))):
                with self.subTest(function=function.__name__),self.assertRaisesRegex(ValueError,'before I/O'): function(*args)

    def test_architecture_magic_feature_bool_and_jobs_are_exact(self):
        self.check(lambda i,v:v['architecture'].update(native_magic='SEKIRW01'))
        self.check(lambda i,v:v['architecture'].update(king_relative_b_small=0))
        self.check(lambda i,v:v.update(build_jobs=True))

    def test_source_scope_and_protected_search(self):
        self.check(lambda i,v:v['source_files_after'].update({'crates/sekirei-core/src/search.rs':ident('changed')}))
        self.check(lambda i,v:v['source_files_after'][c.CHANGED[0]].update(sha256='2'*64))
        self.check(lambda i,v:v['source_files_before'].pop('synthetic/file001.rs'))

    def test_project_cargo_two_endpoints_and_external_dependencies(self):
        self.check(lambda i,v:v['dependency_files_after'].update({'Cargo.lock':ident('new lock')}))
        self.check(lambda i,v:v['dependency_files_after'].update({c.CHANGED[1]:v['dependency_files_before'][c.CHANGED[1]]}))
        self.check(lambda i,v:v['dependency_files_after'].pop('synthetic/a/Cargo.toml'))

    def test_source_dependency_and_compiler_build_changes(self):
        self.check(lambda i,v:v['build_source_files_after'].update({'synthetic/file001.rs':ident('changed')}))
        self.check(lambda i,v:v['build_dependency_files_after'].update({'Cargo.toml':ident('changed')}))
        self.check(lambda i,v:v['compiler_files'].update({'/synthetic/tool/rustc':ident('changed')}))

    def test_teacher_no_symlink_hardlink_rebuild_or_wrong_bytes(self):
        self.check(lambda i,v:v['teacher_copies'][c.TEACHER_WEIGHT].update(different_inode=1))
        self.check(lambda i,v:v['teacher_copies']['bin/yaneuraou']['copy'].update(sha256='3'*64))
        self.check(lambda i,v:v.update(teacher_rebuilt=True))
        self.check(lambda i,v:i['teacher_files'][c.TEACHER_WEIGHT].update(bytes=True))

    def test_probe_link_and_panic_and_membership(self):
        self.check(lambda i,v:v['core_link'].update(panic_strategy='unwind'))
        self.check(lambda i,v:v['core_link'].update(core_dependency_fingerprint=True))
        self.check(lambda i,v:v['release_dependencies_after_probes'].update({'/synthetic/new.rlib':ident('new')}))
        self.check(lambda i,v:v['steps'][-1]['argv'].remove('panic=abort'))

    def test_exact_tests_conflict_failure_and_supervision(self):
        self.check(lambda i,v:v['rust_tests']['white_view']['passed'].pop())
        self.check(lambda i,v:v['feature_conflict'].update(returncode=0))
        self.check(lambda i,v:v['steps'][0].update(reaped=1))
        self.check(lambda i,v:v['steps'][0].update(timeout=True))

    def test_json_pinned_raw_duplicate_nonfinite_and_bool_size(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}'):
            with self.assertRaises(ValueError): c.pinned_json(raw,hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError): c.pinned_json(b'{}','0'*64)
        with self.assertRaises(ValueError): c.info({'bytes':True,'sha256':'1'*64})

    def test_parent_cleanup_on_deferred_spawn_cancel_and_failure_receipt(self):
        child=mock.Mock(pid=123,returncode=None)
        cleanup=mock.Mock(side_effect=lambda obj:setattr(obj,'returncode',-15))
        def spawn(*args,**kwargs):
            handler=signal.getsignal(signal.SIGTERM); self.assertTrue(callable(handler)); handler(signal.SIGTERM,None); return child
        with tempfile.TemporaryDirectory() as directory,w.termination_guard() as state:
            path=Path(directory)/'synthetic.log'
            with mock.patch.object(w.subprocess,'Popen',side_effect=spawn):
                with self.assertRaisesRegex(ValueError,'cancelled') as captured:
                    w.run_step(['synthetic'],Path(directory),path,{},1,state,mock.Mock(cleanup=cleanup))
            cleanup.assert_called_once_with(child)
            outcome=json.loads(path.with_suffix('.outcome.json').read_bytes())
            self.assertEqual(outcome['cleanup_status'],'ok'); self.assertTrue(outcome['reaped']); self.assertTrue(outcome['group_stopped'])
            self.assertIs(captured.exception.build_step_outcome['reaped'],True)

    def test_cleanup_failure_is_not_masked_by_outcome_save_failure(self):
        child=mock.Mock(pid=123,returncode=0); child.wait.return_value=0
        with tempfile.TemporaryDirectory() as directory,w.termination_guard() as state:
            with mock.patch.object(w.subprocess,'Popen',return_value=child),mock.patch.object(w,'write_new',side_effect=OSError('outcome-save')):
                with self.assertRaisesRegex(RuntimeError,'cleanup-failed') as captured:
                    w.run_step(['synthetic'],Path(directory),Path(directory)/'synthetic.log',{},1,state,
                               mock.Mock(cleanup=mock.Mock(side_effect=RuntimeError('cleanup-failed'))))
            outcome=captured.exception.build_step_outcome
            self.assertEqual(outcome['cleanup_status'],'failed'); self.assertEqual(outcome['outcome_write_error'],'outcome-save')

    def test_regular_bytecopy_and_source_alias_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); src=root/'source.txt'; src.write_bytes(b'synthetic only')
            rec=w.copy_regular(src,root/'copy.txt'); self.assertTrue(rec['different_inode']); self.assertEqual((root/'copy.txt').read_bytes(),src.read_bytes())
            alias=root/'alias'; alias.symlink_to(src)
            with self.assertRaises(ValueError): w.copy_regular(alias,root/'bad.txt')
            self.assertFalse((root/'bad.txt').exists())

    def test_budget_rejects_no_reserve_without_any_build(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(w.shutil,'disk_usage',return_value=mock.Mock(free=1)):
                with self.assertRaisesRegex(ValueError,'budget exceeded'): w.budget_guard(Path(directory))

    def test_legacy_teacher_dict_contract_with_regular_synthetic_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); i,_,_=fixture(); i['teacher_runtime']=str(root)
            i['teacher_files']={}; i['teacher_declared_binary_paths']={}
            for name in c.TEACHER_FILES:
                path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'public synthetic fixture')
                i['teacher_files'][name]=w.ref(path)
                if name.startswith('bin/'): i['teacher_declared_binary_paths'][Path(name).name]=str(path)
            old={'lock_sha256':c.LOCK,'sources':i['sources'],'rustflags':c.FLAGS,'g++':i['g++'],
                 'binaries':{n:{'path':p,'sha256':i['teacher_files']['bin/'+n]['sha256']} for n,p in i['teacher_declared_binary_paths'].items()}}
            path=root/'build-manifest.json'; path.write_text(json.dumps(old)); i['teacher_manifest']=w.ref(path)
            w.teacher_guard(i)
            old['binaries']['yaneuraou']=old['binaries']['yaneuraou']['sha256']
            path.write_text(json.dumps(old)); i['teacher_manifest']=w.ref(path)
            with self.assertRaisesRegex(ValueError,'declared binary'): w.teacher_guard(i)


if __name__=='__main__': unittest.main()
