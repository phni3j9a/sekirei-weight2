#!/usr/bin/env python3
"""Disabled dedicated white-view builder. Root alone may enable a new version.

No old source/runtime/helper is modified. No engine, model, dataset, search,
training or fit is executed. Cargo NNUE unit fixtures run only after activation.
verify_runtime returns actual manifest/identity dictionaries after byte checks.
"""
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import time
import white_view_build_contract as c

PROTOTYPE_ONLY = True
R = Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
BOUNDED_SHA = '863ab0c22a65120737df93d91cbd55b12bd19d21e4a7b2d7e560a7d9a1703731'
PRIVATE = Path('/home/server/.local/share/sekirei-weight2')
CONFLICT = 'nnue_white_view_aux_tied and king_relative_b_small are mutually exclusive'


def barrier():
    c.require(not PROTOTYPE_ONLY,'SOURCE ONLY prototype: actual build/runtime verification disabled before I/O')


def canonical(path,exists=True):
    path=Path(path); c.require(path.is_absolute() and path==path.resolve(strict=exists),'absolute canonical non-symlink path required'); return path


def info(path):
    path=canonical(path); c.require(stat.S_ISREG(path.stat().st_mode),'regular input file required')
    h=hashlib.sha256(); size=0
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk); size+=len(chunk)
    return {'bytes':size,'sha256':h.hexdigest()}


def ref(path): return {'path':str(canonical(path)),**info(path)}


def match(path,record):
    c.require(c.exact(info(path),record),'actual file bytes differ: '+str(path))


def read_pinned(path,expected):
    path=canonical(path); raw=path.read_bytes(); result=c.pinned_json(raw,expected)
    c.require(info(path)['sha256']==expected,'file changed while parsing: '+str(path)); return result


def write_new(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def load_helpers():
    path=R/'scripts/prepare_bounded.py'; c.require(info(path)['sha256']==BOUNDED_SHA,'unchanged shared lifecycle/path helper required')
    spec=importlib.util.spec_from_file_location('_white_view_mode_independent_lifecycle',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


@contextmanager
def locked(path,exclusive):
    fd=os.open(path,os.O_RDWR|os.O_CREAT,0o600)
    try:
        fcntl.flock(fd,(fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)|fcntl.LOCK_NB); yield
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN); os.close(fd)


class BuildCancelled(BaseException): pass


@contextmanager
def termination_guard():
    """Same deferred spawn/cleanup policy as the unchanged train_cpu helper."""
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    state={'spawning':False,'cleaning':False,'signal':None}
    def cancel(sig,_frame):
        state['signal']=sig
        if not state['spawning'] and not state['cleaning']: raise BuildCancelled('cancelled by signal '+str(sig))
    try:
        for s in previous: signal.signal(s,cancel)
        yield state
    finally:
        for s,handler in previous.items(): signal.signal(s,handler)


def pending(state): c.require(state['signal'] is None,'cancelled; completed build receipt prohibited')


def run_step(argv,cwd,log_path,env,seconds,state,lifecycle,*,expected_failure=False):
    """The child handle is owned here before a deferred signal can raise."""
    child=None; record={'name':log_path.stem,'argv':list(argv),'timeout':False,'reaped':False,'group_stopped':False,'cleanup_status':'pending'}
    started=time.monotonic(); raised=None
    try:
        with log_path.open('xb') as log:
            try:
                state['spawning']=True
                try: child=subprocess.Popen(argv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                finally: state['spawning']=False
                pending(state)
                try: record['returncode']=child.wait(timeout=seconds)
                except subprocess.TimeoutExpired: record['timeout']=True; raise
            finally:
                state['cleaning']=True
                try:
                    lifecycle.cleanup(child)
                    record['reaped']=child is not None and child.returncode is not None
                    record['group_stopped']=True; record['cleanup_status']='ok'
                except BaseException as error:
                    record['cleanup_status']='failed'; record['cleanup_error']=str(error); raise
                finally: state['cleaning']=False
        pending(state)
        c.require((record['returncode']!=0) if expected_failure else (record['returncode']==0),'build/test command outcome differs')
        return record
    except BaseException as error:
        raised=error; error.build_step_outcome=record; raise
    finally:
        record['wall_seconds']=time.monotonic()-started
        if log_path.exists(): record['log']=ref(log_path)
        # A separate outcome always survives a child/cleanup exception.
        outcome=log_path.with_suffix('.outcome.json')
        state['cleaning']=True
        try:
            try: write_new(outcome,record)
            except BaseException as error:
                record['outcome_write_error']=str(error)
                if raised is None: raise
        finally: state['cleaning']=False


def source_identity(source,lifecycle,*,patched):
    c.require(lifecycle.git(source,'rev-parse','HEAD').decode().strip()==c.BASE,'fixed source commit required')
    c.require(not lifecycle.git(source,'diff','--cached','--name-only','-z'),'staged source edits prohibited')
    names=lifecycle.paths_from_nul(lifecycle.git(source,'ls-files','-z'))
    c.require(len(names)==526 and len(set(names))==526,'complete 526 tracked source files required')
    changed=set(lifecycle.paths_from_nul(lifecycle.git(source,'diff','--name-only','-z',c.BASE)))
    c.require(changed==(set(c.CHANGED) if patched else set()),'source edits outside frozen three-file patch')
    c.require(not lifecycle.git(source,'ls-files','--others','--exclude-standard','-z'),'untracked source input prohibited')
    files={name:info(source/name) for name in names}
    deps={name:files[name] for name in names if Path(name).name in ('Cargo.toml','Cargo.lock','rust-toolchain','rust-toolchain.toml')}
    c.require(len(deps)==8,'complete fixed dependency inventory required')
    expected=c.PATCHED if patched else c.ORIGINAL
    c.require(c.exact({name:files[name] for name in c.CHANGED},expected),'fixed source implementation SHA differs')
    return files,deps


def inventory(root):
    root=canonical(root); files={}
    for path in sorted(root.rglob('*')):
        c.require(not path.is_symlink(),'symlink in dedicated release dependency inventory')
        if path.is_file(): files[str(path)]=info(path)
    c.require(files,'release dependency inventory missing'); return files


def identity_inputs(identity,external_path):
    c.require(Path(c.__file__).resolve()==Path(__file__).resolve().with_name('white_view_build_contract.py'),'dedicated adjacent contract import required; no module substitution')
    pins={str(external_path):info(external_path),str(Path(__file__).resolve()):info(Path(__file__).resolve()),
          str(Path(c.__file__).resolve()):info(Path(c.__file__).resolve()),str(R/'config/toolchain.lock.json'):info(R/'config/toolchain.lock.json'),
          str(R/'scripts/prepare_bounded.py'):info(R/'scripts/prepare_bounded.py')}
    for name in ('helper_files','compiler_files'):
        for path,frozen in identity[name].items(): match(path,frozen); pins[path]=frozen
    for item in [identity['patch'],identity['teacher_manifest'],*identity['teacher_files'].values(),*identity['probe_sources'].values()]:
        path=item['path']; frozen={k:item[k] for k in ('bytes','sha256')}; match(path,frozen)
        c.require(path not in pins or c.exact(pins[path],frozen),'frozen input key collision'); pins[path]=frozen
    for path in (str(Path(__file__).resolve()),str(Path(c.__file__).resolve())):
        c.require(c.exact(identity['helper_files'].get(path),pins[path]),'both dedicated worker and contract must be externally pinned')
    c.require(pins[str(R/'config/toolchain.lock.json')]['sha256']==c.LOCK,'public source lock changed')
    return pins


def teacher_guard(identity):
    teacher=canonical(identity['teacher_runtime']); old=c.pinned_json(Path(identity['teacher_manifest']['path']).read_bytes(),identity['teacher_manifest']['sha256'])
    c.fixed(old,{'lock_sha256':c.LOCK,'sources':identity['sources'],'rustflags':c.FLAGS,'g++':identity['g++']})
    for name in ('sekirei-train','shogiesa','yaneuraou'):
        src=canonical((teacher/'bin'/name).resolve(strict=True))
        c.require(c.exact(ref(src),identity['teacher_files']['bin/'+name]),'canonical teacher binary binding differs')
        declaration=old['binaries'].get(name)
        c.require(c.exact(declaration,{'path':identity['teacher_declared_binary_paths'][name],
                                      'sha256':identity['teacher_files']['bin/'+name]['sha256']}),'legacy teacher declared binary identity differs')
        c.require(canonical(Path(declaration['path']).resolve(strict=True))==src,'legacy declared binary resolves to a different file')
    c.require(c.exact(ref(canonical((teacher/c.TEACHER_WEIGHT).resolve(strict=True))),identity['teacher_files'][c.TEACHER_WEIGHT]),'legacy Beta weight source differs')


def copy_regular(source,destination):
    source=canonical(source); before=ref(source); destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with source.open('rb') as inp,destination.open('xb') as out:
        shutil.copyfileobj(inp,out,length=1024*1024); out.flush(); os.fsync(out.fileno())
    os.chmod(destination,0o700 if os.access(source,os.X_OK) else 0o600)
    c.require(info(destination)=={k:before[k] for k in ('bytes','sha256')} and ref(source)==before,'regular byte-copy changed source/output')
    c.require((source.stat().st_dev,source.stat().st_ino)!=(destination.stat().st_dev,destination.stat().st_ino),'hardlink copy prohibited')
    return {'source':before,'source_before':before,'source_after':ref(source),'copy':ref(destination),'regular_file':True,'different_inode':True}


def budget_guard(runtime):
    total=sum(p.stat().st_size for p in runtime.rglob('*') if p.is_file() and not p.is_symlink())
    c.require(total<=c.BUDGET['maximum_added_bytes'] and shutil.disk_usage(runtime.parent).free>=c.BUDGET['minimum_remaining_bytes'],'8 GiB added/2 GiB reserve build budget exceeded')
    return total


def select_core_link(artifact_log,source,target):
    artifacts=[]
    for line in artifact_log.read_bytes().splitlines():
        try: row=c.pinned_json(line,hashlib.sha256(line).hexdigest())
        except (ValueError,json.JSONDecodeError): continue
        if row.get('reason')=='compiler-artifact': artifacts.append(row)
    cores=[x for x in artifacts if x.get('target',{}).get('name')=='sekirei_core' and x.get('manifest_path')==str(source/'crates/sekirei-core/Cargo.toml') and x.get('profile',{}).get('test') is False]
    usis=[x for x in artifacts if x.get('target',{}).get('name')=='sekirei' and x.get('manifest_path')==str(source/'crates/sekirei-usi/Cargo.toml') and x.get('profile',{}).get('test') is False and x.get('executable')==str(target/'release/sekirei')]
    c.require(len(cores)==len(usis)==1,'actual release USI/core artifacts ambiguous')
    c.require(set(cores[0].get('features',[]))=={c.FEATURE} and set(usis[0].get('features',[]))=={c.FEATURE},'actual USI/core compile features differ')
    paths=[Path(p) for p in cores[0]['filenames'] if p.endswith('.rlib')]; c.require(len(paths)==1,'actual core rlib ambiguous')
    rlib=canonical(paths[0]); tag=rlib.name.removeprefix('libsekirei_core-').removesuffix('.rlib')
    fp=target/'release/.fingerprint'/('sekirei-core-'+tag); corefp=canonical(fp/'lib-sekirei_core')
    corejson=canonical(fp/'lib-sekirei_core.json'); core_record=c.pinned_json(corejson.read_bytes(),info(corejson)['sha256'])
    c.require(json.loads(core_record['features'])==[c.FEATURE],'core fingerprint features differ')
    value=corefp.read_text().strip(); c.require(len(value)==16 and all(x in '0123456789abcdef' for x in value),'malformed core fingerprint')
    fingerprint=int.from_bytes(bytes.fromhex(value),'little')
    matches=[]
    for path in (target/'release/.fingerprint').glob('sekirei-*/bin-sekirei.json'):
        raw=path.read_bytes(); record=c.pinned_json(raw,hashlib.sha256(raw).hexdigest())
        dep=[d for d in record.get('deps',[]) if type(d) is list and len(d)>=4 and d[1]=='sekirei_core' and type(d[3]) is int and d[3]==fingerprint]
        if len(dep)==1 and json.loads(record['features'])==[c.FEATURE]: matches.append(path)
    c.require(len(matches)==1,'actual USI must bind selected core fingerprint uniquely')
    usijson=canonical(matches[0]); tag=usijson.parent.name.removeprefix('sekirei-')
    depbinary=canonical(target/'release/deps'/('sekirei-'+tag)); match(depbinary,info(target/'release/sekirei'))
    return {'rlib':ref(rlib),'core_fingerprint':ref(corefp),'core_fingerprint_json':ref(corejson),
        'usi_fingerprint_json':ref(usijson),'actual_usi_dependency_binary':ref(depbinary),
        'core_dependency_fingerprint':fingerprint,'actual_usi_core_fingerprint_verified':True,
        'core_features':[c.FEATURE],'panic_strategy':'abort','profile':'release'}


def verify_runtime(runtime,expected_manifest_sha256,expected_identity_sha256):
    """Caller holds .build/.prepare shared locks until dependent work ends.

    The teacher runtime/source remains protected by Root's serial operational
    lock set. All declarations here are checked against actual file bytes.
    """
    barrier(); runtime=canonical(runtime); c.require(str(runtime)==c.RUNTIME,'dedicated runtime required')
    lifecycle=load_helpers(); identity_path=runtime/'white-view-build-identity.json'
    identity=c.validate_identity(read_pinned(identity_path,expected_identity_sha256))
    value=read_pinned(runtime/'build-manifest.json',expected_manifest_sha256)
    c.validate_manifest(value,identity,ref(identity_path))
    c.require(c.exact(identity_inputs(identity,canonical(value['external_identity_input']['path'])),value['inputs_before']),'runtime producer/helper actual binding differs')
    files,deps=source_identity(runtime/'source',lifecycle,patched=True)
    c.require(c.exact(files,value['source_files_after']) and c.exact(deps,value['dependency_files_after']),'actual source/dependency inventory differs')
    c.require(lifecycle.compiler_versions()=={k:identity['compiler'][k] for k in ('rustc','cargo')}
        and c.exact(lifecycle.compiler_files(),identity['compiler_files']),'actual compiler changed')
    for path,frozen in value['inputs_before'].items(): match(path,frozen)
    teacher_guard(identity)
    basefiles,basedeps=source_identity(canonical(identity['base_source']),lifecycle,patched=False)
    c.require(c.exact(basefiles,identity['source_files']) and c.exact(basedeps,identity['dependency_files']),'actual protected old base source changed')
    for record in [*value['binaries'].values(),*value['probe_binaries'].values(),*value['core_link'].values()]:
        if type(record) is dict and set(record)=={'path','bytes','sha256'}: match(record['path'],{k:record[k] for k in ('bytes','sha256')})
    c.require(all(os.access(x['path'],os.X_OK) for x in [*value['binaries'].values(),*value['probe_binaries'].values()]),'actual binary is not executable')
    for rec in value['teacher_copies'].values():
        copy=canonical(rec['copy']['path']); src=canonical(rec['source']['path'])
        match(copy,{k:rec['copy'][k] for k in ('bytes','sha256')}); c.require((copy.stat().st_dev,copy.stat().st_ino)!=(src.stat().st_dev,src.stat().st_ino),'copied file replaced by hardlink')
    for step in value['steps']: match(step['log']['path'],{k:step['log'][k] for k in ('bytes','sha256')})
    link=select_core_link(runtime/'usi-build.log',runtime/'source',runtime/'build/white-view')
    c.require(c.exact(link,value['core_link']),'actual USI/core linking evidence changed')
    c.require(c.exact(inventory(runtime/'build/white-view/release/deps'),value['release_dependencies_after_probes']),'release dependencies changed')
    return {'manifest':value,'identity':identity}


def prepare(args):
    barrier(); os.umask(0o077)
    runtime=canonical(args.runtime,False); c.require(str(runtime)==c.RUNTIME and not runtime.exists(),'fresh dedicated fixed runtime required')
    parent=canonical(runtime.parent); lifecycle=load_helpers(); lifecycle.outside_git(parent)
    identity_input=canonical(args.identity); identity=c.validate_identity(read_pinned(identity_input,args.expected_identity_sha256))
    base=canonical(identity['base_source']); teacher=canonical(identity['teacher_runtime'])
    c.require(runtime.is_relative_to(PRIVATE) and not any(runtime.is_relative_to(p) or p.is_relative_to(runtime) for p in (base,teacher,R,identity_input)),'new runtime overlaps a protected input')
    pins=identity_inputs(identity,identity_input)
    c.require(pins[str(Path(__file__).resolve())]['sha256']==args.expected_worker_sha256
        and pins[str(Path(c.__file__).resolve())]['sha256']==args.expected_contract_sha256,'external worker/contract SHA mismatch')
    copy_bytes=sum(x['bytes'] for x in identity['teacher_files'].values())
    c.require(copy_bytes<c.BUDGET['maximum_added_bytes'] and shutil.disk_usage(parent).free>=max(4*2**30+copy_bytes,c.BUDGET['maximum_added_bytes'])+c.BUDGET['minimum_remaining_bytes'],'build budget plus regular teacher-copy bytes and reserve required')
    runtime.mkdir(mode=0o700)
    with ExitStack() as locks,termination_guard() as state:
        for path,exclusive in ((teacher/'.prepare.lock',False),(teacher/'.benchmark.lock',True),
                               (runtime/'.build.lock',True),(runtime/'.prepare.lock',True)):
            locks.enter_context(locked(path,exclusive))
        record={'schema':'sekirei.white-view-runtime-build-attempt.v1','status':'running','runtime':str(runtime),'steps':[]}
        try:
            # Reparse all external input bindings while the coordinating locks hold.
            locked_identity=c.validate_identity(read_pinned(identity_input,args.expected_identity_sha256))
            c.require(c.exact(locked_identity,identity) and c.exact(identity_inputs(identity,identity_input),pins),'inputs changed before lock acquisition')
            producer_commit=lifecycle.git(R,'rev-parse','HEAD').decode().strip()
            c.require(producer_commit==identity['producer_commit'],'build producer commit differs from frozen identity')
            teacher_guard(identity)
            base_before,base_deps=source_identity(base,lifecycle,patched=False)
            c.require(c.exact(base_before,identity['source_files']) and c.exact(base_deps,identity['dependency_files']),'actual original base source differs')
            c.require(lifecycle.compiler_versions()=={k:identity['compiler'][k] for k in ('rustc','cargo')}
                and c.exact(lifecycle.compiler_files(),identity['compiler_files']),'compiler identity changed')
            env=dict(os.environ,LC_ALL='C',CARGO_INCREMENTAL='0',RUSTFLAGS=c.FLAGS,CARGO_BUILD_JOBS='2',RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
            for name in ('RUSTC','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_RUSTFLAGS','CARGO_BUILD_TARGET','CARGO_TARGET_DIR'):
                env.pop(name,None)
            compiler_dirs={str(Path(p).parent) for p in identity['compiler_files']}
            c.require(len(compiler_dirs)==1 and {Path(p).name for p in identity['compiler_files']}=={'rustc','cargo'},'fixed actual Rust/Cargo executable directory required')
            env['PATH']=next(iter(compiler_dirs))+os.pathsep+os.environ.get('PATH','')
            for name in list(env):
                if name.startswith(('CARGO_PROFILE_','CARGO_TARGET_')) or name in ('RUSTC_BOOTSTRAP','RUSTUP_TOOLCHAIN'): env.pop(name)
            def step(argv,cwd,name,seconds=1200,fail=False):
                result=run_step(argv,cwd,runtime/(name+'.log'),env,seconds,state,lifecycle,expected_failure=fail); record['steps'].append(result); budget_guard(runtime); return result
            source=runtime/'source'
            step(['git','clone','--no-hardlinks','--no-checkout',str(base),str(source)],runtime,'clone',120)
            step(['git','checkout','--detach',c.BASE],source,'checkout',120)
            before,db=source_identity(source,lifecycle,patched=False)
            c.require(c.exact(before,identity['source_files']) and c.exact(db,identity['dependency_files']),'original source inventory differs')
            step(['git','apply','--check',identity['patch']['path']],source,'patch-check',120)
            step(['git','apply',identity['patch']['path']],source,'patch-apply',120)
            after,da=source_identity(source,lifecycle,patched=True)
            transition={'source_files_before':before,'source_files_after':after,'dependency_files_before':db,'dependency_files_after':da,
                'changed_source_paths':list(c.CHANGED),'build_source_files_before':after,'build_source_files_after':after,
                'build_dependency_files_before':da,'build_dependency_files_after':da}
            c.validate_source_transition(transition,identity)
            tests={}
            for key,features,names in (('default',[],c.OLD_TESTS),('white_view',[c.FEATURE],sorted(c.OLD_TESTS+c.NEW_TESTS)),('b_small',['king_relative_b_small'],c.OLD_TESTS)):
                argv=['cargo','test','--release','--locked','--offline','-j','2','-p','sekirei-core','--lib','--target-dir',str(runtime/'build'/key)]
                if features: argv+=['--features',','.join(features)]
                argv+=['nnue::','--','--test-threads=1']; step(argv,source,key+'-tests')
                tests[key]=lifecycle.parse_tests(runtime/(key+'-tests.log'),names)
            conflict=step(['cargo','check','--release','--locked','--offline','-j','2','-p','sekirei','--features',c.FEATURE+',king_relative_b_small','--target-dir',str(runtime/'build/conflict')],source,'conflict',fail=True)
            c.require(CONFLICT in (runtime/'conflict.log').read_text(),'dual-feature failed for an unrelated reason')
            conflict.update(expected_failure=True,compile_error_verified=True)
            target=runtime/'build/white-view'
            argv=['cargo','build','--release','--locked','--offline','-j','2','-p','sekirei','--bin','sekirei','--features',c.FEATURE,'--target-dir',str(target),'--message-format=json-render-diagnostics']
            step(argv,source,'usi-build')
            link=select_core_link(runtime/'usi-build.log',source,target)
            copies={n:copy_regular(rec['path'],runtime/n) for n,rec in identity['teacher_files'].items()}
            usi_copy=copy_regular(target/'release/sekirei',runtime/'bin/sekirei')
            dep_before=inventory(target/'release/deps'); probes={}
            for name,src in identity['probe_sources'].items():
                output=runtime/'probes'/name; output.parent.mkdir(exist_ok=True,mode=0o700)
                argv=['rustc','--edition=2024','-C','opt-level=2','-C','target-cpu=x86-64-v3','-C','panic=abort','--extern','sekirei_core='+link['rlib']['path'],'-L','dependency='+str(target/'release/deps'),src['path'],'-o',str(output)]
                step(argv,runtime,name+'-compile',120); probes[name]=ref(output)
            dep_after=inventory(target/'release/deps'); c.require(c.exact(dep_before,dep_after),'probe compile changed release dependency files')
            final_source,final_deps=source_identity(source,lifecycle,patched=True)
            base_after,base_deps_after=source_identity(base,lifecycle,patched=False)
            c.require(c.exact(base_before,base_after) and c.exact(base_deps,base_deps_after),'protected old base source changed during build')
            transition['build_source_files_after']=final_source; transition['build_dependency_files_after']=final_deps
            pin_after={path:info(path) for path in pins}; c.require(c.exact(pins,pin_after),'immutable build input changed')
            c.require(c.exact(lifecycle.compiler_files(),identity['compiler_files']),'compiler changed during build')
            pending(state)
            copy_regular(identity_input,runtime/'white-view-build-identity.json')
            identity_ref=ref(runtime/'white-view-build-identity.json')
            bins={'sekirei':usi_copy['copy'],**{n:copies['bin/'+n]['copy'] for n in ('sekirei-train','shogiesa','yaneuraou')}}
            manifest={'schema':'sekirei.white-view-runtime-build.v1','status':'complete','runtime':str(runtime),'base_commit':c.BASE,
                'plan_sha256':c.PLAN,'patch_sha256':c.PATCH,'lock_sha256':c.LOCK,'sources':identity['sources'],'rustflags':c.FLAGS,
                'rustc':identity['compiler']['rustc'],'cargo':identity['compiler']['cargo'],'g++':identity['g++'],'compiler':identity['compiler'],
                'build_jobs':2,'architecture':c.ARCH,'identity':identity_ref,'external_identity_input':ref(identity_input),**transition,'compiler_files':identity['compiler_files'],
                'binaries':bins,'teacher_copies':copies,'teacher_rebuilt':False,'teacher_copy_method':'regular-byte-copy',
                'rust_tests':tests,'feature_conflict':conflict,'probe_binaries':probes,'core_link':link,
                'release_dependencies_before_probes':dep_before,'release_dependencies_after_probes':dep_after,'steps':record['steps'],
                'source_unchanged':True,'inputs_unchanged':True,'compiler_unchanged':True,'dependency_unchanged_during_build':True,
                'engine_or_model_probe_started':False,'fit_started':False,'inputs_before':pins,'inputs_after':pin_after,
                'base_source_files_before':base_before,'base_source_files_after':base_after,'base_source_unchanged':True,
                'teacher_legacy_declared_binary_paths':identity['teacher_declared_binary_paths'],'budget':c.BUDGET,'actual_added_bytes':budget_guard(runtime),'producer_commit':producer_commit,
                'remaining_free_bytes_before_manifest':shutil.disk_usage(runtime.parent).free,
                'usi_copy':usi_copy,'teacher_copy_bytes':copy_bytes,'worker_sha256':args.expected_worker_sha256,'contract_sha256':args.expected_contract_sha256}
            c.validate_manifest(manifest,identity,identity_ref)
            pending(state); state['spawning']=True
            try: write_new(runtime/'build-manifest.json',manifest)
            finally: state['spawning']=False
            pending(state); budget_guard(runtime)
            return manifest
        except BaseException as error:
            state['cleaning']=True
            try:
                record['status']='failed'; record['error']=str(error); record['cancel_signal']=state['signal']
                if hasattr(error,'build_step_outcome'): record['failed_step']=error.build_step_outcome
                # Nothing can be published as complete on cancellation/write failure.
                (runtime/'build-manifest.json').unlink(missing_ok=True)
                write_new(runtime/'build-failure.json',record)
            finally: state['cleaning']=False
            raise


def main():
    barrier()  # Must precede argparse, runtime/control reads and helper imports.
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',required=True); p.add_argument('--identity',required=True)
    for name in ('identity','worker','contract'): p.add_argument('--expected-'+name+'-sha256',required=True)
    args=p.parse_args(); prepare(args)


if __name__=='__main__': main()
