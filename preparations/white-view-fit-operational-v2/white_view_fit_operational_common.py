"""SOURCE ONLY pinned readers/control gates shared by the three new workers.

Import does no actual I/O. Separate enabled Root copies, strict external SHA
inputs and locked before/after observations are required for actual use.
"""
from contextlib import ExitStack
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import stat
import sys
import white_view_fit_contract as contract

PROTOTYPE_ONLY=True
R=Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
C=contract.C
T=C.parent/'training-15-v1'
B=C.parent/'suisho11beta-sekirei-v0.3.39-v1'
F=C.parent/'training-17-v1/bounded-material-fanin509-trainer-v1'
PYTHON=C.parent/'suisho11beta-v1/venv/bin/python'
SOURCE_INVENTORY=C/'white-view-paired-linear-source-inventory-v1.json'
PUBLIC_DOCS=C/'paired-linear-public-records-published-v1.json'
PUBLIC_DOCS_REF={'path':str(PUBLIC_DOCS),'bytes':3958,'sha256':'70144b20ccab77d3fe2d92e662bf2c9b81b7dd7d05fce267c9ca6401aa84d665'}
ARCHIVE_RESULT=C/'archive-result-paired-linear-constrained-ridge1-l1-39p5-v2.json'
ARCHIVE_RESULT_SHA256='d082e74f672d0b398be56b12b80aa08c3741f26a139b3a1962bac13f03dcb31e'
PUBLIC_RECORDS_COMMIT='c9cb1a044b076a99613aade63d0056098127865a'
NAS_STATUS=Path('/mnt/storage/NAS/sekirei-weight2/receipts/issue17-paired-linear-constrained-ridge1-l1-39p5-v2/status.json')
NAS_DESTINATION=Path('/mnt/storage/NAS/sekirei-weight2/archives/2026-10-04/issue-17-paired-linear-constrained-ridge1-l1-39p5-v2')
PRIOR='paired-linear-constrained-ridge1-l1-39p5-v1'
PRIOR_MODEL_SHA256='12cc820db432fd2677ffcb37d58bab1536f2b03471af0fe14ad26997e2841de3'
PRIOR_FIT_SHA256='54c12735f5047cdb41a2b98634a1639dc7275b449875f8618fe602ad77f1edac'
ARCHIVE_HELPER_SHA256='5520a647ed77f25e47cd7208fd13ee9b6ae10d619a836c8b36709d33308a2816'
PREVIOUS={
 'comparison':{'path':str(C/'comparison-paired-linear-constrained-ridge1-l1-39p5-v2.json'),'bytes':2731906,
    'sha256':'b522d9ec9b6b5b5e2504027794339a9e4797efd4f8502dcac3865ca56483d75f'},
 'independent_review':{'path':str(C/'paired-linear-independent-formal-review-v2.json'),'bytes':29959,
    'sha256':'602166b7c4db25250c0cd63a1c70e4fd35afa3c39e80969eefca639481f1af04'},
 'stopped':{'path':str(C/'paired-linear-formal-stopped-v2.json'),'bytes':4391967,
    'sha256':'94a73fe06d757a5a784566385282faa4c4fd76110c07f5f6d39a5eefcf369402'}}
STAGES={'mae-pilot':102,'mae':1140,'top3-pilot':36,'top3':551}
CHECKS={'execution_except_model','occurrences','teacher_results','teacher_E_cp','top3_occurrences',
    'top3_legal_moves_sha256','top3_denominators','top3_teacher_bestmoves'}
DICT_CHECKS={'execution_except_model','teacher_results','teacher_E_cp','top3_denominators','top3_teacher_bestmoves'}
require,exact,sha=contract.require,contract.exact,contract.strict_sha


def info(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve(strict=True)==path,'canonical actual path required')
    before=path.lstat();require(stat.S_ISREG(before.st_mode),'regular actual file required')
    def fp(s):return(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        require(fp(before)==fp(os.fstat(stream.fileno())),'input replaced before hash')
        digest=hashlib.file_digest(stream,'sha256').hexdigest();after=os.fstat(stream.fileno())
    require(fp(before)==fp(after)==fp(path.lstat()),'input changed while hashing')
    return {'bytes':before.st_size,'sha256':digest}


class Reader:
    def __init__(self):self.files={}
    def pin(self,path,expected=None):
        key=str(path);value=info(path)
        if type(expected) is str:require(value['sha256']==sha(expected),'extern SHA differs: '+key)
        elif expected is not None:
            contract.identity_map({key:expected});require(exact(value,expected),'extern identity differs: '+key)
        require(key not in self.files or exact(self.files[key],value),'bound input rebinding forbidden')
        self.files[key]=value;return value
    def read(self,path,expected=None):
        value=self.pin(path,expected);raw=Path(path).read_bytes()
        require(exact(value,{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}),'raw input changed at read')
        require(exact(info(path),value),'raw input changed after read');return raw
    def document(self,ref):
        contract.fullref(ref);return contract.strict_json(self.read(ref['path'],{k:ref[k] for k in ('bytes','sha256')}))
    def verify(self):
        current={path:info(path) for path in self.files}
        require(exact(self.files,current),'immutable input/source before/after differs');return current


def fullref(path,reader):return {'path':str(path),**reader.pin(path)}

def merge(target,other):
    contract.identity_map(other)
    for path,rec in other.items():
        require(path not in target or exact(target[path],rec),'immutable map collision')
        target[path]=dict(rec)
    return target


def save(path,value):
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def load_source(ref,name):
    contract.fullref(ref);reader=Reader();raw=reader.read(ref['path'],{k:ref[k] for k in ('bytes','sha256')})
    if name in sys.modules:
        loaded=sys.modules[name]
        require(exact(getattr(loaded,'_white_loaded_fullref',None),ref),'source import binding changed')
        reader.verify();return loaded
    spec=importlib.util.spec_from_file_location(name,ref['path']);require(spec is not None,'source loader required')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    exec(compile(raw,ref['path'],'exec'),module.__dict__);module._white_loaded_fullref=dict(ref)
    reader.verify();return module


BUILD_CONTRACT_SHA256='305d4a2148feecfc0b25ca7e6834c108a635e250cf5b42124511ab3c976b8144'


def load_build_validator(reader,ref):
    """Serial, actual-adjacent source import; preserves any public R cache."""
    contract.fullref(ref)
    adjacent_path=Path(ref['path']).with_name('white_view_build_contract.py')
    require(str(adjacent_path) in reader.files,'real adjacent contract must be externally pinned first')
    adjacent_info=reader.files[str(adjacent_path)]
    require(adjacent_info['sha256']==BUILD_CONTRACT_SHA256,'fixed adjacent contract SHA differs')
    adjacent_ref={'path':str(adjacent_path),**adjacent_info}
    adjacent=load_source(adjacent_ref,'_white_operational_adjacent_contract_'+hashlib.sha256(str(adjacent_path).encode()).hexdigest())
    require(Path(adjacent.__file__)==adjacent_path and adjacent_path.resolve()==adjacent_path,'real adjacent contract path required')
    name='_white_operational_build_validator'
    previous_worker=sys.modules.get(name)
    marker=object();previous=sys.modules.get('white_view_build_contract',marker)
    try:
        sys.modules['white_view_build_contract']=adjacent
        worker=load_source(ref,name)
    except BaseException:
        if previous_worker is None:sys.modules.pop(name,None)
        raise
    finally:
        if previous is marker:sys.modules.pop('white_view_build_contract',None)
        else:sys.modules['white_view_build_contract']=previous
    require(getattr(worker,'c',None) is adjacent and callable(getattr(worker,'verify_runtime',None)),
        'dedicated actual adjacent verify_runtime API missing')
    reader.pin(adjacent_path,adjacent_info);reader.pin(ref['path'],{k:ref[k] for k in ('bytes','sha256')})
    return worker


def require_enabled():require(PROTOTYPE_ONLY is False,'SOURCE ONLY operational common disabled before actual I/O')


def lock_paths():
    return [T/'.training.lock',B/'.prepare.lock',B/'.benchmark.lock',F/'.build.lock',F/'.training.lock',contract.Q.parent/'.fit.lock']


def architecture_shared_locks():
    return [contract.RUNTIME/'.build.lock',contract.RUNTIME/'.prepare.lock']


def pin_helpers(reader,helpers):
    contract.helper_map(helpers)
    for name,digest in helpers.items():
        path=R/'scripts'/name;reader.pin(path,digest)
        imported=sys.modules.get(name[:-3])
        if imported is not None:require(Path(imported.__file__).resolve()==path,'shadowed imported helper source')
    from fit_white_view_paired_linear import NUMERIC_HASHES
    for name,digest in NUMERIC_HASHES.items():require(helpers.get(name)==digest,'fixed numeric/original helper differs')


def verify_build(reader,build,feature,expected_manifest,expected_identity,expected_validator):
    contract.build_binding(build);contract.feature_binding(feature)
    require(build['manifest']['sha256']==sha(expected_manifest) and build['identity']['sha256']==sha(expected_identity)
        and build['validator_source']['sha256']==sha(expected_validator),'extern build/source SHA differs')
    for key in ('manifest','identity','validator_source'):
        ref=build[key];reader.pin(ref['path'],{k:ref[k] for k in ('bytes','sha256')})
    worker=load_build_validator(reader,build['validator_source'])
    verified=worker.verify_runtime(Path(build['runtime']),expected_manifest,expected_identity)
    require(type(verified) is dict and set(verified)=={'manifest','identity'},'verified actual build bundle required')
    for key in ('manifest','identity'):require(exact(verified[key],reader.document(build[key])),'verified build raw bundle differs')
    actual=worker.c.feature_binding_for_runtime(verified,contract.HELPER_SHA256)
    require(exact(actual,feature),'actual new source/core/helper feature binding differs')
    immutable=worker.c.immutable_inputs_for_runtime(verified,build['manifest'])
    contract.identity_map(immutable)
    for path,record in immutable.items():reader.pin(path,record)
    reader.verify();return verified


def validate_previous(docs,refs):
    """Recorded complete valid-nonadopt proof; no failed-v1 results are reused."""
    require(type(docs) is dict and set(docs)=={'comparison','independent_review','stopped'},'three previous receipts required')
    cp,au,st=(docs[key] for key in ('comparison','independent_review','stopped'))
    for key,schema in (('comparison','sekirei.formal-comparison.v1'),('independent_review','sekirei.paired-linear-independent-formal-review.v1')):
        record=docs[key];contract.flags(record,{'schema':schema,'status':'complete','comparison_valid':True,
            'adopt':False,'inputs_unchanged':True,'final_used':False})
        require(type(record.get('mae_improved')) is bool and type(record.get('top3_preserved')) is bool
            and record['adopt']==(record['mae_improved'] and record['top3_preserved']), 'previous decision inconsistent')
        checks=record.get('checks');require(type(checks) is dict and set(checks)==CHECKS,'eight previous identities required')
        for name,value in checks.items():
            keys={'equal','baseline_sha256','candidate_sha256'} | ({'mismatch_count'} if name in DICT_CHECKS else set())
            require(type(value) is dict and set(value)==keys and value['equal'] is True
                and sha(value['baseline_sha256'])==sha(value['candidate_sha256']),'previous exact identity differs')
            if name in DICT_CHECKS:require(exact(value['mismatch_count'],0),'previous mismatch nonzero/type')
    for key in ('checks','adopt','mae_improved','top3_preserved'):require(exact(cp[key],au[key]),'previous CP/AU differ')
    require(cp.get('candidate',{}).get('model_identity',{}).get('sha256')==PRIOR_MODEL_SHA256
        and au.get('candidate_model',{}).get('sha256')==PRIOR_MODEL_SHA256,'previous fixed native identity differs')
    contract.flags(au.get('scope'),{'candidate_attempts':1829,'baseline_attempts':1829,'requested_nodes':1000000,
        'games':5,'teacher_E':266,'top3_denominator':551})
    audits=au.get('run_audits');require(type(audits) is dict and set(audits)=={'candidate','baseline'},'both full audits required')
    for records in audits.values():
        require(type(records) is dict and set(records)==set(STAGES),'four previous stages required')
        for stage,count in STAGES.items():contract.flags(records[stage],{'attempts':count,'cleanup_ok':count,
            'runner_exit_zero':count,'raw_sha256_and_lifecycle_verified':count,'technical_failures':0,'timeout_failures':0})
    contract.flags(st,{'schema':'sekirei.paired-linear-formal-stopped.v1','status':'verified-stopped-under-locks',
        'candidate':PRIOR,'all_groups_stopped':True,'comparison_valid':True,'adopt':False,'final_used':False,
        'inputs_unchanged':True,'executor_session_id':91053,'executor_exit_code':0,
        'recorded_cleanup_verified_candidate_attempts':1829,'recorded_cleanup_verified_baseline_attempts':1829})
    require(exact(st.get('exclusive_locks'),[str(x) for x in
        (T/'.training.lock',B/'.prepare.lock',B/'.benchmark.lock',F/'.build.lock',F/'.training.lock',
         C.parent/'training-17-v1/paired-linear-constrained-ridge1-v1/.fit.lock')]), 'previous six stop locks differ')
    contract.identity_map(st.get('inputs_before'));require(exact(st['inputs_before'],st.get('inputs_after')),'previous STOP inputmaps differ')
    for field,name in (('comparison_receipt','comparison'),('independent_review_receipt','independent_review')):
        require(exact(st.get(field),refs[name]),'previous STOP refs differ')
    scans=st.get('process_scans');require(type(scans) is list and len(scans)==2,'two stopped scans required')
    for scan in scans:
        require(type(scan) is dict and scan.get('related_processes')==[]
            and type(scan.get('owned_processes_scanned')) is int and scan['owned_processes_scanned']>=0,'previous stopped scan malformed')
    return True


def validate_archive(record):
    contract.flags(record,{'schema':'sekirei.private-archive.v1','status':'complete','phase':'verified',
        'destination':str(NAS_DESTINATION),'sources_deleted':False,'script_sha256':ARCHIVE_HELPER_SHA256})
    require(type(record.get('mapping')) is dict and len(record['mapping'])==15,'previous 15 roots required')
    sha(record.get('mapping_sha256'));require(type(record.get('locks')) is list,'archive lock record required')
    return record


def validate_published_docs(record):
    contract.flags(record,{'schema':'sekirei.paired-linear-public-records-published.v1',
        'status':'published-verified','candidate':PRIOR,'measurement_attempt':'paired-linear-constrained-ridge1-l1-39p5-v2',
        'result':'valid-rejected','adopt':False,'best_model_updated':False,'goal_complete':False,
        'final_used':False,'draft_pr_merged':False,'public_records_commit':PUBLIC_RECORDS_COMMIT,
        'public_records_commit_is_ancestor':True,'worktree_clean':True})
    files=record.get('files_at_public_records_commit')
    require(type(files) is dict and len(files)==10,'ten published records required')
    for path,rec in files.items():
        require(type(path) is str and not Path(path).is_absolute() and '..' not in Path(path).parts,'public relative path required')
        contract.identity_map({str(R/path):rec})
    for kind in ('public_records_ci','source_preparation_ci'):
        ci=record.get(kind);require(type(ci) is dict and set(ci)=={'push','pull_request'},'both CI contexts required')
        for item in ci.values():
            contract.flags(item,{'status':'completed','conclusion':'success'})
            require(type(item.get('run_id')) is int and item['run_id']>0,'typed completed CI run required')
    require(record.get('nas_status')=={'path':str(NAS_STATUS),'bytes':4092,
        'sha256':'3c85223d2464436b4e155c19f897765bd5eee839574573af726d636380035415'}
        and record.get('archive_result')=={'path':str(ARCHIVE_RESULT),'bytes':1005,'sha256':ARCHIVE_RESULT_SHA256},
        'externally recorded archive dependency differs')
    return record


def verify_git(expected_head, published=None):
    import re,subprocess
    require(type(expected_head) is str and re.fullmatch('[0-9a-f]{40}',expected_head),'extern full Git commit required')
    def git(*args):
        return subprocess.run(['git','-C',str(R),*args],check=True,capture_output=True,timeout=10).stdout
    head=git('rev-parse','HEAD').decode().strip()
    require(head==expected_head and git('status','--porcelain=v1','--untracked-files=all')==b'', 'frozen clean public checkout required')
    require(git('rev-parse','@{upstream}').decode().strip()==head,'local remote tracking branch must match frozen source')
    git('merge-base','--is-ancestor',PUBLIC_RECORDS_COMMIT,head)
    if published is not None:
        validate_published_docs(published)
        for path,rec in published['files_at_public_records_commit'].items():
            raw=git('show',PUBLIC_RECORDS_COMMIT+':'+path)
            require(exact({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},rec),'published commit file bytes differ')
    return {'head':head,'clean':True,'remote_tracking_head':head,'published_records_commit':PUBLIC_RECORDS_COMMIT,
        'published_commit_is_ancestor':True,'remote_network_rechecked':False}


def source_inventory(record):
    contract.flags(record,{'schema':'sekirei.white-view-paired-linear-source-inventory.v1',
        'status':'frozen-before-activation','candidate':contract.MODE,'mode':contract.MODE,
        'plan_sha256':contract.PLAN_SHA256,'fit':contract.FIT,'output':str(contract.Q),
        'fit_started':False,'final_used':False,'adoption_claimed':False})
    contract.helper_map(record.get('source_helpers'));contract.identity_map(record.get('runtime_sources'))
    contract.feature_binding(record.get('feature_binding'));contract.build_binding(record.get('new_build_binding'))
    contract.identity_map(record.get('operational_sources'))
    required={str(C/name) for name in ('white_view_fit_operational_common.py',
        'white-view-paired-linear-activation-worker-v1.py','white-view-paired-linear-source-preflight-worker-v1.py',
        'white-view-paired-linear-fit-launcher-v1.py')}
    require(set(record['operational_sources'])==required,'four fixed enabled operational sources required')
    import re
    require(type(record.get('public_source_commit')) is str and re.fullmatch('[0-9a-f]{40}',record['public_source_commit']),
        'frozen published source commit required')
    return record


def parse_common_arguments(parser, *, inventory=False, preregistration=False, activation=False, source_preflight=False):
    names=['worker','common','build-manifest','build-identity','build-validator']
    if inventory:names.append('source-inventory')
    if preregistration:names.append('preregistration')
    if activation:names.append('activation')
    if source_preflight:names.append('source-preflight')
    for name in names:parser.add_argument('--expected-'+name+'-sha256',required=True)
    parser.add_argument('--expected-public-source-head',required=True)
    return parser


def bootstrap(reader,worker,args):
    require_enabled()
    require(Path(worker)==C/Path(worker).name and Path(worker).resolve()==Path(worker),'canonical Root-enabled worker required')
    reader.pin(worker,args.expected_worker_sha256)
    source=Path(__file__);require(source==C/'white_view_fit_operational_common.py','canonical common copy required')
    reader.pin(source,args.expected_common_sha256)


def pin_operational_sources(reader,values):
    contract.identity_map(values)
    for path,rec in values.items():reader.pin(path,rec)


def verify_activation(reader,prereg,args):
    contract.validate_preregistration(prereg)
    ref=prereg['trigger'];require(ref['sha256']==sha(args.expected_activation_sha256),'extern ACT SHA differs')
    record=reader.document(ref);contract.validate_activation(record,prereg)
    inventory=reader.document(record['source_inventory']);source_inventory(inventory)
    require(exact(inventory['source_helpers'],prereg['source_helpers'])
        and exact(inventory['runtime_sources'],prereg['runtime_sources'])
        and exact(inventory['new_build_binding'],prereg['new_build_binding'])
        and exact(inventory['feature_binding'],prereg['feature_binding'])
        and inventory['public_source_commit']==prereg['public_source_commit']==args.expected_public_source_head,
        'ACT inventory/PR/root source fields differ')
    pin_operational_sources(reader,inventory['operational_sources'])
    for path,rec in record['inputs_before'].items():reader.pin(path,rec)
    docs={name:reader.document(record['previous_completion'][name]) for name in PREVIOUS}
    require(all(exact(record['previous_completion'][name],ref) for name,ref in PREVIOUS.items()),'previous pinned receipts changed')
    validate_previous(docs,PREVIOUS)
    published=reader.document(record['previous_completion']['public_docs'])
    require(exact(record['previous_completion']['public_docs'],PUBLIC_DOCS_REF),'published docs reference differs')
    validate_published_docs(published);validate_archive(reader.document(record['previous_completion']['archive']))
    require(exact(record['previous_completion']['archive'],published['nas_status']),'ACT/archive reference differs')
    verify_git(args.expected_public_source_head,published)
    verify_build(reader,prereg['new_build_binding'],prereg['feature_binding'],
        args.expected_build_manifest_sha256,args.expected_build_identity_sha256,args.expected_build_validator_sha256)
    return {'schema':'sekirei.white-view-paired-linear-activation-validation.v1','status':'verified',
        'plan_sha256':contract.PLAN_SHA256,'new_build_verified':True,'fit_started':False,
        'receipt_and_stopped_input_identities':dict(record['inputs_before']),
        'feature_binding':prereg['feature_binding'],'new_build_binding':prereg['new_build_binding'],
        'final_used':False,'adoption_claimed':False}


def validate_numeric_artifacts(result,gram_bytes,rhs_bytes,f64_bytes,f32_bytes,*,dimension=254,samples=112681):
    """Recompute three exact certificates from sole stored Gram; no second Gram/fit.

    This is O(D²) integer/dyadic math. PSD/design provenance remains the pinned
    producer/driver evidence; this function does not recompute Z.T@Z or replay.
    The optional small dimensions/counts exist only for synthetic unit tests.
    """
    from fractions import Fraction
    import math,struct
    import fit_white_view_paired_linear as driver
    require(type(dimension) is int and 1<=dimension<=254 and type(samples) is int and 1<=samples<=112681,'numeric shape bounds')
    require(all(type(v) is bytes for v in (gram_bytes,rhs_bytes,f64_bytes,f32_bytes)), 'immutable numeric artifacts required')
    require(len(gram_bytes)==dimension*dimension*8 and len(rhs_bytes)==dimension*8
        and len(f64_bytes)==dimension*8 and len(f32_bytes)==dimension*4,'numeric shape mismatch')
    G=tuple(v[0] for v in struct.iter_unpack('<q',gram_bytes));h=tuple(v[0] for v in struct.iter_unpack('<q',rhs_bytes))
    require(all(abs(x)<=samples*1600 for x in G) and all(abs(x)<=samples*40*55779 for x in h),'fixed integer Gram/rhs bound')
    require(all(G[i*dimension+j]==G[j*dimension+i] for i in range(dimension) for j in range(dimension)),'exact Gram symmetry required')
    require(all(G[i*dimension+i]>=0 for i in range(dimension)),'Gram diagonal must be nonnegative')
    vectors=[struct.unpack('<'+str(dimension)+code,data) for code,data in (('d',f64_bytes),('f',f32_bytes))]
    require(all(math.isfinite(x) for vector in vectors for x in vector),'finite coefficients required')
    for i,value in enumerate(vectors[0]):
        converted=struct.unpack('<f',struct.pack('<f',value))[0]
        require(struct.pack('<f',0. if converted==0. else converted)==f32_bytes[4*i:4*i+4],'nearest canonical f32 cast differs')
    K=tuple(tuple(G[i*dimension+j]+(256 if i==j else 0) for j in range(dimension)) for i in range(dimension))
    def certify(vector,radius):
        ratios=[v.as_integer_ratio() for v in vector];Q=max(q for _,q in ratios)
        U=tuple(n*(Q//q) for n,q in ratios)
        KU=tuple(sum(k*u for k,u in zip(row,U)) for row in K)
        gradient=tuple(v-16*Q*hv for v,hv in zip(KU,h));den=256*Q
        norm=Fraction(sum(abs(u) for u in U),Q)
        dot=Fraction(sum(g*u for g,u in zip(gradient,U)),256*Q*Q)
        maximum=Fraction(max(abs(g) for g in gradient),den)
        gap=dot+radius*maximum
        delta=Fraction(sum(u*v for u,v in zip(U,KU)),512*Q*Q)-Fraction(sum(hv*u for hv,u in zip(h,U)),16*Q)
        feasible=norm<=radius
        require(not feasible or gap>=0,'negative exact gap at feasible vector')
        return {'norm':norm,'radius':radius,'feasible':feasible,'fw_gap':gap,'fw_gap_per_sample':gap/samples,
            'solver_threshold_met':feasible and 0<=gap/samples<=Fraction(1,1000000),
            'objective_delta_vs_zero':delta,'gradient_numerators':gradient,'gradient_denominator':den,'coefficient_values':vector}
    radii=(Fraction(161791,4096),Fraction(161791,4096),Fraction(79,2))
    names=('certificate_f64','certificate_f32_solver_radius','certificate_f32_saved_radius')
    computed={name:certify(vectors[0] if index==0 else vectors[1],radii[index]) for index,name in enumerate(names)}
    for name,value in computed.items():require(exact(result.get(name),driver.rational_json(value)),'exact stored certificate differs: '+name)
    flags={'lambda':1,'loss_is_half_unnormalized_sum':True,'fallback_used':False,
        'saved_f32_gap_is_solver_stopping_criterion':False}
    contract.flags(result,flags)
    for name,vector in zip(('coefficient_f64','coefficient_f32'),vectors):
        require(exact(result.get(name),driver.rational_json(vector)),'certificate coefficient artifact differs')
    require(computed[names[0]]['solver_threshold_met'] is True and computed[names[0]]['feasible'] is True
        and computed[names[2]]['feasible'] is True and computed[names[2]]['objective_delta_vs_zero']<=0,
        'fixed exact termination/saved feasibility/objective certificate violated')
    lipschitz=Fraction(max(sum(abs(v) for v in row) for row in K),256)
    require(exact(result.get('lipschitz_exact'),driver.rational_json(lipschitz))
        and exact(result.get('lipschitz_numeric_upper'),driver.rational_json(math.nextafter(float(lipschitz),math.inf))),
        'exact Gershgorin/upward bound differs')
    require(exact(result.get('one_if_needed_f64_correction_factor'),driver.rational_json(1.-2.**-40))
        and type(result.get('one_if_needed_f64_correction_used')) is bool,'fixed single correction rule differs')
    return {'all_three_certificates_exactly_recomputed':True,'nearest_canonical_f32_cast_verified':True,
        'dimension':dimension,'samples':samples,'second_gram_computed':False,'alternate_fit_performed':False,
        'psd_or_design_recomputed':False}


def verify_producer_digest(result,design,zb,db,gb,hb):
    """Content-only verification of inherited issued-producer evidence, no Gram."""
    from types import SimpleNamespace
    import struct,verified_design_gram as producer
    import fit_white_view_paired_linear as driver
    require(type(design) is dict and design.get('schema')=='sekirei.white-view-paired-linear-train-design.v1'
        and exact(design.get('count'),112681) and exact(design.get('dimension'),254), 'fixed TRAIN design receipt required')
    for key,data in (('z_sha256',zb),('d_sha256',db)):require(design.get(key)==hashlib.sha256(data).hexdigest(),'design stored artifact SHA differs')
    require(len(zb)==112681*254 and len(db)==112681*4 and len(gb)==254*254*8 and len(hb)==254*8,'fixed sole producer artifact shapes')
    bound=result.get('verified_design_gram')
    contract.flags(bound,{'schema':producer.SCHEMA,'producer':producer.PRODUCER,
        'producer_source_sha256':driver.NUMERIC_HASHES['verified_design_gram.py'],
        'solver_source_sha256':driver.NUMERIC_HASHES['paired_linear_solver_verified_design.py'],
        'sample_count':112681,'dimension':254,'algebraic_psd_proven_from_issued_design':True,
        'runtime_threadpool_proof_claimed':False,'actual_dataset_provenance_claimed':False,'source_prototype_only':True})
    digest=producer._design_hash(112681,254,zb,db)
    require(bound.get('design_sha256')==digest,'sole immutable design digest differs')
    metadata=bound.get('backend_metadata')
    require(type(metadata) is list and all(type(pair) is list and len(pair)==2 for pair in metadata),'canonical producer backend metadata required')
    metadata=tuple(tuple(pair) for pair in metadata)
    expected_names=('numpy_version','dtype','design_order','transpose','first_numpy_preimported',
        'environment_set_before_first_numpy_import','existing_import_threadpool_evidence_not_claimed')
    require(tuple(name for name,value in metadata)==expected_names,'backend metadata fields differ')
    values=dict(metadata)
    contract.flags(values,{'numpy_version':'1.26.4','dtype':'float64','design_order':'C','transpose':'Z.T@Z;Z.T@d',
        'existing_import_threadpool_evidence_not_claimed':True})
    require(type(values['first_numpy_preimported']) is bool and values['environment_set_before_first_numpy_import']
        is (not values['first_numpy_preimported']), 'backend preimport flags differ')
    flat=tuple(x[0] for x in struct.iter_unpack('<q',gb));G=tuple(flat[i*254:(i+1)*254] for i in range(254))
    h=tuple(x[0] for x in struct.iter_unpack('<q',hb))
    fake=SimpleNamespace(design_sha256=digest,samples=112681,dimension=254)
    require(bound.get('gram_sha256')==producer._gram_hash(fake,G,h,bound['producer_source_sha256'],metadata),
        'sole stored Gram/rhs/source metadata digest differs')
    return {'issued_origin_execution_inherited_from_pinned_driver':True,'stored_design_gram_content_digests_recomputed':True,
        'actual_ZTZ_recomputed':False,'PSD_recomputed':False,'second_gram_or_fit':False}
