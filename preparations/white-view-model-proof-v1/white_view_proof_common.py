"""Disabled white-view model proof source. Old mode-specific guards are untouched.

Only listed mode-independent validators/lifecycle bodies are copied verbatim
from SHA-fixed paired_linear_proof_common_v2.py; original dataset/model scopes
are not delegated to the old fit/Adam/E3 adapters.
"""
from contextlib import ExitStack
from fractions import Fraction
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys

PROTOTYPE_ONLY=True
R=Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
C=Path('/home/server/.local/share/sekirei-weight2/campaign-17-autonomous-v1')
F=C.parent/'training-17-v1'
T=C.parent/'training-15-v1'
B=C.parent/'suisho11beta-sekirei-v0.3.39-v1'
NEW=C.parent/'suisho11beta-sekirei-v0.3.39-white-view-v1'
Q=F/'white-view-paired-linear-constrained-ridge1-v1/run-v1'
REFERENCE=F/'white-view-material-init-seed42-v1/reference03.bin'
PR=C/'white-view-paired-linear-preregistration-v1.json'
ACTIVATION=C/'white-view-paired-linear-activation-v1.json'
SOURCE_PREFLIGHT=C/'white-view-paired-linear-original-source-preflight-v1.json'
PLAN=C/'paired-linear-white-view-next-plan-v1.json'
COMMON=C/'white_view_proof_common.py'
CORE_OUT=C/'white-view-candidate-core-proof-v1'
INCR_OUT=C/'white-view-incremental-candidate-proof-v1'
GATE_OUT=C/'white-view-model-technical-gate-v1.json'
NUMERIC=C/'white-view-paired-linear-independent-numeric-audit-v1.json'
NUMERIC_WORKER=C/'white-view-paired-linear-independent-numeric-audit-worker-v1.py'
CANDIDATE='white-view-paired-linear-constrained-ridge1-l1-39p5-v1'
PLAN_SHA='1f274b187a96b16674814c51f77cb402ca8a80e6a7e5d738a5caf9354127e836'
DATASET=C/'source-input-recovery-v1/dataset'
DATASET_SHA='ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6'
INITIAL=T/'material-init-seed42/material-init.bin'
INITIAL_SHA='bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40'
TEACHER='external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d'
COUNTS={'train':112681,'holdout':5895}
PUBLIC_FIXTURES=R/'tests/fixtures/material_init.json'
PUBLIC_FIXTURES_SHA='18339a0fc5aa274b3f9cc2ae4c8d898aff5929b980cc947d709e3f6330cf1022'
FIXTURES_TSV=T/'material-init-seed42/fixtures.tsv'
RHO=Fraction(161791,4096)
CORE_KEYS={'index','native_core_cp','nearest_core_cp','native_quantized_float_cp','nearest_quantized_float_cp','material_cp'}
INCREMENTAL_KEYS={'positions_checked','fixtures','walks','search_walks','captures','promotions','drops','undos','null_undos','max_material_difference_cp','incremental_refresh_error_cp','accumulator_refresh_equal','parent_restoration_equal','observed_float_intermediates_finite','mxcsr_control','material_bound_enabled'}
CORE_FLAGS={'every_core_material_equals_python':True,'every_reference03_integer_equals_material':True,
    'every_reference03_float_equals_material':True,'every_float_core_bridge_lt_1p001':True,
    'every_candidate_material_difference_le_100':True,'every_ft_prefix_in_range':True}
SHARED_SOURCE_SHA='063ae00387bf81dee2e417ebf27da95f6ca5d0fb0af329b278889d97f435dc30'
SHARED_FUNCTIONS=('require','exact','sha','strict_json','f32_bytes','core_stderr','incremental_result','fixture_tsv_equal','fraction','coefficient_values','validate_solver_certificate','Reader','save','execute')


def runtime_gate():
    if PROTOTYPE_ONLY: raise ValueError('SOURCE ONLY white-view proof: actual entry disabled before any I/O')

def require(ok, reason):
    if not ok: raise ValueError(reason)


def exact(actual, expected):
    if type(actual) is not type(expected):return False
    if type(expected) is dict:return actual.keys()==expected.keys() and all(exact(actual[k],v) for k,v in expected.items())
    if type(expected) is list:return len(actual)==len(expected) and all(exact(a,b) for a,b in zip(actual,expected))
    return actual==expected


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value), 'lowercase SHA256 required')
    return value


def strict_json(raw):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,'duplicate JSON key');value[key]=item
        return value
    def bad(_):raise ValueError('nonfinite JSON constant')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)


def f32_bytes(value):
    require(type(value) in (int,float) and math.isfinite(value),'finite probe float required')
    try:data=struct.pack('<f',value)
    except (struct.error,OverflowError) as e:raise ValueError('invalid binary32 probe float') from e
    require(math.isfinite(struct.unpack('<f',data)[0]),'nonfinite binary32 probe float')
    return data


def core_stderr(raw,count):
    lines=raw.decode('utf-8').splitlines()
    require(len(lines)==2 and re.fullmatch(r'float_subnormal_policy=x86-ftz-daz; mxcsr=0x[0-9a-f]+',lines[0])
            and lines[1]==f'complete count={count}; absolute STM; checked both float bridges and FT prefixes','stock probe policy/completion differs')
    mxcsr=int(lines[0].split('0x')[1],16)
    require(mxcsr&0x8040==0x8040 and not mxcsr&0x6000,'MXCSR policy differs')


def incremental_result(value):
    require(type(value) is dict and set(value)==INCREMENTAL_KEYS,'incremental stdout schema mismatch')
    fixed={'positions_checked':8185,'fixtures':15,'walks':16,'search_walks':8,'incremental_refresh_error_cp':0,
        'accumulator_refresh_equal':True,'parent_restoration_equal':True,'observed_float_intermediates_finite':True,
        'mxcsr_control':'9fc0','material_bound_enabled':True}
    for key,expected in fixed.items():require(exact(value[key],expected),'incremental count/type/property mismatch: '+key)
    for key in ('captures','promotions','drops','undos','null_undos'):require(type(value[key]) is int and value[key]>0,'transition coverage missing')
    require(type(value['max_material_difference_cp']) is int and 0<=value['max_material_difference_cp']<=100,'incremental material bound exceeds100cp')
    return value


def fixture_tsv_equal(raw,fixtures):
    require(type(raw) is bytes and type(fixtures) is list and len(fixtures)==15,'fixed15 fixture contract required')
    expected=[]
    for item in fixtures:
        require(type(item) is dict and type(item['expected_cp_stm']) is int and type(item['sfen']) is str,'typed public fixture required')
        expected.append(str(item['expected_cp_stm'])+'\t'+item['sfen'])
    require(raw.decode('utf-8').splitlines()==expected,'incremental fixture order/content differs from fixed public15')


def fraction(value):
    require(type(value) is dict and set(value)=={'numerator','denominator'}
            and type(value['numerator']) is int and type(value['denominator']) is int and value['denominator']>0,'exact rational certificate required')
    result=Fraction(value['numerator'],value['denominator'])
    require(result.numerator==value['numerator'] and result.denominator==value['denominator'],'noncanonical certificate fraction')
    return result


def coefficient_values(data,code):
    require(type(data) is bytes and len(data)==254*struct.calcsize('<'+code),'fixed254 coefficient artifact required')
    values=struct.unpack('<254'+code,data);require(all(math.isfinite(x) for x in values),'nonfinite coefficient artifact')
    return tuple(Fraction(x) for x in values)


def validate_solver_certificate(result,f64_bytes,f32_bytes):
    """Recheck artifact/certificate arithmetic. Driver establishes Gram/design origin.

    This does not assert that recorded gradients equal H*u-b independently of
    the separately source-pinned driver/Gram producer.
    """
    require(type(result) is dict,'solver result object required')
    for key,want in {'fallback_used':False,'lambda':1,'loss_is_half_unnormalized_sum':True,
        'saved_f32_gap_is_solver_stopping_criterion':False}.items():require(exact(result.get(key),want),'solver policy differs: '+key)
    require(type(result.get('iterations')) is int and 0<=result['iterations']<=20000
        and type(result.get('restart_count')) is int and 0<=result['restart_count']<=result['iterations'],'solver resource counters differ')
    decoded={'f64':coefficient_values(f64_bytes,'d'),'f32':coefficient_values(f32_bytes,'f')}
    try:
        raw_cast=struct.pack('<254f',*struct.unpack('<254d',f64_bytes))
        # The frozen solver canonicalizes rounded zero to +0 before saving.
        cast=b''.join(b'\0'*4 if raw_cast[i:i+4]==b'\0\0\0\x80' else raw_cast[i:i+4] for i in range(0,len(raw_cast),4))
    except (OverflowError,struct.error) as error:raise ValueError('invalid f64 to savedf32 cast') from error
    require(cast==f32_bytes,'savedf32 artifact is not the canonical-zero f64 coefficient cast')
    require(type(result.get('one_if_needed_f64_correction_used')) is bool,'typed f64 correction flag required')
    require(fraction(result['one_if_needed_f64_correction_factor'])==1-Fraction(1,2**40),'fixed one-correction factor differs')
    lip=fraction(result['lipschitz_exact']);upper=fraction(result['lipschitz_numeric_upper'])
    require(lip>0 and upper>=lip,'positive upper-certified Gershgorin step required')
    for kind in ('f64','f32'):
        top=result['coefficient_'+kind];require(type(top) is list and tuple(map(fraction,top))==decoded[kind],'top solver coefficients differ from artifact')
    for key,kind,radius in (('certificate_f64','f64',RHO),('certificate_f32_solver_radius','f32',RHO),('certificate_f32_saved_radius','f32',Fraction(79,2))):
        cert=result[key];u=decoded[kind]
        require(type(cert) is dict and type(cert['coefficient_values']) is list and tuple(map(fraction,cert['coefficient_values']))==u,'certificate coefficients differ')
        norm=sum(map(abs,u),Fraction());require(fraction(cert['norm'])==norm and fraction(cert['radius'])==radius,'certificate norm/radius differs')
        require(exact(cert['feasible'],norm<=radius),'certificate feasibility/type differs')
        nums=cert['gradient_numerators'];den=cert['gradient_denominator']
        require(type(nums) is list and len(nums)==254 and all(type(n) is int for n in nums) and type(den) is int and den>0,'typed gradient records required')
        gradient=tuple(Fraction(n,den) for n in nums);gap=sum((g*x for g,x in zip(gradient,u)),Fraction())+radius*max(map(abs,gradient))
        require(fraction(cert['fw_gap'])==gap and fraction(cert['fw_gap_per_sample'])==gap/112681,'exact FW gap arithmetic differs')
        met=norm<=radius and 0<=gap/112681<=Fraction(1,1000000)
        require(exact(cert['solver_threshold_met'],met),'solver threshold/type differs')
        delta=fraction(cert['objective_delta_vs_zero'])
        if key=='certificate_f64':require(met and gap>=0 and delta<=0,'f64 solver certificate failed')
        if key=='certificate_f32_saved_radius':require(norm<=radius and gap>=0 and delta<=0,'savedf32 certificate failed')
    return {'certificate_artifact_arithmetic_verified':True,'f64_solver_stopping_certificate_passed':True,
        'saved_f32_feasibility_and_deltaJ_passed':True,'saved_f32_gap_used_for_stopping':False,
        'gradient_origin_independently_recomputed_here':False}


class Reader:
    def __init__(self,life):self.life=life;self.files={}
    def read(self,path,expected=None):
        path=self.life.canonical_path(Path(path),exists=True);require(stat.S_ISREG(path.stat().st_mode),'canonical regular input required')
        data=path.read_bytes();identity={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        require(expected is None or identity['sha256']==sha(expected),'externally pinned input differs: '+str(path))
        require(str(path) not in self.files or exact(self.files[str(path)],identity),'input changed while reading')
        self.files[str(path)]=identity;return data
    def pin(self,path,expected):
        self.read(path,expected['sha256']);require(exact(self.files[str(path)],expected),'input bytes/SHA changed')


def save(path,value):
    data=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    Path(path).chmod(0o600)


def execute(command,output,stdin,stdout,stderr,timeout,holder,modules):
    d=modules['diagnose_anchor']
    with stdin.open('rb') as stream,stdout.open('xb') as out,stderr.open('xb') as err:
        with d.deferred_termination():holder['process']=subprocess.Popen(command,stdin=stream,stdout=out,stderr=err,cwd=output,env=dict(os.environ,RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1'),start_new_session=True)
        process=holder['process'];save(output/(stdout.name+'.process.json'),{'argv':command,'pid':process.pid,'pgid':process.pid})
        try:
            code=process.wait(timeout=timeout);require(type(code) is int and code==0,'probe returncode nonzero')
        finally:
            with d.blocked_termination():
                d.cleanup_group(process);require(not d.group_exists(process.pid),'probe group remains after cleanup');holder['process']=None
    return code



def fields(value,expected,label):
    require(type(value) is dict and all(exact(value.get(k),v) for k,v in expected.items()),label+' typed fields differ')


def fullref(value,path=None):
    require(type(value) is dict and set(value)=={'path','bytes','sha256'},'exact fullref required')
    p=Path(value['path']);require(type(value['path']) is str and p.is_absolute() and str(p)==value['path'] and '..' not in p.parts,'absolute canonical path text required')
    require(type(value['bytes']) is int and value['bytes']>=0,'strict bytes required');sha(value['sha256'])
    if path is not None:require(p==path,'fixed artifact path differs')
    return value


def bound_ref(reader,path,expected=None):
    reader.read(path,expected);return {'path':str(path),**reader.files[str(path)]}


def map_subset(actual,expected,label):
    require(type(actual) is dict and type(expected) is dict and all(exact(actual.get(p),v) for p,v in expected.items()),label+' map not fully bound')


def core_row(raw,index,material):
    """Pair argument1=candidate03; argument2=validated material reference03.

    Float scores are actual core observations. Candidate Python forward and
    quantizer reexport are not claimed by this check.
    """
    row=strict_json(raw)
    require(type(row) is dict and set(row)==CORE_KEYS and type(row['index']) is int and row['index']==index,'core schema/order mismatch')
    require(type(material) is int,'integer Python material required')
    for key in ('native_core_cp','nearest_core_cp','material_cp'):require(type(row[key]) is int,'integer cp required')
    require(row['material_cp']==material and row['nearest_core_cp']==material,'new reference03/core material differs from Python M')
    require(abs(row['native_core_cp']-material)<=100,'observed candidate material residual exceeds100cp')
    for prefix in ('native','nearest'):
        name=prefix+'_quantized_float_cp'; row[name]=struct.unpack('<f',f32_bytes(row[name]))[0]
        require(abs(row[name]-row[prefix+'_core_cp'])<1.001,'float/core bridge >=1.001cp')
    require(f32_bytes(row['nearest_quantized_float_cp'])==f32_bytes(material),'reference03 actual float differs from fixed material')
    return row


def reference_bytes(white,original01,reference03,binding):
    require(reference03==white.transform_initializer(original01,binding=binding),'reference03 is not full protected/tied seed42 transform')
    white.validate_hand_ties(reference03)
    require(white.protected_bytes(reference03)==white.protected_bytes(original01),'reference changed protected M/bias bytes')
    return {'full_transform_bytes_equal':True,'hand_full_row_ties_verified':True,'protected_material_bytes_preserved':True,
        'board_ft_bytes_preserved':True,'ft_bias_bytes_preserved':True,'native_magic':'SEKIRW03','header_retag_only_accepted':False}


def numeric_receipt_fields(value,ctx,expected_worker_sha256):
    run=ctx['run'];fields(value,{'schema':'sekirei.white-view-paired-linear-independent-numeric-audit.v1','status':'complete',
        'candidate':CANDIDATE,'plan_sha256':PLAN_SHA,'fit_receipt':ctx['binding']['fit_receipt'],
        'worker_sha256':sha(expected_worker_sha256),'source_helpers_sha256':ctx['preregistration']['source_helpers'],
        'inputs_unchanged':True,'source_unchanged':True,'no_child_process_started':False,'no_engine_or_fit_process_started':True,
        'runtime_validator_may_invoke_readonly_source_compiler_checks':True,'actual_fit_repeated':False,
        'native_core_proof_claimed':False,'final_used':False,'adoption_claimed':False},'independent numeric receipt')
    fields(value,{'native':run['outputs']['weights'],'metadata':ctx['sidecar_ref'],'reference03':ctx['reference_ref'],
        'coefficients_f64':run['outputs']['coefficients_f64'],'coefficients_f32':run['outputs']['coefficients_f32'],
        'solver_certificate':run['outputs']['solver_certificate'],'design_receipt':run['outputs']['design_receipt'],
        'feature_binding':ctx['feature_binding'],'new_build_binding':ctx['preregistration']['new_build_binding'],
        'new_build_verified':True},'numeric forward subject')
    fields(value.get('forward_artifact_binding'),{'native03_full_reconstruction_verified':True,
        'sidecar_complete_run_binding_verified':True,'reference03_full_transform_verified':True,
        'reference03_sha256':ctx['reference_ref']['sha256'],'native_magic':'SEKIRW03','feature_schema':'flat_white_view_aux_tied_v1',
        'coefficient_nearest_cast_core_forward_claimed':False,'actual_native_core_forward_verified':False,
        'actual_native_core_covariance_verified':False,'actual_incremental_verified':False,
        'forward_proof_requires_separate_new_runtime_core_execution':True},'numeric forward scope')
    require(exact(value.get('inputs_before'),value.get('inputs_after')),'numeric inputs changed')
    mathproof=value.get('numerical_math');fields(mathproof,{'schema':'sekirei.white-view-paired-linear-independent-numeric-math.v1','status':'complete',
        'sample_count':112681,'dimension':254,'all_three_certificates_exactly_recomputed':True,
        'saved_f32_nearest_canonical_cast_verified':True,'stored_gradient_matches_Hu_minus_b':True,
        'raw_design_extraction_replayed':False,'gram_product_recomputed':False,'gram_psd_independently_reproved':False,
        'producer_origin_acknowledged_separately':True,'solver_trajectory_replayed':False,'second_fit_run':False,
        'native_core_proof_claimed':False,'final_used':False,'adoption_claimed':False},'independent numeric math')
    for name,filekey in (('gram_bytes_sha256','gram_z'),('rhs_bytes_sha256','rhs_z'),('coefficient_f64_bytes_sha256','coefficients_f64'),
        ('coefficient_f32_bytes_sha256','coefficients_f32'),('solver_certificate_sha256','solver_certificate'),
        ('design_receipt_sha256','design_receipt'),('design_z_sha256','design_z'),('targets_d_sha256','targets_d')):
        require(mathproof.get(name)==run['outputs'][filekey]['sha256'],'numeric artifact SHA differs: '+name)
    require(exact(mathproof.get('recomputed_certificates'),{k:ctx['certificate'][k] for k in
        ('certificate_f64','certificate_f32_solver_radius','certificate_f32_saved_radius')}),'independent three cert exact records differ')
    required={r['path']:{k:r[k] for k in ('bytes','sha256')} for r in run['outputs'].values()}
    required[ctx['binding']['fit_receipt']['path']]={k:ctx['binding']['fit_receipt'][k] for k in ('bytes','sha256')}
    map_subset(value['inputs_before'],required,'numeric audit')
    return value


def load_public(args):
    runtime_gate();sys.dont_write_bytecode=True
    raw=PR.read_bytes();require(hashlib.sha256(raw).hexdigest()==sha(args.expected_preregistration_sha256),'PR source binding before import differs')
    pr=strict_json(raw);helpers=pr['source_helpers']
    required={'functional_anchor.py','material_init.py','diagnose_anchor.py','benchmark.py','export_nearest.py',
        'white_view_paired_linear.py','white_view_fit_contract.py','white_view_build_contract.py'}
    require(type(helpers) is dict and required<=set(helpers),'all new/old imported source dependencies must be pinned')
    for name,digest in helpers.items():
        require(type(name) is str and Path(name).name==name and name.endswith('.py'),'helper basename required')
        path=R/'scripts'/name;require(path==path.resolve(strict=True),'canonical public helper required')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==sha(digest),'public helper SHA changed')
    sys.path.insert(0,str(R/'scripts'))
    modules={name:importlib.import_module(name) for name in ('functional_anchor','material_init','diagnose_anchor','benchmark','export_nearest',
        'white_view_paired_linear','white_view_fit_contract','white_view_build_contract')}
    for name,mod in modules.items():require(Path(mod.__file__).resolve()==R/'scripts'/(name+'.py'),'public import source alias rejected')
    for name,digest in helpers.items():require(hashlib.sha256((R/'scripts'/name).read_bytes()).hexdigest()==digest,'helper changed during import')
    return modules


def load_bound_module(path,expected,name):
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==sha(expected),'private module source SHA differs')
    require(path==path.resolve(strict=True),'canonical private module required')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    require(hashlib.sha256(path.read_bytes()).hexdigest()==expected and Path(module.__file__).resolve()==path,'private import changed source')
    return module


def load_runtime_validator(reader,prereg):
    """Root-approved adjacent import; restore the public same-name module.

    Both real files are read and SHA-bound. No __file__, source identity, guard,
    or compiled producer is rewritten. Root runs imports serially.
    """
    rec=prereg['new_build_binding']['validator_source'];fullref(rec)
    path=Path(rec['path']);reader.pin(path,{k:rec[k] for k in ('bytes','sha256')})
    adjacent=path.with_name('white_view_build_contract.py')
    expected=prereg['source_helpers']['white_view_build_contract.py'];reader.read(adjacent,expected)
    contract=load_bound_module(adjacent,expected,'_white_view_adjacent_build_contract')
    previous=sys.modules.get('white_view_build_contract');had='white_view_build_contract' in sys.modules
    try:
        sys.modules['white_view_build_contract']=contract
        worker=load_bound_module(path,rec['sha256'],'_white_view_frozen_build_validator')
    finally:
        if had:sys.modules['white_view_build_contract']=previous
        else:sys.modules.pop('white_view_build_contract',None)
    require(worker.c is contract and Path(worker.c.__file__).resolve()==adjacent,'actual adjacent contract not retained')
    return worker


def context(args,worker,modules):
    runtime_gate();d=modules['diagnose_anchor'];a=modules['functional_anchor'];white=modules['white_view_paired_linear'];fit=modules['white_view_fit_contract']
    require(Path(__file__).resolve()==COMMON and worker.resolve()==worker and worker.parent==C,'canonical proof producer paths required')
    reader=Reader(d);reader.read(worker,args.expected_worker_sha256);reader.read(COMMON,args.expected_common_sha256)
    plan=strict_json(reader.read(PLAN,PLAN_SHA));pr=strict_json(reader.read(PR,args.expected_preregistration_sha256));fit.validate_preregistration(pr)
    require(plan.get('candidate')==CANDIDATE,'selected source plan candidate differs')
    for name,digest in pr['source_helpers'].items():reader.read(R/'scripts'/name,digest)
    for path,frozen in pr['runtime_sources'].items():reader.pin(path,frozen)
    activation=strict_json(reader.read(ACTIVATION,args.expected_activation_sha256));fit.validate_activation(activation,pr)
    require(exact(pr['trigger'],{'path':str(ACTIVATION),**reader.files[str(ACTIVATION)]}),'actual activation fullref differs')
    for path,frozen in activation['inputs_before'].items():reader.pin(path,frozen)
    pf=strict_json(reader.read(SOURCE_PREFLIGHT,args.expected_source_preflight_sha256));fit.validate_source_preflight(pf,pr,args.expected_preregistration_sha256)
    for path,frozen in pf['inputs_before'].items():reader.pin(path,frozen)
    # Recheck inherited legal/exclusion source receipts without running replay.
    expected_old={'inherited_replay_receipt':'1540bdf1991a174bab774b99ee32c52ceee0eb1d6004557705917f477063dcdd',
        'inherited_source_preflight':'d0d936e7c7c729feadcbedd90944344edd2729b7d5f0523ceac752d732529d3e'}
    replay=None
    for key,digest in expected_old.items():
        rec=fullref(pf[key]);require(rec['sha256']==digest,'fixed inherited source receipt changed')
        inherited=strict_json(reader.read(Path(rec['path']),digest));fields(inherited,{'status':'complete','inputs_unchanged':True,'final_used':False},key)
        if key=='inherited_replay_receipt':replay=inherited
    require(exact(replay['source_inputs_before'],replay['source_inputs_after']) and len(replay['source_inputs_before'])==1042,'inherited1042 source scope changed')
    map_subset(pf['inputs_before'],replay['source_inputs_before'],'white PF original replay')
    rawroot=C.parent/'shogiquest-human-v1/games'; raw={str(p.resolve()) for p in rawroot.glob('*.csa')}
    require(len(raw)==1000 and raw=={p for p in replay['source_inputs_before'] if p.startswith(str(rawroot)+'/') and p.endswith('.csa')},'raw1000 membership changed')
    require(all(Path(p).stem==replay['source_inputs_before'][p]['sha256'] for p in raw),'raw source filename SHA changed')
    validator=load_runtime_validator(reader,pr);bb=pr['new_build_binding']
    verified=validator.verify_runtime(NEW,bb['manifest']['sha256'],bb['identity']['sha256'])
    build=modules['white_view_build_contract'];feature=build.feature_binding_for_runtime(verified,pr['source_helpers']['white_view_paired_linear.py'])
    require(exact(feature,pr['feature_binding']),'actual new runtime/source/helper feature binding differs')
    runtimefiles=build.immutable_inputs_for_runtime(verified,bb['manifest'])
    for path,frozen in runtimefiles.items():reader.pin(path,frozen)
    original={name:reader.read(DATASET/name) for name in a.FILES}
    manifest,rows=a.validate_original(reader.read(DATASET/'manifest.json',DATASET_SHA),original,expected_manifest_sha256=DATASET_SHA)
    require(exact(manifest['positions'],COUNTS) and manifest['teacher_identity']==TEACHER,'original O count/teacher changed')
    initial=reader.read(INITIAL,INITIAL_SHA);reference=reader.read(REFERENCE,args.expected_reference03_sha256)
    reference_proof=reference_bytes(white,initial,reference,feature)
    native=reader.read(Q/'weights.bin',args.expected_native_sha256);coef=reader.read(Q/'coefficients.f32.bin',args.expected_coefficients_f32_sha256)
    f64=reader.read(Q/'coefficients.f64.bin',args.expected_coefficients_f64_sha256);runraw=reader.read(Q/'run.json',args.expected_fit_run_sha256)
    certbytes=reader.read(Q/'solver-certificate.json',args.expected_solver_certificate_sha256);certificate=strict_json(certbytes)
    runref={'path':str(Q/'run.json'),**reader.files[str(Q/'run.json')]};certref={'path':str(Q/'solver-certificate.json'),**reader.files[str(Q/'solver-certificate.json')]}
    binding={'plan_sha256':PLAN_SHA,'preregistration_sha256':args.expected_preregistration_sha256,'activation_sha256':args.expected_activation_sha256,
        'source_preflight_sha256':args.expected_source_preflight_sha256,'fit_receipt':runref,'solver_certificate':certref,
        'source_helpers_sha256':pr['source_helpers'],'feature_binding':feature,'new_build_binding':bb}
    run=fit.validate_fit_run(runraw,native,coef,initial,binding)
    meta=reader.read(Q/'weights.meta.json',args.expected_metadata_sha256);fit.validate_sidecar(meta,native,coef,initial,runraw,binding)
    for rec in run['outputs'].values():reader.pin(rec['path'],{k:rec[k] for k in ('bytes','sha256')})
    for path,frozen in run['inputs_before'].items():reader.pin(path,frozen)
    required=[PR,PLAN,ACTIVATION,SOURCE_PREFLIGHT,INITIAL,DATASET/'manifest.json',*[DATASET/name for name in a.FILES]]
    map_subset(run['inputs_before'],{str(p):reader.files[str(p)] for p in required},'completed fit')
    native_proof=white.validate_native(native,coef,initial,binding=feature);solver_proof=validate_solver_certificate(certificate,f64,coef)
    ctx={'preregistration':pr,'activation':activation,'source_preflight':pf,'run':run,'binding':binding,'certificate':certificate,
        'native_export_proof':native_proof,'solver_artifact_proof':solver_proof,'reference_transform_proof':reference_proof,
        'manifest':manifest,'rows':rows,'modules':modules,'runtime_verified':verified,'validator':validator,'feature_binding':feature,
        'sidecar_ref':{'path':str(Q/'weights.meta.json'),**reader.files[str(Q/'weights.meta.json')]},
        'reference_ref':{'path':str(REFERENCE),**reader.files[str(REFERENCE)]}}
    reader.read(NUMERIC_WORKER,args.expected_numeric_worker_sha256)
    numeric=strict_json(reader.read(NUMERIC,args.expected_numeric_audit_sha256));numeric_receipt_fields(numeric,ctx,args.expected_numeric_worker_sha256)
    for path,frozen in numeric['inputs_before'].items():reader.pin(path,frozen)
    ctx['numeric_audit']=numeric
    return reader,ctx


def stock(reader,modules):
    """Old guards retain original source paths; newfeature never substitutes them."""
    for name,digest in modules['export_nearest'].SOURCE_HASHES.items():reader.read(B/'sources/sekirei'/name,digest)
    corepath=C/'functional-anchor-core-proof-v1/snapshot-before.json';sourcepath=T/'material-fit-112k-v1/snapshot-before.json'
    core=strict_json(reader.read(corepath,'80ddbf5056cbf6778bf3dddc109e3b8957c498f2f511eb6be41735fa64377f1a'))
    source=strict_json(reader.read(sourcepath,'15018dc3b380bb3e2b467dcacb15904e2c88b372db0f719b44c5d07b7c315f1f'))
    require(len(core['files'])==617 and len(source['fixed_source_tree'])==526 and len(source['fixed_build_dependencies'])==63,'original stock membership differs')
    for path,frozen in core['files'].items():reader.pin(path,frozen)
    return core,source


def snapshot(reader,ctx,core,source):
    d=ctx['modules']['diagnose_anchor'];git,names=d.git_source_identity(T/'source')
    require(exact(git,core['git']) and {str(p) for p in names}==set(source['fixed_source_tree']),'stock source membership changed')
    require({str(p) for p in (T/'build/release/deps').iterdir() if p.is_file()}==set(source['fixed_build_dependencies']),'original stock dependency membership changed')
    bb=ctx['preregistration']['new_build_binding'];verified=ctx['validator'].verify_runtime(NEW,bb['manifest']['sha256'],bb['identity']['sha256'])
    require(exact(verified,ctx['runtime_verified']),'dedicated runtime build changed')
    files={path:{'bytes':Path(path).stat().st_size,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in reader.files}
    require(exact(files,reader.files),'proof source/input bytes changed')
    return {'files':files,'stock_git':git,'fixed_stock_core_reference_count':617,'fixed_stock_source_count':526,
        'fixed_stock_build_file_count':63,'new_runtime_manifest':bb['manifest'],'new_runtime_identity':bb['identity']}


def locks(modules):
    stack=ExitStack();b=modules['benchmark']
    try:
        for path,exclusive in ((Q.parent/'.fit.lock',False),(T/'.training.lock',True),(B/'.prepare.lock',False),
            (B/'.benchmark.lock',True),(NEW/'.build.lock',False),(NEW/'.prepare.lock',False),(NEW/'.benchmark.lock',True)):
            require(path.is_file() and not path.is_symlink(),'existing cooperating lock required');stack.enter_context(b.nonblocking_lock(path,exclusive=exclusive))
    except BaseException:stack.close();raise
    return stack


def output_guard(output,reader,modules):
    d=modules['diagnose_anchor'];require(C==C.resolve() and C.is_dir() and C.stat().st_uid==os.getuid() and not C.stat().st_mode&0o077,'private canonical campaign root required')
    require(output.parent==C and output==output.resolve() and not os.path.lexists(output),'new canonical private output required')
    protected=[R,T,B,NEW,Q,REFERENCE,*map(Path,reader.files)]
    require(not any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected),'output overlaps protected input')
    d.output_guard(output,C,[p for p in protected if not p.is_relative_to(C)])
    require(shutil.disk_usage(C).free>=2*2**30,'2 GiB free required')


def record_base(args,ctx,kind,reader):
    return {'schema':'sekirei.white-view-candidate-'+kind+'-proof.v1','status':'running','candidate':CANDIDATE,'plan_sha256':PLAN_SHA,
        'preregistration_sha256':args.expected_preregistration_sha256,'activation_sha256':args.expected_activation_sha256,
        'source_preflight_sha256':args.expected_source_preflight_sha256,'fit_run_sha256':args.expected_fit_run_sha256,
        'source_helpers_sha256':ctx['preregistration']['source_helpers'],'feature_binding':ctx['feature_binding'],
        'new_build_binding':ctx['preregistration']['new_build_binding'],'native_weight':{'path':str(Q/'weights.bin'),**reader.files[str(Q/'weights.bin')]},
        'reference_weight':{'path':str(REFERENCE),**reader.files[str(REFERENCE)]},'native_metadata_sha256':args.expected_metadata_sha256,
        'coefficients_f32_sha256':args.expected_coefficients_f32_sha256,'coefficients_f64_sha256':args.expected_coefficients_f64_sha256,
        'solver_certificate':ctx['binding']['solver_certificate'],'solver_artifact_validation':ctx['solver_artifact_proof'],
        'independent_numeric_audit':{'path':str(NUMERIC),**reader.files[str(NUMERIC)]},'independent_native_export_proof':ctx['native_export_proof'],
        'reference_initializer_transform_proof':ctx['reference_transform_proof'],'worker_sha256':args.expected_worker_sha256,'common_sha256':args.expected_common_sha256,
        'fit_schema':'sekirei.white-view-paired-linear-fit-run.v1','adam_used':False,'epochs':0,'native_magic':'SEKIRW03','float_policy':'x86-ftz-daz',
        'selfload_pair':False,'native_nearest_quantizer_reexport_claimed':False,'original_manifest_sha256':DATASET_SHA,
        'teacher_identity':TEACHER,'observed_integer_material_cap_cp':100,'universal_integer_bound_proven':False,
        'search_benchmark_verified':False,'adoption_claimed':False,'final_used':False}


def common_arguments(parser):
    for name in ('worker','common','preregistration','activation','source-preflight','fit-run','native','metadata','coefficients-f32',
        'coefficients-f64','solver-certificate','reference03','numeric-audit','numeric-worker'):
        parser.add_argument('--expected-'+name+'-sha256',required=True)


def proof_common_fields(value,args,ctx,kind,expected_worker):
    fields(value,{'schema':'sekirei.white-view-candidate-'+kind+'-proof.v1','status':'complete','candidate':CANDIDATE,
        'plan_sha256':PLAN_SHA,'preregistration_sha256':args.expected_preregistration_sha256,
        'activation_sha256':args.expected_activation_sha256,'source_preflight_sha256':args.expected_source_preflight_sha256,
        'fit_run_sha256':args.expected_fit_run_sha256,'source_helpers_sha256':ctx['preregistration']['source_helpers'],
        'feature_binding':ctx['feature_binding'],'new_build_binding':ctx['preregistration']['new_build_binding'],
        'native_weight':ctx['native_ref'],'reference_weight':ctx['reference_ref'],
        'native_metadata_sha256':args.expected_metadata_sha256,'coefficients_f32_sha256':args.expected_coefficients_f32_sha256,
        'coefficients_f64_sha256':args.expected_coefficients_f64_sha256,'solver_certificate':ctx['binding']['solver_certificate'],
        'independent_numeric_audit':ctx['numeric_ref'],'worker_sha256':sha(expected_worker),'common_sha256':args.expected_common_sha256,
        'fit_schema':'sekirei.white-view-paired-linear-fit-run.v1','adam_used':False,'epochs':0,'native_magic':'SEKIRW03',
        'float_policy':'x86-ftz-daz','selfload_pair':False,'native_nearest_quantizer_reexport_claimed':False,
        'original_manifest_sha256':DATASET_SHA,'teacher_identity':TEACHER,'observed_integer_material_cap_cp':100,
        'universal_integer_bound_proven':False,'search_benchmark_verified':False,'adoption_claimed':False,'final_used':False,
        'inputs_unchanged':True,'source_unchanged':True,'stock_build_unchanged':True,'new_build_unchanged':True,'cleanup_verified':True},kind+' proof')
    for key,want in (('solver_artifact_validation',ctx['solver_artifact_proof']),('independent_native_export_proof',ctx['native_export_proof']),
        ('reference_initializer_transform_proof',ctx['reference_transform_proof'])):
        require(exact(value.get(key),want),kind+' native/numeric provenance differs')
    return value


def proof_snapshots(reader,root):
    before=strict_json(reader.read(root/'snapshot-before.json'));after=strict_json(reader.read(root/'snapshot-after.json'))
    require(exact(before,after) and type(before.get('files')) is dict,'proof snapshots changed/missing')
    fields(before,{'fixed_stock_core_reference_count':617,'fixed_stock_source_count':526,'fixed_stock_build_file_count':63},'fixed stock snapshot')
    for path,frozen in before['files'].items():reader.pin(path,frozen)
    return before


def validate_core_results(value):
    require(type(value) is dict and set(value)=={'train','holdout','fixtures'},'all three core splits required')
    fixedkeys={'count','positions_sha256','stdin_sha256','stdout_sha256','stderr_sha256','ordered_sfens_sha256',
        'returncode','timeout','cleanup_status',*CORE_FLAGS,'maximum_observed_candidate_material_difference_cp','maximum_float_core_bridge_cp'}
    for name,count in (('train',112681),('holdout',5895),('fixtures',15)):
        result=value[name];require(type(result) is dict and set(result)==fixedkeys,'strict core result schema differs')
        fields(result,{'count':count,'returncode':0,'timeout':False,'cleanup_status':'ok',**CORE_FLAGS},name+' core split')
        for key in ('positions_sha256','stdin_sha256','stdout_sha256','stderr_sha256','ordered_sfens_sha256'):sha(result[key])
        require(type(result['maximum_observed_candidate_material_difference_cp']) is int
            and 0<=result['maximum_observed_candidate_material_difference_cp']<=100,'core observed residual exceeds100')
        bridge=result['maximum_float_core_bridge_cp'];require(type(bridge) is float and math.isfinite(bridge) and 0<=bridge<1.001,'finite strict float bridge required')
    return value


def bind_proofs(args,reader,ctx):
    """Reparse existing proof outputs; this gate starts no probe or fit; the runtime verifier may run read-only compiler/git commands."""
    d=ctx['modules']['diagnose_anchor'];a=ctx['modules']['functional_anchor']
    ctx['native_ref']=bound_ref(reader,Q/'weights.bin',args.expected_native_sha256)
    ctx['reference_ref']=bound_ref(reader,REFERENCE,args.expected_reference03_sha256)
    ctx['numeric_ref']=bound_ref(reader,NUMERIC,args.expected_numeric_audit_sha256)
    reader.read(C/'white-view-candidate-core-proof-worker-v1.py',args.expected_core_worker_sha256)
    reader.read(C/'white-view-candidate-incremental-proof-worker-v1.py',args.expected_incremental_worker_sha256)
    core=strict_json(reader.read(CORE_OUT/'receipt.json',args.expected_core_sha256))
    proof_common_fields(core,args,ctx,'core',args.expected_core_worker_sha256)
    fields(core,{'total_rows':118591,'counts':COUNTS,'pair_argument_order':['candidate03','reference03'],'seconds_per_split':600,
        'probe':ctx['runtime_verified']['manifest']['probe_binaries']['core_pair'],
        'probe_source':ctx['runtime_verified']['identity']['probe_sources']['core_pair']},'core provenance')
    validate_core_results(core['results']);core_snapshot=proof_snapshots(reader,CORE_OUT)
    fixtures=strict_json(reader.read(PUBLIC_FIXTURES,PUBLIC_FIXTURES_SHA));require(type(fixtures) is list and len(fixtures)==15,'public15 required')
    positions={split:ctx['rows'][split+'.positions.jsonl'] for split in ('train','holdout')}
    positions['fixtures']=[{'sfen':v['sfen']} for v in fixtures]
    for split,rows in positions.items():
        result=core['results'][split];sfens=[v['sfen'] for v in rows]
        inp=reader.read(CORE_OUT/(split+'.sfens'),result['stdin_sha256']);stdout=reader.read(CORE_OUT/(split+'.stdout.jsonl'),result['stdout_sha256'])
        stderr=reader.read(CORE_OUT/(split+'.stderr.txt'),result['stderr_sha256']);core_stderr(stderr,len(sfens))
        require(inp==''.join(s+'\n' for s in sfens).encode(),'core stdin not original ordered split')
        require(result['positions_sha256']==reader.files[str(PUBLIC_FIXTURES if split=='fixtures' else DATASET/(split+'.positions.jsonl'))]['sha256']
            and result['ordered_sfens_sha256']==hashlib.sha256(a.canonical_json_bytes(sfens)).hexdigest(),'core position/order source SHA differs')
        lines=stdout.splitlines();require(len(lines)==len(sfens),'core rows missing/extra')
        maximum,bridge=0,0.0
        for index,(line,sfen) in enumerate(zip(lines,sfens)):
            row=core_row(line,index,a.fixed_material_cp(sfen));maximum=max(maximum,abs(row['native_core_cp']-row['material_cp']))
            bridge=max(bridge,abs(row['native_quantized_float_cp']-row['native_core_cp']))
            if split=='fixtures':require(row['material_cp']==fixtures[index]['expected_cp_stm'],'public fixed material changed')
        require(exact(result['maximum_observed_candidate_material_difference_cp'],maximum)
            and exact(result['maximum_float_core_bridge_cp'],bridge),'core recorded maxima differ from reparse')
    inc=strict_json(reader.read(INCR_OUT/'receipt.json',args.expected_incremental_sha256))
    proof_common_fields(inc,args,ctx,'incremental',args.expected_incremental_worker_sha256)
    fields(inc,{'returncode':0,'timeout':False,'probe':ctx['runtime_verified']['manifest']['probe_binaries']['incremental'],
        'probe_source':ctx['runtime_verified']['identity']['probe_sources']['incremental'],
        'argv':[ctx['runtime_verified']['manifest']['probe_binaries']['incremental']['path'],str(Q/'weights.bin'),str(FIXTURES_TSV),'100']},'incremental provenance')
    inc_snapshot=proof_snapshots(reader,INCR_OUT)
    require(exact(inc['inputs_before'],inc['inputs_after']) and exact(inc['inputs_before'],inc_snapshot['files']),'incremental immutable maps differ')
    require(type(inc['output_files']) is dict and {'stdout.json','stderr.log','stdin.empty','snapshot-before.json','snapshot-after.json'}<=set(inc['output_files']),'incremental raw inventory missing')
    for name,frozen in inc['output_files'].items():
        require(type(name) is str and Path(name).name==name and name!='receipt.json','incremental basename inventory required')
        reader.pin(INCR_OUT/name,frozen)
    require(reader.read(INCR_OUT/'stderr.log')==b'' and reader.read(INCR_OUT/'stdin.empty')==b'','incremental stderr/stdin changed')
    require(exact(incremental_result(strict_json(reader.read(INCR_OUT/'stdout.json'))),inc['result']),'incremental recorded result differs')
    fixture_tsv_equal(reader.read(FIXTURES_TSV),fixtures)
    return {'core':core,'incremental':inc,'core_snapshot':core_snapshot,'incremental_snapshot':inc_snapshot}


def gate_document(args,reader,ctx,proofs,inputs):
    """A technical receipt, with no search/adoption or universal bound claim."""
    records={'fit_run':ctx['binding']['fit_receipt'],'sidecar':bound_ref(reader,Q/'weights.meta.json',args.expected_metadata_sha256),
        'numeric_audit':ctx['numeric_ref'],'core':bound_ref(reader,CORE_OUT/'receipt.json',args.expected_core_sha256),
        'incremental':bound_ref(reader,INCR_OUT/'receipt.json',args.expected_incremental_sha256)}
    return {'schema':'sekirei.white-view-model-technical-gate.v1','status':'complete','candidate':CANDIDATE,
        'mode':CANDIDATE,'plan_sha256':PLAN_SHA,
        'preregistration_sha256':args.expected_preregistration_sha256,'activation_sha256':args.expected_activation_sha256,
        'source_preflight_sha256':args.expected_source_preflight_sha256,'feature_schema':'flat_white_view_aux_tied_v1','native_magic':'SEKIRW03',
        'model':{'kind':'nnue',**ctx['native_ref']},'reference_weight':ctx['reference_ref'],
        'build_manifest':ctx['preregistration']['new_build_binding']['manifest'],
        'feature_binding':ctx['feature_binding'],'new_build_binding':ctx['preregistration']['new_build_binding'],'proofs':records,
        'adam_used':False,'epochs':0,'resume_used':False,'absolute_output':True,'canonical_new_abi_rebuild_equal':True,
        'hand_full_row_ties_verified':True,'board_ft_bytes_preserved':True,'ft_bias_bytes_preserved':True,'material_bytes_preserved':True,
        'all_three_numeric_certificates_independently_verified':True,'core_total_rows':118591,'incremental_observations':8185,
        'observed_core_integer_residual_cap_cp':100,'actual_reference03_core_material_verified':True,
        'observed_float_core_bridge_lt_1p001':True,'all_observed_float_intermediates_finite':True,'mxcsr_policy':'x86-ftz-daz',
        'inputs_before':dict(inputs),'inputs_after':dict(inputs),'inputs_unchanged':True,'source_unchanged':True,
        'stock_build_unchanged':True,'new_build_unchanged':True,'source_and_build_unchanged':True,'cleanup_verified':True,
        'fit_executor_reaped':True,'no_child_process_started':False,'no_engine_or_fit_process_started':True,
        'runtime_validator_may_invoke_readonly_source_compiler_checks':True,'worker_sha256':args.expected_worker_sha256,'common_sha256':args.expected_common_sha256,
        'universal_integer_bound_proven':False,'native_bitexact_covariance_proven':False,'python_native_forward_exactness_proven':False,
        'search_benchmark_verified':False,'final_used':False,'adoption_verified':False}


def validate_gate_document(value,expected):
    require(type(value) is dict and exact(value,expected),'gate full typed document differs from independently reconstructed receipt')
    require(exact(value['inputs_before'],value['inputs_after']),'gate input inventory changed')
    for rec in value['proofs'].values():fullref(rec)
    fullref(value['reference_weight'],REFERENCE);fullref({k:value['model'][k] for k in ('path','bytes','sha256')},Q/'weights.bin')
    fields(value,{'schema':'sekirei.white-view-model-technical-gate.v1','status':'complete','core_total_rows':118591,
        'incremental_observations':8185,'cleanup_verified':True,'final_used':False,'adoption_verified':False},'technical gate')
    return value
