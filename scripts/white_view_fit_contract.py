"""SOURCE ONLY new03 fit/control/absolute sidecar contracts, no filesystem I/O.

External references must be pinned by a later Root driver. Typed receipts and
SHA identity do not themselves rerun legal replay, build, lifecycle or numeric
proof. None of the old native01 fit/run/sidecar validators accepts new03 here.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import math
from fractions import Fraction
import white_view_paired_linear as white
import material_init

PROTOTYPE_ONLY = True
C = Path('/home/server/.local/share/sekirei-weight2/campaign-17-autonomous-v1')
Q = C.parent / 'training-17-v1/white-view-paired-linear-constrained-ridge1-v1/run-v1'
W = Q / 'weights.bin'
FIT_RECEIPT = Q / 'run.json'
PREREGISTRATION = C / 'white-view-paired-linear-preregistration-v1.json'
ACTIVATION = C / 'white-view-paired-linear-activation-v1.json'
SOURCE_PREFLIGHT = C / 'white-view-paired-linear-original-source-preflight-v1.json'
PLAN = C / 'paired-linear-white-view-next-plan-v1.json'
DATASET = C / 'source-input-recovery-v1/dataset'
RUNTIME = C.parent / 'suisho11beta-sekirei-v0.3.39-white-view-v1'
ORIGINAL_INITIALIZER = C.parent / 'training-15-v1/material-init-seed42/material-init.bin'
MODE = white.MODE
PLAN_SHA256 = white.PLAN_SHA256
HELPER_SHA256 = '79845db6f8ceec2e0c73488b705db526813da52bcc601ab6b10eb3f52f0ce6eb'
MANIFEST_SHA256 = 'ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6'
COUNTS = {'train':112681, 'holdout':5895}
FIT = {'train_positions':112681, 'holdout_positions':5895,
    'original_teacher_labels_unchanged':True, 'nonmaterial_coefficients':254,
    'optimizer':'projected-FISTA', 'ridge':1,
    'objective':'unnormalized half SSE plus half squared L2; no constant intercept',
    'solver_l1_radius_exact':'161791/4096', 'saved_f32_l1_max_exact':'79/2',
    'iterations_max':20000, 'global_seconds_max':1200,
    'exact_fw_gap_per_sample_max':'1/1000000', 'adam_used':False, 'epochs':0,
    'resume_used':False, 'parameter_retuning_or_solver_fallback_allowed':False}
FEATURE_HASHES = {
    'crates/sekirei-core/src/nnue.rs':'083c99ed681108d1643fa897aa2a4c3f452a170a73219fa3dc2cf16cdf68de50',
    'crates/sekirei-core/Cargo.toml':'c6c0d7a2e60b988e2975c5bc5b7dff620fe74b4426ff1ffc69474a0fa5cc2fc8',
    'crates/sekirei-usi/Cargo.toml':'f0f9689226fc59c10b7d1adc10a101f9f31914bc6a3e342b29b68917ef73ffea'}
OUTPUT_NAMES = {'design_z':'design.z.i8','targets_d':'targets.d.i32',
    'gram_z':'gram.z.i64','rhs_z':'rhs.z.i64','coefficients_f64':'coefficients.f64.bin',
    'coefficients_f32':'coefficients.f32.bin','weights':'weights.bin',
    'design_receipt':'design-receipt.json','solver_certificate':'solver-certificate.json'}
LIFECYCLE = {'returncode':0,'timed_out':False,'waited':True,'reaped':True,'process_group_stopped':True}
require, exact, strict_sha, sha = white.require, white.exact, white.strict_sha, white.sha


def strict_json(data):
    require(type(data) is bytes, 'immutable JSON bytes required')
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result, 'duplicate JSON key');result[key]=value
        return result
    def finite_float(value):
        result=float(value);require(math.isfinite(result),'nonfinite exponent');return result
    return json.loads(data,object_pairs_hook=pairs,parse_float=finite_float,
        parse_constant=lambda value:require(False,'nonfinite JSON constant'))


def fullref(ref,path=None):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'},'exact fullref fields required')
    require(type(ref['path']) is str and Path(ref['path']).is_absolute()
        and Path(ref['path']).as_posix()==ref['path'] and '..' not in Path(ref['path']).parts,
        'canonical absolute reference required')
    require(type(ref['bytes']) is int and ref['bytes']>=0,'strict reference byte count required')
    strict_sha(ref['sha256'])
    if path is not None:require(ref['path']==str(path),'wrong fixed reference path')
    return ref


def raw_bound(data,ref,path=None):
    fullref(ref,path);require(type(data) is bytes and len(data)==ref['bytes']
        and sha(data)==ref['sha256'],'externally pinned raw bytes differ');return strict_json(data)


def identity_map(value):
    require(type(value) is dict and value,'nonempty immutable identity map required')
    for path,info in value.items():
        require(type(info) is dict and set(info)=={'bytes','sha256'},'exact identity-map fields required')
        fullref({'path':path,**info})
    return value


def helper_map(value):
    require(type(value) is dict and value,'helper map required')
    for name,digest in value.items():
        require(type(name) is str and name==Path(name).name and name.endswith('.py'),'helper basename required')
        strict_sha(digest)
    require(value.get('white_view_paired_linear.py')==HELPER_SHA256,'fixed pure helper changed')
    require({'white_view_fit_contract.py','fit_white_view_paired_linear.py','white_view_build_contract.py'}<=set(value),'new dedicated sources required')
    return value


def feature_binding(value):
    white.validate_binding(value)
    require(exact(value['feature_source_sha256'],FEATURE_HASHES)
        and value['helper_source_sha256']==HELPER_SHA256,'fixed new source/helper identities required')
    return value


def build_binding(value):
    require(type(value) is dict and set(value)=={'runtime','manifest','identity','validator_source'},
        'new runtime/build/source full bindings required')
    require(value['runtime']==str(RUNTIME),'wrong dedicated white runtime')
    fullref(value['manifest'],RUNTIME/'build-manifest.json')
    fullref(value['identity'],RUNTIME/'white-view-build-identity.json')
    fullref(value['validator_source'])
    return value


def flags(record, expected):
    require(type(record) is dict,'typed record required')
    for key,value in expected.items():require(exact(record.get(key),value),'typed contract mismatch: '+key)


def validate_preregistration(record):
    flags(record,{'schema':'sekirei.white-view-paired-linear-preregistration.v1',
        'status':'frozen-before-fit','candidate':MODE,'mode':MODE,'plan_sha256':PLAN_SHA256,
        'dataset':str(DATASET),'dataset_manifest_sha256':MANIFEST_SHA256,
        'teacher_identity':white.ORIGINAL_TEACHER,'counts':COUNTS,'fit':FIT,'output':str(Q),
        'fit_started':False,'final_used':False,'adoption_claimed':False})
    feature_binding(record.get('feature_binding'));build_binding(record.get('new_build_binding'))
    helper_map(record.get('source_helpers'));identity_map(record.get('runtime_sources'))
    fullref(record.get('fit_driver'))
    fullref(record.get('initial_weights'),ORIGINAL_INITIALIZER)
    require(exact(record['initial_weights'],{'path':str(ORIGINAL_INITIALIZER),'bytes':white.WEIGHT_BYTES,
        'sha256':white.STOCK_INITIALIZER_SHA256}),'original stock01 initializer binding differs')
    fullref(record.get('trigger'),ACTIVATION)
    require(exact(record.get('source_preflight'),{'path':str(SOURCE_PREFLIGHT),'state':'required-before-fit'}),
        'source preflight contract required')
    return record


def validate_activation(record,prereg):
    validate_preregistration(prereg)
    flags(record,{'schema':'sekirei.white-view-paired-linear-activation.v1',
        'status':'verified-before-fit','candidate':MODE,'mode':MODE,'plan_sha256':PLAN_SHA256,
        'previous_candidate_valid_nonadopt':True,'previous_candidate_groups_stopped':True,
        'previous_archive_verified':True,'previous_public_docs_pushed':True,
        'new_build_verified':True,'source_preparation_reviewed':True,
        'inputs_unchanged':True,'source_unchanged':True,'fit_started':False,
        'final_used':False,'adoption_claimed':False})
    require(exact(record.get('feature_binding'),prereg['feature_binding'])
        and exact(record.get('new_build_binding'),prereg['new_build_binding']), 'activation new ABI/build differs')
    refs=record.get('previous_completion')
    require(type(refs) is dict and set(refs)=={'comparison','independent_review','stopped','archive','public_docs'},
        'previous valid-negative/STOP/NAS/public docs fullrefs required')
    for ref in refs.values():fullref(ref)
    identity_map(record.get('inputs_before'));require(exact(record['inputs_before'],record.get('inputs_after')),
        'activation unchanged inputs required')
    return record


def validate_source_preflight(record,prereg,prereg_sha):
    validate_preregistration(prereg);strict_sha(prereg_sha)
    flags(record,{'schema':'sekirei.white-view-paired-linear-original-source-preflight.v1',
        'status':'complete','candidate':MODE,'plan_sha256':PLAN_SHA256,
        'preregistration_sha256':prereg_sha,'activation_sha256':prereg['trigger']['sha256'],
        'source_helpers_sha256':prereg['source_helpers'],'counts':COUNTS,
        'teacher_identity':white.ORIGINAL_TEACHER,'original_manifest_sha256':MANIFEST_SHA256,
        'feature_binding':prereg['feature_binding'],'new_build_binding':prereg['new_build_binding'],
        'original_labels_reparsed':True,'raw1000_membership_reverified':True,'raw_replay_rerun':False,
        'original_legal_replay_inherited':True,'whole_pool_exclusions_reverified':True,
        'source_unchanged':True,'inputs_unchanged':True,'new_build_verified':True,
        'final_used':False,'fit_started':False})
    identity_map(record.get('inputs_before'))
    require(exact(record['inputs_before'],record.get('inputs_after')),'preflight unchanged full input map required')
    for key in ('inherited_replay_receipt','inherited_source_preflight'):fullref(record.get(key))
    for path in (PLAN,DATASET/'manifest.json',*(DATASET/name for name in
        ('train.positions.jsonl','train.labels.jsonl','holdout.positions.jsonl','holdout.labels.jsonl')),
        ORIGINAL_INITIALIZER,ACTIVATION,Path(prereg['new_build_binding']['manifest']['path']),
        Path(prereg['new_build_binding']['identity']['path'])):
        require(str(path) in record['inputs_before'],'mandatory original/new build source not bound')
    require(record['inputs_before'][str(DATASET/'manifest.json')]['sha256']==MANIFEST_SHA256,
        'fixed original manifest not bound')
    require(record['inputs_before'][str(PLAN)]['sha256']==PLAN_SHA256,'selected plan not bound')
    for ref in (prereg['initial_weights'],prereg['trigger'],*
        (prereg['new_build_binding'][key] for key in ('manifest','identity','validator_source')),
        record['inherited_replay_receipt'],record['inherited_source_preflight']):
        require(exact(record['inputs_before'].get(ref['path']),{'bytes':ref['bytes'],'sha256':ref['sha256']}),
            'source preflight reference absent or rebound')
    return record


def finite_number(value):
    if type(value) in (int,float):
        require(type(value) is int or math.isfinite(value),'finite numeric scalar required')
        return Fraction(value)
    require(type(value) is dict and set(value)=={'numerator','denominator'}
        and type(value['numerator']) is int and type(value['denominator']) is int
        and value['denominator']>0,'canonical exact rational required')
    q=Fraction(value['numerator'],value['denominator'])
    require(q.numerator==value['numerator'] and q.denominator==value['denominator'], 'noncanonical rational')
    return q


def numeric_execution(value):
    fixed={'schema':'sekirei.white-view-paired-linear-numeric-execution.v1',
        'whole_child_budget_seconds':1200,'parent_process_group_deadline_required':True,
        'boundary_checks_cannot_interrupt_blas':True,'single_gram_produced':True,
        'single_fixed_fit':True,'lambda':1,'max_iterations':20000,
        'loss_half_unnormalized_sum':True,'saved_f32_positive_zero':True,'actual_cleanup_claimed':False}
    extra={'remaining_inner_budget_seconds','fit_wall_seconds','whole_child_elapsed_seconds',
        'iterations','restart_count','openblas'}
    require(type(value) is dict and set(value)==set(fixed)|extra,'exact numeric execution fields required')
    flags(value,fixed)
    require(type(value['iterations']) is int and 0<=value['iterations']<=20000
        and type(value['restart_count']) is int and 0<=value['restart_count']<=value['iterations'],
        'bounded iterations/restart counts required')
    require(0<finite_number(value['remaining_inner_budget_seconds'])<=1200
        and 0<=finite_number(value['fit_wall_seconds'])<1200
        and 0<=finite_number(value['whole_child_elapsed_seconds'])<1200,'child wall budget violated')
    pool=value['openblas']
    flags(pool,{'schema':'sekirei.numpy-openblas-thread-proof.v1','numpy_version':'1.26.4',
        'threads_during':1,'threads_after':1,'rtld_noload':True,'threadpool_verified':True,
        'get_symbol':'openblas_get_num_threads64_','set_symbol':'openblas_set_num_threads64_',
        'config_symbol':'openblas_get_config64_'})
    fullref(pool.get('library'))
    require(type(pool.get('threads_before')) is int and pool['threads_before']>=1
        and type(pool.get('numpy_preimported_when_driver_loaded')) is bool
        and type(pool.get('config')) is str and 'USE64BITINT' in pool['config'], 'actual ILP64 pool proof required')
    expected={'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'}
    require(type(pool.get('environment')) is dict and set(pool['environment'])==expected
        and all(v=='1' for v in pool['environment'].values()), 'all six thread variables must be one')
    return value


def make_run_template(context,outputs,solverref):
    required={'preregistration_sha256','activation_sha256','source_helpers_sha256',
        'source_preflight_sha256','inputs_before','inputs_after','runtime_sources','numeric_execution',
        'feature_binding','new_build_binding','new_build_verified'}
    require(type(context) is dict and set(context)==required,'exact new fit context required')
    identity_map(context['inputs_before']);identity_map(context['runtime_sources'])
    require(exact(context['inputs_before'],context['inputs_after']),'fit input/source maps changed')
    for key in ('preregistration_sha256','activation_sha256','source_preflight_sha256'):strict_sha(context[key])
    helper_map(context['source_helpers_sha256']);feature_binding(context['feature_binding'])
    numeric_execution(context['numeric_execution'])
    build_binding(context['new_build_binding']);require(context['new_build_verified'] is True,'actual new build required')
    require(type(outputs) is dict and set(outputs)==set(OUTPUT_NAMES),'exact nine artifacts required')
    for key,name in OUTPUT_NAMES.items():fullref(outputs[key],Q/name)
    fullref(solverref,Q/'solver-certificate.json')
    require(exact(solverref,outputs['solver_certificate']),'output and solver reference differ')
    result={'schema':'sekirei.white-view-paired-linear-fit-run.v1','status':'complete',
        'candidate':MODE,'mode':MODE,'created_at':datetime.now(timezone.utc).isoformat(),
        'seed':42,'nnue_output':'absolute','native_magic':'SEKIRW03','feature_schema':white.FEATURE_SCHEMA,
        'optimizer':'projected-FISTA','adam_used':False,'epochs':0,'resume_used':False,'trainer_used':False,
        'inputs_unchanged':True,'source_unchanged':True,'cleanup_verified':False,
        'final_used':False,'adoption_claimed':False,'positions':112681,'dimension':254,
        'teacher_identity':white.ORIGINAL_TEACHER,'original_manifest_sha256':MANIFEST_SHA256,
        'material_initializer_sha256':white.STOCK_INITIALIZER_SHA256,
        'fit_holdout_used':False,'fit_development_used':False,'fit_final_used':False,
        'plan_sha256':PLAN_SHA256,'fit':dict(FIT),'outputs':outputs,'solver_certificate':solverref}
    result.update(context);return result


def construct_fit_run(context,outputs,solverref,lifecycle):
    """Parent only: supplied actual wait/reap/stop evidence, not process-table proof."""
    require(exact(lifecycle,LIFECYCLE),'actual zero exit/wait/reap/group-stop required')
    result=make_run_template(context,outputs,solverref)
    result.update(cleanup_verified=True,lifecycle=dict(lifecycle));return result


def fit_binding(binding):
    require(type(binding) is dict and set(binding)=={'plan_sha256','preregistration_sha256',
        'activation_sha256','fit_receipt','solver_certificate','source_helpers_sha256',
        'source_preflight_sha256','feature_binding','new_build_binding'},'external new fit binding fields required')
    require(binding['plan_sha256']==PLAN_SHA256,'wrong selected plan')
    for key in ('plan_sha256','preregistration_sha256','activation_sha256','source_preflight_sha256'):strict_sha(binding[key])
    fullref(binding['fit_receipt'],FIT_RECEIPT);fullref(binding['solver_certificate'],Q/'solver-certificate.json')
    helper_map(binding['source_helpers_sha256']);feature_binding(binding['feature_binding'])
    build_binding(binding['new_build_binding']);return binding


def validate_fit_run(run_bytes,native,coef,original01,binding):
    fit_binding(binding);run=raw_bound(run_bytes,binding['fit_receipt'],FIT_RECEIPT)
    require(type(run) is dict,'typed completed fit run required')
    context_keys={'preregistration_sha256','activation_sha256','source_helpers_sha256','source_preflight_sha256',
        'inputs_before','inputs_after','runtime_sources','numeric_execution','feature_binding','new_build_binding','new_build_verified'}
    template=make_run_template({key:run.get(key) for key in context_keys},run.get('outputs'),run.get('solver_certificate'))
    require(type(run.get('created_at')) is str and datetime.fromisoformat(run['created_at']).tzinfo is not None,
        'aware creation timestamp required')
    template['created_at']=run.get('created_at');template.update(cleanup_verified=True,lifecycle=dict(LIFECYCLE))
    require(exact(run,template),'exact completed new fit schema/fields/lifecycle required')
    for key in ('preregistration_sha256','activation_sha256','source_helpers_sha256','source_preflight_sha256',
        'feature_binding','new_build_binding','solver_certificate'):
        require(exact(run.get(key),binding[key]),'external run binding differs: '+key)
    require(exact(run['outputs']['weights'],{'path':str(W),'bytes':len(native),'sha256':sha(native)})
        and exact(run['outputs']['coefficients_f32'],{'path':str(Q/'coefficients.f32.bin'),
            'bytes':len(coef),'sha256':sha(coef)}),'native/coefficient bytes differ from run')
    white.validate_native(native,coef,original01,binding=binding['feature_binding']);return run


def sidecar_for_fit(native,coef,original01,run_bytes,binding):
    validate_fit_run(run_bytes,native,coef,original01,binding)
    validated=white.validate_native(native,coef,original01,binding=binding['feature_binding'])
    return {'format':'sekirei-nnue-output-v1','nnue_output':'absolute','checkpoint_hash':material_init.fnv1a(native),
        'white_view_paired_linear_constrained_ridge_v1':{
            'schema':'sekirei.white-view-paired-linear-fit-metadata.v1','mode':MODE,'plan_sha256':PLAN_SHA256,
            'native_magic':'SEKIRW03','feature_schema':white.FEATURE_SCHEMA,'seed':42,
            'weight_sha256':sha(native),'weight_bytes':len(native),
            'material_initializer_sha256':sha(original01),'saved_f32_coefficient_l1':validated['saved_f32_coefficient_l1'],
            'coefficient_l1_cap':{'numerator':79,'denominator':2},'fit_binding':binding,
            'optimizer':'projected-FISTA','adam_used':False,'epochs':0,'resume_used':False,'trainer_used':False,
            'canonical_new_abi_rebuild_equal':True,'quantization_reexport_claimed':False,
            'native_bit_exact_covariance_verified':False,'actual_core_bound_verified':False,'adoption_verified':False}}


def validate_sidecar(meta_bytes,native,coef,original01,run_bytes,binding):
    require(exact(strict_json(meta_bytes),sidecar_for_fit(native,coef,original01,run_bytes,binding)),
        'new dedicated absolute sidecar differs');return True


def runtime_entry(*args,**kwargs):
    raise RuntimeError('SOURCE ONLY contract has no actual input/activation/fitting entry')
