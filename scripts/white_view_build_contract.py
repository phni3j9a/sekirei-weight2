"""Pure contracts for a dedicated white-view build; no file/probe/model access.

A validated document is a declaration. The disabled worker's verify_runtime
must check actual bytes, inventory, compiler and lock-protected source state.
"""
import hashlib
import json
from pathlib import PurePosixPath
import re

RUNTIME = '/home/server/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-white-view-v1'
BASE = 'f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9'
PLAN = '1f274b187a96b16674814c51f77cb402ca8a80e6a7e5d738a5caf9354127e836'
PATCH = '38bbf3e47e7a38a25cbd45a2a1cf2dbee602e09ad9fdc518017ee64661baf729'
LOCK = '4343a043250cf300e4b029f3dc9f953f1ec047bd6bcb4024efeda0a3e9b5bca5'
FLAGS = '-C target-cpu=x86-64-v3'
FEATURE = 'nnue_white_view_aux_tied'
MODE = 'white-view-paired-linear-constrained-ridge1-l1-39p5-v1'
CHANGED = ('crates/sekirei-core/src/nnue.rs', 'crates/sekirei-core/Cargo.toml', 'crates/sekirei-usi/Cargo.toml')
CARGO_CHANGED = set(CHANGED[1:])
ORIGINAL = {
    CHANGED[0]: {'bytes':40063,'sha256':'a467e8b1b75f6b82629f369a1c804476b1c610354a561822365f980e6e428561'},
    CHANGED[1]: {'bytes':551,'sha256':'c8d3e853f957c828a9273268f99a25057464ef972a3cf503093a39077f5bb3e8'},
    CHANGED[2]: {'bytes':431,'sha256':'d3d9f4a68af42c234b01dfadcf09bb4e956bcf0f32ad0dd190ee74a71d9a4fb2'}}
PATCHED = {
    CHANGED[0]: {'bytes':52355,'sha256':'083c99ed681108d1643fa897aa2a4c3f452a170a73219fa3dc2cf16cdf68de50'},
    CHANGED[1]: {'bytes':581,'sha256':'c6c0d7a2e60b988e2975c5bc5b7dff620fe74b4426ff1ffc69474a0fa5cc2fc8'},
    CHANGED[2]: {'bytes':500,'sha256':'f0f9689226fc59c10b7d1adc10a101f9f31914bc6a3e342b29b68917ef73ffea'}}
ARCH = {'compile_feature':FEATURE,'feature_schema':'flat_white_view_aux_tied_v1','native_magic':'SEKIRW03',
    'dimensions':{'input':2420,'l1':256,'l2':32},'core_features':[FEATURE],'usi_features':[FEATURE],
    'king_relative_b_small':False,'old_magic_loader_guards_preserved':True,
    'hand_aux_donor':{'0=3':0,'1=2':1},'search_board_eval_usi_main_sources_unchanged':True}
TEACHER_WEIGHT = 'models/suisho11beta-concerto-202512/nn.bin'
TEACHER_FILES = {'bin/sekirei-train','bin/shogiesa','bin/yaneuraou',TEACHER_WEIGHT}
WEIGHT = {'bytes':112887658,'sha256':'d1b16f0adffab3183faeb5740ea11185734236c23aedd715c89c1bb4e8e67785'}
STOCK_INITIALIZER = 'bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40'
# Stock initializer is separately pinned by the pure feature serializer; checked
# against its source declaration by Root before actual feature binding is used.
PROBES = {'core_pair','incremental','serialization'}
PROBE_SOURCE_SHA = {'core_pair':'975073d2aef8518a86ba41b88468e2db0e548a494d6f4a583bc548957e1a6963',
    'incremental':'599335c5852a726d2430c84699578462c9d4657281ad7400aa5d03d72ab9a872',
    'serialization':'eb20fce145b40178ee356e15ceb4b0f891f7bcb1a70d08a4a11e5c1ba444ee09'}
BUDGET = {'maximum_added_bytes':8*2**30,'minimum_remaining_bytes':2*2**30}
SOURCES = {
    'sekirei': {'url':'https://github.com/kent-tokyo/sekirei.git','commit':BASE,'version':'0.3.39','ref':'refs/tags/v0.3.39','packages':['sekirei','sekirei-train']},
    'shogiesa': {'url':'https://github.com/kent-tokyo/shogiesa.git','commit':'dd0317437dc43b4d9b299e11542d82843f538b9d','version':'0.9.2','packages':['shogiesa-cli']},
    'yaneuraou': {'url':'https://github.com/yaneurao/YaneuraOu.git','commit':'a81730f47eefa4d53003ed85034715a28d2437ab','version':'9.20','edition':'YANEURAOU_ENGINE_NNUE_SFNNwoP1536','target_cpu':'AVX2'}}
PROTECTED = {'Cargo.lock','crates/sekirei-core/src/board.rs','crates/sekirei-core/src/search.rs',
             'crates/sekirei-core/src/eval.rs','crates/sekirei-usi/src/main.rs'}
OLD_TESTS = sorted('nnue::tests::'+s for s in (
    'feature_index_separates_square_kind_color_and_perspective','incremental_piece_and_hand_updates_match_full_refresh',
    'board_mailbox_refresh_matches_snapshot_refresh','explicit_weights_preserve_piece_sensitivity_and_side_sign',
    'read_weights_rejects_trailing_bytes','read_weights_rejects_non_finite_values',
    'save_weights_rejects_non_finite_values_before_writing','save_weights_atomically_replaces_existing_file',
    'save_and_read_weights_are_bitwise_deterministic'))
NEW_TESTS = sorted('nnue::white_view_tests::'+s for s in (
    'white_view_exhaustive_board_coordinates_and_physical_color_rotation',
    'white_view_preserves_all_hand_banks_and_ties_lcg_default',
    'white_view_hand_validator_rejects_material_auxiliary_and_shape_mismatch',
    'white_view_native_roundtrip_requires_magic03_and_tied_rows',
    'white_view_save_rejects_untied_rows_before_replacing_destination',
    'white_view_physical_color_rotation_refresh_and_static_forward_match',
    'white_view_incremental_capture_drop_and_undo_match_refresh'))


def require(ok, message):
    if not ok: raise ValueError(message)


def exact(a,b):
    if type(a) is not type(b): return False
    if type(a) is dict: return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if type(a) is list: return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def sha(s):
    require(type(s) is str and re.fullmatch('[0-9a-f]{64}',s),'strict lowercase SHA256 required'); return s


def absolute(s):
    require(type(s) is str and PurePosixPath(s).is_absolute() and str(PurePosixPath(s))==s
            and '..' not in PurePosixPath(s).parts,'canonical absolute path text required'); return s


def relative(s):
    require(type(s) is str and s and not PurePosixPath(s).is_absolute() and str(PurePosixPath(s))==s
            and '..' not in PurePosixPath(s).parts,'canonical relative file required'); return s


def info(v):
    require(type(v) is dict and set(v)=={'bytes','sha256'},'exact bytes/SHA identity required')
    require(type(v['bytes']) is int and v['bytes']>=0,'strict byte count required'); sha(v['sha256']); return v


def fullref(v):
    require(type(v) is dict and set(v)=={'path','bytes','sha256'},'exact full file reference required')
    absolute(v['path']); info({k:v[k] for k in ('bytes','sha256')}); return v


def filemap(v, *, absolute_paths=False):
    require(type(v) is dict and v,'nonempty file inventory required')
    for name,record in v.items():
        (absolute if absolute_paths else relative)(name); info(record)
    return v


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def pinned_json(raw, expected):
    require(type(raw) is bytes and hashlib.sha256(raw).hexdigest()==sha(expected),'parsed bytes must equal external SHA')
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key'); result[k]=v
        return result
    def bad(s): raise ValueError('nonfinite JSON: '+s)
    def finite(s):
        v=float(s); require(abs(v)!=float('inf'),'overflowed JSON float'); return v
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=finite)


def fixed(v, values):
    require(type(v) is dict and all(exact(v.get(k),x) for k,x in values.items()),'fixed typed fields differ')


def validate_identity(v):
    fixed(v,{'schema':'sekirei.white-view-build-identity.v1','status':'frozen-before-build','runtime':RUNTIME,
             'base_commit':BASE,'plan_sha256':PLAN,'lock_sha256':LOCK,'rustflags':FLAGS,'build_jobs':2,
             'architecture':ARCH,'budget':BUDGET})
    require(type(v.get('producer_commit')) is str and re.fullmatch('[0-9a-f]{40}',v['producer_commit']),'producer repository commit required')
    for key in ('base_source','teacher_runtime'): absolute(v.get(key)); require(v[key]!=RUNTIME,'old and new runtime must differ')
    fullref(v.get('patch')); require(v['patch']['sha256']==PATCH and v['patch']['bytes']==16686,'frozen three-file patch required')
    files=filemap(v.get('source_files')); deps=filemap(v.get('dependency_files'))
    require(len(files)==526 and PROTECTED<=files.keys() and len(deps)==8 and set(deps)<=set(files),'fixed complete source/dependency counts required')
    require(exact({n:files.get(n) for n in CHANGED},ORIGINAL),'original three-file source binding differs')
    require(all(exact(files[n],x) for n,x in deps.items()) and CARGO_CHANGED<=deps.keys(),'dependency/source binding differs')
    require(exact(v.get('sources'),SOURCES),'legacy stock source descriptors required')
    compiler=v.get('compiler'); require(type(compiler) is dict and set(compiler)=={'rustc','cargo','target'}
        and all(type(s) is str and s for s in compiler.values()) and compiler['target']=='x86_64-unknown-linux-gnu','compiler/target required')
    filemap(v.get('compiler_files'),absolute_paths=True); filemap(v.get('helper_files'),absolute_paths=True)
    require(type(v.get('g++')) is str and v['g++'],'legacy teacher compiler string required')
    fullref(v.get('teacher_manifest'))
    teachers=v.get('teacher_files'); require(type(teachers) is dict and set(teachers)==TEACHER_FILES,'exact legacy teacher copies required')
    for rec in teachers.values(): fullref(rec)
    require(exact({k:teachers[TEACHER_WEIGHT][k] for k in WEIGHT},WEIGHT),'fixed Beta weight bytes differ')
    declared=v.get('teacher_declared_binary_paths'); require(type(declared) is dict and set(declared)=={'sekirei-train','shogiesa','yaneuraou'},'legacy declared binary paths required')
    for path in declared.values(): absolute(path)
    probes=v.get('probe_sources'); require(type(probes) is dict and set(probes)==PROBES,'all three new core probe sources required')
    for rec in probes.values(): fullref(rec)
    require(exact({p:probes[p]['sha256'] for p in PROBES},PROBE_SOURCE_SHA),'fixed probe sources required')
    return v


def validate_source_transition(v, identity):
    before=filemap(v.get('source_files_before')); after=filemap(v.get('source_files_after'))
    require(exact(before,identity['source_files']) and set(before)==set(after),'original full source inventory differs')
    require({p for p in before if not exact(before[p],after[p])}==set(CHANGED)
            and exact(v.get('changed_source_paths'),list(CHANGED)),'exact three-file source change required')
    require(exact({p:after[p] for p in CHANGED},PATCHED),'patched source SHA differs')
    db=filemap(v.get('dependency_files_before')); da=filemap(v.get('dependency_files_after'))
    require(exact(db,identity['dependency_files']) and set(db)==set(da)
        and {p for p in db if not exact(db[p],da[p])}==CARGO_CHANGED,'only project Cargo2 dependency changes permitted')
    require(all(exact(db[p],before[p]) and exact(da[p],after[p]) for p in db),'both dependency endpoints must match full source')
    for key,wanted in (('build_source_files_before',after),('build_source_files_after',after),
                       ('build_dependency_files_before',da),('build_dependency_files_after',da)):
        require(exact(v.get(key),wanted),'build source/dependency inventory changed')


def expected_commands(identity, rlib):
    source=RUNTIME+'/source'; build=RUNTIME+'/build'
    commands={
        'clone':['git','clone','--no-hardlinks','--no-checkout',identity['base_source'],source],
        'checkout':['git','checkout','--detach',BASE],
        'patch-check':['git','apply','--check',identity['patch']['path']],
        'patch-apply':['git','apply',identity['patch']['path']],
        'conflict':['cargo','check','--release','--locked','--offline','-j','2','-p','sekirei','--features',FEATURE+',king_relative_b_small','--target-dir',build+'/conflict'],
        'usi-build':['cargo','build','--release','--locked','--offline','-j','2','-p','sekirei','--bin','sekirei','--features',FEATURE,'--target-dir',build+'/white-view','--message-format=json-render-diagnostics']}
    for key,features in (('default',[]),('white_view',[FEATURE]),('b_small',['king_relative_b_small'])):
        argv=['cargo','test','--release','--locked','--offline','-j','2','-p','sekirei-core','--lib','--target-dir',build+'/'+key]
        if features: argv+=['--features',','.join(features)]
        commands[key+'-tests']=argv+['nnue::','--','--test-threads=1']
    for name in sorted(PROBES):
        commands[name+'-compile']=['rustc','--edition=2024','-C','opt-level=2','-C','target-cpu=x86-64-v3','-C','panic=abort',
            '--extern','sekirei_core='+rlib,'-L','dependency='+build+'/white-view/release/deps',identity['probe_sources'][name]['path'],'-o',RUNTIME+'/probes/'+name]
    return commands


def declared_identity_inputs(identity):
    result={}
    def add(path,record):
        absolute(path); info(record)
        require(path not in result or exact(result[path],record),'immutable input collision')
        result[path]=record
    for key in ('helper_files','compiler_files'):
        for path,record in identity[key].items(): add(path,record)
    for record in [identity['patch'],identity['teacher_manifest'],*identity['teacher_files'].values(),*identity['probe_sources'].values()]:
        add(record['path'],{k:record[k] for k in ('bytes','sha256')})
    return result


def validate_manifest(v, identity, identity_ref):
    validate_identity(identity); fullref(identity_ref)
    fixed(v,{'schema':'sekirei.white-view-runtime-build.v1','status':'complete','runtime':RUNTIME,
        'base_commit':BASE,'plan_sha256':PLAN,'patch_sha256':PATCH,'lock_sha256':LOCK,'rustflags':FLAGS,
        'build_jobs':2,'architecture':ARCH,'sources':identity['sources'],'compiler':identity['compiler'],
        'rustc':identity['compiler']['rustc'],'cargo':identity['compiler']['cargo'],'g++':identity['g++'],
        'identity':identity_ref,'source_unchanged':True,'inputs_unchanged':True,'compiler_unchanged':True,
        'dependency_unchanged_during_build':True,'engine_or_model_probe_started':False,'fit_started':False,
        'teacher_rebuilt':False,'teacher_copy_method':'regular-byte-copy','base_source_unchanged':True,
        'teacher_legacy_declared_binary_paths':identity['teacher_declared_binary_paths'],'budget':BUDGET,'producer_commit':identity['producer_commit']})
    validate_source_transition(v,identity)
    require(type(v.get('actual_added_bytes')) is int and 0<=v['actual_added_bytes']<=BUDGET['maximum_added_bytes']
        and type(v.get('remaining_free_bytes_before_manifest')) is int and v['remaining_free_bytes_before_manifest']>=BUDGET['minimum_remaining_bytes'],'strict build budget observation required')
    require(exact(v.get('base_source_files_before'),identity['source_files'])
        and exact(v.get('base_source_files_after'),identity['source_files']),'old base source changed during preparation')
    require(exact(v.get('compiler_files'),identity['compiler_files']) and exact(v.get('inputs_before'),v.get('inputs_after')),'immutable input/compiler maps differ')
    filemap(v['inputs_before'],absolute_paths=True)
    external=fullref(v.get('external_identity_input'))
    require(exact({k:external[k] for k in ('bytes','sha256')},{k:identity_ref[k] for k in ('bytes','sha256')}),'copied identity differs from original external bytes')
    required_inputs=declared_identity_inputs(identity)
    required_inputs[external['path']]={k:external[k] for k in ('bytes','sha256')}
    require(all(exact(v['inputs_before'].get(path),record) for path,record in required_inputs.items()),'declared external immutable inputs not fully bound')
    copies=v.get('teacher_copies'); require(type(copies) is dict and set(copies)==TEACHER_FILES,'complete regular teacher copies required')
    for name,record in copies.items():
        fixed(record,{'source':identity['teacher_files'][name],'source_before':identity['teacher_files'][name],
                     'source_after':identity['teacher_files'][name],'regular_file':True,'different_inode':True})
        fullref(record.get('copy')); require(record['copy']['path']==RUNTIME+'/'+name
            and exact({k:record['copy'][k] for k in WEIGHT},{k:identity['teacher_files'][name][k] for k in WEIGHT}),'teacher byte copy differs')
    bins=v.get('binaries'); require(type(bins) is dict and set(bins)=={'sekirei','sekirei-train','shogiesa','yaneuraou'},'legacy binary layout required')
    for name,record in bins.items():
        fullref(record); require(record['path']==RUNTIME+'/bin/'+name,'runtime binary layout differs')
        if name!='sekirei': require(exact(record,copies['bin/'+name]['copy']),'legacy copied binary identity differs')
    tests=v.get('rust_tests'); require(type(tests) is dict and set(tests)=={'default','white_view','b_small'},'three feature test profiles required')
    for key,names in (('default',OLD_TESTS),('white_view',sorted(OLD_TESTS+NEW_TESTS)),('b_small',OLD_TESTS)):
        require(exact(tests[key],{'expected':names,'passed':names,'failed':0,'ignored':0}),'actual NNUE test inventory differs')
    conflict=v.get('feature_conflict'); fixed(conflict,{'expected_failure':True,'compile_error_verified':True,'reaped':True,'group_stopped':True})
    require(type(conflict.get('returncode')) is int and conflict['returncode']!=0,'dual feature check must fail')
    probes=v.get('probe_binaries'); require(type(probes) is dict and set(probes)==PROBES,'newly linked core probes required')
    for record in probes.values(): fullref(record)
    link=v.get('core_link'); fixed(link,{'actual_usi_core_fingerprint_verified':True,'core_features':[FEATURE],'panic_strategy':'abort','profile':'release'})
    for key in ('rlib','core_fingerprint','core_fingerprint_json','usi_fingerprint_json','actual_usi_dependency_binary'): fullref(link.get(key))
    require(exact({k:link['actual_usi_dependency_binary'][k] for k in WEIGHT},{k:bins['sekirei'][k] for k in WEIGHT}),'USI dependency binary differs')
    require(type(link.get('core_dependency_fingerprint')) is int and link['core_dependency_fingerprint']>=0,'actual core dependency integer required')
    filemap(v.get('release_dependencies_before_probes'),absolute_paths=True)
    require(exact(v['release_dependencies_before_probes'],v.get('release_dependencies_after_probes')),'release dependency membership/bytes changed during probe compilation')
    for key in ('rlib','actual_usi_dependency_binary'):
        record=link[key]
        require(exact(v['release_dependencies_after_probes'].get(record['path']),{k:record[k] for k in ('bytes','sha256')}),'USI/core artifact missing from release dependency inventory')
    steps=v.get('steps'); require(type(steps) is list and steps,'parent-supervised steps required')
    commands=expected_commands(identity,link['rlib']['path'])
    require(len(steps)==len(commands) and {s.get('name') for s in steps if type(s) is dict}==set(commands),'exact parent command inventory required')
    for step in steps:
        fixed(step,{'reaped':True,'group_stopped':True,'timeout':False,'cleanup_status':'ok'})
        require(type(step.get('returncode')) is int and type(step.get('argv')) is list
                and step['argv'] and all(type(s) is str for s in step['argv']),'strict child outcome/argv required')
        fullref(step.get('log'))
        require(exact(step['argv'],commands[step['name']]) and step['log']['path']==RUNTIME+'/'+step['name']+'.log','fixed compile/test command/log differs')
        require(step['returncode']==0 or exact(step,conflict),'only feature conflict may fail')
    require(exact(conflict,next((s for s in steps if s.get('name')=='conflict'),None)),'feature conflict outcome not in supervised steps')
    return v


def immutable_inputs_for_runtime(verified, manifest_ref):
    """Pure full map to rehash under caller locks after actual verify_runtime.

    Source/dependency maps retain both original and patched endpoints. The
    caller provides the externally pinned actual manifest reference; no self
    hash is put in the manifest or frozen identity.
    """
    require(type(verified) is dict and set(verified)=={'manifest','identity'},'verified runtime bundle required')
    v=verified['manifest']; i=verified['identity']; validate_manifest(v,i,v['identity']); fullref(manifest_ref)
    require(manifest_ref['path']==RUNTIME+'/build-manifest.json','actual manifest path required')
    result=dict(v['inputs_before'])
    def add(path,record):
        require(path not in result or exact(result[path],record),'runtime inventory collision'); result[path]=record
    for root,key in ((i['base_source'],'source_files_before'),(RUNTIME+'/source','source_files_after')):
        for path,record in v[key].items(): add(root+'/'+path,record)
    def refs(value):
        if type(value) is dict:
            if set(value)=={'path','bytes','sha256'}:
                fullref(value); add(value['path'],{k:value[k] for k in ('bytes','sha256')})
            else:
                for sub in value.values(): refs(sub)
        elif type(value) is list:
            for sub in value: refs(sub)
    refs(v); refs(manifest_ref)
    for path,record in v['release_dependencies_after_probes'].items(): add(path,record)
    return result


def feature_binding_for_runtime(verified, helper_source_sha256):
    """Pure declaration from an already verified bundle, not an I/O certificate."""
    require(type(verified) is dict and set(verified)=={'manifest','identity'},'verified runtime bundle required')
    v=verified['manifest']; validate_manifest(v,verified['identity'],v['identity']); sha(helper_source_sha256)
    return {'schema':'sekirei.white-view-paired-linear-feature-binding.v1','mode':MODE,'plan_sha256':PLAN,
        'feature_schema':ARCH['feature_schema'],'compile_feature':FEATURE,'native_magic':'SEKIRW03',
        'dimensions':ARCH['dimensions'],'stock_initializer_sha256':STOCK_INITIALIZER,
        'feature_source_sha256':{p:v['source_files_after'][p]['sha256'] for p in CHANGED},
        'core_binary_sha256':v['binaries']['sekirei']['sha256'],'helper_source_sha256':helper_source_sha256}
