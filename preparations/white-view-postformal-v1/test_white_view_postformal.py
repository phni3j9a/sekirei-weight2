"""Synthetic-only postformal contract tests; never read any actual artifact."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

D4 = Path(__file__).resolve().parent.parent/'white-view-evaluation-runtime-v4'
sys.path.insert(0, str(D4))
import white_view_evaluation as w
import white_view_production_ports as p
from test_white_view_evaluation import bundle, identity, ref
import white_view_postformal_contract as m


def fixture(adopt=False):
    refs = {k: ref(str(path)) for k,path in m.FIXED_REF_PATHS.items()}
    refs['factory_source'] = ref('/tmp/sekirei-weight2-white-view-root-formal-records-factory-v2.py')
    refs['factory_source']['sha256'] = m.FACTORY_SHA
    refs['opaque_allowlist']['sha256'] = m.ALLOW_SHA
    bind = {'schema':'sekirei.white-view-candidate-gate-binding.v1'}
    for name in w.GATE_BINDING_NAMES: bind[name] = ref('/tmp/public-'+name+'.bin')
    bind['native']['bytes'] = 1305356
    bind['core']=ref('/tmp/public-core-proof/receipt.json')
    bind['incremental']=ref('/tmp/public-incremental-proof/receipt.json')
    bind['gate'] = refs['candidate_gate']
    bind['gate_worker'] = ref(str(p.C/'white-view-candidate-model-gate-worker-v2.py'))
    model = {'kind':'nnue', **bind['native']}
    modules = {n:ref('/tmp/public/'+n+'.py') for n in m.MODULE_NAMES}
    inputs = {}
    for record in [*refs.values(), *modules.values(), *[bind[k] for k in w.GATE_BINDING_NAMES]]:
        inputs[record['path']] = {k:record[k] for k in ('bytes','sha256')}
    build = {'runtime':w.RUNTIME, 'manifest':ref(w.RUNTIME+'/build-manifest.json'),
             'identity':ref('/tmp/public-build-identity.json'), 'validator_source':ref('/tmp/public-builder.py')}
    config = ref(str(p.R/'config/development-benchmark.json'))
    for record in [config,*[build[k] for k in ('manifest','identity','validator_source')]]:
        inputs[record['path']] = {k:record[k] for k in ('bytes','sha256')}
    helpers = {record['path']:record['sha256'] for record in modules.values()}
    for name in p.PUBLIC_NAMES:
        record=ref(str(p.R/'scripts'/(name+'.py')))
        inputs[record['path']]=identity();helpers[record['path']]=record['sha256']
    spec=w.make_spec('candidate','development-public-v1','/tmp/public-candidate-eval',model,
        build['manifest'],build['identity'],helpers,inputs,refs['candidate_gate'],refs['bridge'])
    pf={'schema':'sekirei.white-view-evaluation-launch-preflight.v1','status':'frozen-before-first-write',
        'spec':spec,'config_source':config,'new_build_binding':build,
        'repo_identity':{'commit':m.HEAD,'dirty':False,'status_porcelain':''},
        'decoder_versions':{'numpy':'1.26.4','cshogi':'1.0.4'},'preregistration':bind['preregistration'],
        'candidate_binding':bind,'candidate_validator':bind['gate_worker']}
    req={'schema':'sekirei.white-view-postformal-request.v1','status':'frozen-after-reaped-comparison',
        'candidate':m.MODE,'source_head':m.HEAD,'executor_session_id':123,'comparison_session_id':124,
        'model':model,'refs':refs,'gate_binding':bind,'inputs':inputs,'source_helpers':helpers,
        'modules':modules,'output':str(m.OUT),'final_used':False,'goal_complete':False}
    # Isolate the exact public gate envelope expression in memory only. This is
    # a schema fixture, not a proof producer or actual native/fit verification.
    gate_inputs={k:v for k,v in inputs.items() if k!=bind['gate']['path']}
    env=dict(vars(w));env.update(expected_binding=bind,expected_model=model,expected_build_binding=build,
        inputs=gate_inputs,feature={'core_binary_sha256':'a'*64})
    fn=next(n for n in ast.parse((D4/'white_view_evaluation.py').read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name=='validate_native_gate')
    for target in ('fixed_feature','proofs','fixed'):
        assignment=next(n for n in fn.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id==target for t in n.targets))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'synthetic-envelope','exec'),env)
    gate=env['fixed']
    old=bundle(runtime=str(m.B));fallback=bundle();candidate=bundle(role='candidate',error_delta=-1 if adopt else 0)
    candidate['model']=model
    old['audit_ref']=refs['old_audit'];fallback['audit_ref']=refs['fallback_audit'];candidate['audit_ref']=refs['candidate_audit']
    candidate['run_ids']=spec['run_ids'];candidate['terminal_ref']=refs['candidate_terminal']
    bridge=w.fallback_bridge(old,fallback,{'old_audit':old['audit_ref'],'new_audit':fallback['audit_ref'],
        'old_runtime':str(m.B),'old_build':old['build_manifest'],'new_build':fallback['build_manifest']})
    cp=w.compare_same_runtime(fallback,candidate,bridge,refs['bridge'],model)
    cp.update(baseline_audit=refs['fallback_audit'],candidate_audit=refs['candidate_audit'])
    cr={'schema':'sekirei.white-view-comparison-request.v1','status':'frozen-before-comparison',
        'action':'compare','source_inputs':inputs,'source_helpers':helpers,'output':refs['comparison']['path'],
        'baseline_audit':refs['fallback_audit'],'candidate_audit':refs['candidate_audit'],'bridge':refs['bridge'],'model':model}
    term={'schema':'sekirei.white-view-comparison-terminal.v1','status':'observed-stopped',
        'comparison_session_id':124,'exit_code':0,'session_closed':True,'tool_observed_reaped':True,
        'all_related_groups_observed_stopped':True,'comparison':refs['comparison'],
        'comparison_request':refs['comparison_request'],'source_head':m.HEAD,'final_used':False,'goal_complete':False}
    candidate_term={'schema':'sekirei.white-view-evaluation-terminal.v1','status':'observed-stopped',
        'exit_code':0,'session_closed':True,'tool_observed_reaped':True,'all_related_groups_observed_stopped':True,
        'role':'candidate','model':model,'run_ids':spec['run_ids'],'executor_session_id':123,
        'source_head':m.HEAD,'final_used':False,'goal_complete':False}
    allow={'descriptors':{str(i):{'pid':i,'start_ticks':1} for i in range(1,6)},'excluded_reason':'synthetic fixed service'}
    docs={'comparison_terminal':term,'comparison':cp,'comparison_request':cr,'candidate_preflight':pf,
        'candidate_gate':gate,'candidate_terminal':candidate_term,'opaque_allowlist':allow}
    audits={'old':old,'fallback':fallback,'candidate':candidate}
    for subject,audit in audits.items():
        roots=[str(Path(audit['runtime'])/'runs'/('development-public-'+subject+'-'+stage)) for stage in w.STAGES]
        audit['directory_inventory']={root:['.'] for root in roots}
        files={root+'/manifest.json':identity() for root in roots}
        audit['audit_inputs_before']=copy.deepcopy(files);audit['audit_inputs_after']=copy.deepcopy(files)
        inputs.update(files)
    return req,docs,audits,bridge


class Tests(unittest.TestCase):
    def test_full_schema_normal_valid_negative_and_positive(self):
        for adopted in (False,True):
            q,d,a,b=fixture(adopted)
            self.assertIs(m.validate_documents(q,d,a,b,w,p),adopted)

    def test_external_terminal_flags_sessions_and_int_types(self):
        for key,value in [('exit_code',True),('exit_code',1),('session_closed',False),
                          ('tool_observed_reaped',False),('comparison_session_id',True)]:
            q,d,a,b=fixture();d['comparison_terminal'][key]=value
            with self.assertRaises(ValueError):m.validate_documents(q,d,a,b,w,p)

    def test_comparison_exact_whole_result_and_joint_adoption(self):
        for key,value in [('adopt',True),('mae_improved',True),('top3_preserved',1)]:
            q,d,a,b=fixture();d['comparison'][key]=value
            with self.assertRaises(ValueError):m.validate_documents(q,d,a,b,w,p)

    def test_rational_result_tamper_and_old_comparator_schema(self):
        q,d,a,b=fixture();d['comparison']['candidate_exact']['mae']['numerator']+=1
        with self.assertRaises(ValueError):m.validate_documents(q,d,a,b,w,p)
        q,d,a,b=fixture();d['comparison']['schema']='sekirei.formal-comparison.v1'
        with self.assertRaises(ValueError):m.validate_documents(q,d,a,b,w,p)

    def test_gate19_model_ref_and_typed_envelope(self):
        q,d,a,b=fixture();del q['gate_binding']['numeric_worker']
        with self.assertRaises(ValueError):m.validate_request(q,w)
        q,d,a,b=fixture();d['candidate_gate']['cleanup_verified']=1
        with self.assertRaises(ValueError):m.validate_documents(q,d,a,b,w,p)
        q,d,a,b=fixture();q['model']['sha256']='b'*64
        with self.assertRaises(ValueError):m.validate_request(q,w)

    def test_current_pf_role_head_and_terminal(self):
        for name,key,value in [('candidate_preflight','schema','sekirei.paired-linear-preflight.v1'),
                              ('candidate_terminal','executor_session_id',True)]:
            q,d,a,b=fixture();d[name][key]=value
            with self.assertRaises(ValueError):m.validate_documents(q,d,a,b,w,p)

    def test_full_closure_and_fixed_service_authority(self):
        q,d,a,b=fixture();del q['inputs'][q['refs']['bridge']['path']]
        with self.assertRaises(ValueError):m.validate_request(q,w)
        q,d,a,b=fixture();q['refs']['opaque_allowlist']['sha256']='b'*64
        with self.assertRaises(ValueError):m.validate_request(q,w)
        q,d,a,b=fixture();d['opaque_allowlist']['descriptors']['6']={'pid':6}
        with self.assertRaises(ValueError):m.validate_documents(q,d,a,b,w,p)

    def test_bridge_semantic_projection_not_only_headline(self):
        q,d,a,b=fixture();a['old']['semantic_projections']['sekirei570']['0']['score_cp_sente']+=1
        with self.assertRaises(ValueError):m.validate_documents(q,d,a,b,w,p)

    def test_stop_has_full_five_service_evidence_and_valid_negative(self):
        q,d,a,b=fixture();allow=d['opaque_allowlist'];excluded=[]
        for pid,desc in allow['descriptors'].items():
            excluded.append({'descriptor':desc,'denied_path':'/proc/'+pid+'/exe','errno':13,
                'allowlist':q['refs']['opaque_allowlist'],'reason':allow['excluded_reason'],'identity_reread_unchanged':True})
        scans=[{'related_processes':[],'excluded_preexisting_services':copy.deepcopy(excluded)} for _ in range(2)]
        result=m.stop_document(q,ref('/tmp/public-request.json'),scans,q['inputs'],q['inputs'],False,allow,a)
        self.assertTrue(result['all_groups_stopped']);self.assertFalse(result['adopt']);self.assertFalse(result['goal_complete'])
        scans[1]['excluded_preexisting_services'][0]=scans[1]['excluded_preexisting_services'][1]
        with self.assertRaises(ValueError):m.stop_document(q,ref('/tmp/public-request.json'),scans,q['inputs'],q['inputs'],False,allow,a)

    def test_archive_whole_file_set_size_hash_directories_source_unchanged(self):
        tree={'files':{'a':identity()},'directories':['.']}
        self.assertEqual(m.validate_archive_equality(tree,tree,tree),{'files':1,'directories':1,'bytes':4})
        for altered in ({'files':{'a':identity('b')},'directories':['.']},
                        {'files':{'a':identity(size=5)},'directories':['.']},
                        {'files':{'b':identity()},'directories':['.']},
                        {'files':{'a':identity()},'directories':['.','missing']}):
            with self.assertRaises(ValueError):m.validate_archive_equality(tree,tree,altered)
            with self.assertRaises(ValueError):m.validate_archive_equality(tree,altered,tree)


    def test_archive_request_normal_and_root_terminal_failure(self):
        q,d,a,b=fixture();allow=d['opaque_allowlist']
        excluded=[{'descriptor':desc,'denied_path':'/proc/'+pid+'/exe','errno':13,
            'allowlist':q['refs']['opaque_allowlist'],'reason':allow['excluded_reason'],'identity_reread_unchanged':True}
            for pid,desc in allow['descriptors'].items()]
        scans=[{'related_processes':[],'excluded_preexisting_services':copy.deepcopy(excluded)} for _ in range(2)]
        stopped=m.stop_document(q,ref('/tmp/public-request.json'),scans,q['inputs'],q['inputs'],False,allow,a)
        stopref=ref(str(m.OUT));stoptermref=ref('/tmp/public-stop-terminal.json')
        mapping={root:'runs/'+Path(root).name for root in stopped['audited_run_roots']}
        mapping.update({str(m.Q):'fit/run', '/tmp/public-core-proof':'proofs/core',
                        '/tmp/public-incremental-proof':'proofs/incremental',
                        '/tmp/public-completion-control':'completion/control'})
        records=dict(q['inputs'])
        controls=[stopref,stopped['request'],*stopped['refs'].values(),
                  *[stopped['gate_binding'][k] for k in w.GATE_BINDING_NAMES]]
        controls=list({row['path']:row for row in controls}.values())
        for rec in controls:records[rec['path']]={k:rec[k] for k in ('bytes','sha256')}
        original_paths={};files={};ledger={}
        for name,rec in stopped['audited_raw_files'].items():
            root=str(Path(name).parent);relative=mapping[root]+'/manifest.json'
            original_paths[relative]=name;files[relative]=rec
        for i,rec in enumerate(controls):
            copied=ref('/tmp/public-completion-control/%03d.bin'%i,size=rec['bytes'])
            copied['sha256']=rec['sha256'];relative='completion/control/%03d.bin'%i
            original_paths[relative]=copied['path'];files[relative]={k:rec[k] for k in ('bytes','sha256')}
            ledger[rec['path']]=copied
        directories=['.']+sorted(set(str(Path(name).parent) for name in files))
        inventory={'schema':'sekirei.white-view-selected-archive-inventory.v1','source_roots':mapping,
                   'files':files,'directories':directories,'original_paths':original_paths,'original_copy_ledger':ledger}
        mapref=ref('/tmp/public-map.json');invref=ref('/tmp/public-inventory.json')
        helper=ref(str(m.HELPER));helper['sha256']=m.HELPER_SHA
        for rec in (mapref,invref,helper,stoptermref):records[rec['path']]={k:rec[k] for k in ('bytes','sha256')}
        copied_originals=set(ledger)|set(original_paths.values())
        deps={name:rec for name,rec in records.items() if name not in copied_originals}
        ar={'schema':'sekirei.white-view-archive-request.v1','status':'frozen-after-stopped-before-nas-copy',
            'candidate':m.MODE,'source_head':m.HEAD,'stop':stopref,'stop_terminal':stoptermref,'mapping':mapref,
            'closure_inventory':invref,'source_roots':mapping,'inputs':records,'required_archived_refs':controls,
            'external_dependencies':deps,'destination':str(m.NAS/'archives/2026-10-04/public-white-view'),
            'receipt_name':'public-white-view','helper':helper,
            'parent_exclusive_locks':list(map(str,m.ARCHIVE_OUTER_LOCKS)),
            'helper_exclusive_locks':list(map(str,m.ARCHIVE_HELPER_LOCKS)),
            'source_originals_preserved':True,'standalone_environment_restore':False,'final_used':False,
            'adoption_applied':False,'best_model_updated':False,'goal_complete':False}
        terminal={'schema':'sekirei.white-view-postformal-stop-terminal.v1','status':'observed-stopped',
            'stop_receipt':stopref,'request':stopped['request'],'exit_code':0,'session_closed':True,
            'tool_observed_reaped':True,'all_related_groups_observed_stopped':True,'source_head':m.HEAD,
            'stop_session_id':125,'final_used':False,'goal_complete':False}
        self.assertIs(m.validate_archive_request(ar,stopped,terminal,mapping,inventory),ar)
        terminal['tool_observed_reaped']=False
        with self.assertRaises(ValueError):m.validate_archive_request(ar,stopped,terminal,mapping,inventory)
        terminal['tool_observed_reaped']=True
        del mapping[str(m.Q)]
        with self.assertRaises(ValueError):m.validate_archive_request(ar,stopped,terminal,mapping,inventory)

    def test_archive_overlap_and_shared_mutable_roots_rejected(self):
        for mapping in ({str(m.C):'campaign'}, {str(m.N):'runtime'},
            {'/tmp/public-tree':'tree','/tmp/public-tree/sub':'sub'},
            {'/tmp/public-first':'tree','/tmp/public-second':'tree/sub'}):
            with self.assertRaises(ValueError):m.validate_mapping(mapping)

    def test_factory_process_functions_ast_unchanged(self):
        original=ast.parse((Path(__file__).resolve().parent.parent/'white-view-source-fixtures-v1'/'factory-records-v2.py.txt').read_text())
        worker=ast.parse(Path(__file__).with_name('white-view-postformal-stop-worker-v1.py').read_text())
        def fun(t,n):return ast.dump(next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n),include_attributes=False)
        for n in ('proc_identity','stopped_processes'):self.assertEqual(fun(original,n),fun(worker,n))

    def test_cli_barrier_before_io(self):
        path=Path(__file__).with_name('white-view-postformal-stop-worker-v1.py')
        spec=importlib.util.spec_from_file_location('_synthetic_postformal_worker',path)
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        with mock.patch.object(Path,'read_bytes',side_effect=AssertionError('I/O forbidden')):
            with self.assertRaisesRegex(ValueError,'disabled before any actual I/O'):worker.main()


if __name__=='__main__':unittest.main()
