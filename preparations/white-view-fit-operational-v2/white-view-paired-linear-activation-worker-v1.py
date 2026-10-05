#!/usr/bin/env python3
"""SOURCE ONLY separate new03 activation producer; disabled before actual I/O.

Dependency order is frozen source inventory -> activation -> preregistration ->
original source preflight -> supervised fit. No PR/activation hash cycle exists.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
R=Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
sys.path.insert(0,str(R/'scripts'))
import white_view_fit_operational_common as common
import white_view_fit_contract as contract

PROTOTYPE_ONLY=True
WORKER=contract.C/'white-view-paired-linear-activation-worker-v1.py'


def activate(args):
    common.require(PROTOTYPE_ONLY is False,'SOURCE ONLY activation producer disabled before actual I/O')
    common.require_enabled();os.umask(0o077)
    reader=common.Reader();common.bootstrap(reader,Path(__file__),args)
    from benchmark import nonblocking_lock
    import diagnose_anchor as lifecycle
    with lifecycle.termination_guard(),ExitStack() as stack:
        locks=common.lock_paths()
        for path in locks:stack.enter_context(nonblocking_lock(path,exclusive=True))
        for path in common.architecture_shared_locks():stack.enter_context(nonblocking_lock(path,exclusive=False))
        common.bootstrap(reader,Path(__file__),args)
        inventory=contract.strict_json(reader.read(common.SOURCE_INVENTORY,args.expected_source_inventory_sha256))
        common.source_inventory(inventory)
        common.require(inventory['public_source_commit']==args.expected_public_source_head,'extern source HEAD differs')
        common.pin_helpers(reader,inventory['source_helpers'])
        common.pin_operational_sources(reader,inventory['operational_sources'])
        for path,rec in inventory['runtime_sources'].items():reader.pin(path,rec)
        reader.pin(contract.PLAN,contract.PLAN_SHA256)
        plan=contract.strict_json(reader.read(contract.PLAN,contract.PLAN_SHA256))
        common.require(plan.get('candidate')==contract.MODE and common.exact(plan.get('fit'),contract.FIT)
            and plan.get('actual_build_fit_model_or_probe_started') is False and plan.get('final_used') is False,
            'source-only selected fixed plan differs')
        docs={name:reader.document(ref) for name,ref in common.PREVIOUS.items()}
        common.validate_previous(docs,common.PREVIOUS)
        for path,rec in docs['stopped']['inputs_before'].items():reader.pin(path,rec)
        published=reader.document(common.PUBLIC_DOCS_REF);common.validate_published_docs(published)
        archived=reader.document(published['nas_status']);common.validate_archive(archived)
        archive_result=reader.document(published['archive_result'])
        contract.flags(archive_result,{'schema':'sekirei.paired-linear-archive-result.v1','status':'complete-verified',
            'candidate':common.PRIOR,'attempt':'paired-linear-constrained-ridge1-l1-39p5-v2',
            'source_originals_preserved':True,'standalone_environment_restore':False,
            'child_spawned':True,'child_waited':True,'child_reaped':True,'child_group_stopped':True,
            'helper_returncode':0,'final_used':False,'adoption_claimed':False,'best_model_updated':False,'goal_complete':False})
        common.require(common.exact(archive_result.get('archive_receipt'),published['nas_status']), 'archive result/status binding differs')
        scans=archive_result.get('child_group_scans')
        common.require(type(scans) is list and len(scans)==2 and all(common.exact(x,[]) for x in scans),'archive child not twice observed stopped')
        git_before=common.verify_git(args.expected_public_source_head,published)
        common.verify_build(reader,inventory['new_build_binding'],inventory['feature_binding'],
            args.expected_build_manifest_sha256,args.expected_build_identity_sha256,args.expected_build_validator_sha256)
        common.require(not os.path.lexists(contract.ACTIVATION) and not os.path.lexists(contract.Q),'fresh activation/fit only')
        before=dict(reader.files);after=reader.verify()
        git_after=common.verify_git(args.expected_public_source_head,published)
        common.require(common.exact(git_before,git_after),'source Git changed during activation')
        value={'schema':'sekirei.white-view-paired-linear-activation.v1','status':'verified-before-fit',
            'created_at':datetime.now(timezone.utc).isoformat(),'candidate':contract.MODE,'mode':contract.MODE,
            'plan_sha256':contract.PLAN_SHA256,'previous_candidate_valid_nonadopt':True,
            'previous_candidate_groups_stopped':True,'previous_archive_verified':True,
            'previous_public_docs_pushed':True,'new_build_verified':True,'source_preparation_reviewed':True,
            'inputs_unchanged':True,'source_unchanged':True,'fit_started':False,'final_used':False,'adoption_claimed':False,
            'feature_binding':inventory['feature_binding'],'new_build_binding':inventory['new_build_binding'],
            'previous_completion':{**common.PREVIOUS,'archive':published['nas_status'],'public_docs':common.PUBLIC_DOCS_REF},
            'source_inventory':common.fullref(common.SOURCE_INVENTORY,reader),'source_helpers_sha256':inventory['source_helpers'],
            'runtime_sources':inventory['runtime_sources'],'operational_sources':inventory['operational_sources'],
            'source_git_before':git_before,'source_git_after':git_after,
            'inputs_before':before,'inputs_after':after,'exclusive_locks':list(map(str,locks)), 'architecture_shared_locks':list(map(str,common.architecture_shared_locks())),
            'scope':{'actual_fit':False,'actual_model_created':False,'actual_probe_or_engine_started':False,
                'new_build_receipts_reverified':True,'new_native_core_inference_verified':False,
                'previous_final_read':False,'new_model_adoption_verified':False}}
        common.require(common.exact(before,after),'activation input map differs')
        ref=common.save(contract.ACTIVATION,value);print(ref)


def parser():
    return common.parse_common_arguments(argparse.ArgumentParser(description=__doc__),inventory=True)

if __name__=='__main__':
    if PROTOTYPE_ONLY:raise RuntimeError('SOURCE ONLY activation producer disabled before actual I/O')
    activate(parser().parse_args())
