#!/usr/bin/env python3
"""SOURCE ONLY six-lock/1200s supervisor and new03 run->sidecar finalizer.

Child owns numeric artifacts. Parent owns hard wall deadline, actual wait/reap,
process-group cleanup, before/after snapshots and final run/sidecar. Disabled
before actual files/processes. No search, probe, alternate fit or adoption.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode=True
R=Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
sys.path.insert(0,str(R/'scripts'))
import white_view_fit_operational_common as common
import white_view_fit_contract as contract

PROTOTYPE_ONLY=True
WORKER=contract.C/'white-view-paired-linear-fit-launcher-v1.py'
LAUNCH=contract.C/'white-view-paired-linear-fit-launch-v1.json'
OP=contract.C/'white-view-paired-linear-fit-operational-verification-v1.json'
FAILURE=contract.C/'white-view-paired-linear-fit-failure-v1.json'
STDOUT=contract.C/'white-view-paired-linear-fit.stdout.log'
STDERR=contract.C/'white-view-paired-linear-fit.stderr.log'


def within(started):common.require(time.monotonic()-started<1200.,'whole child/finalization hard wall exhausted')


def group_scans(pid,lifecycle):
    import diagnose_anchor as guard
    scans=[guard.group_exists(pid),guard.group_exists(pid)]
    lifecycle['process_group_scans']=scans
    lifecycle['process_group_stopped']=all(value is False for value in scans)
    common.require(lifecycle['process_group_stopped'],'child group remains after wait/cleanup')
    return scans


def run_child(command,output,stdout,stderr,holder,lifecycle):
    """Retained handle and bounded cleanup before any six-lock release."""
    import diagnose_anchor as guard
    started=time.monotonic();lifecycle['started_monotonic']=started
    try:
        guard.spawn_process(command,output,stdout,stderr,holder)
        process=holder['process'];lifecycle['spawned']=True;lifecycle['pid']=process.pid
        wait_error=None
        try:
            code=process.wait(timeout=max(.001,1200.-(time.monotonic()-started)))
            lifecycle.update(returncode=code,waited=True,reaped=True)
        except BaseException as error:
            wait_error=error
            if isinstance(error,subprocess.TimeoutExpired):lifecycle['timed_out']=True
        try:
            with guard.blocked_termination():
                guard.cleanup_group(process);group_scans(process.pid,lifecycle)
                holder['process']=None
        except BaseException as cleanup:
            lifecycle['cleanup_error']=type(cleanup).__name__+': '+str(cleanup)
            if wait_error is None:raise
        if wait_error is not None:raise wait_error
        common.require(type(code) is int and code==0 and lifecycle['timed_out'] is False,'fit child unsuccessful')
        within(started);return started
    except BaseException as original:
        lifecycle['original_error']=type(original).__name__+': '+str(original)
        with guard.blocked_termination():
            process=holder['process'];lifecycle['spawned']=process is not None or lifecycle['spawned']
            if process is not None:
                try:
                    guard.cleanup_group(process)
                    # cleanup_group has actually waited, including a failed or
                    # cancelled spawn; do not set these on a cleanup exception.
                    lifecycle.update(waited=True,reaped=True,returncode=process.returncode)
                    group_scans(process.pid,lifecycle);holder['process']=None
                except BaseException as error:
                    lifecycle['cleanup_error']=type(error).__name__+': '+str(error)
                    lifecycle['process_group_stopped']=False
            lifecycle['process_handle_retained']=holder['process'] is not None
        raise


def validate_and_finalize(reader,draft,prereg,args,lifecycle,started):
    """Root-only actual finalization after completed wait/reap/group observation."""
    import fit_white_view_paired_linear as driver
    import white_view_paired_linear as white
    common.require(type(draft) is dict and set(draft)=={'schema','status','run_template','numeric_execution',
        'cleanup_verified','final_used','adoption_claimed'},'strict child draft fields required')
    contract.flags(draft,{'schema':'sekirei.white-view-paired-linear-fit-result.v1','status':'numeric-stage-complete',
        'cleanup_verified':False,'final_used':False,'adoption_claimed':False})
    template=draft['run_template'];common.require(type(template) is dict and template.get('cleanup_verified') is False,
        'child cannot claim own reaping')
    outputs=template['outputs'];common.require(type(outputs) is dict and set(outputs)==set(contract.OUTPUT_NAMES),'nine numeric outputs required')
    values={}
    for key,name in contract.OUTPUT_NAMES.items():
        within(started);ref=outputs[key];contract.fullref(ref,contract.Q/name)
        values[key]=reader.read(ref['path'],{k:ref[k] for k in ('bytes','sha256')})
    initial=reader.read(contract.ORIGINAL_INITIALIZER,{'bytes':white.WEIGHT_BYTES,'sha256':white.STOCK_INITIALIZER_SHA256})
    white.validate_native(values['weights'],values['coefficients_f32'],initial,binding=prereg['feature_binding'])
    result=contract.strict_json(values['solver_certificate']);design=contract.strict_json(values['design_receipt'])
    common.require(common.exact(design.get('feature_binding'),prereg['feature_binding'])
        and design.get('teacher_identity')==white.ORIGINAL_TEACHER
        and design.get('original_manifest_sha256')==contract.MANIFEST_SHA256,'stored design source/feature identity differs')
    math=common.validate_numeric_artifacts(result,values['gram_z'],values['rhs_z'],values['coefficients_f64'],values['coefficients_f32'])
    provenance=common.verify_producer_digest(result,design,values['design_z'],values['targets_d'],values['gram_z'],values['rhs_z'])
    context_keys={'preregistration_sha256','activation_sha256','source_helpers_sha256','source_preflight_sha256',
        'inputs_before','inputs_after','runtime_sources','numeric_execution','feature_binding','new_build_binding','new_build_verified'}
    context={key:template.get(key) for key in context_keys}
    expected_template=contract.make_run_template(context,outputs,outputs['solver_certificate'])
    expected_template['created_at']=template.get('created_at')
    common.require(common.exact(expected_template,template),'child template fields/recipe differ')
    common.require(common.exact(context['numeric_execution'],draft['numeric_execution']),'child numeric execution disagrees')
    for key,expected in {'preregistration_sha256':args.expected_preregistration_sha256,
        'activation_sha256':args.expected_activation_sha256,'source_preflight_sha256':args.expected_source_preflight_sha256,
        'source_helpers_sha256':prereg['source_helpers'],'runtime_sources':prereg['runtime_sources'],
        'feature_binding':prereg['feature_binding'],'new_build_binding':prereg['new_build_binding']}.items():
        common.require(common.exact(context[key],expected),'child external context differs: '+key)
    before=context['inputs_before'];common.require(common.exact(before,context['inputs_after']),'child immutable map changed')
    for path,rec in before.items():reader.pin(path,rec)
    for path,rec in reader.files.items():
        # Numeric outputs/draft are newly created. Every original control/source
        # held by the parent must already be in the child's completed snapshot.
        if not Path(path).is_relative_to(contract.Q):
            common.require(path in before and common.exact(before[path],rec),'child omitted parent frozen input')
    current={path:common.info(path) for path in before}
    common.require(common.exact(before,current),'parent actual input/source rehash differs')
    complete={key:lifecycle[key] for key in contract.LIFECYCLE}
    run=contract.construct_fit_run(context,outputs,outputs['solver_certificate'],complete)
    raw=driver.json_bytes(run)
    binding={'plan_sha256':contract.PLAN_SHA256,'preregistration_sha256':args.expected_preregistration_sha256,
        'activation_sha256':args.expected_activation_sha256,'source_preflight_sha256':args.expected_source_preflight_sha256,
        'source_helpers_sha256':prereg['source_helpers'],'feature_binding':prereg['feature_binding'],
        'new_build_binding':prereg['new_build_binding'],'solver_certificate':outputs['solver_certificate'],
        'fit_receipt':{'path':str(contract.FIT_RECEIPT),'bytes':len(raw),'sha256':driver.sha(raw)}}
    contract.validate_fit_run(raw,values['weights'],values['coefficients_f32'],initial,binding)
    within(started);driver.write_new(contract.FIT_RECEIPT,raw)
    meta=contract.sidecar_for_fit(values['weights'],values['coefficients_f32'],initial,raw,binding)
    meta_raw=driver.json_bytes(meta);contract.validate_sidecar(meta_raw,values['weights'],values['coefficients_f32'],initial,raw,binding)
    within(started);driver.write_new(contract.W.with_suffix('.meta.json'),meta_raw)
    common.require(common.exact(before,{path:common.info(path) for path in before}),'finalization source/input changed')
    within(started);return {'run':binding['fit_receipt'],'metadata':{'path':str(contract.W.with_suffix('.meta.json')),
        **common.info(contract.W.with_suffix('.meta.json'))},'mathematical_checks':math,'producer_checks':provenance}


def launch(args):
    common.require(PROTOTYPE_ONLY is False,'SOURCE ONLY fit launcher disabled before actual I/O')
    common.require_enabled();os.umask(0o077)
    reader=common.Reader();common.bootstrap(reader,Path(__file__),args)
    from benchmark import nonblocking_lock
    import diagnose_anchor as guard
    holder={'process':None};lifecycle={'spawned':False,'waited':False,'reaped':False,'returncode':None,
        'timed_out':False,'process_group_stopped':False,'process_group_scans':[],'process_handle_retained':False}
    started=None
    with guard.termination_guard(),ExitStack() as stack:
        locks=common.lock_paths()
        for path in locks:stack.enter_context(nonblocking_lock(path,exclusive=True))
        for path in common.architecture_shared_locks():stack.enter_context(nonblocking_lock(path,exclusive=False))
        common.bootstrap(reader,Path(__file__),args)
        pr=contract.strict_json(reader.read(contract.PREREGISTRATION,args.expected_preregistration_sha256))
        contract.validate_preregistration(pr)
        common.require(pr['public_source_commit']==args.expected_public_source_head,'extern source HEAD differs')
        common.pin_helpers(reader,pr['source_helpers'])
        common.verify_activation(reader,pr,args)
        pf=contract.strict_json(reader.read(contract.SOURCE_PREFLIGHT,args.expected_source_preflight_sha256))
        contract.validate_source_preflight(pf,pr,args.expected_preregistration_sha256)
        for path,rec in pf['inputs_before'].items():reader.pin(path,rec)
        for path,rec in pr['runtime_sources'].items():reader.pin(path,rec)
        common.verify_git(args.expected_public_source_head)
        common.require(common.exact(pr['fit_driver'],common.fullref(R/'scripts/fit_white_view_paired_linear.py',reader)), 'driver PR reference differs')
        common.require(pr['fit_driver']['sha256']==pr['source_helpers']['fit_white_view_paired_linear.py'],'driver helper SHA conflict')
        for path in (contract.Q,LAUNCH,OP,FAILURE,STDOUT,STDERR):common.require(not os.path.lexists(path),'fresh output/records required')
        common.require(contract.Q.parent.resolve()==contract.Q.parent and contract.Q.parent.is_dir()
            and not contract.Q.parent.stat().st_mode&0o077,'private canonical run parent required')
        common.require(shutil.disk_usage(contract.Q.parent).free>=2*2**30,'fit needs 2 GiB free')
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        before=dict(reader.files);reader.verify()
        command=[str(common.PYTHON),'-B',pr['fit_driver']['path']]
        for name,value in (('preregistration',args.expected_preregistration_sha256),('activation',args.expected_activation_sha256),
            ('source-preflight',args.expected_source_preflight_sha256),('driver',pr['fit_driver']['sha256']),
            ('build-manifest',args.expected_build_manifest_sha256),('build-identity',args.expected_build_identity_sha256),
            ('build-validator',args.expected_build_validator_sha256)):
            command+=['--expected-'+name+'-sha256',value]
        try:
            with STDOUT.open('xb') as stdout,STDERR.open('xb') as stderr:
                common.save(LAUNCH,{'schema':'sekirei.white-view-paired-linear-fit-launch.v1','status':'frozen-before-spawn',
                    'argv':command,'preregistration_sha256':args.expected_preregistration_sha256,
                    'source_preflight_sha256':args.expected_source_preflight_sha256,'hard_wall_budget_seconds':1200,
                    'inputs_before':before,'exclusive_locks':list(map(str,locks)), 'architecture_shared_locks':list(map(str,common.architecture_shared_locks())),'final_used':False,'adoption_claimed':False})
                started=run_child(command,contract.Q.parent,stdout,stderr,holder,lifecycle)
            after=reader.verify();common.require(common.exact(before,after),'source/input changed during child')
            common.verify_git(args.expected_public_source_head)
            common.verify_build(reader,pr['new_build_binding'],pr['feature_binding'],args.expected_build_manifest_sha256,
                args.expected_build_identity_sha256,args.expected_build_validator_sha256)
            draft=contract.strict_json(reader.read(contract.Q/'fit-result.json'))
            proof=validate_and_finalize(reader,draft,pr,args,lifecycle,started)
            within(started)
            common.save(OP,{'schema':'sekirei.white-view-paired-linear-fit-operational-verification.v1','status':'complete',
                'candidate':contract.MODE,'preregistration_sha256':args.expected_preregistration_sha256,
                'source_preflight_sha256':args.expected_source_preflight_sha256,'activation_sha256':args.expected_activation_sha256,
                'feature_binding':pr['feature_binding'],'new_build_binding':pr['new_build_binding'],
                'source_helpers_sha256':pr['source_helpers'],'fit_receipt':proof['run'],'native_metadata':proof['metadata'],
                'native':{'path':str(contract.W),**common.info(contract.W)},'solver_certificate':draft['run_template']['solver_certificate'],
                'inputs_before':before,'inputs_after':after,'inputs_unchanged':True,'source_unchanged':True,
                'new_build_verified':True,'cleanup_verified':True,'lifecycle':lifecycle,'numeric_reverification':proof['mathematical_checks'],
                'producer_origin_reverification':proof['producer_checks'],'exclusive_locks':list(map(str,locks)), 'architecture_shared_locks':list(map(str,common.architecture_shared_locks())),
                'adam_used':False,'epochs':0,'actual_native_core_verified':False,'final_used':False,'adoption_claimed':False})
        except BaseException as original:
            with guard.blocked_termination():
                cleanup=None
                if holder['process'] is not None:
                    process=holder['process']
                    try:
                        guard.cleanup_group(process);lifecycle.update(waited=True,reaped=True,returncode=process.returncode)
                        group_scans(process.pid,lifecycle);holder['process']=None
                    except BaseException as error:
                        cleanup=type(error).__name__+': '+str(error);lifecycle['process_group_stopped']=False
                lifecycle['process_handle_retained']=holder['process'] is not None
                if not os.path.lexists(FAILURE):common.save(FAILURE,{'schema':'sekirei.white-view-paired-linear-fit-failure.v1',
                    'status':'failed','candidate':contract.MODE,'error':type(original).__name__+': '+str(original),
                    'cleanup_error':cleanup,'lifecycle':lifecycle,'originals_preserved':True,'final_used':False,'adoption_claimed':False})
            raise


def parser():
    return common.parse_common_arguments(argparse.ArgumentParser(description=__doc__),preregistration=True,activation=True,source_preflight=True)

if __name__=='__main__':
    if PROTOTYPE_ONLY:raise RuntimeError('SOURCE ONLY fit launcher disabled before actual I/O')
    launch(parser().parse_args())
