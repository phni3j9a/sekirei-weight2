"""Disabled SOURCE launcher for dedicated white-view archive requests.

The unchanged helper owns copy/mount/four locks. This parent owns five distinct
locks through wait/reap and bounded helper-group cleanup. The parent observes
only its own child; Root must independently observe this launcher's terminal.
"""
import argparse
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import white_view_archive_runtime_v1 as m

PROTOTYPE_ONLY = True


def guard():
    m.c.require(PROTOTYPE_ONLY is False, 'SOURCE ONLY launcher disabled before I/O')
    m.guard()


def lifecycle_new():
    return {'child_spawned':False,'child_waited':False,'child_reaped':False,
        'helper_returncode':None,'child_group_stopped':False,'child_group_scans':[]}


def group_members(group):
    guard(); m.c.require(type(group) is int and group > 0, 'explicit helper PG required')
    members = []
    for entry in sorted(Path('/proc').iterdir()):
        if not entry.name.isdigit(): continue
        try:
            if entry.stat().st_uid != os.getuid(): continue
            if os.getpgid(int(entry.name)) == group: members.append(int(entry.name))
        except (FileNotFoundError,ProcessLookupError): continue
    return members


def stop_child_group(child,lifecycle):
    guard()
    if group_members(child.pid):
        try: os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError: pass
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        child.wait(timeout=10)
    lifecycle.update(child_waited=True,child_reaped=True,helper_returncode=child.returncode)
    # Leader reap does not establish that its descendants stopped.
    if group_members(child.pid):
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
    deadline = time.monotonic()+10
    while group_members(child.pid) and time.monotonic() < deadline: time.sleep(.05)
    scans = [group_members(child.pid),group_members(child.pid)]
    lifecycle.update(child_group_scans=scans,child_group_stopped=all(not row for row in scans))
    m.c.require(lifecycle['child_group_stopped'], 'bounded helper cleanup left related processes')


def spawn_held(command,holder,lifecycle):
    guard(); received = []
    prior = {n:signal.getsignal(n) for n in (signal.SIGINT,signal.SIGTERM)}
    def defer(number,frame): received.append(number)
    error = None
    try:
        for n in prior: signal.signal(n,defer)
        # Only the parent defers Python exceptions. No signal mask is changed
        # before Popen, so helper/rsync inherit normal TERM/INT delivery.
        holder['child'] = subprocess.Popen(command,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
        lifecycle['child_spawned'] = True
    except BaseException as original: error = original
    finally:
        with m.blocked_termination():
            for n,handler in prior.items(): signal.signal(n,handler)
    if error is not None: raise error
    if received: raise m.ArchiveCancelled('deferred termination signals '+repr(received))


def run_helper(command,holder,lifecycle):
    guard(); spawn_held(command,holder,lifecycle)
    child = holder['child']
    lifecycle['helper_returncode'] = child.wait(timeout=m.HELPER_TIMEOUT_SECONDS)
    lifecycle.update(child_waited=True,child_reaped=True)
    scans = [group_members(child.pid),group_members(child.pid)]
    lifecycle.update(child_group_scans=scans,child_group_stopped=all(not row for row in scans))
    m.c.require(lifecycle['child_group_stopped'], 'helper descendants remain after leader wait')
    m.c.require(type(lifecycle['helper_returncode']) is int and lifecycle['helper_returncode'] == 0,
        'helper failed; partial NAS destination is retained invalid')


def cleanup_held(holder,lifecycle):
    guard(); child = holder['child']
    if child is None: return
    with m.blocked_termination():
        try: stop_child_group(child,lifecycle)
        except BaseException as cleanup:
            lifecycle['cleanup_error'] = type(cleanup).__name__+': '+str(cleanup)
            try:
                lifecycle['child_group_scans'] = [group_members(child.pid),group_members(child.pid)]
                lifecycle['child_group_stopped'] = all(not row for row in lifecycle['child_group_scans'])
            except BaseException as scan:
                lifecycle['child_group_scan_error'] = type(scan).__name__+': '+str(scan)
                lifecycle.update(child_group_scans=[],child_group_stopped=False)
            raise
        # Retain the handle when cleanup is unconfirmed; never synthesize reap.
        holder['child'] = None


def controls(request,expected_stop_session):
    guard()
    stopped = m.document(request['stop']); terminal = m.document(request['stop_terminal'])
    mapping = m.document(request['mapping']); selected = m.document(request['closure_inventory'])
    m.c.validate_archive_request(request,stopped,terminal,mapping,selected)
    m.c.require(type(expected_stop_session) is int and expected_stop_session > 0
        and terminal['stop_session_id'] == expected_stop_session, 'external Root stop-session binding differs')
    return stopped,terminal,mapping,selected


def launch(args):
    guard(); os.umask(0o077)
    m.c.require(Path(__file__).resolve() == m.WORKER and Path(m.__file__).resolve() == m.SOURCE
        and Path(m.c.__file__).resolve() == m.CONTRACT, 'canonical separately enabled sources required')
    request_ref = {'path':args.request,'bytes':args.expected_request_bytes,'sha256':m.c.sha(args.expected_request_sha256)}
    m.c.fullref(request_ref)
    sources = {str(m.WORKER):m.c.sha(args.expected_worker_sha256),
        str(m.SOURCE):m.c.sha(args.expected_runtime_source_sha256),
        str(m.CONTRACT):m.c.sha(args.expected_contract_sha256),str(m.c.HELPER):m.c.HELPER_SHA}
    request = m.document(request_ref)
    before = copy.deepcopy(m.c.filemap(request['inputs']))
    for name,digest in sources.items():
        m.c.require(before.get(name,{}).get('sha256') == digest, 'frozen source input missing/external SHA differs')
    m.c.require(os.path.abspath(sys.executable) == str(m.PYTHON)
        and Path(sys.prefix) == Path(str(m.PYTHON)).parent.parent, 'fixed AUD interpreter/prefix required')
    for runtime_input in (Path(sys.executable).resolve(strict=True),Path(sys.prefix)/'pyvenv.cfg'):
        m.c.require(m.c.exact(before.get(str(runtime_input)),m.info(runtime_input)),
                    'actual interpreter/config input missing from frozen closure')
    m.c.require(Path(request_ref['path']).parent == Path(str(m.c.C)), 'dedicated campaign request required')
    holder = {'child':None}; lifecycle = lifecycle_new()
    with ExitStack() as locks:
        locks.enter_context(m.cancellation_handlers())
        for path in m.c.ARCHIVE_OUTER_LOCKS: locks.enter_context(m.exclusive_lock(path))
        # Lock-held reparsing closes stale pre-lock reads. The child never
        # receives these FDs, so the parent's ownership is not duplicated.
        request = m.document(request_ref)
        m.c.require(m.c.exact(request['inputs'],before), 'locked immutable input map differs')
        m.verify_map(before)
        stopped,terminal,mapping,selected = controls(request,args.expected_stop_session_id)
        helper = m.load_helper(request['helper'])
        for output in (m.LAUNCH,m.RESULT,m.FAILURE):
            m.reject_output_overlap(output,mapping,[*before,request_ref['path']])
            m.c.require(not os.path.lexists(output), 'exclusive launch/result output required')
        m.c.require(not os.path.lexists(request['destination']) and
            not os.path.lexists(Path(str(m.c.NAS))/'receipts'/request['receipt_name']), 'fresh NAS destination and receipt required')
        _,full = m.inventory_sources(helper,mapping,selected)
        m.c.require(m.c.exact(helper.load_mapping(Path(request['mapping']['path'])),
            [(Path(root),m.PurePosixPath(target)) for root,target in sorted(mapping.items())]), 'helper actual mapping differs')
        m.verify_map(before); m.read_ref(request_ref)
        launch_ref = m.save_new(m.LAUNCH,{'schema':'sekirei.white-view-nas-archive-launch.v1',
            'status':'frozen-before-copy','candidate':m.c.MODE,'source_head':m.c.HEAD,
            'request':request_ref,'stop':request['stop'],'stop_terminal':request['stop_terminal'],
            'sources':sources,'helper_timeout_seconds':m.HELPER_TIMEOUT_SECONDS,
            'parent_exclusive_locks':list(map(str,m.c.ARCHIVE_OUTER_LOCKS)),
            'helper_exclusive_locks':list(map(str,m.c.ARCHIVE_HELPER_LOCKS)),
            'source_originals_preserved':True,'destination_reused':False,
            'standalone_environment_restore':False,'final_used':False,'adoption_applied':False,
            'best_model_updated':False,'goal_complete':False,'created_at':m.utc()})
        original = None
        try:
            # Do not execute a helper whose current bytes changed during the
            # source inventory or launch-receipt write.
            m.read_ref(request['helper'])
            run_helper(m.command(request),holder,lifecycle)
            cleanup_held(holder,lifecycle)
            m.verify_map(before); m.read_ref(request_ref)
            again = controls(request,args.expected_stop_session_id)
            m.c.require(m.c.exact(again,(stopped,terminal,mapping,selected)), 'archive controls changed')
            _,after_parent = m.inventory_sources(helper,mapping,selected)
            m.c.require(m.c.exact(after_parent,full), 'parent whole source inventory changed during archive')
            receipt = Path(str(m.c.NAS))/'receipts'/request['receipt_name']
            names = ('status.json','source-before.json','source-after.json','expected-destination.json','destination.json')
            refs = {name:{'path':str(receipt/name),**m.info(receipt/name)} for name in names}
            docs = {name:m.document(ref) for name,ref in refs.items()}
            stats = m.validate_helper_receipts(request,mapping,selected,*[docs[name] for name in names])
            actual_destination = helper.inventory(Path(request['destination']))
            m.c.require(m.c.exact(actual_destination,full), 'current full destination differs from frozen selected closure')
            m.verify_map(before); m.read_ref(request_ref)
            for ref in refs.values(): m.read_ref(ref)
            result = {'schema':'sekirei.white-view-nas-archive-result.v1','status':'complete-verified',
                'candidate':m.c.MODE,'source_head':m.c.HEAD,'request':request_ref,'launch':launch_ref,
                'adopt':stopped['adopt'],'destination':request['destination'],'helper_receipts':refs,
                'parent_whole_source_before_after_and_destination_equal':True,'stats':stats,
                'inputs_before':before,'inputs_after':copy.deepcopy(before),'inputs_unchanged':True,
                'source_originals_preserved':True,'standalone_environment_restore':False,
                **lifecycle,'final_used':False,'adoption_applied':False,'best_model_updated':False,
                'goal_complete':False,'created_at':m.utc()}
            m.save_new(m.RESULT,result)
            return result
        except BaseException as error:
            original = error
            try: cleanup_held(holder,lifecycle)
            except BaseException: pass # Preserve original failure; cleanup_error stays separately recorded.
            with m.blocked_termination():
                m.save_new(m.FAILURE,{'schema':'sekirei.white-view-nas-archive-result.v1','status':'failed',
                    'candidate':m.c.MODE,'source_head':m.c.HEAD,'request':request_ref,'launch':launch_ref,
                    'error':type(original).__name__+': '+str(original),**lifecycle,
                    'helper_handle_retained':holder['child'] is not None,
                    'partial_destination_if_created_retained_invalid':True,'destination_reused':False,
                    'nas_copy_verified':False,'source_originals_preserved':True,
                    'standalone_environment_restore':False,'final_used':False,'adoption_applied':False,
                    'best_model_updated':False,'goal_complete':False,'created_at':m.utc()})
            raise
        finally:
            # Includes cancellation after return-value construction, still
            # within all five parent locks; holder ownership never escapes.
            if holder['child'] is not None:
                try: cleanup_held(holder,lifecycle)
                except BaseException:
                    if original is None: raise


def main():
    guard() # Includes --help: stop before argparse, actual read, or write.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True)
    parser.add_argument('--expected-request-bytes',type=int,required=True)
    parser.add_argument('--expected-stop-session-id',type=int,required=True)
    for name in ('request','worker','runtime-source','contract'):
        parser.add_argument('--expected-'+name+'-sha256',required=True)
    args = parser.parse_args()
    result = launch(args)
    print(m.json.dumps({'status':result['status'],'result':str(m.RESULT),'sha256':m.info(m.RESULT)['sha256']}))


if __name__ == '__main__': main()
