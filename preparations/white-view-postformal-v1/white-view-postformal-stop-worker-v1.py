"""Disabled dedicated white-view postformal stop worker, source preparation only."""
from contextlib import ExitStack
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
import time
import white_view_postformal_contract as m

PROTOTYPE_ONLY = True
R, C = Path(m.R), Path(m.C)
OPAQUE_ALLOWLIST = Path(m.ALLOW)
OPAQUE_ALLOWLIST_SHA = m.ALLOW_SHA
WORKER = C/'white-view-postformal-stop-worker-v1.py'
SOURCE = C/'white_view_postformal_contract.py'
_reader = _frozen = None


def barrier():
    m.require(not PROTOTYPE_ONLY and not m.PROTOTYPE_ONLY, 'SOURCE ONLY disabled before any actual I/O')


def load_modules(request):
    """Execute only externally pinned bytes; restore unrelated canonical caches."""
    global w, io, p, E
    prior = {name: sys.modules.get(name) for name in m.MODULE_NAMES}
    loaded = {}
    try:
        for name in m.MODULE_NAMES[:3]:
            rec = m.fullref(request['modules'][name]); raw = Path(rec['path']).read_bytes()
            m.require(len(raw) == rec['bytes'] and hashlib.sha256(raw).hexdigest() == rec['sha256'], 'actual source raw SHA differs')
            spec = importlib.util.spec_from_file_location('_white_view_postformal_'+name, rec['path'])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            exec(compile(raw, rec['path'], 'exec'), module.__dict__)
            loaded[name] = module
        w, io, p = (loaded[n] for n in m.MODULE_NAMES[:3])
        m.require(io.w is w and p.w is w and p.io is io, 'actual source module objects differ')
        E = Path(request['modules']['white_view_operations']['path']).parent
    finally:
        for name, value in prior.items():
            if value is None: sys.modules.pop(name, None)
            else: sys.modules[name] = value
    return loaded


def ref(path, expected=None):
    text = str(path); record = _frozen[text]
    if expected is not None: m.require(record['sha256'] == expected, 'external scan authority differs')
    _reader.pin(text, record)
    return {'path': text, **record}


def document(path):
    return _reader.json_ref(ref(path))


def verify_git():
    for command, expected in ((['git','-C',str(R),'rev-parse','HEAD'],m.HEAD),
                              (['git','-C',str(R),'status','--porcelain=v1'],'')):
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        m.require(result.returncode == 0 and result.stdout.strip() == expected, 'measurement Git source identity changed')


def proc_identity(proc):
    stat=(proc/'stat').read_text();a=stat[stat.rfind(')')+2:].split()
    status={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in (proc/'status').read_text().splitlines() if ':' in x}
    return {'pid':int(proc.name),'ppid':int(a[1]),'pgid':int(a[2]),'sid':int(a[3]),'start_ticks':int(a[19]),'uids':[int(v) for v in status['Uid'].split()],'gids':[int(v) for v in status['Gid'].split()],'name':status['Name'],'comm':(proc/'comm').read_text().rstrip('\n'),'cmdline_hex':(proc/'cmdline').read_bytes().hex(),'cgroup':(proc/'cgroup').read_text()}

def stopped_processes():
    """Actual same-user runtime/runner processes; ignore this reader itself."""
    runtime_roots=(p.B,p.N,p.Q.parent,p.OLD_FAN)
    executables={str(E/'white_view_operations.py'),str(R/'scripts/benchmark.py'),str(R/'scripts/top3.py')}
    allow_ref=ref(OPAQUE_ALLOWLIST,OPAQUE_ALLOWLIST_SHA);allow=document(allow_ref['path'])
    w.require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==allow['boot_id'],'opaque-service boot identity changed')
    found=[];excluded=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            exe=(proc/'exe').resolve(strict=True);cwd=(proc/'cwd').resolve(strict=True)
            argv=[v.decode('utf-8','surrogateescape') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
            scoped=any(exe==root or root in exe.parents or cwd==root or root in cwd.parents for root in runtime_roots)
            scoped=scoped or any(a in executables for a in argv)
            scoped=scoped or any(a==str(root) or a.startswith(str(root)+'/') for a in argv for root in runtime_roots)
            if scoped:found.append({'pid':int(proc.name),'exe':str(exe),'cwd':str(cwd),'argv':argv})
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError as error:
            if error.errno!=13 or error.filename not in (str(proc/'exe'),str(proc/'cwd')):raise
            before=proc_identity(proc);frozen=allow['descriptors'].get(proc.name)
            w.require(frozen is not None and w.exact(before,frozen),'unknown or changed opaque process')
            w.require(before['start_ticks']<(allow['campaign_started_unix']-allow['btime_unix'])*allow['clock_ticks_per_second'],'opaque process did not predate campaign')
            after=proc_identity(proc);w.require(w.exact(before,after),'opaque process identity changed while classified')
            excluded.append({'descriptor':before,'denied_path':error.filename,'errno':error.errno,'allowlist':allow_ref,'reason':allow['excluded_reason'],'identity_reread_unchanged':True})
    return found,excluded

def stop(args):
    barrier()
    os.umask(0o077); sys.dont_write_bytecode = True
    m.require(Path(__file__) == WORKER and Path(m.__file__) == SOURCE, 'separate canonical enabled source required')
    # Read request with stdlib first; the strict parser and exact same raw are
    # bound again through Reader under all nine locks before it is used.
    raw = Path(args.request).read_bytes()
    m.require(hashlib.sha256(raw).hexdigest() == m.sha(args.expected_request_sha256), 'external request SHA differs')
    import json
    request = json.loads(raw)
    load_modules(request)
    m.validate_request(request,w)
    m.require(m.exact(tuple((str(path), ex) for path,ex in p.LOCKS),
                      tuple((str(path), ex) for path,ex in m.LOCKS[:7])), 'actual seven-lock contract differs')
    m.require(request['executor_session_id'] == args.expected_executor_session_id
        and request['comparison_session_id'] == args.expected_comparison_session_id, 'external terminal sessions differ')
    global _reader, _frozen
    _reader = io.Reader(); _frozen = request['inputs']
    m.require(args.request == str(C/'white-view-postformal-stop-request-v1.json'), 'fixed postformal request path required')
    request_ref = {'path':args.request, 'bytes':len(raw), 'sha256':args.expected_request_sha256}
    for source, digest in ((WORKER,args.expected_worker_sha256),(SOURCE,args.expected_source_sha256)):
        m.require(_frozen.get(str(source),{}).get('sha256') == m.sha(digest), 'actual stop source not externally pinned')
    with ExitStack() as held:
        for lock, exclusive in m.LOCKS: held.enter_context(io.existing_lock(Path(lock), exclusive))
        locked_request = _reader.json_ref(request_ref)
        m.require(m.exact(locked_request,request), 'request changed before lock acquisition')
        m.validate_request(locked_request,w)
        for name, rec in request['inputs'].items(): _reader.pin(name,rec)
        before = _reader.current(); verify_git()
        names = ('comparison_terminal','comparison','comparison_request','candidate_preflight',
                 'candidate_gate','candidate_terminal','opaque_allowlist')
        docs = {name:_reader.json_ref(request['refs'][name]) for name in names}
        audits = {name:p.read_audit_bundle(_reader,request['refs'][key])
                  for name,key in (('old','old_audit'),('fallback','fallback_audit'),('candidate','candidate_audit'))}
        bridge = p.read_verified_bridge(_reader, request['refs']['bridge'])
        adopt = m.validate_documents(request,docs,audits,bridge,w,p)
        scans = []
        for _ in range(2):
            rows, excluded = stopped_processes()
            m.require(not rows, 'related runtime/runner process still exists')
            scans.append({'related_processes':rows,'excluded_preexisting_services':excluded})
            time.sleep(.1)
        after = _reader.current(); verify_git()
        m.require(m.exact(after,before), 'postformal immutable closure changed or expanded')
        value = m.stop_document(request,request_ref,scans,before,after,adopt,docs['opaque_allowlist'],audits)
        io.private_output(Path(m.OUT), list(before))
        return io.write_new(Path(m.OUT), io.json_bytes(value))


def main():
    barrier()  # Before argparse, including --help and file/proc/lock access.
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request',required=True)
    for name in ('request','worker','source'): ap.add_argument('--expected-'+name+'-sha256',required=True)
    ap.add_argument('--expected-executor-session-id',type=int,required=True)
    ap.add_argument('--expected-comparison-session-id',type=int,required=True)
    print(stop(ap.parse_args()))


if __name__ == '__main__': main()
