"""SOURCE ONLY archive runtime; dedicated white-view contract stays unchanged.

All filesystem/process entry points reject before I/O while disabled. Pure
receipt validation is not a new measurement or a standalone restore proof.
"""
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import types

import white_view_postformal_contract as c

PROTOTYPE_ONLY = True
WORKER = Path(str(c.C/'white-view-nas-archive-launch-worker-v1.py'))
SOURCE = Path(str(c.C/'white_view_archive_runtime_v1.py'))
CONTRACT = Path(str(c.C/'white_view_postformal_contract.py'))
LAUNCH = Path(str(c.C/'white-view-nas-archive-launch-v1.json'))
RESULT = Path(str(c.C/'white-view-nas-archive-result-v1.json'))
FAILURE = Path(str(c.C/'white-view-nas-archive-failure-v1.json'))
PYTHON = c.C.parent/'suisho11beta-v1/venv/bin/python'
HELPER_TIMEOUT_SECONDS = 3600


def guard():
    c.require(PROTOTYPE_ONLY is False, 'SOURCE ONLY archive runtime disabled before I/O')


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            c.require(key not in value, 'duplicate JSON key'); value[key] = item
        return value
    def bad(_): raise ValueError('nonfinite JSON constant')
    def finite(text):
        value = float(text); c.require(math.isfinite(value), 'nonfinite JSON exponent'); return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad, parse_float=finite)


def info(path):
    guard(); path = Path(path)
    c.path(str(path)); c.require(path.resolve(strict=True) == path, 'canonical actual input required')
    before = path.lstat(); c.require(stat.S_ISREG(before.st_mode), 'regular actual input required')
    def fp(s): return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        c.require(fp(before) == fp(os.fstat(stream.fileno())), 'input replaced before reading')
        digest = hashlib.file_digest(stream, 'sha256').hexdigest(); after = os.fstat(stream.fileno())
    c.require(fp(before) == fp(after) == fp(path.lstat()), 'input changed during reading')
    return {'bytes': before.st_size, 'sha256': digest}


def read_ref(ref):
    guard(); c.fullref(ref); record = {k:ref[k] for k in ('bytes','sha256')}
    c.require(c.exact(info(ref['path']),record), 'external file identity differs')
    raw = Path(ref['path']).read_bytes()
    c.require(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256']
        and c.exact(info(ref['path']),record), 'parsed bytes differ from external identity')
    return raw


def document(ref):
    guard(); return strict_json(read_ref(ref))


def verify_map(values):
    guard(); c.filemap(values)
    for path, record in values.items():
        c.require(c.exact(info(path),record), 'frozen archive input changed')
    return copy.deepcopy(values)


def save_new(path, value):
    guard(); path = Path(path)
    c.require(path.is_absolute() and path.parent.resolve(strict=True) == path.parent
        and path.parent.is_dir() and not os.path.lexists(path), 'fresh canonical output required')
    parent = path.parent.stat()
    c.require(parent.st_uid == os.getuid() and stat.S_IMODE(parent.st_mode) == 0o700,
              'private output parent must be owned mode0700')
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


@contextmanager
def exclusive_lock(path):
    guard(); path = Path(path)
    c.require(path.resolve(strict=True) == path and not path.is_symlink(), 'existing canonical lock required')
    fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        c.require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'regular lock required')
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream,fcntl.LOCK_UN)


class ArchiveCancelled(BaseException): pass


@contextmanager
def blocked_termination():
    guard()
    previous = signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
    try: yield
    finally: signal.pthread_sigmask(signal.SIG_SETMASK,previous)


@contextmanager
def cancellation_handlers():
    guard()
    def cancel(number,frame): raise ArchiveCancelled('termination signal '+str(number))
    prior = {n:signal.getsignal(n) for n in (signal.SIGINT,signal.SIGTERM)}
    try:
        for n in prior: signal.signal(n,cancel)
        yield
    finally:
        with blocked_termination():
            for n, handler in prior.items(): signal.signal(n,handler)


def load_helper(ref):
    guard(); c.fullref(ref)
    c.require(ref['path'] == str(c.HELPER) and ref['sha256'] == c.HELPER_SHA,
              'unchanged generic helper5520 required')
    raw = read_ref(ref)
    module = types.ModuleType('_white_view_unchanged_archive_helper5520')
    module.__file__ = ref['path']
    exec(compile(raw,ref['path'],'exec'),module.__dict__)
    c.require(module.NAS == Path(str(c.NAS)) and
        all(callable(getattr(module,name,None)) for name in ('inventory','combined_inventory','load_mapping')),
        'source-pinned actual helper APIs differ')
    return module


def command(request):
    c.require(set(c.ARCHIVE_OUTER_LOCKS).isdisjoint(c.ARCHIVE_HELPER_LOCKS)
        and len(c.ARCHIVE_OUTER_LOCKS) == 5 and len(c.ARCHIVE_HELPER_LOCKS) == 4,
        'five outer and four helper locks must be distinct')
    result = [str(PYTHON),'-B',str(c.HELPER),'--mapping',request['mapping']['path'],
              '--destination',request['destination'],'--receipt-name',request['receipt_name']]
    for path in c.ARCHIVE_HELPER_LOCKS: result += ['--lock',str(path)]
    return result


def validate_source_inventory(mapping, sources, selected):
    c.validate_mapping(mapping)
    c.require(type(sources) is dict and set(sources) == set(mapping), 'every selected source inventory required')
    combined = {'files':{},'directories':{'.'},'symlinks':{}}
    for root,target in mapping.items():
        value = sources[root]
        c.require(type(value) is dict and set(value) == {'files','directories','symlinks'}
            and type(value['symlinks']) is dict and value['symlinks'] == {}, 'white archive refuses every symlink')
        tree = {k:value[k] for k in ('files','directories')}
        c.validate_archive_equality(tree,tree,tree)
        relative = PurePosixPath(target)
        combined['directories'].update(str(p) for p in (relative,*relative.parents))
        combined['directories'].update(str(relative) if d == '.' else str(relative/d) for d in value['directories'])
        for name,record in value['files'].items(): combined['files'][str(relative/name)] = copy.deepcopy(record)
    combined['directories'] = sorted(combined['directories'])
    expected = {k:selected[k] for k in ('files','directories')}
    c.validate_archive_equality(expected,expected,{k:combined[k] for k in ('files','directories')})
    return combined


def inventory_sources(helper,mapping,selected):
    guard()
    current = {root:helper.inventory(Path(root)) for root in mapping}
    validated = validate_source_inventory(mapping,current,selected)
    api_value = helper.combined_inventory([(Path(root),PurePosixPath(target)) for root,target in sorted(mapping.items())],current)
    c.require(c.exact(validated,api_value), 'whole inventory disagrees with unchanged helper producer')
    return current,validated


def validate_helper_receipts(request,mapping,selected,status,before,after,expected,destination):
    c.fields(status,{'schema':'sekirei.private-archive.v1','status':'complete','phase':'verified',
        'destination':request['destination'],'mapping':mapping,'mapping_sha256':request['mapping']['sha256'],
        'script_sha256':c.HELPER_SHA,'sources_deleted':False})
    c.require(type(status.get('locks')) is list and len(status['locks']) == 4
        and set(status['locks']) == set(map(str,c.ARCHIVE_HELPER_LOCKS)), 'helper lock ownership differs')
    full = validate_source_inventory(mapping,before,selected)
    c.require(c.exact(before,after) and c.exact(expected,full) and c.exact(destination,full),
              'all source-before/source-after/expected/destination inventories must match')
    stats = c.validate_archive_equality({k:full[k] for k in ('files','directories')},
        {k:expected[k] for k in ('files','directories')},{k:destination[k] for k in ('files','directories')})
    c.fields(status,{'files':stats['files'],'bytes':stats['bytes'],'directories':stats['directories'],'symlinks':0})
    return stats


def reject_output_overlap(path,mapping,inputs):
    target = c.path(str(path))
    for root in [*mapping,*inputs]:
        source = c.path(root)
        c.require(not target.is_relative_to(source) and not source.is_relative_to(target),
                  'new archive control output overlaps frozen source/input')


def utc(): return datetime.now(timezone.utc).isoformat()
