"""SOURCE ONLY strict I/O and parent lifecycle primitives.

Main entry is disabled. Synthetic tests use only new temporary files and fake
process objects. No actual input/output/runtime/model is opened on import.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
import white_view_evaluation as w

PROTOTYPE_ONLY = True


def info(path):
    path = Path(path)
    w.require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink()
              and stat.S_ISREG(path.stat().st_mode), 'canonical regular file required')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            size += len(chunk); digest.update(chunk)
        after = os.fstat(stream.fileno())
    w.require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
              == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
              and size == after.st_size, 'file changed during hashing')
    return {'bytes': size, 'sha256': digest.hexdigest()}


class Reader:
    def __init__(self):
        self.files, self.cache = {}, {}
    def pin(self, path, expected):
        path = str(w.absolute(str(path)))
        w.file_identity(expected)
        current = info(path)
        w.require(w.exact(current, expected), 'externally pinned file identity differs')
        w.require(path not in self.files or w.exact(self.files[path], expected), 'input identity rebinding forbidden')
        self.files[path] = dict(expected)
        return current
    def read(self, path, expected):
        path = str(w.absolute(str(path)))
        self.pin(path, expected)
        raw = Path(path).read_bytes()
        w.require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256']
                  and w.exact(info(path), expected), 'parsed raw bytes/current file differ')
        self.cache[path] = raw
        return raw
    def read_ref(self, ref):
        w.fullref(ref)
        return self.read(ref['path'], {k: ref[k] for k in ('bytes', 'sha256')})
    def json_ref(self, ref):
        return w.read_pinned_json_bytes(self.read_ref(ref), ref['sha256'])
    def current(self):
        result = {path: info(path) for path in self.files}
        w.require(w.exact(result, self.files), 'immutable input/source inventory changed')
        return result


def union_exact(left, right):
    result = dict(left)
    for path, record in right.items():
        w.absolute(path); w.file_identity(record)
        w.require(path not in result or w.exact(result[path], record), 'identity overwrite forbidden')
        result[path] = dict(record)
    return result


def private_output(path, protected, minimum_free=2*2**30):
    """No directory creation; new canonical private output outside all inputs."""
    path = Path(str(w.absolute(str(path))))
    w.require(not os.path.lexists(path), 'output already exists; preserve all failed/completed evidence')
    parent = path.parent
    w.require(parent.resolve(strict=True) == parent and parent.is_dir()
              and not parent.stat().st_mode & 0o077, 'canonical private 0700 parent required')
    for item in protected:
        item = Path(str(w.absolute(str(item))))
        w.require(not path.is_relative_to(item) and not item.is_relative_to(path), 'output/input overlap forbidden')
    result = subprocess.run(['git', '-C', str(parent), 'rev-parse', '--is-inside-work-tree'],
        capture_output=True, timeout=10, env=dict(os.environ, LC_ALL='C'))
    w.require(result.returncode != 0 and b'not a git repository' in result.stderr,
              'output inside Git or Git status unverified')
    import shutil
    w.require(shutil.disk_usage(parent).free >= minimum_free, 'insufficient SSD capacity before first write')
    return path


def write_new(path, raw):
    w.require(type(raw) is bytes, 'immutable bytes writer required')
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {'path': str(path), **info(path)}


def json_bytes(value):
    import json
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


class OwnedOutput:
    """Only self-created mutable config/state files can be replaced."""
    def __init__(self, root):
        self.root, self.owned = Path(root), set()
    def create(self):
        self.root.mkdir(mode=0o700, parents=False)
    def write(self, name, value, mutable=False):
        w.require(type(name) is str and Path(name).name == name and name not in ('', '.', '..'), 'fixed output basename required')
        path = self.root/name
        raw = json_bytes(value)
        if path in self.owned:
            w.require(mutable and name in ('candidate.json', 'evaluation.json'), 'immutable output cannot be overwritten')
            w.require(path.resolve(strict=True) == path and not path.is_symlink(), 'owned output path changed')
            temp = self.root/(name+'.atomic-'+str(os.getpid()))
            write_new(temp, raw)
            try: os.replace(temp, path)
            finally:
                if temp.exists(): temp.unlink()
        else:
            write_new(path, raw); self.owned.add(path)
        return {'path': str(path), **info(path)}


def inventory(directories, paths=()):
    files, children = set(), {}
    for path in paths:
        files.add(Path(path))
    for root in directories:
        root = Path(root)
        w.require(root.resolve(strict=True) == root and root.is_dir() and not root.is_symlink(), 'canonical audit directory required')
        names = []
        for path in sorted(root.rglob('*')):
            w.require(not path.is_symlink(), 'symlink in audit inventory forbidden')
            if path.is_file():
                files.add(path); names.append(str(path.relative_to(root)))
            else:
                w.require(path.is_dir(), 'special file in audit inventory forbidden')
        children[str(root)] = names
    return {'files': {str(p): info(p) for p in sorted(files)}, 'directories': children}


@contextmanager
def existing_lock(path, exclusive):
    import fcntl
    path = Path(path)
    w.require(type(exclusive) is bool and path.resolve(strict=True) == path and path.is_file()
              and not path.is_symlink(), 'existing canonical cooperating lock required')
    with path.open('rb') as stream:
        fcntl.flock(stream.fileno(), (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)|fcntl.LOCK_NB)
        try: yield stream
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Cancelled(BaseException):
    pass


@contextmanager
def termination_guard(state):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def cancel(sig, frame):
        state['signal'] = sig
        if not state['spawning'] and not state['cleaning']:
            raise Cancelled('cancelled by signal '+str(sig))
    try:
        for sig in previous: signal.signal(sig, cancel)
        yield
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)


def group_exists(pid):
    try: os.killpg(pid, 0); return True
    except ProcessLookupError: return False


def bounded_cleanup(process, exists=group_exists, send=os.killpg,
                    clock=time.monotonic, pause=time.sleep):
    """Reap the direct child and independently stop all members of its group.

    A reaped leader does not imply that descendants honored TERM. KILL any
    remaining group, then observe its disappearance within a second bounded
    deadline. An unverified group raises and the caller retains its holder.
    """
    def kill(sig):
        try: send(process.pid, sig)
        except ProcessLookupError: pass
    if exists(process.pid): kill(signal.SIGTERM)
    killed = False
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        kill(signal.SIGKILL); killed = True
        process.wait(timeout=5)
    if exists(process.pid):
        if not killed: kill(signal.SIGKILL)
        deadline = clock() + 5.0
        while exists(process.pid):
            remaining = deadline - clock()
            w.require(remaining > 0, 'process group remains after bounded KILL observation')
            pause(min(0.05, remaining))
    w.require(type(process.returncode) is int, 'direct child not stopped/reaped after bounded cleanup')
    return {'waited': True, 'reaped': True, 'process_group_stopped': True}


def supervise(command, cwd, stdout, stderr, seconds, holder, save_failure,
              popen=subprocess.Popen, cleanup=bounded_cleanup, pass_fds=()):
    """Parent-held locks surround this function and every failure save.

    Store Popen directly into caller holder under deferred parent handler;
    no signal mask is inherited by the child. Cleanup errors retain holder.
    """
    w.require(type(seconds) is int and seconds > 0 and type(holder) is dict
              and set(holder) == {'process'} and holder['process'] is None, 'bounded fresh parent holder required')
    state = {'spawning': False, 'cleaning': False, 'signal': None}
    result = {'status': 'not-started', 'waited': False, 'reaped': False, 'process_group_stopped': False}
    primary = None
    with termination_guard(state):
        try:
            state['spawning'] = True
            try:
                holder['process'] = popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                    env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                    start_new_session=True, pass_fds=pass_fds)
            finally: state['spawning'] = False
            if state['signal'] is not None: raise Cancelled('cancelled during spawn')
            process = holder['process']
            result.update(pid=process.pid, pgid=process.pid, status='running')
            try:
                result['returncode'] = process.wait(timeout=seconds)
                result.update(waited=True, reaped=True, status='finished', timed_out=False)
            except subprocess.TimeoutExpired:
                result.update(status='timeout', timed_out=True)
                raise
            w.require(type(result['returncode']) is int and result['returncode'] == 0, 'four-stage child exited nonzero')
        except BaseException as error:
            primary = error
            result.update(error_type=type(error).__name__, error=str(error), status='failed')
        finally:
            state['cleaning'] = True
            try:
                if holder['process'] is not None:
                    result.update(cleanup(holder['process']))
                    result['returncode'] = holder['process'].returncode
                    holder['process'] = None
                else:
                    result.update(status='not-started', process_group_stopped=False)
            except BaseException as error:
                result.update(cleanup_error_type=type(error).__name__, cleanup_error=str(error),
                              process_group_stopped=False, status='failed')
                if primary is None: primary = error
            if state['signal'] is not None and primary is None:
                primary = Cancelled('cancelled during cleanup')
                result.update(status='failed', error_type=type(primary).__name__, error=str(primary))
            try:
                if primary is not None:
                    try: save_failure(dict(result))
                    except BaseException as error:
                        # Keep the original process/cancellation error and
                        # expose failure-persistence loss without a success.
                        result['failure_save_error'] = str(error)
                        setattr(primary, 'failure_save_error', str(error))
            finally: state['cleaning'] = False
        if primary is not None:
            setattr(primary, 'parent_process_outcome', dict(result))
            raise primary
    return result


def main():
    raise RuntimeError('SOURCE ONLY I/O/lifecycle CLI is disabled')


if __name__ == '__main__': main()
