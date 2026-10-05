"""Disabled actual orchestration for normal white-view four-stage evaluation.

This is SOURCE preparation. No input/runtime/model/control is read on import.
Root enables a separately frozen copy only after external build/model proof
contracts and a launch preflight exist. The parent retains all serial flocks
through child wait, group cleanup, failure persistence and input rehash.
"""
from contextlib import ExitStack
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import white_view_evaluation as w
import white_view_runtime_io as io
import white_view_production_ports as p

PROTOTYPE_ONLY = True
LOCAL_MODULES = ('white_view_evaluation.py', 'white_view_runtime_io.py',
                 'white_view_production_ports.py', 'white_view_operations.py')


def barrier():
    w.require(not PROTOTYPE_ONLY, 'SOURCE ONLY orchestration disabled before I/O')
    p.barrier()


def pinned_reference(path, expected_sha):
    w.sha(expected_sha); path = str(w.absolute(str(path)))
    current = io.info(path)
    w.require(current['sha256'] == expected_sha, 'external SHA does not match actual raw bytes')
    return {'path': path, **current}


def read_launch(path, expected_sha, expected_worker_sha):
    barrier()
    os.umask(0o077)
    reader = io.Reader(); ref = pinned_reference(path, expected_sha)
    pf = p.validate_launch_preflight(reader.json_ref(ref))
    worker = Path(__file__).resolve(strict=True)
    w.require(worker.name == 'white_view_operations.py' and io.info(worker)['sha256'] == w.sha(expected_worker_sha), 'frozen canonical orchestration source required')
    for name in LOCAL_MODULES:
        source = str(worker.with_name(name))
        expected = pf['spec']['inputs'].get(source)
        w.file_identity(expected)
        w.require(pf['spec']['source_helpers'].get(source) == expected['sha256'], 'all adjacent source modules must be externally bound')
        reader.pin(source, expected)
    w.require(pf['spec']['inputs'][str(worker)]['sha256'] == expected_worker_sha, 'worker SHA not in launch input map')
    w.require(sys.executable == w.AUD and sys.prefix == str(Path(w.AUD).parent.parent), 'exact audit venv invocation required')
    return pf, ref, reader


def owned_state_replace(path, expected, value):
    """Replace only known mutable state after checking its exact prior bytes."""
    path = Path(path)
    w.require(path.name == 'evaluation.json' and w.exact(io.info(path), expected), 'state changed before parent completion')
    temp = path.with_name(path.name+'.parent-'+str(os.getpid()))
    io.write_new(temp, io.json_bytes(value))
    try:
        w.require(w.exact(io.info(path), expected), 'mutable state changed during parent replacement')
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def child(args):
    state = {'spawning': False, 'cleaning': False, 'signal': None}
    with io.termination_guard(state):
        return _child(args)


def _child(args):
    pf, ref, reader = read_launch(args.preflight, args.expected_preflight_sha256, args.expected_worker_sha256)
    lock_records = w.read_pinned_json_bytes(args.inherited_locks_json.encode(),
        hashlib.sha256(args.inherited_locks_json.encode()).hexdigest())
    p.validate_inherited_locks(lock_records)
    ports = p.ProductionPorts(ref, inherited_locks=lock_records)
    base = ports.reader.json_ref(pf['config_source'])
    draft = w.four_stage_call_sequence(pf['spec'], base, ports)
    w.require(draft['cleanup_verified'] is False and draft['executor_reaped'] is False,
              'child cannot claim parent wait/reap')
    ports.writer.write('draft.json', draft)
    ports.state.update(status='measured-pending-parent-terminal', cleanup_verified=False,
                       executor_reaped=False, stage_evidence=draft['stage_evidence'])
    ports.writer.write('evaluation.json', ports.state, mutable=True)
    return draft


def launch(args):
    barrier()
    state = {'spawning': False, 'cleaning': False, 'signal': None}
    with io.termination_guard(state):
        return _launch(args, state)


def _launch(args, state):
    pf, ref, reader = read_launch(args.preflight, args.expected_preflight_sha256, args.expected_worker_sha256)
    spec = pf['spec']; holder = {'process': None}
    # Parent output is separate from child-owned E and every bound input.
    parent_root = Path(args.parent_output)
    io.private_output(parent_root, [*spec['inputs'], spec['evaluation_path'], str(ref['path']), str(p.N)])
    with ExitStack() as stack:
        records = []
        for path, exclusive in p.LOCKS:
            stream = stack.enter_context(io.existing_lock(path, exclusive))
            records.append({'path': str(path), 'exclusive': exclusive, 'fd': stream.fileno()})
        # Full locked source/runtime/native/bridge/decoder preflight happens
        # before parent output/log creation or the first engine command.
        pre = p.ProductionPorts(ref, inherited_locks=records)
        with pre.coordinating_locks(spec):
            base = pre.reader.json_ref(pf['config_source'])
            pre.preflight(spec, w.config_for_role(base, spec))
        before = pre.reader.current()
        parent_root.mkdir(mode=0o700)
        command = [w.AUD, '-B', str(Path(__file__).resolve()), 'child', '--preflight', ref['path'],
            '--expected-preflight-sha256', ref['sha256'], '--expected-worker-sha256', args.expected_worker_sha256,
            '--inherited-locks-json', json.dumps(records, separators=(',', ':'))]
        def save_failure(outcome):
            frozen = {'schema': 'sekirei.white-view-evaluation-parent-failure.v1', 'status': 'failed',
                'role': spec['role'], 'preflight': ref, 'run_ids': spec['run_ids'], 'model': spec['model'],
                'parent_process_outcome': outcome, 'inputs_before': before,
                'final_used': False, 'comparison_valid': False, 'adopt': False,
                'executor_tool_reaped': False}
            try:
                frozen['inputs_after'] = pre.reader.current()
                frozen['inputs_unchanged'] = w.exact(before, frozen['inputs_after'])
            except BaseException as error:
                frozen.update(inputs_unchanged=False, inputs_after_error=str(error))
            io.write_new(parent_root/'failure.json', io.json_bytes(frozen))
        try:
            with (parent_root/'child.stdout').open('xb') as out, (parent_root/'child.stderr').open('xb') as err:
                outcome = io.supervise(command, str(p.R), out, err, args.timeout_seconds,
                    holder, save_failure, pass_fds=tuple(row['fd'] for row in records))
            w.require(holder['process'] is None and outcome['returncode'] == 0
                and outcome['reaped'] is True and outcome['process_group_stopped'] is True, 'normal child group cleanup required')
            after = pre.reader.current()
            w.require(w.exact(before, after), 'inputs/source/build/model changed during child')
            draft_ref = pinned_reference(str(Path(spec['evaluation_path'])/'draft.json'), args.expected_draft_sha256) if args.expected_draft_sha256 else {'path': str(Path(spec['evaluation_path'])/'draft.json'), **io.info(Path(spec['evaluation_path'])/'draft.json')}
            draft = reader.json_ref(draft_ref)
            w.require(w.exact(draft['inputs_before'], spec['inputs']) and w.exact(draft['inputs_after'], spec['inputs'])
                and draft['executor_reaped'] is False and draft['cleanup_verified'] is False, 'child draft subject/input contract differs')
            record = {'schema': 'sekirei.white-view-evaluation-parent-outcome.v1', 'status': 'child-stopped-awaiting-tool-terminal',
                'role': spec['role'], 'preflight': ref, 'draft': draft_ref, 'run_ids': spec['run_ids'], 'model': spec['model'],
                'child_outcome': outcome, 'inputs_before': before, 'inputs_after': after,
                'inputs_unchanged': True, 'cleanup_verified': True, 'executor_tool_reaped': False,
                'final_used': False, 'comparison_valid': False, 'adopt': False}
            io.write_new(parent_root/'parent-outcome.json', io.json_bytes(record))
            return record
        except BaseException as error:
            # A post-wait input/draft/write failure also persists as invalid.
            state['cleaning'] = True
            try:
                cleanup_error = None
                retained = holder['process']
                if retained is not None:
                    try:
                        io.bounded_cleanup(retained)
                        holder['process'] = None
                    except BaseException as cleanup:
                        cleanup_error = str(cleanup)
                if not (parent_root/'failure.json').exists():
                    save_failure({'status': 'failed', 'error': str(error),
                        'cleanup_error': cleanup_error, 'process_handle_retained': holder['process'] is not None,
                        'reaped': bool(retained is not None and retained.returncode is not None),
                        'process_group_stopped': holder['process'] is None})
            finally: state['cleaning'] = False
            raise


def complete(args):
    """Separate invocation after Root's wait tool observes launcher exit0."""
    pf, ref, reader = read_launch(args.preflight, args.expected_preflight_sha256, args.expected_worker_sha256)
    spec = pf['spec']
    with ExitStack() as stack:
        for path, exclusive in p.LOCKS: stack.enter_context(io.existing_lock(path, exclusive))
        ports = p.ProductionPorts(ref)
        # Successful completion must not run the fresh-output preflight.
        ports.verify_sources(); ports.verify_new_runtime(w.config_for_role(reader.json_ref(pf['config_source']), spec))
        if spec['role'] == 'candidate': ports.verify_candidate()
        for path, record in spec['inputs'].items(): reader.pin(path, record)
        parent_ref = pinned_reference(args.parent_outcome, args.expected_parent_outcome_sha256)
        parent = reader.json_ref(parent_ref)
        fixed = {'schema': 'sekirei.white-view-evaluation-parent-outcome.v1', 'status': 'child-stopped-awaiting-tool-terminal',
            'role': spec['role'], 'preflight': ref, 'run_ids': spec['run_ids'], 'model': spec['model'],
            'inputs_unchanged': True, 'cleanup_verified': True, 'executor_tool_reaped': False,
            'final_used': False, 'comparison_valid': False, 'adopt': False}
        w.require(all(w.exact(parent.get(k), v) for k, v in fixed.items()), 'normal externally pinned parent subject required')
        outcome = parent['child_outcome']
        w.require(outcome.get('returncode') == 0 and type(outcome.get('returncode')) is int
            and outcome.get('waited') is True and outcome.get('reaped') is True
            and outcome.get('process_group_stopped') is True and outcome.get('timed_out') is False,
            'parent wait/reap/group-stop outcome required')
        w.require(w.exact(parent['inputs_before'], parent['inputs_after']), 'parent input maps changed')
        for path, record in parent['inputs_after'].items(): reader.pin(path, record)
        terminal_ref = pinned_reference(args.terminal, args.expected_terminal_sha256)
        terminal = reader.json_ref(terminal_ref)
        p.validate_audit_terminal(terminal, spec['role'], spec['model'], spec['run_ids'])
        draft = reader.json_ref(parent['draft'])
        w.require(w.exact(draft['run_ids'], spec['run_ids']) and w.exact(draft['candidate_model'], spec['model']), 'draft subject differs')
        current = reader.current()
        op = {'schema': 'sekirei.white-view-evaluation-operational-verification.v1', 'status': 'complete',
            'role': spec['role'], 'terminal': terminal_ref, 'parent_outcome': parent_ref,
            'preflight': ref, 'inputs_before': spec['inputs'], 'inputs_after': spec['inputs'],
            'completion_inputs': current, 'inputs_unchanged': True, 'source_and_build_unchanged': True,
            'cleanup_verified': True, 'executor_reaped': True, 'final_used': False}
        completed = w.finalize_parent(draft, terminal, op, terminal_ref)
        # Fresh completed OP first, mutable state last. A write error does not
        # create a complete descriptor; neither output is reused on retries.
        io.write_new(Path(spec['evaluation_path'])/'operational-verification.json', io.json_bytes(op))
        state = Path(spec['evaluation_path'])/'evaluation.json'
        owned_state_replace(state, io.info(state), completed)
        return completed


def audit(args):
    barrier()
    os.umask(0o077)
    reader = io.Reader()
    request_ref = pinned_reference(args.audit_request, args.expected_audit_request_sha256)
    request = p.validate_audit_request(reader.json_ref(request_ref))
    for name in LOCAL_MODULES:
        path = str(Path(__file__).resolve().with_name(name))
        w.require(path in request['source_inputs'] and path in request['source_helpers'], 'all actual auditor source modules required')
        reader.pin(path, request['source_inputs'][path])
    w.require(request['source_inputs'][str(Path(__file__).resolve())]['sha256'] == w.sha(args.expected_worker_sha256), 'external audit source SHA differs')
    return p.audit_role(reader, request)


def bridge_or_compare(args, action):
    barrier()
    os.umask(0o077)
    reader = io.Reader()
    request_ref = pinned_reference(args.comparison_request, args.expected_comparison_request_sha256)
    request = reader.json_ref(request_ref)
    common_keys = {'schema', 'status', 'action', 'source_inputs', 'source_helpers', 'output'}
    keys = common_keys | ({'old_audit', 'new_audit', 'old_build', 'new_build', 'old_runtime'}
        if action == 'bridge' else {'baseline_audit', 'candidate_audit', 'bridge', 'model'})
    w.require(type(request) is dict and set(request) == keys
        and request['schema'] == 'sekirei.white-view-comparison-request.v1'
        and request['status'] == 'frozen-before-comparison' and request['action'] == action,
        'typed externally pinned comparison request required')
    w.require(type(request['source_inputs']) is dict and request['source_inputs'], 'full comparison input map required')
    for name in LOCAL_MODULES:
        source = str(Path(__file__).resolve().with_name(name))
        w.require(source in request['source_inputs'] and request['source_helpers'].get(source)
            == request['source_inputs'][source]['sha256'], 'all comparison source bytes must be externally pinned')
    w.require(request['source_inputs'][str(Path(__file__).resolve())]['sha256'] == w.sha(args.expected_worker_sha256), 'external comparison source SHA differs')
    with ExitStack() as locks:
        for path, exclusive in p.LOCKS: locks.enter_context(io.existing_lock(path, exclusive))
        for path, rec in request['source_inputs'].items(): reader.pin(path, rec)
        before = reader.current()
        if action == 'bridge':
            old = p.read_audit_bundle(reader, request['old_audit'])
            new = p.read_audit_bundle(reader, request['new_audit'])
            pins = {key: request[key] for key in ('old_audit', 'new_audit', 'old_build', 'new_build', 'old_runtime')}
            result = w.fallback_bridge(old, new, pins)
        else:
            baseline = p.read_audit_bundle(reader, request['baseline_audit'])
            candidate = p.read_audit_bundle(reader, request['candidate_audit'])
            verified_bridge = p.read_verified_bridge(reader, request['bridge'])
            result = w.compare_same_runtime(baseline, candidate, verified_bridge, request['bridge'], request['model'])
            result.update(baseline_audit=request['baseline_audit'], candidate_audit=request['candidate_audit'])
        for path, rec in reader.files.items():
            if path != request_ref['path']:
                w.require(w.exact(request['source_inputs'].get(path), rec), 'complete raw/terminal/audit payload closure missing from external request')
        w.require(w.exact(reader.current(), before), 'comparison input set/bytes changed')
        output = Path(request['output'])
        io.private_output(output, list(before))
        return io.write_new(output, io.json_bytes(result))


def main():
    barrier()  # Disabled before argparse/file access, including --help.
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('launch', 'child', 'complete', 'audit', 'bridge', 'compare'))
    parser.add_argument('--preflight')
    parser.add_argument('--expected-preflight-sha256')
    parser.add_argument('--expected-worker-sha256', required=True)
    parser.add_argument('--inherited-locks-json')
    parser.add_argument('--parent-output')
    parser.add_argument('--timeout-seconds', type=int, default=20000)
    parser.add_argument('--expected-draft-sha256')
    parser.add_argument('--parent-outcome')
    parser.add_argument('--expected-parent-outcome-sha256')
    parser.add_argument('--terminal')
    parser.add_argument('--expected-terminal-sha256')
    parser.add_argument('--audit-request')
    parser.add_argument('--expected-audit-request-sha256')
    parser.add_argument('--comparison-request')
    parser.add_argument('--expected-comparison-request-sha256')
    args = parser.parse_args()
    if args.action == 'launch':
        w.require(type(args.timeout_seconds) is int and 1 <= args.timeout_seconds <= 20000, 'bounded formal wall timeout')
        return launch(args)
    if args.action == 'child': return child(args)
    if args.action == 'complete': return complete(args)
    if args.action == 'audit': return audit(args)
    return bridge_or_compare(args, args.action)


if __name__ == '__main__': main()
