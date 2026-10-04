#!/usr/bin/env python3
"""SOURCE ONLY native03 finite transition proof; no old Adam/E3 mode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import white_view_proof_common_v2 as p
PROTOTYPE_ONLY=True
OUT=p.INCR_OUT
WORKER=p.C/'white-view-candidate-incremental-proof-worker-v2.py'


def run(args):
    p.require(PROTOTYPE_ONLY is False,'SOURCE ONLY incremental prototype; runtime entry is disabled before any I/O')
    p.runtime_gate();p.require(Path(__file__)==WORKER and Path(__file__).resolve()==WORKER,'canonical private proof worker required')
    modules=p.load_public(args);d=modules['diagnose_anchor'];os.umask(0o077)
    holder={'process':None};record=None
    with d.termination_guard():
        with p.locks(modules):
            try:
                reader,ctx=p.context(args,Path(__file__),modules)
                core,source=p.stock(reader,modules)
                probe=ctx['runtime_verified']['manifest']['probe_binaries']['incremental']
                p.fixture_tsv_equal(reader.read(p.FIXTURES_TSV),p.strict_json(reader.read(p.PUBLIC_FIXTURES,p.PUBLIC_FIXTURES_SHA)))
                p.output_guard(OUT,reader,modules);before=p.snapshot(reader,ctx,core,source)
                OUT.mkdir(mode=0o700);started=time.monotonic()
                record=p.record_base(args,ctx,'incremental',reader)
                record.update(probe=probe,probe_source=ctx['runtime_verified']['identity']['probe_sources']['incremental'],
                    reuse_reason='Unchanged fixed fixture/walk APIs; the probe is linked to the actual new white-view USI core and reads only validated candidate03.',
                    scope='15 public fixtures/16 deterministic walks/8 search move APIs; no search or learning')
                p.save(OUT/'launch.json',record);p.save(OUT/'snapshot-before.json',before)
                inp=OUT/'stdin.empty';inp.write_bytes(b'')
                command=[probe['path'],str(p.Q/'weights.bin'),str(p.FIXTURES_TSV),'100'];record['argv']=command
                p.execute(command,OUT,inp,OUT/'stdout.json',OUT/'stderr.log',60,holder,modules)
                stdout=(OUT/'stdout.json').read_bytes();stderr=(OUT/'stderr.log').read_bytes()
                result=p.incremental_result(p.strict_json(stdout))
                p.require(stderr==b'','unexpected incremental stderr')
                after=p.snapshot(reader,ctx,core,source);p.require(p.exact(before,after),'candidate proof inputs changed')
                p.require((OUT/'stdout.json').read_bytes()==stdout and (OUT/'stderr.log').read_bytes()==stderr
                    and inp.read_bytes()==b'','validated incremental raw output changed')
                p.save(OUT/'snapshot-after.json',after)
                record.update(status='complete',returncode=0,timeout=False,result=result,
                    inputs_before=before['files'],inputs_after=after['files'],inputs_unchanged=True,source_unchanged=True,
                    stock_build_unchanged=True,new_build_unchanged=True,cleanup_verified=True,wall_seconds=time.monotonic()-started,
                    output_files={q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sorted(OUT.iterdir()) if q.is_file()})
                p.save(OUT/'receipt.json',record)
            except BaseException as error:
                if record is not None:
                    record.update(status='cancelled' if isinstance(error,d.DiagnosticCancelled) else 'timeout' if isinstance(error,p.subprocess.TimeoutExpired) else 'failed',timeout=isinstance(error,p.subprocess.TimeoutExpired),error=type(error).__name__+': '+str(error))
                    try:
                        if holder['process'] is not None:d.cleanup_group(holder['process']);holder['process']=None
                    except BaseException as cleanup_error:record.update(status='cleanup_failure',cleanup_error=str(cleanup_error))
                    finally:
                        with d.blocked_termination():p.save(OUT/'failure.json',record)
                raise
            finally:
                if holder['process'] is not None:
                    with d.blocked_termination():d.cleanup_group(holder['process']);p.require(not d.group_exists(holder['process'].pid),'incremental group remains after outer cleanup')
    print(json.dumps({'status':'complete','receipt_sha256':hashlib.sha256((OUT/'receipt.json').read_bytes()).hexdigest()}),flush=True)


if __name__=='__main__':
    p.require(PROTOTYPE_ONLY is False,'SOURCE ONLY entry disabled before I/O');p.runtime_gate()
    parser=argparse.ArgumentParser(description=__doc__);p.common_arguments(parser);run(parser.parse_args())
