#!/usr/bin/env python3
"""SOURCE ONLY native03/reference03 full-row proof; synthetic preparation only."""
from contextlib import ExitStack
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
OUT=p.CORE_OUT
WORKER=p.C/'white-view-candidate-core-proof-worker-v2.py'


def run(args):
    p.require(PROTOTYPE_ONLY is False,'SOURCE ONLY core prototype; runtime entry is disabled before any I/O')
    p.runtime_gate();p.require(Path(__file__)==WORKER and Path(__file__).resolve()==WORKER,'canonical private proof worker required')
    modules=p.load_public(args);d=modules['diagnose_anchor'];a=modules['functional_anchor']
    os.umask(0o077);holder={'process':None};record=None
    with d.termination_guard():
        with p.locks(modules):
            try:
                reader,ctx=p.context(args,Path(__file__),modules)
                core,source=p.stock(reader,modules)
                probe=ctx['runtime_verified']['manifest']['probe_binaries']['core_pair']
                fixtures_path=p.PUBLIC_FIXTURES
                fixtures=p.strict_json(reader.read(fixtures_path,p.PUBLIC_FIXTURES_SHA));p.require(type(fixtures) is list and len(fixtures)==15,'15 fixed public fixtures required')
                positions={split:ctx['rows'][split+'.positions.jsonl'] for split in ('train','holdout')}
                positions['fixtures']=[{'sfen':v['sfen']} for v in fixtures]
                p.output_guard(OUT,reader,modules);before=p.snapshot(reader,ctx,core,source)
                OUT.mkdir(mode=0o700);started=time.monotonic()
                record=p.record_base(args,ctx,'core',reader)
                record.update(counts=p.COUNTS,probe=probe,probe_source=ctx['runtime_verified']['identity']['probe_sources']['core_pair'],
                    pair_argument_order=['candidate03','reference03'],seconds_per_split=600,results={},
                    reuse_reason='Unchanged source-level probe interface; newly linked dedicated white-view core, full transformed reference03 and candidate03 are separately loaded.')
                p.save(OUT/'launch.json',record);p.save(OUT/'snapshot-before.json',before)
                for split in ('train','holdout','fixtures'):
                    sfens=[r['sfen'] for r in positions[split]];count=len(sfens)
                    p.require(count==(15 if split=='fixtures' else p.COUNTS[split]) and all(type(s) is str and s and '\n' not in s and '\r' not in s for s in sfens),'split count/SFEN differs')
                    stdin=''.join(s+'\n' for s in sfens).encode()
                    inp,out,err=(OUT/(split+suffix) for suffix in ('.sfens','.stdout.jsonl','.stderr.txt'))
                    with inp.open('xb') as stream:stream.write(stdin)
                    command=[probe['path'],str(p.Q/'weights.bin'),str(p.REFERENCE),'x86-ftz-daz']
                    rc=p.execute(command,OUT,inp,out,err,600,holder,modules)
                    stdout,stderr=out.read_bytes(),err.read_bytes();p.core_stderr(stderr,count)
                    lines=stdout.splitlines();p.require(len(lines)==count and inp.read_bytes()==stdin,'ordered stdin/output count changed')
                    maximum,bridge=0,0.0
                    for index,(line,sfen) in enumerate(zip(lines,sfens)):
                        value=p.core_row(line,index,a.fixed_material_cp(sfen))
                        maximum=max(maximum,abs(value['native_core_cp']-value['material_cp']))
                        bridge=max(bridge,abs(value['native_quantized_float_cp']-value['native_core_cp']))
                        if split=='fixtures':p.require(value['material_cp']==fixtures[index]['expected_cp_stm'],'public fixture material differs')
                    record['results'][split]={'count':count,
                        'positions_sha256':reader.files[str(fixtures_path if split=='fixtures' else p.DATASET/(split+'.positions.jsonl'))]['sha256'],
                        'stdin_sha256':hashlib.sha256(stdin).hexdigest(),'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'stderr_sha256':hashlib.sha256(stderr).hexdigest(),
                        'ordered_sfens_sha256':hashlib.sha256(a.canonical_json_bytes(sfens)).hexdigest(),'returncode':rc,'timeout':False,'cleanup_status':'ok',
                        **p.CORE_FLAGS,
                        'maximum_observed_candidate_material_difference_cp':maximum,'maximum_float_core_bridge_cp':bridge}
                    print(json.dumps({'stage':split+'-complete','count':count}),flush=True)
                for split,result in record['results'].items():
                    for suffix,key in (('.sfens','stdin_sha256'),('.stdout.jsonl','stdout_sha256'),('.stderr.txt','stderr_sha256')):
                        p.require(hashlib.sha256((OUT/(split+suffix)).read_bytes()).hexdigest()==result[key],'validated raw output changed')
                after=p.snapshot(reader,ctx,core,source);p.require(p.exact(before,after),'proof input/source changed')
                p.save(OUT/'snapshot-after.json',after)
                p.require(sum(v['count'] for v in record['results'].values())==118591,'118591 totalrows required')
                record.update(status='complete',total_rows=118591,inputs_unchanged=True,source_unchanged=True,cleanup_verified=True,stock_build_unchanged=True,new_build_unchanged=True,wall_seconds=time.monotonic()-started)
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
                    with d.blocked_termination():d.cleanup_group(holder['process']);p.require(not d.group_exists(holder['process'].pid),'probe group remains after outer cleanup')
    print(json.dumps({'status':'complete','receipt_sha256':hashlib.sha256((OUT/'receipt.json').read_bytes()).hexdigest()}),flush=True)


if __name__=='__main__':
    p.require(PROTOTYPE_ONLY is False,'SOURCE ONLY entry disabled before I/O');p.runtime_gate()
    parser=argparse.ArgumentParser(description=__doc__);p.common_arguments(parser);run(parser.parse_args())
