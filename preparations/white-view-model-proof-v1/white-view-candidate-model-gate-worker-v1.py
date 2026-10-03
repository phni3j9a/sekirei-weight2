#!/usr/bin/env python3
"""Disabled producer and read-only consumer for the dedicated native03 gate.

The evaluation adapter owns the cooperating locks when calling the consumer.
The producer owns its dedicated proof locks itself; these differ from the
evaluation lock set. Neither path launches a probe or fit. The actual build
verifier may run read-only compiler/git child commands.
"""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import white_view_proof_common as p

PROTOTYPE_ONLY=True
WORKER=p.C/'white-view-candidate-model-gate-worker-v1.py'
BINDING_KEYS={'schema','common','preregistration','activation','source_preflight','fit_run','native','metadata',
    'coefficients_f32','coefficients_f64','solver_certificate','reference03','numeric_audit','numeric_worker',
    'core','core_worker','incremental','incremental_worker','gate','gate_worker'}
BINDING_PATHS={'common':p.COMMON,'preregistration':p.PR,'activation':p.ACTIVATION,'source_preflight':p.SOURCE_PREFLIGHT,
    'fit_run':p.Q/'run.json','native':p.Q/'weights.bin','metadata':p.Q/'weights.meta.json',
    'coefficients_f32':p.Q/'coefficients.f32.bin','coefficients_f64':p.Q/'coefficients.f64.bin',
    'solver_certificate':p.Q/'solver-certificate.json','reference03':p.REFERENCE,'numeric_audit':p.NUMERIC,
    'numeric_worker':p.NUMERIC_WORKER,'core':p.CORE_OUT/'receipt.json','core_worker':p.C/'white-view-candidate-core-proof-worker-v1.py',
    'incremental':p.INCR_OUT/'receipt.json','incremental_worker':p.C/'white-view-candidate-incremental-proof-worker-v1.py',
    'gate':p.GATE_OUT,'gate_worker':WORKER}


def barrier():
    p.require(PROTOTYPE_ONLY is False,'SOURCE ONLY model gate entry disabled before I/O')
    p.runtime_gate()


def validate_binding(binding):
    p.require(type(binding) is dict and set(binding)==BINDING_KEYS,'exact new candidate proof binding required')
    p.require(binding['schema']=='sekirei.white-view-candidate-gate-binding.v1','candidate proof binding schema differs')
    for key,path in BINDING_PATHS.items():p.fullref(binding[key],path)
    return binding


def arguments_from_binding(binding):
    validate_binding(binding)
    mapping={'worker':'gate_worker','common':'common','preregistration':'preregistration','activation':'activation',
        'source_preflight':'source_preflight','fit_run':'fit_run','native':'native','metadata':'metadata',
        'coefficients_f32':'coefficients_f32','coefficients_f64':'coefficients_f64','solver_certificate':'solver_certificate',
        'reference03':'reference03','numeric_audit':'numeric_audit','numeric_worker':'numeric_worker',
        'core':'core','core_worker':'core_worker','incremental':'incremental','incremental_worker':'incremental_worker'}
    return SimpleNamespace(**{'expected_'+key+'_sha256':binding[value]['sha256'] for key,value in mapping.items()})


def reconstruct(args,modules):
    reader,ctx=p.context(args,WORKER,modules)
    core,source=p.stock(reader,modules)
    proofs=p.bind_proofs(args,reader,ctx)
    before=p.snapshot(reader,ctx,core,source);after=p.snapshot(reader,ctx,core,source)
    p.require(p.exact(before,after),'technical proof sources/inputs changed')
    expected=p.gate_document(args,reader,ctx,proofs,before['files'])
    return reader,expected


def verify_candidate_inputs(reader,binding,spec):
    """Called under the evaluation parent's seven held cooperating locks.

    The supplied launch inventory must cover every transitive verified input.
    Actual raw gate bytes and parsed document are both externally SHA-bound.
    """
    barrier();validate_binding(binding)
    p.require(type(spec) is dict and p.exact(spec.get('candidate_gate'),binding['gate'])
        and type(spec.get('inputs')) is dict,'externally frozen candidate launch spec required')
    args=arguments_from_binding(binding);modules=p.load_public(args)
    local,expected=reconstruct(args,modules)
    for key,rec in ((key,binding[key]) for key in BINDING_PATHS if key!='gate'):
        local.pin(rec['path'],{k:rec[k] for k in ('bytes','sha256')})
    p.require(p.exact(spec.get('model'),expected['model']) and p.exact(spec.get('build_manifest'),expected['build_manifest'])
        and p.exact(spec.get('build_identity'),expected['new_build_binding']['identity']),'candidate/build identity differs')
    raw=local.read(p.GATE_OUT,binding['gate']['sha256'])
    p.require(len(raw)==binding['gate']['bytes'],'gate fullref bytes differ')
    actual=p.validate_gate_document(p.strict_json(raw),expected)
    for path,rec in local.files.items():
        p.require(p.exact(spec['inputs'].get(path),rec),'all transitive model proof inputs must be launch-bound')
        reader.pin(path,rec)
    return actual


def run(args):
    barrier();p.require(Path(__file__)==WORKER and Path(__file__).resolve()==WORKER,'canonical gate source required')
    modules=p.load_public(args);d=modules['diagnose_anchor']
    with d.termination_guard():
        with p.locks(modules):
            reader,expected=reconstruct(args,modules)
            p.output_guard(p.GATE_OUT,reader,modules)
            # Recheck every input immediately before the one final receipt write.
            for path,frozen in list(reader.files.items()):reader.pin(path,frozen)
            p.validate_gate_document(expected,expected)
            p.save(p.GATE_OUT,expected)
    print(hashlib.sha256(p.GATE_OUT.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':
    barrier()
    parser=argparse.ArgumentParser(description=__doc__);p.common_arguments(parser)
    for name in ('core','core-worker','incremental','incremental-worker'):
        parser.add_argument('--expected-'+name+'-sha256',required=True)
    run(parser.parse_args())
