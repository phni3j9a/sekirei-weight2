# Version 2 source wiring

Version 1 bytes and all fit/input bindings remain preserved. This version only
selects canonical `white_view_proof_common_v2.py`, the three worker-v2 sources,
and independent numeric worker-v2/receipt-v2. Core/incremental/gate output paths
remain v1 because they have not been created. Every schema, math/guard body and
API remains unchanged; normalized reverse replacement reproduces all four v1
source files byte for byte and their 46 top-level function/class ASTs.

The gate requires externally pinned numeric-v2 receipt and worker hashes and
rejects binding refs to the v1 paths. The evaluator loader must temporarily bind
`sys.modules['white_view_proof_common_v2']` to the SHA-verified real canonical
common-v2 object during gate-worker execution, then restore the previous value
or absence. It must not alias v2 as the old common import name.

No actual artifacts/control/model/runtime/raw were read or executed in this
source preparation. Source-only failures/history are not new measurements.

# White-view native03 model proof source preparation

This directory contains disabled source preparation only. No actual model,
dataset, runtime, receipt, raw output, engine, compiler or fit was opened or
started during preparation. Root alone enables separately preserved copies,
freezes all source hashes and supplies actual external SHA arguments.

The dedicated contracts are:

- `sekirei.white-view-candidate-core-proof.v1`: original train 112681, original
  holdout 5895 and fixed public 15, totaling 118591 actual core observations.
- `sekirei.white-view-candidate-incremental-proof.v1`: fixed 8185 transition
  observations, undo/null/refresh equality, finite float observations, MXCSR
  and observed integer material difference at most 100 cp.
- `sekirei.white-view-model-technical-gate.v1`: three independently recomputed
  numerical certificates, full native03 reconstruction, full reference03
  transformation and strict raw reparse of both completed proof receipts.
- Numeric producer is separate, under
  `sekirei.white-view-paired-linear-independent-numeric-audit.v1` and inner
  `sekirei.white-view-paired-linear-independent-numeric-math.v1`.

The original material initializer remains native01. The dedicated reference
is `training-17-v1/white-view-material-init-seed42-v1/reference03.bin` and must
match the full `white_view_paired_linear.transform_initializer()` output,
including all hand ties and protected material/board/bias bytes. Q retains its
exact nine fit outputs. Header replacement alone is rejected.

Core pair argument 1 is the coefficient-rebuilt candidate03 and argument 2
is the separately transformed reference03. `native_*` fields belong to the
candidate, while the stock probe's historical `nearest_*` field names belong
to the reference. This is neither a selfload pair nor quantizer reexport.
Reference integer and float CP must equal Python material; observed candidate
integer residual must be at most 100 cp, and the actual float/integer bridge
must be strictly below 1.001 cp. Strict schemas reject missing/extra fields,
bools used as integers, nonfinite floats, duplicate JSON fields and wrong order.

`white_view_proof_common_v2.py` preserves 14 mode-independent function/class ASTs
from the fixed `paired_linear_proof_common_v2.py` source SHA
`063ae00387bf81dee2e417ebf27da95f6ca5d0fb0af329b278889d97f435dc30`.
No old Adam/E3/checkpoint, mode-specific build/metadata or adopted-model guard
is called or weakened. The old export source hashes still point to the original
stock source. Fixed provenance is 617 files including 526 tracked source files
and 63 old release dependency files, plus a distinct, fully verified new runtime
inventory. New release dependency membership is not forced to 63.

The new probes reuse the unchanged Rust source interfaces but are selected
from the actual dedicated build manifest and linked to that USI executable's
core dependency. Their source, compiled binary, compiler, build identity and
all transitive inputs are reverified by the frozen build verifier. Its private
adjacent contract is imported with real SHA-pinned bytes; only the import cache
entry is temporarily bound and restored, including import-failure cases. No
source path, `__file__`, compiled producer or old guard is spoofed.

Runtime CLI source bodies:

1. `white-view-candidate-core-proof-worker-v2.py`: canonical C output directory
   `white-view-candidate-core-proof-v1`; invokes only the new verified pair probe.
2. `white-view-candidate-incremental-proof-worker-v2.py`: canonical C output
   `white-view-incremental-candidate-proof-v1`; invokes only the new incremental
   probe using fixed public fixtures.
3. `white-view-candidate-model-gate-worker-v2.py`: canonical C output file
   `white-view-model-technical-gate-v1.json`; starts no probe/fit and reparses
   both complete proof outputs and all before/after snapshots.

All three require `--expected-{worker,common,preregistration,activation,
source-preflight,fit-run,native,metadata,coefficients-f32,coefficients-f64,
solver-certificate,reference03,numeric-audit,numeric-worker}-sha256`.
The gate additionally requires `--expected-{core,core-worker,incremental,
incremental-worker}-sha256`. Unknown actual values are mandatory external pins.
Every entry rejects before actual I/O while `PROTOTYPE_ONLY=True`.

Producer locks: fit SH, original trainer EX, original runtime prepare SH and
benchmark EX, new runtime build SH, prepare SH and benchmark EX. Locks stay held
through Popen caller-holder assignment, wait/reap, bounded process-group cleanup,
failure receipt and source/input rechecks. Paths must be canonical/private,
new, outside Git, nonoverlapping with every protected input and have 2 GiB free.
Shared campaign-root sibling inputs are allowed without weakening the original
output guard. Success is emitted only after full checks. Failure/cancellation
never becomes success.

Read-only evaluation API:

`verify_candidate_inputs(reader, binding, spec)` in the model-gate worker is
called **inside the evaluation parent's already held cooperating locks**; it
does not reacquire them. The evaluation parent holds its own lock set; it is
not the producer/probe lock set listed above. `binding` has schema
`sekirei.white-view-candidate-gate-binding.v1` and the exact fullrefs listed by
`BINDING_PATHS`. Every transitive input must appear identically in `spec.inputs`.
The expected gate is reconstructed, raw gate bytes are externally SHA-bound and
its complete typed JSON document must equal the reconstructed one. `model`,
`build_manifest`, `build_identity` and gate fullrefs must equal the launch spec.
The consumer returns that exact verified document for the evaluation adapter.
The runtime verifier can invoke read-only compiler/git child commands, so the
gate truthfully records no_child_process_started=false,
no_engine_or_fit_process_started=true and
runtime_validator_may_invoke_readonly_source_compiler_checks=true.

Tests are synthetic bytes/JSON and source AST fixtures. The reference fixture
is a synthetic transform mock; actual protected native-byte transformation is
the existing white-view pure helper's separate fixture and runtime gate. The
certificate arithmetic check does not independently prove Gram extraction or
gradient origin: the separate numeric audit computes H*u-b, DeltaJ, Gershgorin
and all three exact certificates from saved artifacts. Design/PSD/trajectory
replay scope remains false. Finite probes and observed 100 cp caps are evidence
for observed rows/transitions, not universal covariance, all-position bounds,
Python/native full-forward equality, search success or adoption.
