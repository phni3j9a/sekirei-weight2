# Development 1M-node baseline validation

This is a validation report for the fixed development split; it is not the final holdout.

- run type: `formal`
- games / occurrences: 5 / 570
- cshogi conversion check: {'available': False, 'checked_legal_moves': 0, 'checked_occurrences': 0}
- teacher-E exact points: 266
- candidate exact coverage on E: 266 / 266 (1.0)
- candidate headline MAE: 1087.0461722818245 cp (valid)
- formal headline eligible: True / evidence-valid run: True
- exact attempt matrix: 1140 / 1140 (missing 0, technical failures 0)
- retained repetitions / attempts: 1 / 1140; observed max node evidence: 1001086
- position types: normal=563, mate_in_one_available=1, forced_single_legal_move=5, terminal_checkmate=1, terminal_stalemate=0
- position selections: canonical=0, regression=0, formal=570
- positive node evidence: teacher: 11039 positive observations / 569 attempts (has_positive=True); sekirei: 563 positive observations / 563 attempts (has_positive=True)
- all-evidence M distribution: teacher: evidence=570 positive=569 zero=1 missing=0 invalid=0 p50=1000337.5 p95=1000719.0 p99=1000821.6 max=1001086 >N=545 >C=0 max_overrun=1086 rate=0.001086; sekirei: evidence=569 positive=563 zero=6 missing=1 invalid=0 p50=1000000 p95=1000002.0 p99=1000003.0 max=1000004 >N=228 >C=0 max_overrun=4 rate=4e-06
- M quantiles: sorted finite M values; linear interpolation at (n - 1) * p
- diagnostic overlap micro MAE: 1136.8684210526317 cp; mean signed error: 60.85338345864662 cp; max absolute error: 32941 cp

## Fixed masks and diagnostics

The headline is defined only when every teacher-E point in all five games has a candidate exact cp result. Bounds and mates are retained as diagnostics and are not converted to cp.

- bound interval checks: 273 teacher-B points; lower violations 67; upper violations 78
- mate winner counts: {'same': 8, 'opposite': 0, 'nonmate': 22, 'unknown': 1}; known distance pairs 8

## All-U type cross-table

| teacher \ candidate | exact_cp | bound_cp | mate | no_score | failure | missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_cp | 266 | 0 | 0 | 0 | 0 | 0 |
| bound_cp | 273 | 0 | 0 | 0 | 0 | 0 |
| mate | 22 | 0 | 8 | 1 | 0 | 0 |
| no_score | 0 | 0 | 0 | 0 | 0 | 0 |
| failure | 0 | 0 | 0 | 0 | 0 | 0 |
| missing | 0 | 0 | 0 | 0 | 0 | 0 |

## All-U operational/type cross-table

| teacher operational state \ candidate type | exact_cp | bound_cp | mate | no_score | failure | missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| operational | 561 | 0 | 8 | 1 | 0 | 0 |
| failure | 0 | 0 | 0 | 0 | 0 | 0 |
| missing | 0 | 0 | 0 | 0 | 0 | 0 |

| teacher operational state \ candidate operational state | operational | failure | missing |
| --- | ---: | ---: | ---: |
| operational | 570 | 0 | 0 |
| failure | 0 | 0 | 0 |
| missing | 0 | 0 | 0 |

## Reproducibility hashes

- benchmark_manifest_sha256: `d7ee0cce928be4a45b43733dc8d34d568583b0451586078e19df98892ad9e5fc`
- development_csa_aggregate_sha256: `0e02b6319cbf908761dde7326ab6a1bfc4b647e2601ca1e3643fa6a207f48ee2`
- position_classifications_sha256: `a244a2206fd2b25b6fe475a9794c07eb2c5fc99f996e891dbfed1e89a0a89427`
- universe_sha256: `33ce54f3ff9c40687e2304a7dc222ceddc0d6843180020a68262dd1762ec5fa6`
