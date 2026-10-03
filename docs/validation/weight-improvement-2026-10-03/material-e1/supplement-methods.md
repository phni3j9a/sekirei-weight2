# Three-series development supplement

The five panels show Teacher, Reference model (the strict comparison's baseline), and Candidate model using one shared raw centipawn axis from Black's perspective. The reference label does not independently assert that a model was adopted.

Only exact cp observations appear as lines and points. A bound, mate, no-score, or nonconsecutive ply breaks a segment. Bound and mate values are never converted to cp. The per-panel E/B/M/N counts retain their types.

The first, middle and last thirds are mathematical subdivisions of each complete game: for ply 1..N, third index is floor(3*(ply-1)/N). They are not claims about opening, middlegame or endgame. Dotted boundaries fall between the last ply of one third and the first ply of the next.

Every stage error metric uses all teacher-exact positions in that game's third as its fixed denominator. If a model has a missing/nonexact score on any such point, its metric is undefined; the denominator never shrinks. Zero Teacher-E points also yield undefined metrics. Pooled third MAEs are explicitly micro-weighted diagnostic summaries, separate from the existing five-game equal-weight formal headline.

Large-error diagnostics publish absolute-error quantiles and maxima with coverage counts, without identifying positions. Quantiles use linear interpolation at (n-1)*p in the sorted sample. No position-level JSON is published. The SVG's curve coordinates follow the same disclosure scope as the existing public evaluation graph.

The completed strict comparison, its input file hashes and file sets, teacher semantic projections, occurrence projections, fixed Teacher-E 266, and exact rational per-game/five-game MAEs are verified before this derivative report is produced. It does not rerun search or change formal headlines, Top3 results, or adoption decisions.

## Per-game mathematical thirds

| Game | Third | Teacher-E denominator | Reference exact | Candidate exact | Reference MAE (cp) | Candidate MAE (cp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | first third | 6 | 6 | 6 | 64.17 | 85.83 |
| 1 | middle third | 12 | 12 | 12 | 323.00 | 186.25 |
| 1 | last third | 24 | 24 | 24 | 465.42 | 180.50 |
| 2 | first third | 28 | 28 | 28 | 896.14 | 766.14 |
| 2 | middle third | 12 | 12 | 12 | 988.33 | 641.08 |
| 2 | last third | 32 | 32 | 32 | 2252.69 | 2536.97 |
| 3 | first third | 8 | 8 | 8 | 184.00 | 128.00 |
| 3 | middle third | 20 | 20 | 20 | 1259.15 | 933.05 |
| 3 | last third | 38 | 38 | 38 | 1173.11 | 1680.24 |
| 4 | first third | 9 | 9 | 9 | 313.11 | 692.22 |
| 4 | middle third | 10 | 10 | 10 | 1054.00 | 594.10 |
| 4 | last third | 21 | 21 | 21 | 1692.81 | 789.62 |
| 5 | first third | 13 | 13 | 13 | 475.62 | 441.15 |
| 5 | middle third | 17 | 17 | 17 | 483.82 | 544.35 |
| 5 | last third | 16 | 16 | 16 | 2660.50 | 3011.44 |

## Pooled thirds (micro diagnostic only)

These pool Teacher-E positions across games. They are not the formal five-game equal-weight headline.

| Third | Teacher-E denominator | Reference MAE (cp) | Candidate MAE (cp) |
| --- | ---: | ---: | ---: |
| first third | 64 | 561.72 | 546.19 |
| middle third | 71 | 840.62 | 616.68 |
| last third | 131 | 1572.15 | 1634.57 |

## Absolute-error distribution (pooled Teacher-E)

| Model | Exact / fixed denominator | p50 (cp) | p90 (cp) | p95 (cp) | p99 (cp) | Maximum (cp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 266 / 266 | 883.50 | 1814.00 | 1993.00 | 2394.40 | 32941.00 |
| Candidate | 266 / 266 | 756.00 | 1828.50 | 2048.00 | 2464.65 | 33669.00 |
