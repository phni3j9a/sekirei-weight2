# functional-anchor half epoch3最近傍の正式比較

**採用条件を満たさなかった。** 正式MAEとTop3の証拠は有効で、固定教師・対象集合・モデル以外の実行条件が比較基準と一致した。

| 採用指標 | 駒得fallback | 候補 | 候補 − 基準 |
| --- | ---: | ---: | ---: |
| MAE | 1084.479 cp | 995.804 cp | -88.675 cp |
| Top3入り率 | 55.3888% | 55.3828% | -0.0059ポイント |

採用条件は「MAEが厳密に改善し、Top3入り率が下がらないこと」。各局の整数cp誤差・hit数から5局等重みの有理数で判定した。未丸め値と分数は comparison.json に保存する。

両者100万ノード指定、Threads=1、Hash=128 MiB。MAEはMultiPV=1、Top3はSekireiの別runでMultiPV=3。最終評価用データは使っていない。

| 局 | 教師exact点数 | 基準MAE | 候補MAE | 基準Top3 hit / 対象 | 候補Top3 hit / 対象 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 42 | 367.405 | 166.690 | 49 / 84 | 48 / 84 |
| 2 | 72 | 1514.417 | 1491.375 | 83 / 122 | 81 / 122 |
| 3 | 66 | 1079.288 | 1180.439 | 55 / 127 | 55 / 127 |
| 4 | 40 | 1222.675 | 881.275 | 55 / 104 | 57 / 104 |
| 5 | 46 | 1238.609 | 1259.239 | 62 / 114 | 63 / 114 |

MAEは570局面・1,140 attemptを完了し、固定Teacher-E 266点を全て採点した。Top3は対象551点を全て測定した。比較検証では教師の型付き評価・bestmove・PV、Teacher-Eのcp、Top3の対象・合法手・分母・参照bestmoveを照合した。

候補：`functional-anchor half epoch3最近傍`。重みSHA-256：`54e388057db60cab61f7d4110565cf2456564aa28dea11af0456cf0c4fa6d111`。

この結果は固定development 5局の測定であり、最終評価や一般的な棋力の改善は示さない。今回の条件と次の判断は[自律改善](../../../AUTONOMOUS_WEIGHT_IMPROVEMENT.md)、前回の学習・診断条件は[改善実験](../../../WEIGHT_IMPROVEMENT.md)を参照。

- [候補の評価値集計とグラフ](mae/validation.md)
- [比較値とidentity](comparison.json)

MAE公開exportの4ファイルは元のまま保持した。生成見出しは共通の「baseline validation」だが、fingerprintと数値はこの候補のもの。
