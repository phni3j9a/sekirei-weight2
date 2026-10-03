# 112681局面・13変数駒価値学習の正式比較

**採用条件を満たさなかった。** 正式MAEとTop3の証拠は有効で、固定教師・対象集合・モデル以外の実行条件が比較基準と一致した。

| 採用指標 | 駒得fallback | 候補 | 候補 − 基準 |
| --- | ---: | ---: | ---: |
| MAE | 1084.479 cp | 1068.398 cp | -16.081 cp |
| Top3入り率 | 55.3888% | 53.9054% | -1.4833ポイント |

採用条件は「MAEが厳密に改善し、Top3入り率が下がらないこと」。各局の整数cp誤差・hit数から5局等重みの有理数で判定した。未丸め値と分数は comparison.json に保存する。

両者100万ノード指定、Threads=1、Hash=128 MiB。MAEはMultiPV=1、Top3はSekireiの別runでMultiPV=3。最終評価用データは使っていない。

| 局 | 教師exact点数 | 基準MAE | 候補MAE | 基準Top3 hit / 対象 | 候補Top3 hit / 対象 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 42 | 367.405 | 280.738 | 49 / 84 | 49 / 84 |
| 2 | 72 | 1514.417 | 1629.194 | 83 / 122 | 78 / 122 |
| 3 | 66 | 1079.288 | 1120.803 | 55 / 127 | 51 / 127 |
| 4 | 40 | 1222.675 | 973.275 | 55 / 104 | 53 / 104 |
| 5 | 46 | 1238.609 | 1337.978 | 62 / 114 | 64 / 114 |

MAEは570局面・1,140 attemptを完了し、固定Teacher-E 266点を全て採点した。Top3は対象551点を全て測定した。比較検証では教師の型付き評価・bestmove・PV、Teacher-Eのcp、Top3の対象・合法手・分母・参照bestmoveを照合した。

候補：`material13-112k-lambda0.01`。重みSHA-256：`cb0406e7f24ef700da9977d0f284f0e6c80d1b00d8c20be2d69ae02e8b0e84cc`。

この結果は固定development 5局の測定であり、最終評価や一般的な棋力の改善は示さない。学習・診断条件と次の判断は[改善実験](../../../WEIGHT_IMPROVEMENT.md)を参照。

- [候補の評価値集計とグラフ](mae/validation.md)
- [比較値とidentity](comparison.json)

MAE公開exportの4ファイルは元のまま保持した。生成見出しは共通の「baseline validation」だが、fingerprintと数値はこの候補のもの。

補助資料： [教師・基準・候補の3系列図](supplement-three-series.svg) / [手数の三分割と誤差分布](supplement-methods.md) / [補助集計](supplement-phase-summary.json) / [生成manifest](supplement-manifest.json)。正式指標と採否は変更しない。
