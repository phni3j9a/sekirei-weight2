# 初回weightの正式比較（2026-10-02）

初回候補は**不採用**。事前に定めた「両指標が有効で、MAEが改善し、Top3入り率が下がらない」を満たさなかった。比較対象は同じSekirei v0.3.39の駒得fallbackであり、探索エンジンは変更していない。

| 採用指標 | baseline | 初回候補 | 候補 − baseline |
| --- | ---: | ---: | ---: |
| MAE（低いほどよい） | 1084.479 cp | 1314.481 cp | +230.002 cp |
| Top3入り率（高いほどよい） | 55.3888% | 36.1241% | -19.2646ポイント |

両者とも100万ノード指定。両指標は棋譜ごとの値を5局で等重み平均する。MAEは固定教師exact-cp 266点、Top3は事前固定551局面を対象とし、水匠MultiPV=1のbestmoveがSekireiの別測定MultiPV=3の候補に含まれるかを測る。Top1一致率は採用しない。

MAE正式runは各1,140/1,140 attempt、技術失敗0、教師E coverage 266/266。Top3正式runは各551/551、invalid/missing 0。各モデルのMAE pilotは102/102、34/34 stable、Top3 pilotは36/36、12/12 stableで事前検証した。両MAE runの教師側570局面のスコア・境界・詰み・bestmove・PV・最大nodeは一致した。

| development | baseline MAE (cp) | 候補 MAE (cp) | baseline Top3 hit / 対象 | 候補 Top3 hit / 対象 |
| --- | ---: | ---: | ---: | ---: |
| 01 | 367.405 | 888.571 | 49 / 84 | 39 / 84 |
| 02 | 1514.417 | 1765.181 | 83 / 122 | 49 / 122 |
| 03 | 1079.288 | 1436.076 | 55 / 127 | 31 / 127 |
| 04 | 1222.675 | 901.750 | 55 / 104 | 35 / 104 |
| 05 | 1238.609 | 1580.826 | 62 / 114 | 41 / 114 |

この5局はdevelopmentであり、一般的な棋力や未使用finalへの改善を示す結果ではない。学習済みweightを生成・読み込めたことと、モデル性能の改善を区別する。重みは非公開runtime/NASに保持する。

- [学習条件・資源実測・採否・次の仮説](../../FIRST_WEIGHT.md)
- [baselineの評価値集計とグラフ](baseline/validation.md)
- [候補の評価値集計とグラフ](candidate/validation.md)
- [比較値・run fingerprint・weight hash](comparison.json)
- Top3集計：[baseline](baseline-top3.json) / [候補](candidate-top3.json)
