# Raw master / nearest03診断の結果

[公開コードと再現条件](../../../../preparations/raw-master-quantization-diagnostic-v1/README.md)は、第1候補MSEの保存raw masterと同じ候補のnearest03を、train256＋固定holdout256のlabel/prediction非依存indexで比較する。v3の実parentはexit0で完了した。この診断はHuber候補や現bestの誤差、100万node探索の正式MAE/Top3、モデル採用を測った結果ではない。

| 差（手番視点cp） | train平均絶対差 | train最大絶対差 | holdout平均絶対差 | holdout最大絶対差 |
| --- | ---: | ---: | ---: | ---: |
| raw master − native dequant | 1.041513689 | 10.280944824 | 1.198571399 | 9.785339355 |
| native dequant − integer core | 0.493230104 | 0.996322632 | 0.499285597 | 0.998382568 |
| raw master − integer core | 1.293795891 | 9.996826172 | 1.392100502 | 9.297668457 |

[sampled-summary.json](sampled-summary.json)は実producerの2,569 Bをそのまま保存し、signed meanを含む全数値とFP観測を保持する。[公開集計](summary.json)はRootの実native exit0、六childのexit0とwait/reap・ECHILD・二回空scan・残存なし、元input/source/modelの前後一致、requestの512行との一致、全state bits不変、nearest03のfullbyte再export、11 locksの全解放を記録する。Rootは512出力を同じpure aggregatorで再集計して全summary一致を確認したが、Rustのmodel計算やoriginal initialized readerを独立に再実行した結果ではない。raw master側の最大差はnative dequant/coreの1 cp未満bridgeへ読み替えない。

v1のRust module path E0583と、v2のstartup bannerに対するstrict aggregator拒否を失敗として保持する。v2の5段階と512行の数値childはexit0だったが、aggregateがexit1のため全体完了にはしていない。v3は既存FTZ/DAZ startup bannerを先頭に一度だけ要求するparser部分のみを修正し、Rust計算本体はv2と同じである。v1/v2を成功へretagせず、生の局面・ラベル・診断log・モデル・私有pathを公開しない。

512 sampleの範囲での実測であり、全局面上限・汎化・モデル改善を証明しない。モデル更新・採用はなく、development/final未使用、教師queryなし、optimizer updateは0。Linux x86_64の同じCPU/platform、固定source/model/runtime/compiler/deps・OS/loader等を前提とする。診断記録のNAS保存はpendingで、この結果から保存完了を推定しない。
