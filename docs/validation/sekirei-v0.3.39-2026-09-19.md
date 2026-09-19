# Sekirei v0.3.39移行検証（2026-09-19）

## 結論

Sekirei / sekirei-trainをupstream v0.3.39、commit `f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9`へ固定し、Mac mini上でlocked release buildと100万ノードsmokeを完了した。今後は、v0.3.37で入ったsingular-extension verification searchのTT cutoff修正を含むこの版を探索基準にする。

今回、upstream v0.3.38の公開NNUE checkpointは読み込んでいない。Sekireiは学習済み重みなしのmaterial fallbackであり、探索版更新と評価器更新を分離している。このsmokeは環境疎通の確認で、評価値グラフの改善や棋力を示すものではない。

## 固定と分離

| 対象 | 固定値 |
| --- | --- |
| Sekirei / sekirei-train | v0.3.39 / `f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9` |
| upstream tag | `refs/tags/v0.3.39` |
| runtime profile | `suisho11beta-sekirei-v0.3.39-v1` |
| Rust flags | `-C target-cpu=x86-64-v3` |
| build | `cargo build --locked --release`、jobs=2 |
| toolchain lock SHA-256 | `4343a043250cf300e4b029f3dc9f953f1ec047bd6bcb4024efeda0a3e9b5bca5` |

旧v0.3.36 runtime `suisho11beta-v1`、raw run、公開baselineは変更していない。v0.3.39は新しいruntimeにsource・build・bin・runを分離した。

上流の[v0.3.39 release](https://github.com/kent-tokyo/sekirei/releases/tag/v0.3.39)と[変更履歴](https://github.com/kent-tokyo/sekirei/blob/v0.3.39/CHANGELOG.md)を確認した。今回の移行理由であるTT修正は[該当commit](https://github.com/kent-tokyo/sekirei/commit/e819dca65db4726e5a6247eb8eca0ade54155aed)にあり、v0.3.37以降に含まれる。

## 実機ビルド

| 対象 | SHA-256 |
| --- | --- |
| build manifest | `13d1116d61d9e60358a0738e8d3127e9ca122925965dc3107de6085584623420` |
| `sekirei` | `1fcb3e9e67d3a926eed6cd172824ef9e4336af80f3ac62e02fa58abd0e865582` |
| `sekirei-train` | `31602b951f30646fe7f55575484d184181e577ed4cf1d8ae181b99477cc104a1` |
| `shogiesa` | `134f76a8f1a2844208b4c9f42314889024426d830951dbb2ad0653a6c00f3653` |
| 水匠11β用やねうら王 | `5de16bb5070b4bcd2ee4150ab08d7dcf1c65dcee1ebc072ff4463539a4b1595c` |

使用したコンパイラはrustc/cargo 1.96.0、g++ 13.3.0。`sekirei-train --help`も正常に起動し、v0.3.39で追加・維持されたpositions-mode、teacher cache、mate label診断を含むCLIを確認した。`sekirei-train`自体には`--version` optionがないため、版の根拠は固定source commitとbuild manifestである。

## 100万ノードsmoke

入力は自作fixtureの `startpos moves 7g7f 3c3d 2g2f`（後手番）。Sekireiは `Threads=1, Hash=128, SearchMode=Speculative, SpecTopN=0, MultiPV=1, Ponder=false, UseBook=false`、教師は従来の固定水匠11β設定で実行した。

| 項目 | 結果 |
| --- | --- |
| Sekirei | 1,000,000 nodes、depth 13、exact 0 cp、bestmove `2b8h+`、0.670秒 |
| 水匠11β | 1,000,500 nodes、depth 20、手番視点 -64 cp lowerbound、bestmove `2b8h+`、2.351秒 |
| shogiesa | 1件extract・label、strict validate成功、1,000,500 nodes lowerbound |
| startup warning | 両エンジンともなし |
| summary SHA-256 | `d1af1edadebc0144c5be2b5a3f7c89447308fb4d1183d7fdc5ff9d99c797e785` |

Sekireiは要求した全USI optionを広告し、`usi`、`isready`、`usinewgame`、`position`、`go nodes 1000000`、対応するscore/PV/bestmove、`quit`まで正常終了した。教師の最終値はlowerboundなので確定cpへ変換していない。

## baselineの扱いと次の停止条件

公開済み `development-baseline-20260916-v2` はv0.3.36＋material fallbackの履歴として保持する。エンジンcommitとtoolchain lockが変わったため、そのpilot fingerprintとnode evidenceをv0.3.39へ流用しない。

`config/development-benchmark.json` の `formal.pilot_evidence` は `null` に戻した。次はv0.3.39 runtimeで17局面×両エンジン×3 repetitionの102-attempt pilotを取得し、technical failure、再現性、全node evidenceと上限をレビューする。入力集合を確認するformal planは作成できるが、その証拠を固定するまでformal実行はfail-closedで停止する。

実際にpilot planが17局面・3 repetition・両エンジンの102 attemptsになること、formal planが570局面・1 repetition・両エンジンの1,140 attemptsになることを確認した。未凍結状態でformalを起動すると、run directoryを作る前に `formal.pilot_evidence is required` で終了することも確認した。Python unit testは92件すべて成功し、JSON構文、Python構文、`git diff --check`も通過した。
