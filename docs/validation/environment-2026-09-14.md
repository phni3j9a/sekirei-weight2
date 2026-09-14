# 初期環境の検証 — 2026-09-14

Issue #1。これは環境疎通の検証であり、モデル改善・棋力・グラフ一致の測定ではない。

## 最終構成

- Sekirei / 学習器: v0.3.36、`aeb6ea30d58f93cad84ffe98bc13441feb807fa8`
- shogiesa: 0.9.2、`dd0317437dc43b4d9b299e11542d82843f538b9d`
- やねうら王: 9.70git、`76d58ef2e4cf64116784f41fd4816425ab6817ee`
- Suisho11Plus nn.bin SHA-256: `a78b7f889843037d344f482623b3febd124ead5c1f34f134d9f1c2c78cd0f829`
- Ubuntu 24.04 / i5-8500B / 32 GiB RAM。計算は CPU 内で実施。
- ビルド設定は config/toolchain.lock.json。コンパイラとバイナリのハッシュはローカル build-manifest.json および run の summary.json に保存。

## 実行結果

入力は自作 fixture の `startpos moves 7g7f 3c3d 2g2f`（後手番）。Threads=1、Hash=128 MiB、定跡なし、MultiPV=1、100万ノード指定。

| 項目 | 結果 |
| --- | --- |
| Sekirei の USI・探索 | 正常終了、1,000,002 nodes、depth 13、bestmove `2b8h+` |
| Sekirei の評価 | 駒得評価、0 cp、exact。学習済み重みなし |
| Sekirei の探索実時間 | 約0.563秒 |
| 教師の USI・探索 | 正常終了、1,000,131 nodes、depth 21、bestmove `4c4d` |
| 教師の最終出力 | 手番視点 -63 cp **upperbound**。先手視点では +63 cp の lowerbound |
| 教師の確定評価値 | `null`。境界値を確定評価値に変換していない |
| 教師の探索実時間 | 約1.258秒 |
| shogiesa extract | 一局面を抽出、skip=0 |
| shogiesa label | 一局面をラベル、1,000,131 nodes、upperbound。起動失敗=0 |
| shogiesa validate --strict | 成功。JSON/SFEN/tag/重複の異常なし |
| 学習器 | ビルドと `--help` 成功。本格学習は未実施 |
| Python テスト | 6件成功。先後反転、境界値、PV不整合、詰み、MultiPVの誤受理を確認 |

時間はこの一局面の観測値であり、全棋譜や学習済みモデルの速度予測ではない。

最終 run: `/home/server/.local/share/sekirei-weight2/runs/environment-2jmr28ip/`

環境ディレクトリは約644 MiB。準備時に作った9.80gitの比較用ソース・ビルドは `sources/yaneuraou-9.80git-unselected` に保持されているが、bin と最終 manifest は9.70gitを指している。手元の配布資料は元の場所に保持した。

## 確認した注意点

### 教師の識別ハッシュ警告

手元アーカイブと展開重みの SHA-256 は固定値に一致した。一方、やねうら王はモデル内の古い識別ハッシュについて警告を出す。

- モデル内の全体識別ハッシュ: `1008746012`
- エンジン側の期待値: `1008745266`
- 表示名もファイル内 `HalfKA` と現在の `HalfKA2`、旧ネットワーク表記と現在の表記で異なる。

固定 upstream の `source/eval/nnue/evaluate_nnue.cpp` は、古い評価関数ファイルでは識別ハッシュが一致しない場合があるとして、警告後にパラメータを読み込む実装になっている。パラメータ読込とファイル末尾確認は成功し、USI ready と探索も完了した。同梱ヘッダの HalfKA2・1024/7/64・9 stacks と、指定して生成した構造を照合した。同梱の名前が `sfnnwop-1536.h` でも実際の幅は1024。

警告は `summary.json` の startup_warnings と USI ログに保存した。警告がないとは報告しない。配布済み macOS/Windows バイナリとの数値同一性までは検証していない。正式な教師ベースラインを確定する際の照合事項として残す。

### 最終出力の境界値

100万ノード到達時の最後の info が upperbound だった。古い exact info を選んで最新結果として扱うことも、境界を外して -63 cp の確定値にすることもしていない。今回の smoke は型を保持した正常なプロトコル疎通を確認する。

正式ベンチマークは、境界値・詰み・欠測を含めて候補によって採点分母が変わらない規則を定める必要がある。

### shogiesa の履歴

今回の直接 USI smoke は指し手履歴を渡すが、shogiesa label は単一 SFEN を渡す。両者は一般の棋譜で同じ意味とは限らない。正式なグラフ比較に shogiesa label をそのまま流用しない。

### 最初に試した Sekirei main

当初取得した main `08107a2`（package 0.3.5）は、USI `go nodes` と探索のノード上限を実装していなかった。smoke が120秒でタイムアウトし、不合格として検出した。公開タグ v0.3.36 の該当実装を確認して固定し直した。

失敗ログは `runs/environment-n3a0lqm_/`。9.80gitを試した中間ログは `runs/environment-445bqica/`。どちらも最終成功の根拠には使っていない。
