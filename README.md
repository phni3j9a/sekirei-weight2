# sekirei-weight2

水匠11β（Suisho Concerto 202512）と Sekirei をそれぞれ **100万ノード指定**で解析し、未学習棋譜の評価値グラフをできるだけ重ねるための研究プロジェクト。Suisho11Plus は主目標からは外し、節目で確認する参考教師として残す。

探索実装を当面固定し、評価モデルの重み・特徴量・NNUE 構造・学習データ・学習方法を改善する。旧 sekirei-weight の方針や実験履歴を前提にしない。

教師解析・学習は現在の Mac mini の CPU・32 GiB RAM・既存ストレージの範囲で進める。今後も追加機材は導入せず、外付け GPU・別 PC・クラウド GPU・有料計算基盤への移行を計画に含めない。公開リポジトリと GitHub Actions の軽量 CI を利用する。

## 現在の段階

Issue [#1](https://github.com/phni3j9a/sekirei-weight2/issues/1) / [PR #2](https://github.com/phni3j9a/sekirei-weight2/pull/2) で初期環境を整備した。Issue [#3](https://github.com/phni3j9a/sekirei-weight2/issues/3) で主教師を水匠11βへ切り替え、生成済み教師データを監査している。

- Sekirei と付属学習器、shogiesa、水匠11β用やねうら王V9.20のソースを commit 単位で固定。
- ビルド・重み照合・USI疎通はPython標準ライブラリで実行。`.pack` の局面復号だけは専用venvに固定したcshogi / NumPyを使う。
- β・100万ノードとして配布された `.pack` 15本を取り込み、同一内容の2本を除いた13本（534,175,084 bytes）をローカルで管理。ゲーム境界を保つストリーム復号と標本再解析が可能。
- 実機で両エンジンの100万ノード指定探索と shogiesa の一局面ラベル生成を確認済み。[β環境と教師監査](docs/validation/suisho11beta-2026-09-15.md)。旧Plus環境の結果は[初期検証](docs/validation/environment-2026-09-14.md)に残す。
- 10標本の固定V9.20再解析では、確定値6件のMAE 3.167 cp（最大11 cp）、境界値4件、保存指し手一致6件。互換性の小規模確認であり、元の生成環境との完全同一性の証明ではない。
- 正式ベンチマークの棋譜選定、`.pack` から学習器への入力経路、本格学習、モデル採用判定、定期自動実行は次の段階。
- Sekirei の今回の初期疎通は **駒得評価へのフォールバック**。学習済みモデルはまだない。

## 使い始める

Linux x86_64 / AVX2・BMI2 対応 CPU、Rust/Cargo、Python 3.11+、GCC C++、make、7z が必要。準備スクリプトは sudo や OS パッケージ変更を行わない。Ubuntu 24.04 の現ホストには必要なツールがある。

専用 worktree 内から実行する。

```sh
python3 scripts/prepare.py doctor
python3 scripts/prepare.py audit-deps
python3 scripts/prepare.py build --jobs 2
python3 scripts/prepare.py import-teacher \
  --archive '/absolute/path/to/suisho11beta-suisho_concerto202512.7z'
python3 scripts/prepare.py import-corpus --archive '/absolute/path/to/kif20260630-pack-1M.7z'
python3 scripts/prepare.py import-corpus --archive '/absolute/path/to/kif20260731-pack-1M.7z'
python3 scripts/smoke.py
~/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python \
  scripts/audit_pack.py --samples 10
```

`--runtime /absolute/path` で保存先を変更できる。既定は `~/.local/share/sekirei-weight2/suisho11beta-v1`。旧Plus環境を上書きせず、複数 worktree で固定したβ環境を共有する。実験でソース版・構造を変える場合は別 runtime を使う。

`smoke.py` は短い自作棋譜の一局面を両エンジンで100万ノード指定解析し、さらに shogiesa から教師を起動してラベルを一件生成する。結果と USI ログは runtime の `runs/environment-*/` に保存される。これはモデル品質や棋譜全体の一致率を測るベンチマークではない。

`import-corpus` はアーカイブ全体を既知SHA-256で照合し、β用の `1000000a/` / `1000000b/` だけを抽出する。各ファイルを内容SHAで保存するため重複は一つの実体になる。アーカイブ、重み、生の `.pack`、実行結果はGitやActionsへ入れない。

`audit_pack.py` は各固有ファイルから有限評価値を1局面ずつ選び、局面・履歴・手番を復元して固定教師で再解析する。元データに探索の境界種別はないため、再解析が upperbound / lowerbound の場合は点差を計算しない。また `go nodes` は上限であり、合法手を読み切るなどして上限前に完了した探索の実ノード数もそのまま記録する。

## 次の到達点

まず、生成済み `.pack` をゲーム単位で train / development に分割し、全量JSONL展開を避けた学習入力経路を作る。同時に 5〜10 棋譜程度の独立した比較用棋譜で現状の重ね合わせグラフを作り、時間・再現性・誤差を把握する。その後、CPUで小規模学習を一周通す。

主指標は棋譜ごとの MAE を棋譜間で平均する案。教師のみ詰み／候補のみ詰みなどで採点対象が候補依存に変わらない仕様を、正式な採用判定の前に確定する。

## 文書

- [研究方針・比較条件](docs/RESEARCH.md)
- [固定環境・再現手順・制約](docs/ENVIRONMENT.md)
- [開発運用](AGENTS.md)
- [外部ソフト・資料の出典](docs/PROVENANCE.md)

コード・設定・検証手順を公開 Git に置き、配布記事・アーカイブ・重み・大規模データ・実行結果はローカルに置く。Issue/PR には変更理由と検証結果を残す。モデル公開・定期実行の有効化は今回の準備に含まれない。
