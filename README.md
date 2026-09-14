# sekirei-weight2

Suisho11Plus と Sekirei をそれぞれ **100万ノード指定**で解析し、未学習棋譜の評価値グラフをできるだけ重ねるための研究プロジェクト。

探索実装を当面固定し、評価モデルの重み・特徴量・NNUE 構造・学習データ・学習方法を改善する。旧 sekirei-weight の方針や実験履歴を前提にしない。

教師解析・学習は現在の Mac mini の CPU・32 GiB RAM・既存ストレージの範囲で進める。今後も追加機材は導入せず、外付け GPU・別 PC・クラウド GPU・有料計算基盤への移行を計画に含めない。公開リポジトリと GitHub Actions の軽量 CI を利用する。

## 現在の段階

Issue [#1](https://github.com/phni3j9a/sekirei-weight2/issues/1) で初期方針と実行環境を整備し、[PR #2](https://github.com/phni3j9a/sekirei-weight2/pull/2) でレビューする。

- Sekirei と付属学習器、shogiesa、Suisho11Plus 用やねうら王のソースを commit 単位で固定。
- Python 標準ライブラリによるビルド・重み照合・USI 疎通確認スクリプト。
- 実機で両エンジンの100万ノード指定探索、shogiesa の一局面ラベル生成を確認済み。[検証結果と制約](docs/validation/environment-2026-09-14.md)。
- 正式ベンチマークの棋譜選定、本格学習、モデル採用判定、定期自動実行は次の段階。
- Sekirei の今回の初期疎通は **駒得評価へのフォールバック**。学習済みモデルはまだない。

## 使い始める

Linux x86_64 / AVX2・BMI2 対応 CPU、Rust/Cargo、Python 3.11+、GCC C++、make、7z が必要。準備スクリプトは sudo や OS パッケージ変更を行わない。Ubuntu 24.04 の現ホストには必要なツールがある。

専用 worktree 内から実行する。

```sh
python3 scripts/prepare.py doctor
python3 scripts/prepare.py build --jobs 2
python3 scripts/prepare.py import-teacher --archive '/absolute/path/to/Suisho11Plus.7z'
python3 scripts/smoke.py
```

`--runtime /absolute/path` で保存先を変更できる。既定は `~/.local/share/sekirei-weight2`。複数 worktree で固定環境を共有し、実験でソース版・構造を変える場合は別 runtime を使う。

`smoke.py` は短い自作棋譜の一局面を両エンジンで100万ノード指定解析し、さらに shogiesa から教師を起動してラベルを一件生成する。結果と USI ログは runtime の `runs/environment-*/` に保存される。これはモデル品質や棋譜全体の一致率を測るベンチマークではない。

教師の旧形式の識別ハッシュ警告は記録に残る。今回の最終教師出力は upperbound で、確定評価値には変換していない。正式な教師ベースラインの照合と境界値の採点規則は次の確認事項。

## 次の到達点

まず 5〜10 棋譜程度で現状の重ね合わせグラフを作り、時間・再現性・誤差を把握する。そのために、棋譜の出典と分割、履歴を保持した局面入力、詰み・境界値・解析失敗の扱いを確定する。その後、新規データによる小規模学習を一周通す。

主指標は棋譜ごとの MAE を棋譜間で平均する案。教師のみ詰み／候補のみ詰みなどで採点対象が候補依存に変わらない仕様を、正式な採用判定の前に確定する。

## 文書

- [研究方針・比較条件](docs/RESEARCH.md)
- [固定環境・再現手順・制約](docs/ENVIRONMENT.md)
- [開発運用](AGENTS.md)
- [外部ソフト・資料の出典](docs/PROVENANCE.md)

コード・設定・検証手順を公開 Git に置き、配布記事・アーカイブ・重み・大規模データ・実行結果はローカルに置く。Issue/PR には変更理由と検証結果を残す。PR マージ・モデル公開・定期実行の有効化は今回の準備に含まれない。
