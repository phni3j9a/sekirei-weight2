# sekirei-weight2

水匠11β（Suisho Concerto 202512）と Sekirei をそれぞれ **100万ノード指定**で解析し、未学習棋譜の評価値グラフをできるだけ重ねるための研究プロジェクト。Suisho11Plus は主目標からは外し、節目で確認する参考教師として残す。

既知のsingular-extension TT cutoff修正を含むupstream Sekirei v0.3.39へ探索実装を更新して固定し、評価モデルの重み・特徴量・NNUE構造・学習データ・学習方法を改善する。旧sekirei-weightの方針や実験履歴を前提にしない。

教師解析・学習は現在の Mac mini の CPU・32 GiB RAM・既存ストレージの範囲で進める。既存の `/mnt/storage/NAS/sekirei-weight2` を資料と成果物の非公開保管先に使い、実行環境と使用中のデータは内蔵SSDに置く。今後も追加機材は導入せず、外付け GPU・別 PC・クラウド GPU・有料計算基盤への移行を計画に含めない。公開リポジトリと GitHub Actions の軽量 CI を利用する。

## 現在の段階

Issue [#13](https://github.com/phni3j9a/sekirei-weight2/issues/13) / [PR #14](https://github.com/phni3j9a/sekirei-weight2/pull/14) で、配布packから最初の学習済みweightを生成し、固定v0.3.39で比較した。採用指標は **評価値MAE** と **水匠の最善手がSekireiのTop3に入る割合** の二つ。

初回は55,404局面・3エポックをCPUで学習し、445.376秒、最大RSS約327 MiBで完了した。ゲーム単位で分離した保留5,895局面は学習に使わず、取得済み独立1,000局の全盤面を学習・保留から機械的に除外した。学習器の専用checkoutだけに外部ラベル入力とFTZ/DAZ設定を追加し、探索エンジンは固定版を維持している。

正式比較は完了し、**初回候補は不採用**。事前に固定したMAE改善の条件を満たさなかった。

| 採用指標 | v0.3.39駒得fallback | 初回weight |
| --- | ---: | ---: |
| MAE（低いほどよい） | 1084.479 cp | 1314.481 cp |
| Top3入り率（高いほどよい） | 55.39% | 36.12% |

両指標ともdevelopment 5局の等重み平均で、正式測定の検証は有効。[比較集計と採否](docs/validation/first-weight-2026-10-02/comparison.md)、[初回weightの条件・結果・次の仮説](docs/FIRST_WEIGHT.md)と、[baseline](docs/validation/first-weight-2026-10-02/baseline/validation.md) / [候補](docs/validation/first-weight-2026-10-02/candidate/validation.md)の評価値グラフを参照。

- Sekirei / sekirei-trainはv0.3.39、shogiesaと水匠11β用やねうら王V9.20もcommit単位で固定。
- 教師packは内容重複を除いた13本（534,175,084 bytes）を非公開で保持。既存の教師を再利用し、全量JSONL展開を避けて学習入力を作る。
- 独立した人間同士・平手・合法手・重複なしの1,000局から、解析前にdevelopment 5局 / final 5局を固定した。今回の正式評価はdevelopmentだけを使用した。[取得・分割の記録](docs/validation/shogiquest-corpus-2026-09-15.md)。
- 生ログ、配布データ、学習weightはSSDと検証済みNAS保管先に保持し、公開Git/Actionsには集計・コード・設定だけを置く。
- PR #14はマージ済み。初回weightは不採用のままで、最良モデルの更新と、実験コード・知見の統合は別に判断する。

初期環境・教師監査は [PR #2](https://github.com/phni3j9a/sekirei-weight2/pull/2) / [PR #4](https://github.com/phni3j9a/sekirei-weight2/pull/4)、v0.3.39移行は[検証記録](docs/validation/sekirei-v0.3.39-2026-09-19.md)に残す。v0.3.36の旧baseline（MAE 1,087.046 cp）は[履歴](docs/validation/development-baseline-2026-09-19/validation.md)として保持し、今回の候補との比較にはv0.3.39のbaselineを使う。

## 開発用100万ノード比較

固定development 5局の各記録指し手直後、計570局面を入力にする。MAEは両エンジンをMultiPV=1・100万ノード指定で測定し、教師のexact cp集合266点を固定したまま、棋譜ごとの平均絶対誤差を5局で等重み平均する。bound・mate・no-scoreをcpへ変換せず、候補の欠測で分母を減らさない。

Top3は別runでSekireiをMultiPV=3・100万ノード指定で測定する。対象は事前分類がnormalかつ合法手4手以上の551局面。正式MAE runの水匠のbestmoveが3候補に含まれる割合を棋譜ごとに求め、5局で等重み平均する。詳細な対象集合・候補不足・同点・pilotの契約は [初回weight](docs/FIRST_WEIGHT.md#評価指標)に固定する。

MAEのpilotは17局面×両エンジン×3反復、Top3のpilotはその対象集合との共通部分12局面×3反復。完全性・再現性と同じweight identityを確認してからformalを実行する。モデルのハッシュと明示的な重み読込応答を検証し、異なるweightのpilotやMAE runを流用しない。

```sh
python3 scripts/benchmark.py plan --run-type pilot
python3 scripts/benchmark.py plan --run-type formal
python3 scripts/benchmark.py status
```

planはエンジンを起動せず入力集合を検査する。現在の `config/development-benchmark.json` にはv0.3.39 fallback用のpilot evidenceを凍結済み。候補用configは個別runtimeに保持する。[学習から正式比較までの実行例](docs/FIRST_WEIGHT.md#実行例と資源上限)を参照。

ノード判定は `one-sided-1-percent` v1の `C(N)=N+floor(N/100)` を使い、全有効node evidenceの最大値 `M <= 1,010,000` を要求する。これは片側1%の運用上の比較・異常検出ガードレールであり、停止上限の数学的保証や内部仕事量の同値性を示さない。hash-boundのforced-single・mate-in-one・terminal例外、USI lifecycle、timeout、cleanup、resumeの厳密な検証は [研究方針](docs/RESEARCH.md) / [環境文書](docs/ENVIRONMENT.md#開発baseline-runnerのruntime)に記録する。

公開exportは履歴・source game ID・ローカルパス・モデルパスを除いた4ファイルだけを出力する。生USI、attempt、詳細reportは非公開runtimeに保持する。旧v0.3.36 runを再検証する場合は、当時の固定版 `ed76730` の専用worktreeを使い、現行のtoolchain lockやfingerprintを書き換えない。

## 使い始める

USI node grammar は、`go nodes <[0-9]+>`（ASCII 十進数字列、符号なし）から対応する `bestmove` までの各 structured `info` 行を左から解釈する。`info string` は全体を free text として無視し、`pv`・`string`・`refutation`・`currline` の可変長 payload に入った後の token は解釈しない。payload 前の各行には `nodes <[0-9]+>` を高々一組だけ許容し、欠落・重複・負値・符号付き／非整数値は無効な node evidence として扱う。

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
~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/venv/bin/python \
  scripts/audit_pack.py --samples 10
~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/venv/bin/python \
  scripts/acquire_quest.py crawl --dry-run
~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/venv/bin/python \
  scripts/acquire_quest.py crawl
~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/venv/bin/python \
  scripts/acquire_quest.py verify
~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/venv/bin/python \
  scripts/acquire_quest.py snapshot
~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/venv/bin/python \
  scripts/acquire_quest.py verify-snapshot
```

`--runtime /absolute/path` で保存先を変更できる。既定は `~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1`。v0.3.36の旧β環境や旧Plus環境を上書きせず、複数worktreeでv0.3.39環境を共有する。実験でソース版・構造を変える場合も別runtimeを使う。

`smoke.py` は短い自作棋譜の一局面を両エンジンで100万ノード指定解析し、さらに shogiesa から教師を起動してラベルを一件生成する。結果と USI ログは runtime の `runs/environment-*/` に保存される。これはモデル品質や棋譜全体の一致率を測るベンチマークではない。

`import-corpus` はアーカイブ全体を既知SHA-256で照合し、β用の `1000000a/` / `1000000b/` だけを抽出する。各ファイルを内容SHAで保存するため重複は一つの実体になる。アーカイブ、重み、生の `.pack`、実行結果はGitやActionsへ入れない。

`audit_pack.py` は各固有ファイルから有限評価値を1局面ずつ選び、局面・履歴・手番を復元して固定教師で再解析する。元データに探索の境界種別はないため、再解析が upperbound / lowerbound の場合は点差を計算しない。また `go nodes` は上限であり、合法手を読み切るなどして上限前に完了した探索の実ノード数もそのまま記録する。

`acquire_quest.py` は、現在公開されている第三者の棋譜検索画面から履歴とCSAを直列・既定2秒間隔で取得し、公式棋譜ページの `opp:human` 属性、一覧上のBot印、平手初期局面、cshogiによる全手再生を照合する。取得状態と応答cacheは `~/.local/share/sekirei-weight2/shogiquest-human-v1` に1局ごとに保存され、同じコマンドで再開できる。Webサービスの非公開通信を解析・利用しない。公開画面の仕様は変わり得るため、異常な応答では停止し、取得済みcacheを再利用する。

`snapshot` は1,000局が揃った後、対局者の重複、手数、対局時レーティング差を制約し、固定hash順位だけで development 5局 / final 5局を選ぶ。CSA内の対局者名とレーティングは置換し、出典を追跡するため対局IDはmanifestに残す。最終評価用5局はモデルや閾値の選択には使わない。

## ストレージ

資料原本は `/mnt/storage/NAS/sekirei-weight2/materials/pixiv_fanbox_yaneurao` に置く。main checkoutの従来パス `docs/pixiv_fanbox_yaneurao` はGit対象外の互換リンクとして保持する。教師 `.pack` の保管コピーは `datasets/suisho11beta-1m`、既存runtimeの重み・実験記録・build manifestの保管コピーは `archives/2026-10-02/runtimes` にある。

Git/worktree・build・venv・使用中のデータ/重み・実行中の出力は内蔵SSDを使う。完了した成果物はファイル集合・サイズ・SHA-256を照合して手動で保管する。既存runは絶対パスを含むため、NASコピーを直接実行用runtimeとせず、SSD側の参照パスも維持する。[配置・コピーと復元の手順](docs/ENVIRONMENT.md#内蔵ssdとnasの運用)、[移設の検証記録](docs/validation/storage-2026-10-02.md)を参照。

## 現在の改善実験

Issue [#15](https://github.com/phni3j9a/sekirei-weight2/issues/15) / [PR #16](https://github.com/phni3j9a/sekirei-weight2/pull/16)で、約8時間を目安にした手動改善実験を終えた。固定100万ノードの正式比較は4候補で完了した。**採用基準のMAE改善とTop3維持を同時に満たす候補はなく、最良モデルは駒得fallbackのまま**である。

| モデル | 正式MAE | 正式Top3入り率 | 採用 |
| --- | ---: | ---: | --- |
| 駒得fallback | 1084.479 cp | 55.39% | 現行を維持 |
| constant LR・epoch12 | 1201.084 cp | 29.90% | 不採用 |
| 駒得初期化・epoch1 | 1011.750 cp | 48.19% | 保留 |
| 同epoch1・最近傍丸め | 985.098 cp | 53.54% | 保留 |
| 112,681局面・13駒価値学習 | 1068.398 cp | 53.91% | 保留 |

両指標はdevelopment 5局の等重み平均。最近傍丸めは同じcheckpointの切り捨て版から両指標を改善したが、現行fallbackのTop3には達していない。13駒価値学習もMAEは改善したが、Top3維持条件を満たさなかった。[正式比較・手順・各グラフ](docs/WEIGHT_IMPROVEMENT.md)に採否と仮説を記録する。

参考教師Suisho11Plusは、主教師βだけで確定した最良モデルについて17局面×3反復を完了した。[参考確認](docs/validation/weight-improvement-2026-10-03/plus-reference/reference.md)は型付き結果の安定性を示すが、正式な採用判断には使っていない。112,681局面のNNUE追加学習と静的診断も完了し、選択済みepoch3の最近傍丸め版は固定holdout MAE **644.599 cp**となった。この追加候補の正式比較は未実施のため採用せず、次回検証用に保持する。

探索実装、教師、正式比較条件、採用基準を維持し、final 5局はモデル選択や採点に使用していない。実験成果物20,765ファイル（2,417,083,627 bytes）は、ファイル集合・サイズ・SHA-256等の一致を確認してNASへ保存した。SSD原本と参照パスも保持している。[保存の検証集計](docs/validation/weight-improvement-2026-10-03/archive.json)を参照。PRは未マージ。

## 文書

- [研究方針・比較条件](docs/RESEARCH.md)
- [固定環境・再現手順・制約](docs/ENVIRONMENT.md)
- [Sekirei v0.3.39移行検証](docs/validation/sekirei-v0.3.39-2026-09-19.md)
- [将棋クエスト独立棋譜の取得・分割](docs/validation/shogiquest-corpus-2026-09-15.md)
- [初回weightの学習条件・正式比較・採否](docs/FIRST_WEIGHT.md)
- [旧v0.3.36 baselineの検証値・グラフ](docs/validation/development-baseline-2026-09-19/validation.md)
- [開発運用](AGENTS.md)
- [外部ソフト・資料の出典](docs/PROVENANCE.md)

コード・設定・検証手順を公開 Git に置き、配布記事・アーカイブ・重み・大規模データ・実行結果はローカルに置く。Issue/PR には変更理由と検証結果を残す。モデル公開・定期実行の有効化は今回の準備に含まれない。
