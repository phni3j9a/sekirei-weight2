# sekirei-weight2

水匠11β（Suisho Concerto 202512）と Sekirei をそれぞれ **100万ノード指定**で解析し、未学習棋譜の評価値グラフをできるだけ重ねるための研究プロジェクト。Suisho11Plus は主目標からは外し、節目で確認する参考教師として残す。

既知のsingular-extension TT cutoff修正を含むupstream Sekirei v0.3.39へ探索実装を更新して固定し、評価モデルの重み・特徴量・NNUE構造・学習データ・学習方法を改善する。旧sekirei-weightの方針や実験履歴を前提にしない。

教師解析・学習は現在の Mac mini の CPU・32 GiB RAM・既存ストレージの範囲で進める。今後も追加機材は導入せず、外付け GPU・別 PC・クラウド GPU・有料計算基盤への移行を計画に含めない。公開リポジトリと GitHub Actions の軽量 CI を利用する。

## 現在の段階

Issue [#1](https://github.com/phni3j9a/sekirei-weight2/issues/1) / [PR #2](https://github.com/phni3j9a/sekirei-weight2/pull/2) で初期環境を整備し、Issue [#3](https://github.com/phni3j9a/sekirei-weight2/issues/3) / [PR #4](https://github.com/phni3j9a/sekirei-weight2/pull/4) で主教師を水匠11βへ切り替えて生成済み教師データを監査した。Issue [#5](https://github.com/phni3j9a/sekirei-weight2/issues/5) では独立評価棋譜を固定し、Issue [#7](https://github.com/phni3j9a/sekirei-weight2/issues/7) で初期baselineを確定した。Issue [#9](https://github.com/phni3j9a/sekirei-weight2/issues/9) ではSekireiをv0.3.39へ更新する。

- Sekirei / sekirei-trainはv0.3.39、shogiesaと水匠11β用やねうら王V9.20もcommit単位で固定。
- ビルド・重み照合・USI疎通はPython標準ライブラリで実行。`.pack` の局面復号と外部CSAの合法手確認だけは専用venvに固定したcshogi / NumPyを使う。
- β・100万ノードとして配布された `.pack` 15本を取り込み、同一内容の2本を除いた13本（534,175,084 bytes）をローカルで管理。ゲーム境界を保つストリーム復号と標本再解析が可能。
- 実機でv0.3.39 Sekireiと水匠11βの100万ノード指定探索、shogiesaの一局面ラベル生成を確認済み。[v0.3.39移行検証](docs/validation/sekirei-v0.3.39-2026-09-19.md)。教師データ監査は[β環境と教師監査](docs/validation/suisho11beta-2026-09-15.md)、旧Plus環境は[初期検証](docs/validation/environment-2026-09-14.md)に残す。
- 10標本の固定V9.20再解析では、確定値6件のMAE 3.167 cp（最大11 cp）、境界値4件、保存指し手一致6件。互換性の小規模確認であり、元の生成環境との完全同一性の証明ではない。
- 正式ベンチマーク用に、将棋クエストの公開棋譜から人間同士・平手・合法手・重複なしの1,000局をローカルへ固定し、その中から解析前に development 5局 / final 5局を選定済み。[取得・分割の検証記録](docs/validation/shogiquest-corpus-2026-09-15.md)。
- `.pack` から学習器への入力経路、本格学習、モデル採用判定、定期自動実行は次の段階。
- Sekirei の今回の初期疎通は **駒得評価へのフォールバック**。学習済みモデルはまだない。
- Issue #7 のSekirei v0.3.36系列では、`development-pilot-20260916-v5` をreviewed/formal launch evidenceとして凍結し、同じexecution identityの正式測定 `development-baseline-20260916-v2` を完了した。正式測定は1,140/1,140 attempt、technical failure 0、teacher-E 266/266 coverageで、棋譜ごとのMAEを等重み平均したheadlineは **1,087.046 cp**。公開用の[検証値・グラフ・hash manifest](docs/validation/development-baseline-2026-09-19/validation.md)を履歴として追跡する。本正式測定とreport/export経路はfinal 5局へアクセスしていない。
- v0.3.39は新しいruntimeへ分離し、上流v0.3.38の公開NNUE重みを入れずに駒得fallbackでまず探索差分だけを確認する。v0.3.36のpilot evidenceはv0.3.39へ流用せず、`formal.pilot_evidence` は未凍結に戻してある。
- 旧 `development-baseline-20260916` は1,140/1,140 attemptを収集したが、現行の正式根拠にはできない。Teacher development-04 p077 の all-evidence max 1,001,086 は旧1,001,024を超え、Sekirei development-05 p122 は旧分類にない mate 1 / nodes 0 だった。この旧1,001,024をsource-derived guaranteeとして扱う主張は撤回する。`status=complete` は formal valid を意味せず、診断上の exact coverage 265/265 も headline やモデル採用の根拠ではない。
- 現行契約は `go nodes 1000000` を両エンジンへ送り、`one-sided-1-percent` v1 の整数式 `C(N)=N+floor(N/100)` により all-evidence max `M <= C(N)` を判定する。これは片側1%の運用上の比較・異常検出ガードレールであり、YaneuraOuの停止上限、内部仕事量の同値性、最低ノード目標ではない。pilot/formal の `max_reported_nodes` はともに 1,010,000 を事前登録する。旧v2/v3/v4はprior diagnostic、旧v5はv0.3.36 formalだけのlaunch evidenceである。

## 開発用 100万ノード baseline

以下の完了済み結果はSekirei v0.3.36の履歴である。v0.3.39は別のexecution identityなので、同じ入力・契約でpilotを再取得してから新しいformal baselineを作る。

Issue #7 の実装は、`benchmarks/shogiquest-v1/development` の5局だけを固定入力にする。初期局面は含めず、各棋譜の記録指し手の直後を一つの occurrence として扱うため、合計570局面になる。同じ指し手列が別の棋譜に現れても occurrence は共有しない。`final` を選べるパス引数は用意していない。

標準ライブラリのboard trackerは固定hashのCSAについて、手番・所有者・成り・駒取り・打ち駒を含む厳密な移動遷移／擬似合法性を検査してUSI履歴へ変換する。盤面分類は `config/development-position-classifications.json` に入力・履歴・position hashを束縛して保持する。これは pinned cshogi 1.0.4 で全570 occurrence・全46,668合法root moveを再生して生成・検証した監査メタデータで、強制1手5件、mate-in-one available の development-05 p122（合法手217、mating move `2g4g`）、終端詰み1件を含む。CIの標準ライブラリ検査だけから完全合法性を主張しない。計画とhashを確認するコマンドは次のとおり。実行結果は外部runtimeへ保存され、CIはネットワークや教師重みを必要としない。

```sh
python3 scripts/benchmark.py plan --run-type pilot
python3 scripts/benchmark.py plan --run-type formal
```

planは実行せず入力集合を検査するため、未凍結でも作成できる。実際の `formal` 起動は、v0.3.39 pilotをレビューして `formal.pilot_evidence` を凍結するまでfail-closedで拒否する。

pilotは各棋譜の `{1, ceil(L/2), L}` のcanonical 15局面に regression の development-04:77 と development-05:122 を加えた17局面を、両エンジン・3回ずつ（102 attempt / 34 engine-position triple）計画する。formalは570局面を各1回ずつ（1,140 attempt）計画する。両計画の `max_reported_nodes` は、`one-sided-1-percent` v1 の `C(1,000,000)=1,010,000` を使う。plan時に固定audit runtimeの cshogi 1.0.4 が利用できれば、全570 occurrence・46,668合法root moveと分類を照合して結果を記録する（`--cshogi` は互換用に受理する）。これはCIの必須依存ではない。

v0.3.36の実機では `development-pilot-20260916-v5`（17局面・102 attempt）を完了し、その全attemptから `observed_max_reported_nodes=1,001,086` を再計算した。technical failure 0、34/34 engine-position triple stable、complete evidence validである。`go nodes 1000000` は両エンジンへ送り、`one-sided-1-percent` v1 の整数式 `C(N)=N+floor(N/100)` により all-evidence max `M <= C(N)` を受理した。`C(1,000,000)=1,010,000` は inclusive だが、片側1%の運用上の比較・異常検出ガードレールであり、YaneuraOuの数学的停止上限、内部仕事量の同値性、最低ノード目標ではない。`reported_nodes_at_score`、`last_reported_nodes`、全有効値の `max_reported_nodes_evidence` は別々に保持し、current-go の欠落・負値・malformed・重複と earlier overrun を後続値で隠さない。この旧v5 evidenceはv0.3.39のformal gateには使わない。

正式測定 `development-baseline-20260916-v2`（fingerprint `d1708915e2de8bfd22d1d3c29eb10cc10f45926da4f0f2357dd87a0e407ab084`）は570局面×2エンジンの1,140/1,140 attemptを完了した。全attemptがdeadline内完了、cleanup/supervisor正常、engine return code 0で、technical failure・missing/extra/duplicateは0。Teacherは `exact_cp=266 / bound_cp=273 / mate=31`、Sekireiは `exact_cp=561 / mate=8 / no_score=1` だった。Teacher-EのSekirei exact coverageは266/266、5局の等重みheadline MAEは `1,087.0461722818245 cp`（全E点のmicro MAEは `1,136.8684210526317 cp`）。Mの最大値はTeacher `1,001,086`、Sekirei `1,000,004` で、両方とも `>C=0`。公開exportは [`docs/validation/development-baseline-2026-09-19`](docs/validation/development-baseline-2026-09-19/validation.md) に固定した。これは学習前の駒得fallback baselineであり、モデル品質が十分という意味ではない。final 5局は本測定・report/export経路では未アクセスである。

nonterminal の Sekirei には hash-bound の0 nodes例外が二つある。forced-single 5件は exact cp（`status=exact_cp`、`score_kind=cp`、`score_bound_stm=exact`、整合するcp値）に限り、正常lifecycle、全structured node evidenceが存在して全て0、reported/last/maxが0、in-check・合法手1・sole move一致、normal bestmoveとPV head一致を要求する（PV全体はmate-in-oneの一手PVとは区別し、head以降を許容する）。mate-in-one available は development-05 p122 のみ（黒番・非チェック・合法手217・mating move `2g4g`）で、さらに exact raw `score mate 1`、合法normal bestmove、PV一手、PV headとbestmove一致、凍結リスト一致を要求する。どちらもpositive/missing/invalid/mixed/duplicate/集計不整合を受理しない。別枠のTeacher terminal checkmateは、同一要求内の厳密な mate -1/resign responseだけを許容する。説明不能な normal、その他のTeacher/engine、bound cp・mate・no-score等のforced救済、resignの不正な組み合わせ、PV不一致、分類改ざんは受理しない。forced/mateはcpやexact-cp headlineへ変換しない。旧 v3/v4 と旧 `development-baseline-20260916` はprior diagnosticとしてのみ保持し、final 5局は未読・未変更である。

`development-pilot-20260916-v4` は102/102 attempt、technical failure 0、34/34 stable、observed max `M=1,001,086 <= C(1,000,000)=1,010,000` だったが、forced例外のnode summary集計検査を強化する前のparser identityで生成されたprior diagnosticである。formal freezeには使わず、v5とは区別する。`development-pilot-20260916-v5` は parser `usi-observation-v4` のv0.3.36 identityでreviewed/formal launch evidenceとして凍結した。Teacherは `exact_cp=9 / bound_cp=30 / mate=12`、M evidence `51/48/3/0/0`（evidence/positive/zero/missing/invalid）、max `1,001,086`、`>N=39`、`>C=0`。Sekireiは `exact_cp=45 / mate=3 / no_score=3`、M evidence `48/42/6/3/0`、max `1,000,001`、`>N=3`、`>C=0` だった。Sekireiのmissing 3件はterminal no_score、zero 6件はforced-single 3件とmate-in-one 3件、Teacherのzero 3件はterminalである。pilotなのでheadlineは定義しない。対応するv0.3.36 formal v2は上記のとおり完了したが、final 5局は本測定経路で未アクセスである。旧 `development-baseline-20260916` formalは実行済みだがinvalidであり、formal v2やv4/v5とは別枠で扱う。監査条件でforced例外を取り落として付された旧 findingは現行仕様の誤読として撤回済みである。

```sh
python3 scripts/benchmark.py status
python3 scripts/benchmark_report.py report \
  --run-dir ~/.local/share/sekirei-weight2/suisho11beta-v1/runs/development-baseline-20260916-v2 \
  --output /tmp/development-baseline-v2-report
python3 scripts/benchmark_report.py export \
  --run-dir ~/.local/share/sekirei-weight2/suisho11beta-v1/runs/development-baseline-20260916-v2 \
  --output /tmp/development-baseline-v2-export
```

raw USI log、attempt JSON、manifest、重み・モデル・絶対パスを含むレポートはruntime外へ出さない。attemptとreportには position type、canonical/regression selection、分類別coverage、engine別のpositive node evidenceと全structured nodeの `M` 分布（evidence/positive/zero/missing/invalid、p50/p95/p99/max、`>N`、`>C(N)`、最大overrun/rate）を保持する。`reported_nodes_at_score`、`last_reported_nodes`、`max_reported_nodes_evidence` は別フィールドである。`export` は空の出力ディレクトリ直下に4つのredacted public fileだけを書き、履歴・source game ID・ローカルパス・モデルパスをレビュー時に拒否する。quantileはソート済み有限M値の `(n-1)*p` 位置を線形補間する。supervisor、resume、cleanupの厳密な契約は従来どおり維持する。v5はv0.3.36 formalのreviewed evidenceであり、旧v2/v3/v4と `development-baseline-20260916` は旧semanticsのprior diagnosticとして区別する。

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

## 次の到達点

v0.3.36の固定development 5局baselineと重ね合わせグラフは履歴として確定した。次はv0.3.39でpilotとformal baselineを再取得し、その後、生成済み `.pack` をゲーム単位で train / development に分割して全量JSONL展開を避けた学習入力経路を作る。このMac miniのCPUで小規模学習を一周通し、同じ正式契約でv0.3.39 baselineとの差を測る。

主指標は、教師がexact cpを返した固定E点を分母とする棋譜ごとのMAEを5局で等重み平均する。教師bound/mate、候補mate/no-scoreはcpへ変換せず別診断に残し、候補の出力でE点の分母を変えない。

## 文書

- [研究方針・比較条件](docs/RESEARCH.md)
- [固定環境・再現手順・制約](docs/ENVIRONMENT.md)
- [Sekirei v0.3.39移行検証](docs/validation/sekirei-v0.3.39-2026-09-19.md)
- [将棋クエスト独立棋譜の取得・分割](docs/validation/shogiquest-corpus-2026-09-15.md)
- [開発用100万ノードbaselineの検証値・グラフ](docs/validation/development-baseline-2026-09-19/validation.md)
- [開発運用](AGENTS.md)
- [外部ソフト・資料の出典](docs/PROVENANCE.md)

コード・設定・検証手順を公開 Git に置き、配布記事・アーカイブ・重み・大規模データ・実行結果はローカルに置く。Issue/PR には変更理由と検証結果を残す。モデル公開・定期実行の有効化は今回の準備に含まれない。
