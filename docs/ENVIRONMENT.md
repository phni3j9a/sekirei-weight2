# 実行環境

## 利用範囲

今後も現在の Mac mini と GitHub Actions の軽量 CI の範囲で進める。追加機材（外付け GPU・別 PC・ストレージ増設等）は導入せず、クラウド GPU・有料計算基盤への移行も計画に含めない。教師解析・学習は CPU で実行し、32 GiB RAM と既存ストレージに収まるよう実測に基づいて規模を調整する。詳細は [計算資源の固定条件](RESEARCH.md#計算資源の固定条件) を参照。

## 配置

| パス | 用途 |
| --- | --- |
| `/home/server/projects/sekirei-weight2` | main 同期用 checkout、既存資料の保持 |
| `/home/server/worktrees/sekirei-weight2/<作業名>` | Issueごとの開発worktree。統合後は安全確認して削除 |
| `/home/server/projects/sekirei-weight2/docs/pixiv_fanbox_yaneurao` | ユーザー提供のローカル資料。Git 対象外 |
| `~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/sources` | v0.3.39比較系列のupstreamソース |
| `~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/build` | v0.3.39比較系列のビルド生成物 |
| `~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/bin` | v0.3.39比較系列の実行ファイルへのリンク |
| `~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/models/suisho11beta-concerto-202512` | 主教師重み |
| `~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/data/teachers/suisho11beta-1m` | 内容ハッシュで重複除外した教師 `.pack` とmanifest |
| `~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/venv` | `.pack` 監査・CSA合法手確認用の固定Python環境 |
| `~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/runs` | v0.3.39のsmoke・pilot・今後の実験成果物 |
| `~/.local/share/sekirei-weight2/suisho11beta-v1` | v0.3.36 baselineを保持する旧runtime。上書きしない |
| `~/.local/share/sekirei-weight2/shogiquest-human-v1` | 公開棋譜1,000局、取得cache、再開状態、ローカルmanifest |

大規模資料を worktree にコピーしない。独立した研究実験では専用のソース・出力先を使い、共通 runtime を改造しない。`prepare.py` は排他ロック、`smoke.py` と `audit_pack.py` は共有ロックを取り、スクリプト同士のビルド／解析の競合を防ぐ。手動でのソース変更や直接ビルドは別途利用状況を確認する。

## 固定ソフト

完全な commit とハッシュは [toolchain.lock.json](../config/toolchain.lock.json) にある。

| ソフト | 固定版 | 役割 |
| --- | --- | --- |
| Sekirei | v0.3.39 / `f09c130` | 改善対象エンジン。初期はdefault特徴量・SEKIRW01形式、重みなしのmaterial fallback |
| sekirei-train | 同上 | upstream の学習器。実ファイル名 `train` を `bin/sekirei-train` にリンク |
| shogiesa | 0.9.2 / `dd03174` | 棋譜からの局面抽出・教師ラベル・データ整形 |
| やねうら王 | V9.20 / `a81730f` | 水匠11βに対応する公開版の教師エンジン |
| 水匠11β | Suisho Concerto 202512 | SFNNwoP1536、FV_SCALE=28。第1目標 |
| Suisho11Plus | 2026-08-01 配布記事に対応する手元アーカイブ | 旧固定環境を残す参考教師 |

水匠11βは重みであり、やねうら王の実行ファイルとは別。配布記事はV9.20以降と `NNUE_SFNNwoP1536` を指定しているため、公開履歴のV9.20版更新commitをLinux向けにビルドした。配布バイナリとのビット単位の同一性は主張しない。旧Plusのruntimeは上書きせず、新しい比較系列としてβ専用profileへ分離した。

最初の環境では、必要なUSI機能の実装と実機結果からv0.3.36を固定した。当時調査した `main` (`08107a2`, package 0.3.5) は `go nodes` の解釈と探索のノード上限がなく、smokeが120秒でタイムアウトした一方、公開タグv0.3.36はその機能を持っていた。

その後、上流v0.3.37でsingular-extension verification searchが通常探索用TT entryによりshort-circuitされる問題が修正された。旧sekirei-weightでも影響を受けた既知問題なので、今後の比較基準はこの修正を含むv0.3.39へ更新する。固定commit `f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9` は確認時のv0.3.39 tagとupstream mainの両方に一致する。v0.3.38で公開された任意のA-flat NNUE checkpointは今回読み込まず、探索版更新と評価器更新を分離する。v0.3.36のbuild・run・公開baselineは旧runtimeに保持する。ビルドhashと100万ノード疎通結果は[v0.3.39移行検証](validation/sekirei-v0.3.39-2026-09-19.md)に固定した。

エンジンcommit変更によりexecution identityが変わるため、v0.3.36の `development-pilot-20260916-v5` はv0.3.39のformal launch evidenceとして使えない。現行 `formal.pilot_evidence` は `null` で、v0.3.39のpilotを実測・レビューするまでformalはfail-closedで停止する。今後も「latest」へ自動追従せず、採用版の完全なcommitと実機結果を固定する。

### ビルド条件

- Ubuntu 24.04 / Intel i5-8500B（6 core）、32 GiB RAM。
- 初回確認: rustc/cargo 1.96.0、Python 3.12.3、g++ 13.3.0。
- Rust: `--release --locked`、default features、`RUSTFLAGS='-C target-cpu=x86-64-v3'`。既存プロジェクトのグローバル設定を変更しない。
- やねうら王: `normal COMPILER=g++ PYTHON=python3 TARGET_CPU=AVX2 YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_SFNNwoP1536`。
- コンパイラ自体はインストール／自動更新しない。実際のバージョンと各バイナリの SHA-256 を `build-manifest.json` に記録する。同じソースでもコンパイラ変更後の完全なバイナリ同一性は主張しない。
- β環境構築前の空きは約57 GiB。ビルド・venv・重み・重複除外済みpackを含むβ runtimeは実測998 MiB。大規模な中間形式への全量展開やGPU学習環境は今回導入しない。

### 主教師の重み

```sh
python3 scripts/prepare.py import-teacher --archive \
  '/home/server/projects/sekirei-weight2/docs/pixiv_fanbox_yaneurao/suisho11beta-suisho_concerto202512.7z'
```

アーカイブSHA-256 `989d292d...b69aab8a8` を照合して `eval/nn.bin` だけを抽出し、112,887,658 bytes、SHA-256 `d1b16f0a...e8e67785` を再照合する。

配布記事の指定どおり、USIで `FV_SCALE=28` と絶対パスの `EvalDir` を渡す。V9.20の固定ソースにある `sfnnwop-1536.h` は HalfKA_hm、変換後1536次元、隠れ層15/32、LayerStacks=9。アーキテクチャをファイル名から推測せず、当時のMakefileにある専用editionを使う。

### 生成済み教師 `.pack`

```sh
python3 scripts/prepare.py import-corpus --archive \
  '/home/server/projects/sekirei-weight2/docs/pixiv_fanbox_yaneurao/kif20260630-pack-1M.7z'
python3 scripts/prepare.py import-corpus --archive \
  '/home/server/projects/sekirei-weight2/docs/pixiv_fanbox_yaneurao/kif20260731-pack-1M.7z'
```

それぞれのアーカイブ全体を固定SHA-256で照合し、記事で水匠11β・100万ノードと説明された `1000000a/` と `1000000b/` の `.pack` だけを標準出力経由で安全に抽出する。個々のファイルはSHA-256名で保存し、同じ内容を複数回保持しない。実機では収録15本のうち2本が重複し、13本、534,175,084 bytesになった。由来と重複関係はローカルの `manifest.json` に残る。

`.pack` の復号と取得したCSAの完全な合法手再生には cshogi 1.0.4 / NumPy 1.26.4 を専用venvへ固定する。benchmark runner内の標準ライブラリboard trackerは、固定hashに対する厳密な移動遷移／擬似合法性を検査するが、それだけで完全合法性を主張しない。development runnerの `config/development-position-classifications.json` はこの pinned cshogi 1.0.4 で全570 occurrence・全46,668合法root moveを検証した hash-bound metadata であり、通常の実行依存にはしない。これはCPU用で、GPU環境は導入しない。

```sh
python3 scripts/prepare.py audit-deps
~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/venv/bin/python \
  scripts/audit_pack.py --samples 10
```

## 疎通検証

```sh
python3 scripts/smoke.py
```

- 専用の空の run ディレクトリから起動し、呼出元にある `engine_options.txt` 等の混入を避ける。
- バイナリ・重みを照合し、広告された USI オプションを確認してから設定する。
- `usi` / `isready` を待ち、局面履歴を渡して100万ノード指定で解析する。
- cp 形式、最終 PV と bestmove の一致、node 指定の実行、正常終了を確認する。最後が upperbound/lowerbound の場合はその種別を保持し、確定評価値を `null` にする。以前の exact スコアへのすり替えはしない。
- accepted score の node evidence は存在する整数 `>=0` とし、通常は正の値を要求する。非終端Sekireiの0 nodesにはhash-boundの例外が二つある。forced-singleの5件は exact cp（`status=exact_cp`、`score_kind=cp`、`score_bound_stm=exact`、整合するcp値）に限り、正常lifecycle、全structured node evidenceが存在して全て0、reported/last/maxが0、in-check・合法手1・sole move一致、normal bestmoveとPV head一致を要求する（PV全体はmate-in-oneの一手PVとは区別し、head以降を許容する）。`mate_in_one_available` は development-05 p122のみで、さらに exact raw `score mate 1`、合法normal bestmove、PV一手、PV headとbestmove一致、凍結mating move一致を要求する。両方ともpositive/missing/invalid/mixed/duplicate/集計不整合は受理しない。別枠のTeacher terminal checkmateは同一要求内の厳密な `score mate -1 nodes 0 pv resign` と `bestmove resign` だけを mate として扱う。裸の resign や stale score は no_score にする。bound cp・mate・no-score等をforced例外として救済しない。pilot gate は両エンジンそれぞれの正の node evidence と技術失敗ゼロを要求する。
- 自作の3手の CSA fixture を shogiesa で一局面に抽出し、教師ラベルを生成する。
- 両エンジンの USI ログ、shogiesa の JSONL・manifest、全体の summary.json を保存する。

### reviewed pilot v2 の診断結果

`development-pilot-20260916-v2` は15局面・3 repetition・2 engineの90/90 attemptを完了した。strict validationでtechnical failureは0件、30個の engine-position tripleは30/30が安定した。status内訳はTeacherが `exact_cp=6 / bound_cp=30 / mate=9`、Sekireiが `exact_cp=42 / no_score=3` である。

positive node evidenceはTeacherが42/45 attempt、positive observation 870、zero 3、missing/invalid 0、observed max 1,000,692 nodes、Sekireiが39/45 attempt、positive observation 39、zero 3、missing 3、invalid 0、observed max 1,000,001 nodesだった。Sekireiのmissing 3件はterminal checkmateのno-scoreであり、技術失敗には数えない。

development-04 p108は `forced_single_legal_move`（in check、合法手1、sole move `2i1g`）で、Teacherはmate、Sekireiは300 cpだった。development-05 p123は `terminal_checkmate`（in check、合法手0）で、Teacherは mate -1、Sekireiは `no_score` だった。pilotはbounds/matesを保持する再現性診断であり、五局完全なexact-cp headlineは作らない。

v2のpilot fingerprint `6eb64a82e1673510438e3dcbc9e7ba1533c5b7042d9c04ec3c5e209e1c3904e4` と strict all-evidence observed maximum 1,000,692は、記録済みrunnerに対するstrict-validなprior diagnosticとして保持する。旧 `requested + 1024` は当時の診断設定であり、YaneuraOuの数学的停止上限・正確な仕事量同値性として扱わない。現行は `one-sided-1-percent` v1 の整数式 `C(N)=N+floor(N/100)` による片側1%の運用上のsanity envelopeへ更新し、`formal.max_reported_nodes` を `C(1,000,000)=1,010,000` に固定する。次に凍結する `formal.pilot_evidence.max_reported_nodes` も同じ値でなければならない。`go nodes` は上限なので最低ノード目標は設けない。runner/parserの変更でexecution identityも変わったため、v2/v3/v4をformal launch evidenceへ使わない。

### reviewed pilot v5 と改訂前診断の扱い

`development-pilot-20260916-v3` は改訂前runnerのprior diagnosticであり、現行の分類・parser identityに対するformal launch evidenceではない。現行pilotはcanonical 15局面に development-04:77 と development-05:122 の regression 2局面を加え、17局面×3 repetition×2 engine = 102 attempt / 34 engine-position tripleとする。

`development-pilot-20260916-v4` は102/102 attempt、technical failure 0、34/34 engine-position tripleがstable、observed max `M=1,001,086 <= C(1,000,000)=1,010,000` だった。ただし forced-single zero-node 例外のnode summary集計検査を強化する前のparser identityで生成されたprior diagnosticなので、formal freezeには使わない。v5とは区別して保持する。旧 `development-baseline-20260916` formalは実行済みだがinvalidであり、v4とは別枠で扱う。

旧 `development-baseline-20260916` formal は1,140/1,140 attemptを収集したが invalidである。Teacher development-04 p077 の all-evidence max 1,001,086 は旧1,001,024を超え、Sekirei development-05 p122 は旧分類にない mate 1 / nodes 0 だった。`status=complete` は formal valid を意味せず、診断上の exact coverage 265/265 も有効なheadlineやモデル採用の根拠ではない。v0.3.36当時のconfigはv5のrun ID・fingerprint・observed max 1,001,086・ceiling 1,010,000を凍結し、そのidentityでformal v2を完了した。final 5局は同formalとreport/export経路で未アクセスである。現行v0.3.39 configではこのevidenceを解除しており、旧1,001,024をsource-derived guaranteeとして扱う主張も撤回済みである。

typed resultは、既存のforced-single 5件とterminal checkmate 1件を維持し、development-05 p122を `mate_in_one_available`（黒番・非チェック・合法手217・mating move `2g4g`）として扱う。非終端Sekireiのzero-node例外はforced-singleとmate-in-oneの二つのhash-bound例外である。forced-singleは exact cp（`status=exact_cp`、`score_kind=cp`、`score_bound_stm=exact`、整合するcp値）に限り、正常lifecycle、全structured node evidence 0、reported/last/max 0、sole moveとnormal bestmove/PV head一致を要求し、PV全体は一手に限定しない。mate-in-oneはさらにexact raw mate 1と一手PV・凍結mating move一致を要求する。別枠のTeacher terminal checkmateは厳密なmate -1/resign responseだけを許容する。bound/mateをforced例外としてcpへ変換せず、v5もpilotなので五局完全なexact-cp headlineは定義しない。`development-pilot-20260916-v5` は17局面・102/102 attempt、technical failure 0、34/34 stable、complete evidence validである。Teacherは `exact_cp=9 / bound_cp=30 / mate=12`、M evidence `51/48/3/0/0`、max 1,001,086、`>N=39`、`>C=0`、Sekireiは `exact_cp=45 / mate=3 / no_score=3`、M evidence `48/42/6/3/0`、max 1,000,001、`>N=3`、`>C=0`。missing 3件はterminal Sekirei no_score、zero 6件はforced-single 3件とmate-in-one 3件、Teacher zero 3件はterminalである。これはv0.3.36 formalのreviewed launch evidenceで、対応するformal v2も完了した。旧 `development-pilot-20260916-v2`、v3、v4と旧invalid formalはprior diagnosticとしてのみ保持する。

### reviewed formal v2 の実機結果

`development-baseline-20260916-v2`（fingerprint `d1708915e2de8bfd22d1d3c29eb10cc10f45926da4f0f2357dd87a0e407ab084`）は570 occurrence×2 engineの1,140/1,140 attemptを完了した。strict validatorではmissing/extra/duplicate 0、technical failure 0。全attemptで `completed_before_deadline=true`、`cleanup_status=ok`、`supervisor_status=ok`、engine return code 0を確認した。Teacherは `exact_cp=266 / bound_cp=273 / mate=31`、Sekireiは `exact_cp=561 / mate=8 / no_score=1` である。

Teacher-EのSekirei exact coverageは266/266、5局を等重みで平均した正式headline MAEは `1,087.0461722818245 cp`。all-evidence MのmaxはTeacher `1,001,086`、Sekirei `1,000,004` で、両engineとも `>C=0`、invalid=0だった。Sekireiのzero 6件はforced-single 5件とmate-in-one 1件、missing 1件はterminal checkmateのno-score、Teacherのzero 1件は同terminalで、型付き例外契約どおりである。redactedな公開成果物は [`validation/development-baseline-2026-09-19`](validation/development-baseline-2026-09-19/validation.md) に固定し、raw log・attempt・絶対パス・重みは外部runtimeにだけ残す。final 5局は本正式測定とreport/export経路で未アクセスである。

### 現時点の制約

USI node grammar は、`go nodes <[0-9]+>`（ASCII 十進数字列、符号なし）から対応する `bestmove` までの各 structured `info` 行を左から解釈する。`info string` は free text として無視し、`pv`・`string`・`refutation`・`currline` の可変長 payload 内の token は解釈しない。payload 前の各行には `nodes <[0-9]+>` を高々一組だけ許容し、欠落・重複・負値・符号付き／非整数値は無効な node evidence として扱う。

Sekirei の `isready` は重み読込失敗後でも応答するため、将来のモデル評価ではファイルハッシュと読込成功の確認が必要。今回の smoke は明示的な駒得評価であり、旧モデルやランダム重みを学習済みとして扱わない。

ノード上限到達が aspiration 探索の途中になると、やねうら王の最終 `info` に上限・下限が付くことがある。`OutputFailLHPV=false` でも最後の報告には付く場合がある。これはノード指定の疎通失敗ではないが、確定値の採点には使えない。smoke と監査では境界の向きと生の値を保存し、先後反転では上限／下限も反転する。また `go nodes` は上限であり、探索が完了すれば100万より手前で正常終了しうる。current-go の対応bestmoveまでの structured `info` を走査し、`info string` と `pv` payload は境界として nodes を読まない。各行の nodes は一組だけを許容し、重複・欠落・負値・malformedを技術失敗として、選択score行・最後の有効値・全有効値の最大 `M` を別保存する。`M <= C(N)` を `one-sided-1-percent` v1 のinclusiveな運用ガードレールとして適用するが、これは数学的停止上限・内部仕事量の同値性・最低ノード目標ではない。説明不能な0、timeout/cleanup/protocol failureは成功扱いにしない。

GenSfen `.pack` はゲーム境界を保持するが、評価関数SHA、エンジンcommit、全option、score boundを持たない。今回の対象ファイルにはV9.20公開前の日付のものもあり、生成に使った開発版を特定できない。GenSfenは通常、1局につき先後用のプロセスを再利用する一方、監査は局面ごとに新しいプロセスと空のhashを使う。したがって監査は符号・尺度・大きな取り違えの検出であり、同一生成環境の証明ではない。

shogiesa の固定版は `position sfen ...` で局面を渡し、同じプロセスを再利用する。棋譜履歴を保持し、局面ごとに探索状態を独立させる正式なグラフ比較と意味が異なる。学習用のラベル生成には活用するが、今の label コマンドをそのまま最終採点器にはしない。

独立評価用には、将棋クエストの公開棋譜1,000局をGit外runtimeへ固定し、エンジン解析前にdevelopment 5局 / final 5局を選定した。取得状態を再利用するため、公開Web画面へ繰り返しアクセスする必要はない。出典、filter、実測、snapshot hash、制約は[将棋クエスト独立棋譜corpusの固定](validation/shogiquest-corpus-2026-09-15.md)に記録する。

固定版の `sekirei-train` と `shogiesa` はGenSfen `.pack` を直接は読まない。ストリーム復号から学習入力への接続、固定棋譜の100万ノード比較、複数棋譜の採点、CPUでの小規模学習は次の到達点。βへの切替と監査結果は [β環境と教師監査](validation/suisho11beta-2026-09-15.md)、旧Plus環境は [初期検証](validation/environment-2026-09-14.md) に記録する。

公開 GitHub Actions では Python の構文、USI スコア受理のテスト、設定 JSON を検証する。教師重みを CI へアップロードせず、実エンジンと教師の smoke は Mac mini で実行する。

## 開発baseline runnerのruntime

Issue #7の計画・実行は `scripts/benchmark.py`、採点・図・公開境界は `scripts/benchmark_report.py` が担当する。既定の保存先は次のruntime配下で、リポジトリへコピーしない。

USI supervisorのcleanup契約は、engineの終了コードと監督結果を分離する。監督は `waitid(WNOWAIT)` でengine exitを観測し、`/proc/<supervisor>/task/<supervisor>/children` の直接子をreap前に識別してpidfdを取得する。再親化で元のsession/PGIDを離れた子にも `signal.pidfd_send_signal` でTERM→KILLを送り、全子のreap/ECHILDと入力・出力relay閉鎖を、一つのcleanup deadline内で確認する。能力・列挙・signal・reapの証明失敗、監督terminal status欠落、runnerによるanchorのforce-killはcleanup failureとしてraw outcomeに保存し、成功扱いにしない。

```text
~/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-v1/
  .prepare.lock       # prepare.pyと共有するロック
  .benchmark.lock     # benchmark全体の排他ロック
  runs/<run-id>/
    .run.lock
    manifest.json     # fingerprint、execution identity、attempt matrix、commit/dirty、toolchain、host、option、limit
    universe.json     # canonical development occurrence universe
    plan.json         # run type固有の位置集合とhash
    attempts/*.json   # 成功・失敗を含む確定attempt
    logs/*.jsonl      # send/receive各行、runner outcome、monotonic offset
```

prepareの共有non-blocking lock、benchmark-wide排他non-blocking lock、run lockを同時に保持する。attempt JSONとraw logは一時ファイルへ書いてfsync後にreplaceする。resume・report・exportは同じ厳密validatorで、現在のdevelopment configのsplit/benchmark ID、全position/repetition/engineのexact attempt matrix、options/requested nodes、raw JSONLのgo-bound score evidenceとhashを照合する。成功・no-scoreの raw transcript については、広告された全option、`usi` → `setoption`（設定順）→ `isready` → `usinewgame` → exact `position` → exact `go nodes` →対応`bestmove`→`quit` の lifecycle も検証し、recorded binary identity をruntime manifestのpath/hash/bytesに束縛する。raw log は engine 行を改変せず、cleanup後に一つだけ structured runner outcome を末尾へ置く。各USIはrunner所有のLinux supervisor/subreaperをsession leader・PGID anchorとして起動し、supervisorがengineをshellなしの直接argvでexecして子孫をreapする。engine親が即時終了してもanchorは子孫cleanupまで存続するため、runnerのscheduler依存pollで正当な子を発見する必要がない。validな成功・失敗は再実行せず、欠落したattemptだけを実行する。破損・不一致は上書きせず停止する。timeoutのdeadline後に遅れて届いた score/bestmove は採点へ昇格しない。timeoutや親プロセス先行終了でも、捕捉したsupervisorのsession leader PID/PGIDを検証してTERM→KILLし、実行可能な子が消えたことを確認してからpipeを閉じる。未知の同一PGIDが現れて所有権が曖昧になった場合は一切signalせず、cleanup failureとして以後のbenchmarkを中止する。Popen前のstartup failureにもrunner eventを含むraw logを残す。

resume・report・exportの再検証では、position type、canonical/regression selection、分類別coverage、accepted node evidence（存在する整数 `>=0`、非終端Sekireiのforced-singleまたはmate-in-oneのhash-bound 0例外、terminal teacherの厳密なmate/resign例外）を同じparser semanticsで照合する。forced-singleは exact cp（`status=exact_cp`、`score_kind=cp`、`score_bound_stm=exact`、整合するcp値）に限り、全structured node evidenceの存在・全値0・reported/last/max 0・集計一致・sole moveとnormal bestmove/PV head一致、mate-in-oneはこれに加えてexact raw mate 1・一手PV・凍結mating move一致を要求する。bound cp・mate・no-score等をforced例外として救済しない。timeout・cleanup・protocol failureの遅い出力や保存値を成功へ救済しない。旧 `development-pilot-20260916` と改訂前v2/v3/v4は旧parser semanticsの診断証跡として immutable に扱い、formal evidenceには使わない。

実行前の確認は次のとおり。

```sh
python3 scripts/benchmark.py plan --run-type pilot
python3 scripts/benchmark.py plan --run-type formal
```

planはエンジンを起動せず入力集合を確認するため、未凍結でも作成できる。実際の `formal` 実行はv0.3.39 pilotのreview後に `formal.pilot_evidence` を凍結するまで拒否する。

planの固定値はpilot 17 positions / 102 attempts / 34 engine-position triples、formal 570 positions / 1,140 attempts、development CSA aggregate SHA-256 `0e02b6319cbf908761dde7326ab6a1bfc4b647e2601ca1e3643fa6a207f48ee2`、分類manifest SHA-256 `a244a2206fd2b25b6fe475a9794c07eb2c5fc99f996e891dbfed1e89a0a89427`、分類を含むcanonical universe SHA-256 `33ce54f3ff9c40687e2304a7dc222ceddc0d6843180020a68262dd1762ec5fa6`。両計画は `go nodes 1000000` と `one-sided-1-percent` v1、`C(1,000,000)=1,010,000` を事前登録する。`reported_nodes_at_score`、`last_reported_nodes`、全有効値の最大 `M` を保存し、`M <= C(N)` をinclusiveに判定する。observed maxがrequested nodes未満でも早期完了として許容するが、各engineに正のnode evidenceが必要で、技術失敗や片側だけの証拠ではgateを通さない。formal gateは同じruntimeの102-attempt pilotについて `pilot_run_id`、`pilot_fingerprint`、全attemptから再計算した observed maximum、policy limit、完全なattempt matrixを要求する。v0.3.39ではまだ未凍結なのでformalは停止する。execution identityにはrunner/parser、分類・development hash、requested nodes、node policy id/version/rate、timeout、環境、全option、バイナリ・build/toolchain、教師重み、候補モデル、hostを束縛する。

`benchmark_report.py report` はlocal詳細を書けるが、`export` は空の出力ディレクトリ直下へ `validation.md`、`reviewed.svg`、`validation.json`、`manifest.json` の4 redacted public fileだけを書く。local/や局面別ファイルは作らず、絶対パス、ユーザー名、source game ID、raw position履歴、model path、free-form provenanceを入れない。v0.3.36 formal v2の生成物はMainがSVGを目視レビューし、[`validation/development-baseline-2026-09-19`](validation/development-baseline-2026-09-19/validation.md) に追跡した。v5は同系列の17局面・102 attemptのreviewed/formal launch evidence、formal v2はそのidentityでの初期baselineであり、旧v2/v3/v4 pilotと旧invalid formalから区別する。local/public reportはengine別に全有効値Mのevidence/positive/zero/missing/invalid、p50/p95/p99/max、`>N`、`>C(N)`、最大positive overrun/rateを保持し、quantileはソート済み有限値の `(n-1)*p` 位置を線形補間する。pilotのTeacher exact coverageはサンプル診断に限られ、正式headlineはformal v2の固定Teacher-E 266点が全てSekirei exactになった場合にだけ定義した。final 5局は未アクセスである。
