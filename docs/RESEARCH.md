# 研究方針

## 目標

未学習棋譜の同じ局面・同じ指し手履歴について、水匠11β（Suisho Concerto 202512）と Sekirei をそれぞれ `go nodes 1000000` で探索させ、先手視点の評価値グラフを近づける。対局勝率や学習 loss は目標の代替にしない。

水匠11βを第1目標とした理由は、手元に同モデル・100万ノードと説明された生成済み教師データがあり、Mac miniで大量の教師探索を再生成せず学習へ進めるため。Suisho11Plusは参考教師として節目で確認するが、初期モデルの採否をPlusへの一致で決めない。

人間が目的・評価方法・計算予算・変更可能範囲を定め、AI が実測から次の改善仮説を選ぶ。特徴量・NNUE 構造の変更は検討対象。既知のsingular-extension TT cutoff問題を解消したupstream v0.3.39への移行後は、探索実装の追加変更を当面対象外とする。

## Fresh に進める

- 原典、固定版の実装、今回の実験結果を参照する。
- 旧 sekirei-weight のロードマップ・採否基準・学習済みモデル・独自パッチを自動で継承しない。
- 新しいフォルダに用意された配布資料は利用可能。内容の正確さは実物の形式・設定・動作と照合する。
- 標準の upstream ツールを使うことは、旧プロジェクトの研究方針を継承することを意味しない。

## 初期の比較条件

| 項目 | 初期値・扱い |
| --- | --- |
| ノード指定 | 両者 `go nodes 1000000`。報告されたノード数も保存 |
| スレッド | 両者 1。Sekirei は `SearchMode=Speculative`、`SpecTopN=0`、`RAYON_NUM_THREADS=1` |
| ハッシュ容量 | 両者 128 MiB。USI オプション名の違いを明示 |
| 定跡・ponder | 無効 |
| MultiPV | 1 |
| 主教師 | 固定やねうら王 V9.20 + 固定水匠11β Concerto 202512、SFNNwoP1536、FV_SCALE=28 |
| 参考教師 | Suisho11Plus。旧固定環境を保持するが、初期の合否判定には使わない |
| Sekirei | upstream v0.3.39の固定commit。上流公開NNUE重みを入れず、最初はmaterial fallbackで探索差分を分離 |
| 局面入力 | 正式評価では `position startpos moves ...` または初期 SFEN + 全履歴 |
| 探索状態 | 最初の正式評価は局面ごとに新規プロセスを作り、他局面の TT/history を持ち越さない案。生成済み `.pack` の探索状態とは区別 |
| 縦軸 | 先手視点の cp。棋譜ごとの倍率変更・オフセット補正・平滑化を採点に使わない |

同じノード指定でも、エンジン間で数え方や探索内容は等価ではない。ここではユーザーが選んだ観測条件として固定する。初期 smoke の ±2% は疎通用の異常検知幅であり、正式ベンチマークのノード許容規則ではない。

## 生成済み教師データ

最初の学習系列では、配布記事で水匠11β・100万ノードと説明された GenSfen `.pack` の `1000000a/` と `1000000b/` を使う。アーカイブ全体の固定ハッシュを照合し、個々の `.pack` は内容ハッシュで重複を除く。生データは利用条件に従ってローカルに置き、そのまま再配布しない。

`.pack` は開始局面とゲーム境界、各局面の選択手、手番視点の符号付き16bit評価値、終局情報を持つ。train / development の分割は局面ごとではなくゲームごとに行う。全量を巨大なJSONLへ展開せず、ストリーム復号して学習器が必要とする形式へ渡す。

配布 `.pack` には次の情報が含まれないため、「水匠11β・100万ノード」という記事上の由来と、今回固定した環境の完全同一性は証明できない。

- 評価関数ファイルのSHA-256
- 生成に使ったやねうら王のcommitと全USI option
- 各探索結果が exact / upperbound / lowerbound のどれだったか
- 各手番用プロセスに残っていたTT・探索履歴

そこで少量の局面を固定V9.20・同一重み・100万ノードで再評価し、符号、尺度、評価値差、指し手、実ノード数を監査する。この監査は取り違え検出と互換性確認であり、由来の証明や学習データ自体を開発・最終評価データとして使うことではない。

## 評価と分割

- 候補選択に使う開発棋譜と、節目だけで使う最終評価棋譜を分ける。学習・開発・最終評価の間で同一棋譜、変化枝、重複局面の混入を点検する。
- 生成済み `.pack` は学習用の候補母集団であり、正式グラフ比較の棋譜には使わない。
- 独立棋譜は将棋クエストの公開棋譜から、公式ページが人間対局と示し、平手初期局面から全手を合法に再生できる1,000局を先に固定する。既知Bot印のないことも別に確認し、対局IDと指し手列の両方で重複を除く。
- 今回のrunnerの標準ライブラリboard trackerは、固定hashに対する手番・所有者・移動経路・成り・駒取り・打ち駒の厳密な遷移／擬似合法性を検査する補助証跡である。盤面分類は `config/development-position-classifications.json` に occurrence ID、履歴・局面 hash、cshogi 1.0.4 の verifier version とともに固定し、全570手の cshogi 再計算で照合する。全合法性の根拠はこの比較結果とし、CIのtrackerだけから完全合法性を主張しない。
- 教師・Sekireiの解析前に、固定hash順位、時間設定、手数、レーティング条件、対局者非重複の機械的規則で development 5局 / final 5局を選ぶ。選ばれたCSAから名前とレーティングを除き、出典追跡用の対局IDはmanifestに残す。
- final 5局は改善仮説・重み・閾値の選択に使わず、節目の確認に限定する。公開リポジトリにある以上、データ秘匿ではなく運用上のholdoutである。
- 主指標案は、棋譜ごとの評価値の平均絶対誤差を求め、棋譜間で等重み平均する。
- 棋譜ごとに教師・現在の最良・候補を同じ縦軸で重ねる。序中終盤の内訳と大きな誤差も見る。
- 詰みを任意の巨大 cp に変換して MAE に混ぜない。候補が詰みや解析失敗を出すことで難しい局面が分母から消えないようにする。
- 最終スコアと bestmove の対応、上限・下限だけの score、timeout、千日手・連続王手、投了・入玉宣言の扱いを正式採点前に確定する。
- 閾値や完了目標値は現在地の実測前に捏造しない。最初に再現性と分布を測る。

## 実験の一周

1. 開発データの誤差を調べ、次の仮説を一つ選ぶ。
2. 変更理由、比較対象、計算上限を Issue に記録する。
3. 専用 worktree で実装・学習し、条件と成果物を保存する。
4. 安価な確認を経た候補を、同じ 100万ノード条件で比較する。
5. 最良モデルの採否と知見を記録する。改善しない実験の記録も残す。

コードの統合とモデルの採用を区別する。同時に進行させる実験課題は原則一つ。数回改善しない場合にはデータ・表現力・学習方法・探索のどこが制約かを調べ、単なる再試行を続けない。

## 計算資源の固定条件

現在の Mac mini（Intel i5-8500B、6 core、32 GiB RAM、既存ストレージ）の範囲で継続する。今後も追加機材は導入しない。外付け GPU・別 PC・クラウド GPU・有料計算基盤への移行は計画に含めず、ユーザーが方針を変更するまで固定条件とする。公開 GitHub リポジトリと Actions による軽量 CI を使う。

CPU で学習から評価まで小規模に一周し、教師生成・学習・比較それぞれの所要時間、メモリ使用量、保存容量を測る。その実測に合わせて、教師データの再利用、学習データの選定、モデル容量、実験数、CPU 実装の効率化を検討する。効率化によって当面固定している探索の意味や比較設定を変えない。

正式な採用判定は100万ノード条件を維持する。安価な予備確認は正式評価と区別し、計算資源の都合で採用基準を無断で緩めない。目標への到達を保証せず、この環境で得られた改善と制約を実測から報告する。

## 自動化の範囲

rfkit-rs の Planner → 一つの Issue → Worker → 検証済み PR の骨格を参考にする。具体的な自動実行スケジュール、モデル配役、自動マージ、長時間計算の予算はまだ設定しない。

## Issue #9 Sekirei v0.3.39への移行

現在のSekirei / sekirei-trainはupstream v0.3.39、commit `f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9`へ固定する。v0.3.37で入ったsingular-extension verification searchのTT cutoff修正を含む版を、今後の探索基準にする。upstream v0.3.38で公開されたA-flat NNUE checkpointは評価器そのものを変えるため、この移行には含めない。まず学習済み重みなしのmaterial fallbackで、v0.3.36との差を探索版だけに限定する。

v0.3.36のruntime `suisho11beta-v1` と完了済みbaselineは上書きしない。v0.3.39は `suisho11beta-sekirei-v0.3.39-v1` に分離する。エンジンcommitとtoolchain lockが変わるため、旧 `development-pilot-20260916-v5` は新しいformal launch evidenceにならない。`config/development-benchmark.json` の `formal.pilot_evidence` は `null` に戻し、v0.3.39の102-attempt pilotを取得・レビューしてから新しい値を凍結する。入力集合を確認するformal planは作成できるが、それまではformal実行をfail-closedで停止させる。

## Issue #7 v0.3.36開発baselineの固定契約

### reviewed pilot v2 の診断結果

`development-pilot-20260916-v2` は、固定された15局面を両エンジン・3 repetitionで解析し、90/90 attemptを完了した。strict validatorで technical failure は0件、30個の engine-position triple（15局面×2エンジン）は30/30がrepetition間で安定した。Teacherの内訳は `exact_cp=6`、`bound_cp=30`、`mate=9`、Sekireiは `exact_cp=42`、`no_score=3` である。

positive node evidence もengine別に保持する。Teacherは45 attempt中42 attemptが正の値を持ち、positive observationは870、zeroは3、missing/invalidは0、observed maxは1,000,692 nodesだった。Sekireiは45 attempt中39 attemptが正の値を持ち、positive observationは39、zeroは3、missingは3、invalidは0、observed maxは1,000,001 nodesだった。Sekireiのmissing 3件は terminal checkmate の no-score responseであり、technical failureではない。

typed occurrenceでは、development-04 p108を `forced_single_legal_move`（in check、合法手1、sole move `2i1g`）として保持し、Teacherはmate distance 2、Sekireiは300 cpを返した。development-05 p123は `terminal_checkmate`（in check、合法手0）として保持し、Teacherは mate -1、Sekireiは `no_score` の裸の resignとして扱った。詰みやboundを巨大cpへ変換せず、pilotのTeacher exact coverageだけで五局の正式headlineを定義しない。

このpilotのfingerprintとobserved maximumは、記録済みrunnerに対する改訂前のstrict-validな診断結果として保持する。旧 `requested + 1024` をYaneuraOuの停止上限とみなすsource-derived guaranteeは撤回する。現行は `one-sided-1-percent` v1 の整数式 `C(N)=N+floor(N/100)` を採用し、`M <= C(N)` をinclusiveに判定する。これは片側1%の運用上の比較・異常検出ガードレールであり、数学的停止上限、内部仕事量の同値性、最低ノード目標ではない。現行のpilot/formal計画は `C(1,000,000)=1,010,000` を両方に事前登録し、改訂前artifactは再解釈しない。

### reviewed pilot v5 と改訂前診断の扱い

`development-pilot-20260916-v3` は改訂前runnerのprior diagnosticであり、現行の分類・parser identityに対するformal launch evidenceではない。現行pilotはcanonical 15局面に development-04:77 と development-05:122 の regression 2局面を加え、17局面×3 repetition×2 engine = 102 attempt / 34 engine-position tripleとする。

`development-pilot-20260916-v4` は102/102 attempt、technical failure 0、34/34 engine-position tripleがstable、observed max `M=1,001,086 <= C(1,000,000)=1,010,000` だった。ただし forced-single zero-node 例外のnode summary集計検査を強化する前のparser identityで生成されたprior diagnosticなので、formal freezeには使わない。v5とは区別して保持する。旧 `development-baseline-20260916` formalは実行済みだがinvalidであり、v4とは別枠で扱う。

旧 `development-baseline-20260916` formal は1,140/1,140 attemptを収集したが invalidである。Teacher development-04 p077 の all-evidence max 1,001,086 は旧1,001,024を超え、Sekirei development-05 p122 は旧分類にない mate 1 / nodes 0 だった。`status=complete` は formal valid を意味せず、診断上の exact coverage 265/265 も有効なheadlineやモデル採用の根拠ではない。v0.3.36当時のconfigはv5のrun ID・fingerprint・observed max 1,001,086・ceiling 1,010,000を凍結し、そのidentityでformal v2を完了した。final 5局は同formalとreport/export経路で未アクセスである。v0.3.39の現行configではこのevidenceを解除している。

typed resultは、既存のforced-single 5件とterminal checkmate 1件を維持し、development-05 p122だけを `mate_in_one_available`（黒番・非チェック・合法手217・sorted unique mating move `2g4g`）として追加する。非終端Sekireiの0 nodes例外はhash-boundで二つある。forced-singleは exact cp（`status=exact_cp`、`score_kind=cp`、`score_bound_stm=exact`、整合するcp値）に限り、正常lifecycle、全structured node evidenceが存在して全て0、reported/last/maxが0、in-check・合法手1・sole move一致、normal bestmoveとPV head一致を要求する（PV全体は一手に限定しない）。mate-in-oneはさらにexact raw `score mate 1`、合法normal bestmove、PV一手、PV headとbestmove一致、凍結リスト一致を要求する。両方ともpositive/missing/invalid/mixed/duplicate/集計不整合は受理しない。別枠のTeacher terminal checkmateは厳密な mate -1/resign responseだけを許容し、bound cp・mate・no-score等をforced例外として救済せず、forced/mateはcpやexact-cp headlineへ変換しない。`development-pilot-20260916-v5` は17局面・102/102 attempt、technical failure 0、34/34 stable、complete evidence validである。Teacherは `exact_cp=9 / bound_cp=30 / mate=12`、M evidence `51/48/3/0/0`、max 1,001,086、`>N=39`、`>C=0`、Sekireiは `exact_cp=45 / mate=3 / no_score=3`、M evidence `48/42/6/3/0`、max 1,000,001、`>N=3`、`>C=0`。missing 3件はterminal Sekirei no_score、zero 6件はforced-single 3件とmate-in-one 3件、Teacher zero 3件はterminalである。pilotなのでheadlineは定義せず、v5をv0.3.36 formalのreviewed launch evidenceとして凍結した。旧v2/v3/v4と旧invalid formalはprior diagnosticとしてv0.3.36 formal v2から区別する。

### reviewed formal v2 の初期baseline

`development-baseline-20260916-v2`（fingerprint `d1708915e2de8bfd22d1d3c29eb10cc10f45926da4f0f2357dd87a0e407ab084`）は、固定development 570 occurrenceをTeacher/Sekireiで各1回解析し、1,140/1,140 attemptを完了した。strict validatorはmissing/extra/duplicate 0、technical failure 0を確認した。全attemptがdeadline内に完了し、cleanup/supervisorは正常、engine return codeは0である。Teacherは `exact_cp=266 / bound_cp=273 / mate=31`、Sekireiは `exact_cp=561 / mate=8 / no_score=1` だった。

固定Teacher-E 266点はSekireiでも266/266がexact cpで、各局のMAEは `374.286 / 1,522.514 / 1,081.561 / 1,222.175 / 1,234.696 cp`。正式headlineである5局等重み平均は `1,087.0461722818245 cp`、全266点を直接平均するmicro MAEは `1,136.8684210526317 cp`、最大絶対誤差は `32,941 cp` だった。bound/mate/no-scoreはこれらのcp指標へ混ぜていない。これは学習済みモデルのない駒得fallbackの初期baselineであり、十分な一致を示す値ではない。

all-evidence MはTeacherが evidence/positive/zero/missing/invalid=`570/569/1/0/0`、p50/p95/p99/max=`1,000,337.5/1,000,719/1,000,821.6/1,001,086`、Sekireiが `569/563/6/1/0`、`1,000,000/1,000,002/1,000,003/1,000,004`。`>C(1,000,000)` は両方0である。Sekireiのzero 6件はforced-single 5件とmate-in-one 1件、missing 1件はterminal checkmateのno-score、Teacherのzero 1件は同terminalで、いずれも事前登録した型契約に一致した。redactedな集計・SVG・JSON・hash manifestは [`docs/validation/development-baseline-2026-09-19`](validation/development-baseline-2026-09-19/validation.md) に固定する。raw logと局面別recordは外部runtimeだけに残し、final 5局は本測定・report/export経路で未アクセスである。

node evidence の厳密文法は、`go nodes <[0-9]+>`（ASCII 十進数字列、符号なし）から対応する `bestmove` までの各 structured `info` 行を対象にする。`info string` は free text として無視し、`pv`・`string`・`refutation`・`currline` の可変長 payload 内は読まない。payload 前の各行には `nodes <[0-9]+>` を高々一組だけ許容し、欠落・重複・負値・符号付き／非整数値は無効または曖昧として失敗させる。

最初の重み改善の前に、現在のSekirei（学習済みモデルなしの駒得fallback）と固定水匠11βを、独立したdevelopment 5局で比較できる実装を用意する。初期位置は採点対象にせず、各記録指し手後の570 occurrenceだけを使う。入力は常に `position startpos moves ...` とし、別棋譜の同一履歴をcacheしない。final 5局はこのrunnerの入力集合に含めない。development-05 p123 は終端詰みとして universe に残す。

USI supervisorのcleanup契約では、engineの終了コードと監督結果を分離する。監督は `waitid(WNOWAIT)` でengine exitを観測し、`/proc/<supervisor>/task/<supervisor>/children` の直接子をreap前に識別してpidfdを取得する。再親化で元のsession/PGIDを離れた子にもpidfd TERM/KILLを送り、全子をreapしてECHILDになり、入出力relayが閉じるまでanchorを破棄しない。能力・列挙・signal・reapの証明失敗、terminal status欠落、runnerによるanchor force-killはcleanup failureとしてraw outcomeに残し、成功扱いにしない。

`config/development-benchmark.json` は5局のCSA SHA-256、正規化CSA hash、benchmark manifest hash、分類manifest hash、全履歴を含むcanonical universe hash、手数合計570を固定する。分類manifestは forced single legal move 5件、`mate_in_one_available` 1件（development-05 p122）、terminal checkmate 1件を含み、cshogi 1.0.4で全570 occurrence・46,668合法root moveを照合する。`scripts/benchmark.py plan --run-type pilot` はcanonical 15局面に明示的な regression の development-04:77 と development-05:122 を加えた17局面を3 fresh-process repetitionで102 attempt / 34 engine-position tripleとする。pilot/formalの `max_reported_nodes` は named policy `one-sided-1-percent` v1 の `C(1,000,000)=1,010,000` を事前登録する。formalは同じ570局面を両エンジン1回ずつで1,140 attemptとする。formal実行には、同じruntimeの102-attempt pilotについて `pilot_run_id`、`pilot_fingerprint`、全attemptから再計算した `observed_max_reported_nodes`、policy limitを `formal.pilot_evidence` に凍結する。v0.3.36ではv5 evidenceを使ってformal v2を完了したが、v0.3.39の現行値は未凍結（null）である。未凍結・型不正・実測不一致のevidenceは引き続き停止条件とする。

両エンジンの正式optionは、Sekireiが `Threads=1, Hash=128, SearchMode=Speculative, SpecTopN=0, MultiPV=1, Ponder=false, UseBook=false`、教師が `Threads=1, USI_Hash=128, MultiPV=1, USI_Ponder=false, BookFile=no_book, FV_SCALE=28, OutputFailLHPV=false, PvInterval=0`。送信順もrunner証跡に固定し、`usi` / option広告 / `setoption` / `isready` / `usinewgame` / exact position / exact `go nodes 1000000` / matching bestmove / quitのraw lifecycleを検証する。教師の `EvalDir` は照合済み重みから導出し、モデル・重み・optionの不一致はfallbackせず失敗にする。pilotとformalは、runner/parser、分類・development hash、requested nodes、named node policy id/version/rate、timeout、thread環境、全engine設定とresolved option、binary/build/toolchain、teacher weight、candidate model、hostを含むcanonical execution identityを共有する。run type・repetition・pilot sample/full positionsはidentityから除外し、freeze時のconfig digestやrepo commit変更は再利用を無効化しない。runnerはLinuxのsession leader supervisor/subreaperを各attemptの所有anchorにし、engineをshellなしで直接argv execする。親engineが即時終了してもsupervisorは子孫をreapし、cleanup完了までPGIDを保持するため、runnerは終了後の子発見をscheduler pollに依存しない。

USI結果は最後のprimary（`multipv`なしまたは1）PVだけを採用し、最終行がboundなら以前のexactへ戻さない。cpの符号とbound方向は手番から先手視点へ変換するが、mate tokenは `+`、`-`、`-0` を含むlexical値のまま保存する。current-go の `go nodes ...` から対応する `bestmove` までを対象に、`info string` と `pv` payloadの境界より前にある structured `info` の全 `nodes` tokenを収集する。各structured info行の `nodes` は値組を高々一組だけ許容し、重複・欠落・負値・malformedは失敗にする。選択score行の値、最後の有効値、全有効値の最大 `M` は別フィールドに保存し、`M <= C(N)` をinclusiveに判定する。これはminimumではなく、source-derived guaranteeや内部仕事量の同値性でもない。非終端Sekireiの0 nodesはforced-singleまたはmate-in-oneのhash-bound例外だけを許容する。forced-singleは exact cp（`status=exact_cp`、`score_kind=cp`、`score_bound_stm=exact`、整合するcp値）に限り、正常lifecycle、全structured node evidenceが存在して全て0、reported/last/maxが0、in-check・合法手1・sole move一致、normal bestmoveとPV head一致を要求する。PV全体は一手に限定せず、mate-in-oneの一手PVとは区別する。mate-in-oneはさらにexact raw `score mate 1`、合法normal bestmove、PV一手、PV headとbestmove一致、凍結mating move一致を要求する。両方ともpositive/missing/invalid/mixed/duplicate/集計不整合は拒否する。bound cp・mate・no-score等をforced例外として救済しない。terminal checkmateは teacher の同一要求内 `score mate -1 nodes 0 pv resign` と `bestmove resign` の組だけを mate とし、裸の resign・stale score・win は no_score にする。`reported_nodes_at_score`、`last_reported_nodes`、`max_reported_nodes_evidence`、engine `time`、goからbestmoveまでのwall時間を別フィールドにする。teacherのexact cpだけをE、boundをB、mateをMとして候補MAEを計算し、E全点の候補exactが揃う5局のときだけheadlineを定義する。B/M/terminal/forced/mate-in-oneは別の型・coverage・node診断に留める。レポートのM分布はengine別にevidence/positive/zero/missing/invalid、p50/p95/p99/max、`>N`、`>C(N)`、最大positive overrun/rateを保持し、quantileはソート済み有限値の `(n-1)*p` 線形補間とする。

グラフは標準ライブラリの決定的SVGで5パネルを生成し、全エンジン・全棋譜で共通の対称線形raw sente-cp軸を使う。全exact観測には点markerを置き、線は同一repetition内の連続ply間だけを結ぶ。boundは矢印、mateは余白marker、no-score/failureはgap markerとして表す。pilotの採点はrep1を暗黙のbaselineにせず、各repetitionの採点結果とengine/position-type別の状態・coverage・node/time/wall分布を保持する。deadline後の遅延 score/bestmoveはrunnerがtimeoutとして凍結した場合に採点へ戻さない。実行結果とraw logは外部runtimeに残し、公開exportはaggregate Markdown・レビュー済みSVG・validation JSON・hash manifestの4ファイルだけを空ディレクトリ直下に作る。旧 `development-pilot-20260916` は旧 semantics の診断証跡として immutable に扱う。
