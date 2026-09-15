# 研究方針

## 目標

未学習棋譜の同じ局面・同じ指し手履歴について、水匠11β（Suisho Concerto 202512）と Sekirei をそれぞれ `go nodes 1000000` で探索させ、先手視点の評価値グラフを近づける。対局勝率や学習 loss は目標の代替にしない。

水匠11βを第1目標とした理由は、手元に同モデル・100万ノードと説明された生成済み教師データがあり、Mac miniで大量の教師探索を再生成せず学習へ進めるため。Suisho11Plusは参考教師として節目で確認するが、初期モデルの採否をPlusへの一致で決めない。

人間が目的・評価方法・計算予算・変更可能範囲を定め、AI が実測から次の改善仮説を選ぶ。特徴量・NNUE 構造の変更は検討対象。探索実装の変更は当面対象外。

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
| Sekirei | 固定 commit、初期 build は upstream の default NNUE 形式 |
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

## Issue #7 開発baselineの固定契約

node evidence の厳密文法は、`go nodes <[0-9]+>`（ASCII 十進数字列、符号なし）から対応する `bestmove` までの各 structured `info` 行を対象にする。`info string` は free text として無視し、`pv`・`string`・`refutation`・`currline` の可変長 payload 内は読まない。payload 前の各行には `nodes <[0-9]+>` を高々一組だけ許容し、欠落・重複・負値・符号付き／非整数値は無効または曖昧として失敗させる。

最初の重み改善の前に、現在のSekirei（学習済みモデルなしの駒得fallback）と固定水匠11βを、独立したdevelopment 5局で比較できる実装を用意する。初期位置は採点対象にせず、各記録指し手後の570 occurrenceだけを使う。入力は常に `position startpos moves ...` とし、別棋譜の同一履歴をcacheしない。final 5局はこのrunnerの入力集合に含めない。development-05 p123 は終端詰みとして universe に残す。

USI supervisorのcleanup契約では、engineの終了コードと監督結果を分離する。監督は `waitid(WNOWAIT)` でengine exitを観測し、`/proc/<supervisor>/task/<supervisor>/children` の直接子をreap前に識別してpidfdを取得する。再親化で元のsession/PGIDを離れた子にもpidfd TERM/KILLを送り、全子をreapしてECHILDになり、入出力relayが閉じるまでanchorを破棄しない。能力・列挙・signal・reapの証明失敗、terminal status欠落、runnerによるanchor force-killはcleanup failureとしてraw outcomeに残し、成功扱いにしない。

`config/development-benchmark.json` は5局のCSA SHA-256、正規化CSA hash、benchmark manifest hash、分類manifest hash、全履歴を含むcanonical universe hash、手数合計570を固定する。分類manifestは forced single legal move 5件（development-04 p106/p108、development-05 p115/p117/p121）と terminal checkmate 1件（development-05 p123）を含み、source hash と verifier version が変わると fail-closed になる。`scripts/benchmark.py plan --run-type pilot` の計画は1局あたり先頭・中央（切り上げ）・最終の3点、3 fresh-process repetitionで90 attempt。pilotには上限を置かず、`reported_nodes_at_score` と `last_reported_nodes` を全attemptで記録する。formalは同じ570局面を両エンジン1回ずつで1,140 attemptとする。formal実行には、同じruntimeの完全なpilotについて `pilot_run_id`、`pilot_fingerprint`、全90 attemptから再計算した `observed_max_reported_nodes`、採用する絶対整数 `max_reported_nodes` を `formal.pilot_evidence` に凍結する。`go nodes` は上限なので observed max が requested nodes 未満でも合法な早期完了として許容するが、各engineに少なくとも一つの正の node observation が必要で、全attemptの技術失敗や片側だけの正の証拠では gate を通さない。formalの上限や証跡がnull・型不正・実測不一致の間は停止する。

両エンジンの正式optionは、Sekireiが `Threads=1, Hash=128, SearchMode=Speculative, SpecTopN=0, MultiPV=1, Ponder=false, UseBook=false`、教師が `Threads=1, USI_Hash=128, MultiPV=1, USI_Ponder=false, BookFile=no_book, FV_SCALE=28, OutputFailLHPV=false, PvInterval=0`。送信順もrunner証跡に固定し、`usi` / option広告 / `setoption` / `isready` / `usinewgame` / exact position / exact go nodes / matching bestmove / quitのraw lifecycleを検証する。教師の `EvalDir` は照合済み重みから導出し、モデル・重み・optionの不一致はfallbackせず失敗にする。PvIntervalは最後の観測を増やすための明示設定であり、内部探索が最終値を計算したという主張ではない。pilotとformalは、runner/parser、development hash、requested nodes、timeout、thread環境、全engine設定とresolved option、binary/build/toolchain、teacher weight、candidate model、hostを含むcanonical execution identityを共有する。run type・repetition・pilot sample/full positions・post-pilot ceilingはidentityから除外し、freeze時のconfig digestやrepo commit変更は再利用を無効化しない。runnerはLinuxのsession leader supervisor/subreaperを各attemptの所有anchorにし、engineをshellなしで直接argv execする。親engineが即時終了してもsupervisorは子孫をreapし、cleanup完了までPGIDを保持するため、runnerは終了後の子発見をscheduler pollに依存しない。

USI結果は最後のprimary（`multipv`なしまたは1）PVだけを採用し、最終行がboundなら以前のexactへ戻さない。cpの符号とbound方向は手番から先手視点へ変換するが、mate tokenは `+`、`-`、`-0` を含むlexical値のまま保存する。current-go の `go nodes ...` から対応する `bestmove` までを対象に、`info string` と `pv` payload の境界より前にある structured `info` の全 `nodes` tokenを収集する。各 structured info 行の `nodes` は値組を高々一つとする厳密文法で、重複は曖昧として失敗にし、選択score行の値・最後の有効値・全有効値の最大を別フィールドに保存する。非終端の accepted score は node evidence が存在する整数 `>=0` を要求し、0 は固定分類上の Sekirei forced-single で sole move とPV/bestmoveが一致する場合だけ許容する。全有効値の最大を ceiling・pilot観測値に使い、負値・malformed・欠落は後続の有効値で救済しない。terminal checkmate は teacher の同一要求内 `score mate -1 nodes 0 pv resign` と `bestmove resign` の組だけを mate とし、裸の resign・stale score・win は no_score にする。`reported_nodes_at_score`、`last_reported_nodes`、`max_reported_nodes_evidence`、engine `time`、goからbestmoveまでのwall時間を別フィールドにする。teacherのexact cpだけをE、boundをB、mateをMとして候補MAEを計算し、E全点の候補exactが揃う5局のときだけheadlineを定義する。B/M/terminal/forced は別の型・coverage・node診断に留める。

グラフは標準ライブラリの決定的SVGで5パネルを生成し、全エンジン・全棋譜で共通の対称線形raw sente-cp軸を使う。全exact観測には点markerを置き、線は同一repetition内の連続ply間だけを結ぶ。boundは矢印、mateは余白marker、no-score/failureはgap markerとして表す。pilotの採点はrep1を暗黙のbaselineにせず、各repetitionの採点結果とengine/position-type別の状態・coverage・node/time/wall分布を保持する。deadline後の遅延 score/bestmoveはrunnerがtimeoutとして凍結した場合に採点へ戻さない。実行結果とraw logは外部runtimeに残し、公開exportはaggregate Markdown・レビュー済みSVG・validation JSON・hash manifestの4ファイルだけを空ディレクトリ直下に作る。旧 `development-pilot-20260916` は旧 semantics の診断証跡として immutable に扱う。
