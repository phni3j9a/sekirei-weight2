# 1週間の継続ウェイト改善（Issue #19）

## 期間と起点

運転期間は **2026-10-05 18:52:22〜2026-10-12 18:52:22 JST**（UTCでは09:52:22、168時間）。ユーザーの許可によりgoalを開始した。採用が一件得られた後も、残り期間に次の改善を進める。期間終了とモデル改善の成功を区別し、改善未達も記録する。

起点は[第8候補の採用モデル](validation/autonomous-weight-improvement-2026-10-04/best-model.json)、白視点・対非線形固定E3。正式MAEは907.6716211807516 cp、Top3入り率は59.951319613787%。モデルSHA-256は `f647864fa17a7e9d06ed44aed6894527128c721208bac1f761741c3c5ec1e042`。既存重みの実体を照合した。新しい採用モデルが得られるまではこのモデルを最良として保持する。

## 採用と継続

規定のdevelopment 5局・100万ノード正式比較を維持する。採用には、最新bestに対するMAEの厳密低下とTop3の非低下を同時に要求する。保存した各局の整数から5局等重みの有理数で判定し、表示丸めや学習lossを採用理由にしない。候補自身のMAE pilot・正式MAE・Top3 pilot・正式Top3を検証し、重み読込・入力集合・教師・比較identity・全attempt・USI lifecycleを確認する。

モデル更新のたびに比較基準を新bestへ進める。旧best、技術失敗、不採用候補の条件と成果物を保存する。既存fallback専用consumerをそのまま学習済みbestとの比較へ流用しない。探索実装・教師・比較設定は固定し、重み・特徴量・NNUE構造・学習方法・学習データを改善対象とする。final 5局を日々の仮説選択や局面採掘に使わない。

原典と今回の実測から一仮説ずつ選ぶ。不採用が続いたときはデータ・表現・量子化・学習目的等を診断し、単純なepoch/LRの再試行に偏らない。同じdevelopmentと既存holdoutの反復使用を、新しい棋譜への汎化の証明とは扱わない。

## 最初の準備と仮説

開始時の準備と最初の仮説は次のとおり。技術検証をモデル改善の成功に読み替えない。

- 前回の学習後親処理に残った未定義変数と、未実行のRust native fixtureの型不整合を、過去snapshotを保持した新しい実行経路で修復する。
- 現採用E3を明示的なincumbentとして、候補対候補の正式証拠を比較するconsumerを追加する。
- 教師packの各400ゲームprefixという抽出規則を見直す。最初はpack全体のゲーム境界から固定hash順位でゲームを選び、モデル構造・学習量を維持してデータの多様化を検証する案を準備する。固定holdout、独立1000局の全盤面除外、ゲーム分離、盤面重複除外を維持する。候補の選択規則は生成と正式比較前に固定する。

学習済みbestを明示する `scripts/compare_incumbent.py` を追加し、公開16 fixtureと保存済みE3の4段階rawの再検証に成功した。E3対E3の自己比較は8 identityが一致し、入力前後不変、MAE `1446737797/1593900`、Top3 `321208691/535782520` を再確認した。同一モデルなのでadopt=falseであり、新しい改善や採用ではない。最初のconsumer実機確認はcanonical occurrence IDと教師metric keyの表記差で失敗し、回帰fixtureと対応付けを修正した。失敗原本を保持し、新しいv2出力で成功を確認した。[公開再検証集計](validation/weekly-weight-improvement-2026-10-05/incumbent-revalidation.json)にはrawやローカルパスを含めない。

教師データの抽出は実機で完了した。最初に固定した各pack500局は、1個のpackが全392局のため31.87秒で失敗し、学習を開始しなかった。全13packの境界索引を確認し、12個各500局・短い1個392局、計6,392局を明記した新profileと事前登録v3を生成前に固定した。選択数を暗黙で縮めず、全局数・選択数・pack集合・source closureを検証するv2 producerへ改めた。

再生成は53.85秒で成功し、112,681学習局面とbyte一致の5,895 holdout局面を維持した。実際に学習行へ使ったゲームは4,537局、旧学習盤面との重複は14,782局面。手数と評価値帯の分布は旧データと近かった。以下の正式比較では改善に至らず、今回の選択規則と学習条件による多様化仮説は不採用となった。[公開データ準備集計](validation/weekly-weight-improvement-2026-10-05/diverse-data-preparation.json)は失敗と成功のreceipt SHAを含み、生局面・ラベルを含めない。

固定upstreamと旧complete patchを全postimageで確認し、専用trainer sourceを新runtimeへ展開した。新Readerは実manifestのmetadata検査を通過した。数値更新・scalar Adam・export・initialized I/O本体と旧snapshotを維持し、Python親にはgateを明示注入する。新親の小実プロセスfixtureで正常終了、exit失敗、timeout、SIGTERM時のwait/reapとgroup停止を確認した。長時間学習の前に、この候補自身の読込・初期状態保存/読込とFTZ未設定時のI/O前拒否を実行した。

専用trainerのCargo testは33件（failure/ignoreとも0）、buildは成功した。test 43.39秒・build 40.69秒、両childのwait/reapとgroup空、538 source・compiler2・dependency2,268件の前後不変を確認した。[公開build集計](validation/weekly-weight-improvement-2026-10-05/trainer-build.json)を残した。別agentの読取レビューでも、旧snapshotのnumeric update/Adam/native/export/floatと固定white coreのbyte維持を確認した。このbuild集計は学習完了やモデル採用の証拠と分けて扱う。

新しい入力証明の実作者runは59.66秒で成功した。全packの境界と選択順位、固定holdoutの全byte、独立1000局の全盤面除外、完全label join、reference03の独立再構成、4,435件の入力、source/compiler/dependency集合を再照合した。[公開preflight集計](validation/weekly-weight-improvement-2026-10-05/preflight.json)を残した。選択ゲームの合法手再生は固定producer側で行い、このconsumerはraw境界・順位・入力と出力のprovenanceを再確認する。preflight公開fixture13件と、launch親の公開fixture9件、全体549件が通過した。独立レビューで見つかったlaunch時のprocess読取不能の見逃しを修正し、同じallowlist検査を起動前と完了後にも維持する。

候補自身の実input routeと初期状態codec検査を通過し、3エポックCPU学習を開始した。FTZ/DAZ未設定の負例はI/O前にexit=1で拒否し、正例では全state bits、nearest03全byte、compiled native readerの読込一致、child wait/reapとgroup停止を確認した。[公開学習開始集計](validation/weekly-weight-improvement-2026-10-05/training-start.json)を記録した。実fit childの起動と実command一致を確認した。開始時の記録と、以下の学習完了・技術検証の記録を区別する。開始時の段階では候補自身の4段階比較は未完了だった。現在の結果と採否は以下に記録する。

正式評価helperはruntime・base config・モデルSHAを明示できるようにした。固定development 5局と整数100万ノードを要求し、候補自身のpilotを用いる。各段階の前後の実重みと、MAE/Top3の実manifestに記録された全4件のモデルidentityを同じpinへ照合する。公開7 fixtureと、既存bestの保存済み4件の実manifestを使った読取smokeが通過した。白視点runtimeを省略せず指定して使用する。

学習後の週次candidate専用proof helperも準備し、公開18 fixtureと独立読取レビューを通過した。[公開proof準備集計](validation/weekly-weight-improvement-2026-10-05/proof-preparation.json)に予定の全118,591 core行と8,185 incremental観測を記録する。レビューで固定White manifestの削除済み旧公開worktree参照8件を見つけ、元manifest/identityを保持したまま、両build endpointと同じsize/SHAの現公開sourceへ明示的に束縛するreceiptを追加した。private入力・compiler・依存・logにはこの対応を適用しない。元/current各1,161件の入力集合と対応8件、欠落0件を独立確認した。この準備集計は新candidateでの全量compile/probeを含まない。学習後の実機結果は[公開技術検証集計](validation/weekly-weight-improvement-2026-10-05/technical-proof.json)に別に記録し、fixture成功と区別する。初期pin検査で拒否した既存directoryを変更せず、新規作成した出力だけへ失敗記録を保存する条件も実CLIのfixtureで確認した。

学習・固定holdoutの静的誤差診断helperを用意し、公開12 fixtureが通過した。[公開診断準備集計](validation/weekly-weight-improvement-2026-10-05/static-residual-preparation.json)に固定集計規則とsource SHAを残す。候補自身の全量proof成功後に、その保存済みcore出力を教師ラベルへ結び、pack・手数帯・教師評価値帯・旧学習盤面との重複・手番別の誤差と、train二乗誤差への上位1／5／10／20%の寄与を集計する。新たな探索やdevelopment/finalの参照は行わず、診断結果は次の仮説選定に使う。実データの診断も完了し、下記に集計を記録した。採用判定は正式比較で行う。

## 第1候補の学習完了と技術検証

第1候補のfresh seed42・固定3epoch学習は、各112,681局面、計338,043更新を完了した。実wall timeは13,189.706秒（約3時間40分）で、childはreturncode=0、wait/reap完了、group空を2回確認した。resume・shuffleは使わず、入力・source・buildの前後不変も確認した。[公開学習完了集計](validation/weekly-weight-improvement-2026-10-05/training-completion.json)にcompletionと出力のsize/SHAを残す。nearest03重みは1,305,356 bytes、SHA-256 `44d1b412da22688542a406599dd40b40bc3243285aaaeae67b21d003db9a0429`。重み本体と完全master状態は非公開に保持する。

浮動小数点制御は、各epochの開始・完了の計6回とexport時の1回で、FTZ/DAZ・MXCSR control `0x9fc0`を確認した。これは338,043更新すべてで制御wordを個別観測したという主張ではない。

候補自身の全量技術検証も完了した。coreはtrain 112,681行・固定holdout 5,895行・自作技術fixture 15行、計118,591行を検証した。保存nativeを復号したfloat forwardと同じnativeの整数coreとのbridgeは全対象で1.001 cp未満であり、raw master floatからnearest03への量子化差を測った結果とは区別する。incrementalは8,185観測でrefresh error=0、accumulatorとfresh refreshの一致、undo後のparent復元一致を確認した。compile 2件・core 3件・incremental 1件の計6 childはすべてreturncode=0、wait/reap完了、group空を2回確認した。[公開技術検証集計](validation/weekly-weight-improvement-2026-10-05/technical-proof.json)には件数と検証範囲だけを投影する。全局面での整数bit単位の色交換共変性や、全board FTの保存を新たに主張しない。

候補自身の100万ノード・4段階比較は **2026-10-06 00:05:20 JST（10-05 15:05:20 UTC）** に開始し、**01:27:08 JST**に全段階を完了した。最新bestの登録を比較直前に凍結し、全4段階のraw再検証、8項目の比較identity一致、入力とbest登録の前後不変を確認した。正式比較は有効で、**第1候補は不採用**。現bestを維持する。

| 指標 | 現best | 多様化データE3 |
| --- | ---: | ---: |
| 5局等重みMAE（cp） | 907.671621 | 966.372955 |
| 5局等重みTop3入り率 | 59.951320% | 57.054274% |

MAEは`513433951/531300`、Top3は`114632561/200918445`。MAEは58.701334 cp悪化し、Top3は2.897045ポイント低下した。[正式比較・各局集計・評価値グラフ](validation/weekly-weight-improvement-2026-10-05/diverse-games-e3/comparison.md)を保存する。データの多様化一般ではなく、今回の固定選択規則・構造・更新量の組合せの結果とする。

初回の親処理は通常Pythonで起動したため、Top3 pilot前に固定decoder依存の欠落で停止した。候補自身の完了済みMAE rawを再検証し、固定audit環境でTop3 pilot・formalを新規実行した。最新best照合の補助処理でも、表示用floatを含む辞書と整数だけの登録形式の違いで停止し、別記録で分子・分母と有理数の同値、登録前後不変を再検証した。元の失敗記録を保持し、成功へ書き換えていない。正式比較の生データを再取得・変更した結果ではない。

保存済みnative core出力を使った[静的誤差診断](validation/weekly-weight-improvement-2026-10-05/diverse-games-e3/static-residuals.json)も完了した。trainの静的MAEは607.573806 cp、固定holdoutは651.689737 cp。train残差の上位1%（1,127行）が二乗誤差の46.233841%、上位5%（5,635行）が67.257528%を占めた。教師の絶対評価値10,000 cp以上の155行（0.137556%）は二乗誤差の31.4715%を占める。これは学習完了後のnative静的残差であり、各更新時のmaster lossや勾配を個別測定した結果ではない。次は大きな残差への感度を抑える学習目的を検討する。新たな探索・development/final参照は行わなかった。

第1候補の完了記録14,994 regular files・803 directories・2,325,993,562 bytesをNASへ保存し、独立verifyと新しいSSD領域への物理復元で集合・mode・size・SHAを照合した。復元コピーからRustとaudit venvの2,473 files・1,027,715,260 bytesを再構成し、4リンクの再作成と5件の実動作確認を通過した。Pythonの実import先も新しいvenvだった。同じplatformの固定OS・Python stdlib・loader/shared libraries・linkerを前提とする復元であり、OS全体を独立復元したという主張ではない。

コピーした8個の元parserだけで7,371 inputs・180,568,115 bytesを再集計し、8比較identity・入力前後不変・整数集計・不採用判定が元receiptと一致した。live原本へのfallback、engine起動、採用反映は行わなかった。さらに元source binding、元compiled readerの初期step 0保存/読込、元8段階raw比較を実行し、元参照パス、最新best revision 0、exact metrics、入力/source不変、lock解放を再確認した。学習更新は0。この確認からtrained masterの量子化差を推定しない。[保存・復元集計](validation/weekly-weight-improvement-2026-10-05/diverse-games-e3/archive-recovery.json)に件数とreceipt SHAを残す。

復元環境の初版はリンクの生成順で停止した。全リンクを作ってから検査する別sourceを用意し、実4リンクの8回帰fixtureと独立レビューを経て成功した。再集計コマンドのSHA誤記、元reader補助の4-key metadata / 3-key identity形式差による停止も別記録で保持した。後者はpath・bytes・SHAの厳密一致と実modeを維持する修正を31 fixtureで確認した。修正controller 7 files・54,670 bytesと、失敗/成功・実child出力等71 files・30,918,032 bytesも別NAS補足として照合保存した。base archiveと原本は上書きせず、SSD重複の削除はまだ行っていない。

評価helperはdecoder依存を出力作成・最初のMAE実行の前に検査するよう修正した。固定audit venvの11 fixture、構文、差分、CLI helpが通過した。第1候補の凍結済みprivate script/config/weights/runtime controlは保持した。

## 第2候補: Huber1000

第2候補は同じ多様化データ、fresh seed42、3epoch、構造、scalar Adam、LR、保護パラメータ、export、探索条件を固定し、損失だけをδ=1,000 cpのHuberへ変えた。実装は[標準Huber](https://docs.pytorch.org/docs/2.14/generated/torch.nn.HuberLoss.html)の2倍とし、絶対誤差が1,000 cp以下では旧MSEのloss・gradientと演算順を維持する。大きい残差域で二乗増加を線形増加に変える仮説で、ラベルやデータは変更しない。δは今回の事前選択値である。

候補自身のfresh seed42・固定3epoch学習は112,681局面×3、338,043更新、13,466.234221秒で完了した。resume・shuffleなし、入力/source/buildの前後不変、FTZ/DAZ・MXCSRのepoch境界6回とexport1回を確認した。候補自身のcore 118,591行とincremental 8,185観測も完了し、compile2・core3・incremental1のchildは実exit0、wait/reap、group空の2回確認を通過した。bridgeの対象は保存nearest03を復号したfloatと同じnative整数coreであり、raw masterの量子化差や全boardの整数bit共変性を証明したという主張ではない。

候補自身のMAE pilot・正式MAE・Top3 pilot・正式Top3を102 / 1,140 / 36 / 551 attemptで完了した。全4段階のHuber重みidentity、両候補のraw再検証、8比較identity、最新best snapshotとのexact fraction、登録前後不変を確認した。正式比較は有効で、**第2候補も不採用**。

| 指標 | 現best | Huber1000 E3 |
| --- | ---: | ---: |
| 5局等重みMAE（cp） | 907.671621 | 1022.504590 |
| 5局等重みTop3入り率 | 59.951320% | 58.590726% |

候補のMAEは `1086513377/1062600`、Top3は `941756597/1607347560`。MAEは114.832969 cp悪化し、Top3は1.360594ポイント低下した。[正式比較・学習/proof/実親終了集計・評価値グラフ](validation/weekly-weight-improvement-2026-10-05/huber1000-e3/comparison.md)を保存する。今回の固定データ・構造・更新量・δによる結果であり、Huber一般の有効性や未評価棋譜への汎化を結論しない。現best revision 0を維持し、final5局は未使用。

[Huber専用sourceと補助CLI](../preparations/white-view-paired-nonlinear-huber1000-v1/README.md)の準備時manifestはsource-only記録として保持する。今回の学習・技術検証・正式比較は別receiptに記録した。Huber候補のNAS保存は15,451ファイル・2,486,486,375 bytes、902 directory・419 rootを照合し、独立verifyと2回目のSSD物理復元まで完了した。コピー環境は2,473 regular files・1,027,715,260 bytesと4 symlinkを検証し、Rust identity／sysroot／std link／runとPython cshogi・NumPyの5 smokeが通過した。コピーだけからの再集計は7,371ファイル・180,547,431 bytes、8 identityと正式receiptを再現してrejectを確認した。元Huber数値reader v4も7,416 logical files・19 source roleを読み、専用raw proof consumerと入力前後不変、コピー由来decoder、blocked attempt／boundary denial各0を確認した。コピーvenvのbefore/after対象は2,259ファイル・120,770,606 bytes・269 directory・4 link。実親はexit0、wait/reap・ECHILD・子/group空の2回確認と26 pinsの前後一致を通過した。

初回SSD復元は1ファイルの1バイトSHA不一致で失敗した。原本とNASは固定SHAに一致し、その後の失敗stageの観測では内容が変化してNAS同値となったが原因は未確定。観測した不一致bytesは記録した1バイト差から明示的に再構成して保管し、元の失敗stageの不変コピーとは扱わない。reader v3はclosure receiptの4-key metadata（modeを含む）と3-key identityの比較で数値処理前に失敗した。v4はuniqueな復元manifest entryへmodeを束縛するbootstrapだけを修正し、数値・cache・境界関数と旧62 fixtureのASTを維持した74 fixture、および実数値consumerを通過した。失敗記録・v4修復・完了記録126ファイル・11,984,951 bytes・26 directory metadataを別の補足NAS packetへ保存し、base archiveを変更せずSSD原本を保持した。[保存・復元集計](validation/weekly-weight-improvement-2026-10-05/huber1000-e3/archive-recovery.json)を参照。復元は同じplatformの固定OS・Python stdlib・loader/shared libraries・linkerを前提とし、OS全体の独立復元を主張しない。

## 計算と保存

現在のMac mini CPU・32 GiB RAM・既存SSD/NASだけを使う。解析jobs=1、Threads=1、build jobs=2、重い実験は直列。開始時SSD空き約27 GiB、NAS空き約5.9 TiB。初期の追加SSD作業領域は8 GiBを目安とし、各候補前に空き容量と時間を再確認する。

前回の非線形3epoch学習は、保存済みchild outcomeとcompletionの一致するwall timeで13,200.321秒（約3時間40分）。今回第1候補の実測は13,189.706秒だった。旧sparse trainerの時間は流用しない。正式4段階の過去実測は約66〜72分。今回は起動依存の修復と待機を含め、開始から全段階完了まで81分48秒だった。MAE/Top3の有効raw測定と失敗・復旧を分けて保存する。

Git/worktree・build・venv・使用中データと重み・実行中出力はSSDに置く。今回新規の完了runは、停止、ファイル集合・size・SHA-256一致、参照依存の復元可能性、元パスからの参照を確認して `/mnt/storage/NAS/sekirei-weight2` に保管する。確認済みの今回のSSD重複は整理できる。過去原本と既存runの参照を保持し、単なるsymlink置換でvalidatorが通るとは仮定しない。詳細receipt、モデル、教師、生局面・ラベル・ログは非公開に保持する。

NASの小fixture保存は初回のpublishで失敗した。実mergerfsが `renameat2(RENAME_NOREPLACE)` をEINVALで拒否したため、SSD原本とNAS stageを保持し、archive成功として扱わなかった。非対応filesystemの場合だけ、既存先を上書きしない排他mkdir・全集合物理コピー・manifest最終公開へ切り替える方式を追加した。この方式はdirectory全体のatomic renameではなく、完全なmanifest SHAと全集合の照合を成功条件とし、コピー元stageも保持する。

新しい実機v2試験では44 bytesの1ファイルと空directoryをNASへ保存し、独立したverifyとSSDへの物理復元を通過した。集合・mode・size・SHA一致と、SSD復元inodeが原本と異なることを確認した。公開15 fixtureでは破損、集合差、特殊file、出力競合、途中コピー失敗などの拒否も確認した。[公開NAS疎通集計](validation/weekly-weight-improvement-2026-10-05/nas-copy-smoke.json)を残す。これは小fixtureの保存・復元確認である。第1候補の実験記録と依存環境については上記の実検証を完了した。SSD原本と過去資料の削除は行っていない。

## 記録と終了

Issue [#19](https://github.com/phni3j9a/sekirei-weight2/issues/19)を起点に専用worktreeを用いる。意味のある成果単位でcommit/push・PR作成または更新を行い、READMEを現状へ更新する。モデル採用とGit統合を区別する。PRマージと定期実行は今回の許可に含まれない。

期限前は、新しい長時間処理を控えて検証・保存・停止確認・報告の時間を確保する。期末に最新bestと更新履歴、不採用と技術失敗、残事項、未完了処理、再開状態を残す。利用上限、容量や技術障害が生じた場合も状態を保存し、期間内に達成した範囲を報告する。
