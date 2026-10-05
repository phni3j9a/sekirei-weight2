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

再生成は53.85秒で成功し、112,681学習局面とbyte一致の5,895 holdout局面を維持した。実際に学習行へ使ったゲームは4,537局、旧学習盤面との重複は14,782局面。手数と評価値帯の分布は旧データと近く、pack内prefix限定による偏りの有無や改善効果は未検証である。[公開データ準備集計](validation/weekly-weight-improvement-2026-10-05/diverse-data-preparation.json)は失敗と成功のreceipt SHAを含み、生局面・ラベルを含めない。

固定upstreamと旧complete patchを全postimageで確認し、専用trainer sourceを新runtimeへ展開した。新Readerは実manifestのmetadata検査を通過した。数値更新・scalar Adam・export・initialized I/O本体と旧snapshotを維持し、Python親にはgateを明示注入する。新親の小実プロセスfixtureで正常終了、exit失敗、timeout、SIGTERM時のwait/reapとgroup停止を確認した。長時間学習の前に、この候補自身の読込・初期状態保存/読込とFTZ未設定時のI/O前拒否を実行した。

専用trainerのCargo testは33件（failure/ignoreとも0）、buildは成功した。test 43.39秒・build 40.69秒、両childのwait/reapとgroup空、538 source・compiler2・dependency2,268件の前後不変を確認した。[公開build集計](validation/weekly-weight-improvement-2026-10-05/trainer-build.json)を残した。別agentの読取レビューでも、旧snapshotのnumeric update/Adam/native/export/floatと固定white coreのbyte維持を確認した。このbuild集計は学習完了やモデル採用の証拠と分けて扱う。

新しい入力証明の実作者runは59.66秒で成功した。全packの境界と選択順位、固定holdoutの全byte、独立1000局の全盤面除外、完全label join、reference03の独立再構成、4,435件の入力、source/compiler/dependency集合を再照合した。[公開preflight集計](validation/weekly-weight-improvement-2026-10-05/preflight.json)を残した。選択ゲームの合法手再生は固定producer側で行い、このconsumerはraw境界・順位・入力と出力のprovenanceを再確認する。preflight公開fixture13件と、launch親の公開fixture9件、全体549件が通過した。独立レビューで見つかったlaunch時のprocess読取不能の見逃しを修正し、同じallowlist検査を起動前と完了後にも維持する。

候補自身の実input routeと初期状態codec検査を通過し、3エポックCPU学習を開始した。FTZ/DAZ未設定の負例はI/O前にexit=1で拒否し、正例では全state bits、nearest03全byte、compiled native readerの読込一致、child wait/reapとgroup停止を確認した。[公開学習開始集計](validation/weekly-weight-improvement-2026-10-05/training-start.json)を記録した。実fit childの起動と実command一致を確認した。開始時の記録と、以下の学習完了・技術検証の記録を区別する。候補自身の4段階比較とモデル採用はまだ未完了である。

正式評価helperはruntime・base config・モデルSHAを明示できるようにした。固定development 5局と整数100万ノードを要求し、候補自身のpilotを用いる。各段階の前後の実重みと、MAE/Top3の実manifestに記録された全4件のモデルidentityを同じpinへ照合する。公開7 fixtureと、既存bestの保存済み4件の実manifestを使った読取smokeが通過した。白視点runtimeを省略せず指定して使用する。

学習後の週次candidate専用proof helperも準備し、公開18 fixtureと独立読取レビューを通過した。[公開proof準備集計](validation/weekly-weight-improvement-2026-10-05/proof-preparation.json)に予定の全118,591 core行と8,185 incremental観測を記録する。レビューで固定White manifestの削除済み旧公開worktree参照8件を見つけ、元manifest/identityを保持したまま、両build endpointと同じsize/SHAの現公開sourceへ明示的に束縛するreceiptを追加した。private入力・compiler・依存・logにはこの対応を適用しない。元/current各1,161件の入力集合と対応8件、欠落0件を独立確認した。この準備集計は新candidateでの全量compile/probeを含まない。学習後の実機結果は[公開技術検証集計](validation/weekly-weight-improvement-2026-10-05/technical-proof.json)に別に記録し、fixture成功と区別する。初期pin検査で拒否した既存directoryを変更せず、新規作成した出力だけへ失敗記録を保存する条件も実CLIのfixtureで確認した。

学習・固定holdoutの静的誤差診断helperを用意し、公開12 fixtureが通過した。[公開診断準備集計](validation/weekly-weight-improvement-2026-10-05/static-residual-preparation.json)に固定集計規則とsource SHAを残す。候補自身の全量proof成功後に、その保存済みcore出力を教師ラベルへ結び、pack・手数帯・教師評価値帯・旧学習盤面との重複・手番別の誤差と、train二乗誤差への上位1／5／10／20%の寄与を集計する。新たな探索やdevelopment/finalの参照は行わず、診断結果は次の仮説選定に使う。実データの診断はまだ実行しておらず、採用判定は正式比較で行う。

## 第1候補の学習完了と技術検証

第1候補のfresh seed42・固定3epoch学習は、各112,681局面、計338,043更新を完了した。実wall timeは13,189.706秒（約3時間40分）で、childはreturncode=0、wait/reap完了、group空を2回確認した。resume・shuffleは使わず、入力・source・buildの前後不変も確認した。[公開学習完了集計](validation/weekly-weight-improvement-2026-10-05/training-completion.json)にcompletionと出力のsize/SHAを残す。nearest03重みは1,305,356 bytes、SHA-256 `44d1b412da22688542a406599dd40b40bc3243285aaaeae67b21d003db9a0429`。重み本体と完全master状態は非公開に保持する。

浮動小数点制御は、各epochの開始・完了の計6回とexport時の1回で、FTZ/DAZ・MXCSR control `0x9fc0`を確認した。これは338,043更新すべてで制御wordを個別観測したという主張ではない。

候補自身の全量技術検証も完了した。coreはtrain 112,681行・固定holdout 5,895行・自作技術fixture 15行、計118,591行を検証した。保存nativeを復号したfloat forwardと同じnativeの整数coreとのbridgeは全対象で1.001 cp未満であり、raw master floatからnearest03への量子化差を測った結果とは区別する。incrementalは8,185観測でrefresh error=0、accumulatorとfresh refreshの一致、undo後のparent復元一致を確認した。compile 2件・core 3件・incremental 1件の計6 childはすべてreturncode=0、wait/reap完了、group空を2回確認した。[公開技術検証集計](validation/weekly-weight-improvement-2026-10-05/technical-proof.json)には件数と検証範囲だけを投影する。全局面での整数bit単位の色交換共変性や、全board FTの保存を新たに主張しない。

候補自身の100万ノード・4段階比較は **2026-10-06 00:05:20 JST（10-05 15:05:20 UTC）** に開始した。正式比較の結果と採用判断、実データの静的誤差診断、実candidateのNAS保管はまだ未完了である。学習と技術検証の成功をモデル改善へ読み替えず、採用条件を満たす新結果が得られるまで起点のbestを維持する。final 5局は未使用。

## 計算と保存

現在のMac mini CPU・32 GiB RAM・既存SSD/NASだけを使う。解析jobs=1、Threads=1、build jobs=2、重い実験は直列。開始時SSD空き約27 GiB、NAS空き約5.9 TiB。初期の追加SSD作業領域は8 GiBを目安とし、各候補前に空き容量と時間を再確認する。

前回の非線形3epoch学習は、保存済みchild outcomeとcompletionの一致するwall timeで13,200.321秒（約3時間40分）。今回第1候補の実測は13,189.706秒だった。旧sparse trainerの時間は流用しない。正式4段階の過去実測は約66〜72分で、今回の完了時間はまだ未確定である。

Git/worktree・build・venv・使用中データと重み・実行中出力はSSDに置く。今回新規の完了runは、停止、ファイル集合・size・SHA-256一致、参照依存の復元可能性、元パスからの参照を確認して `/mnt/storage/NAS/sekirei-weight2` に保管する。確認済みの今回のSSD重複は整理できる。過去原本と既存runの参照を保持し、単なるsymlink置換でvalidatorが通るとは仮定しない。詳細receipt、モデル、教師、生局面・ラベル・ログは非公開に保持する。

NASの小fixture保存は初回のpublishで失敗した。実mergerfsが `renameat2(RENAME_NOREPLACE)` をEINVALで拒否したため、SSD原本とNAS stageを保持し、archive成功として扱わなかった。非対応filesystemの場合だけ、既存先を上書きしない排他mkdir・全集合物理コピー・manifest最終公開へ切り替える方式を追加した。この方式はdirectory全体のatomic renameではなく、完全なmanifest SHAと全集合の照合を成功条件とし、コピー元stageも保持する。

新しい実機v2試験では44 bytesの1ファイルと空directoryをNASへ保存し、独立したverifyとSSDへの物理復元を通過した。集合・mode・size・SHA一致と、SSD復元inodeが原本と異なることを確認した。公開15 fixtureでは破損、集合差、特殊file、出力競合、途中コピー失敗などの拒否も確認した。[公開NAS疎通集計](validation/weekly-weight-improvement-2026-10-05/nas-copy-smoke.json)を残す。これは小fixtureの保存・復元確認であり、実験run全体や依存環境の復元検証は今後の完了runで行う。SSD原本と過去資料の削除は行っていない。

## 記録と終了

Issue [#19](https://github.com/phni3j9a/sekirei-weight2/issues/19)を起点に専用worktreeを用いる。意味のある成果単位でcommit/push・PR作成または更新を行い、READMEを現状へ更新する。モデル採用とGit統合を区別する。PRマージと定期実行は今回の許可に含まれない。

期限前は、新しい長時間処理を控えて検証・保存・停止確認・報告の時間を確保する。期末に最新bestと更新履歴、不採用と技術失敗、残事項、未完了処理、再開状態を残す。利用上限、容量や技術障害が生じた場合も状態を保存し、期間内に達成した範囲を報告する。
