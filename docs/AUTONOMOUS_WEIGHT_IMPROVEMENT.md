# 採用可能なウェイトを得るまでの自律改善（Issue #17）

## 目標と現在地

ユーザーの依頼により、採用可能な評価モデルが得られるまでgoalで改善を進める。数日という見込みを達成条件にはせず、規定のdevelopment 5局・100万ノード正式比較で現行最良よりMAEが下がり、Top3入り率が下がらないことを要求する。最初の比較基準は固定v0.3.39のmaterial fallbackで、MAEは1084.4786006022964 cp、Top3は0.5538875742592971。表示上の丸め値ではなく保存済み正式結果の有理数で採否を照合する。

Issue #15 / PR #16の実装と実験記録はマージ済みで、今回の専用worktreeは最新mainから開始した。実験コードの統合とモデル採用を区別し、今回のPRは別途許可があるまでマージしない。

## 最初の候補

- 112,681局面・constant LR=0.0001・3エポック・駒得初期化・fresh Adam・seed42。
- 前回の事前選択規則で選択済みのepoch3を、FT最近傍丸めでexportした重みをそのまま使用する。
- 重みSHA-256: `81b1c80a884349fbe8c32d74053e1b3d70a69a191611d3a3d9d2ee0febd1305f`。
- 静的holdout MAE 644.599 cpは診断値で、正式探索後のMAEや採用成功を示さない。
- 同stemのmetadataは`nnue_output=absolute`。過去の学習・診断・export・比較runは書き換えない。

候補自身のMAE pilot（17局面×両エンジン×3反復）を検証・凍結してからMAE formal（570局面×両エンジン）を行う。その同一重みのMAE formalを参照してTop3 pilot（12局面×3反復）、Top3 formal（551局面）を実施する。weight identity、明示的読込成功、全attempt、型付き結果、node evidence、USI lifecycle、cleanup、固定Teacher-E 266点とTop3対象を既存validatorで照合する。

直近の1候補全体の実測は約69〜72分。今回の所要時間は記録して見積りを更新する。既存のfallback正式runは読み取り再検証して比較基準にするが、fallbackや別候補のpilotを今回の候補のformal gateへ流用しない。

## 継続方針と資源

正式結果を得てから次の仮説を一つ選ぶ。学習量、量子化、教師データ、学習目的、表現力のどれが制約かを今回の実測と原典から調べる。単純な学習延長やdevelopmentの特定局面に合わせる調整を反復しない。候補生成・静的選択規則は正式比較前に記録する。固定holdoutは既に候補選択に使われた検証集合であり、新しい汎化の証明として扱わない。final 5局は日々の仮説選択・局面採掘に使わない。

最初の112k epoch3最近傍候補の正式比較が有効に完了し、採用条件を満たさなかった場合に限り、既存の112k・13駒価値候補ridge=1を次に評価する。最初の候補が採用された場合は評価しない。データ・特徴・prior・solver・制約・丸めを維持し、既評価のridge=0.01から正則化だけを強める。正式結果を見てlambdaを選び直さず、候補自身のpilot・正式MAE・Top3で判断する。技術的にinvalidな測定の場合はこの分岐へ進まず、証拠の問題を解消する。

次候補のSHA-256は`db1e4c4c910194709a3b7edbbb601fa1c0962de9314f7f6c7b39e8fd0ad1672e`。前回の生成・core一致記録と実体hashを再確認した。元の13係数からのL2距離は851.140→297.610 cpと小さくなる一方、固定holdout MAEは662.717→719.088 cpへ悪化する。これは静的誤差を犠牲にして駒得保持を強めたとき探索Top3が回復するかを調べる新しい仮説であり、前回のholdout選択規則の書換えではない。位置的な優劣を表せないモデルの限界と、同じ5局での反復選択による過適合を踏まえて一候補だけを試す。

探索実装・教師・100万ノード条件・二指標の採用基準を維持する。重み・特徴量・NNUE構造・学習方法・学習データは改善対象。現在のMac mini CPU・32 GiB RAM・既存SSD/NASだけを使い、重い実験は直列、解析jobs=1/Threads=1、build jobs=2とする。開始時SSD空き約27 GiB。初期の追加作業領域は8 GiB以内を目安にし、候補ごとに容量と時間見積りを確認する。既存原本を削除して容量を作らない。

今回のprivate campaignは`campaign-17-autonomous-v1`。前回の実行・比較補助を新campaignへコピーし、削除済みIssue #15 worktreeへの参照だけを今回のworktreeへ変更した。比較を担う既存6スクリプトの内容hashは前回の正式比較と同じで、固定cshogi 1.0.4 / NumPy 1.26.4のvenvで実行する。適応補助のsource・変更後hashと候補identityはprivate receiptに残す。途中停止時は既存の厳密resumeで欠落attemptだけを再開し、技術失敗・破損・identity不一致を削除や書換えで救済しない。

生局面・ラベル・weight・raw log・詳細receiptはGit外に保持する。完了成果物はSSD原本を維持し、ファイル集合・サイズ・SHA-256を照合して既存NASへ保管する。公開Git/Actionsにはコード・集計・条件・ハッシュだけを載せる。利用上限や技術的な障害は率直に報告し、未達のままgoal完了とは扱わない。

## 比較の実行経路

前回private運用に使った測定・厳密比較・公開化の補助を、それぞれ`scripts/evaluate_candidate.py`、`scripts/compare_candidates.py`、`scripts/publish_comparison.py`へ移した。repository参照をスクリプト位置から解決し、公開Markdownから今回の自律改善文書へのリンクを加えた。既存の測定・比較・公開境界を保持し、今回の最初のrunは開始時に保存したprivateコピーを使う。実行中のsourceは差し替えない。

固定audit venvのPythonで、次の順に実行できる。`CAMPAIGN`と`WEIGHT`はそれぞれ新しいprivate保存先、候補重みの絶対パスとする。測定は既定の固定runtimeを使う。

```sh
python scripts/evaluate_candidate.py --weight "$WEIGHT" \
  --output "$CAMPAIGN/evaluation" --prefix development-candidate-unique-v1
python scripts/compare_candidates.py "$CAMPAIGN/evaluation" \
  --output-json "$CAMPAIGN/comparison.json"
python scripts/publish_comparison.py --comparison "$CAMPAIGN/comparison.json" \
  --title '候補の正式比較' --candidate-label '候補モデル' \
  --output "$CAMPAIGN/public"
```

測定コマンドは新しい出力先を要求し、採用は行わない。比較コマンドのexit 0は比較の有効性で、採用には`adopt=true`が必要。公開化はこのevaluation-directory経路に対応する。既定の比較基準は今回のfallback正式runであり、最良モデルが更新された後に追加比較する場合は`--baseline-evaluation`を明示し、公開化にも`--baseline-label`を渡す。途中再開は`benchmark.py resume`または`top3.py --resume`を使い、全体補助を同じ出力先へ再実行しない。

## 次の学習案の入力検証

ridge=1の結果待ちに、`scripts/functional_anchor.py`を準備した。元教師のcpと固定fallbackの駒得cpを1/2ずつ混ぜ、符号付き整数のnearest-evenで丸める候補案に対応する。元manifestと4ファイルのhash・size・count・source metadata、全SFEN/cache、train/holdoutの盤面・ゲーム分離を検証し、入力だけからcanonical specと派生target identityを導出する。train labelsは元行順・追加metadataを保ち、train positionsとholdout 2ファイルはbyte一致で保持する。

これはbyte入力を受け取る純粋moduleで、実データadapterや学習・診断wrapperを持たない。recipeの`real_generation_ready`は常にfalseで、source/initializer実体と全train core一致、holdout証拠のbinding、private output/lockなどのpreflightを別に要求する。元教師holdoutは元identityのまま保持し、既存診断器のguardも変更しない。14件の公開synthetic fixtureで、丸め・不正cache・hash/identity改変・既存split guardとの境界を確認した。ridge=1の有効な採用未達後、以下の固定学習案を事前登録した。実データの派生target生成・probe・学習・診断は未実施で、実体preflightを通してから進める。

## 実行状態

最初の候補は2026-10-03 04:12:16〜05:22:25 UTC、4,208.758秒（約70.1分）で全段階を完了した。MAE pilotは102/102・両engine17/17局面×3反復安定、正式MAEは1,140/1,140・Teacher-E 266/266、Top3 pilotは36/36・安定、正式Top3は551/551。全1,829 attemptで技術失敗0、期限内終了、supervisor/cleanup成功を確認した。最大報告1,001,086 nodesは規定上限1,010,000以内だった。

| 採用指標（5局等重み） | 現行fallback | 112k epoch3・FT最近傍 |
| --- | ---: | ---: |
| MAE | 1084.478601 cp | 949.416682 cp |
| Top3入り率 | 55.388757% | 54.067494% |

厳密比較は8項目のidentity一致、入力hash不変、developmentのみ使用を確認し、有効に完了した。MAEは約135.062 cp改善したが、Top3は約1.321ポイント低下した。採用条件は満たさず、最良モデルはfallbackを維持する。独立監査でも各局の保存値から有理数を再集計し、同じ採否を確認した。[正式比較・identity](validation/autonomous-weight-2026-10-03/expanded-e3-nearest/comparison.md)と[候補グラフ](validation/autonomous-weight-2026-10-03/expanded-e3-nearest/mae/validation.md)を公開した。これはdevelopmentでの比較であり、final評価や一般的な棋力改善を示さない。

有効な採用未達という事前分岐条件を満たしたため、次は上記のridge=1候補を評価する。今回の結果を見てlambdaを選び直すことはしない。完了候補のNAS保存と次候補の容量・identity確認を済ませてから、候補自身のpilotを開始する。goalは継続中。

完了候補のevaluation・4run・モデル/metadata・凍結したsource/receipt/publication snapshotを既存NASへ保存し、source-before/source-after/destinationの集合・サイズ・SHA-256、directory集合を照合した。3,726 files・30 directories・0 symlinks・33,657,398 bytesで一致し、SSD原本は保持した。[保存集計](validation/autonomous-weight-2026-10-03/expanded-e3-nearest/archive.json)だけを公開する。このコピーは固定runtime・audit venv・fallback比較・前回の学習archiveへの依存を持ち、単独で全環境を復元できるものではない。

ridge=1は候補自身のMAE pilotを2026-10-03 05:36:27 UTCに開始し、05:40:19に正式MAE、06:29:58にTop3 pilot、06:31:03に正式Top3へ進み、06:46:56に全段階を完了した。4,229.379秒（約70.5分）。MAE pilot 102/102・両engine17局面×3反復安定、正式MAE 1,140/1,140・Teacher-E 266/266、Top3 pilot 36/36・12局面×3反復安定、正式Top3 551/551。全1,829 attemptのraw SHA・lifecycle・options・cleanup・timeoutを独立監査し、技術失敗0、正式の最大報告はSekirei 1,000,004 / teacher 1,001,086 nodesで規定内だった。MAE pilot fingerprintは`2ad7b90124295ab38456a5d3c25c9e2027394cc97a65bc51e0d6b1a0330e0806`。

| 採用指標（5局等重み） | 現行fallback | 112k 13駒価値ridge=1 |
| --- | ---: | ---: |
| MAE | 1084.478601 cp | 993.941540 cp |
| Top3入り率 | 55.388757% | 53.970998% |

8項目のidentity一致と7,365入力fileのhash・directory集合不変を確認した。各局から独立に再集計した有理数はMAE `3168486841/3187800`、Top3 `433750763/803673780`。MAEは約90.537 cp改善したが、Top3は約1.418ポイント低下し、採用しない。正則化だけを強める案はTop3保持に届かず、このlambdaの追加gridは行わない。[正式比較とグラフ](validation/autonomous-weight-2026-10-03/material-ridge1-112k/comparison.md)を公開し、最良fallbackを維持する。formal・学習の実行プロセスは終了し、goalは継続中。

完了ridge=1のevaluation・4run・model/metadata・凍結snapshotをNASへ保管し、3,737 files / 33 directories / 0 symlinks / 34,969,726 bytesの集合・size・SHA-256をsource-before/source-after/destinationで照合した。SSD原本を保持し、[保存集計](validation/autonomous-weight-2026-10-03/material-ridge1-112k/archive.json)のみ公開する。snapshotには入力不一致・復元receiptと次候補の未実行事前登録も記録し、固定runtime/venv/前回archiveへの依存を明示した。

## 次候補の事前登録と入力復元

次は同じ112,681局面で、元teacherの手番側cp Tと固定material fallback cp Mをnearest-evenで`(T+M)/2`へ加工する。位置の特徴を学ぶNNUEの学習目標に固定駒得の保持を加える仮説で、探索Top3の改善はまだ証明していない。比率1/2は一つだけ、固定material init・fresh Adam・seed/shuffle 42・LR 0.0001・step-half 3epoch・epoch3固定・FT nearest-even exportを事前登録した。元holdout 5,895局面はbyteとteacher identityを保持し、学習target Dのcheckpointを元teacher Oへ診断する専用wrapperを準備する。既存guardは変更しない。

canonical spec SHAは`378072a024a17d64719eb9b7519b6679d98d1c0a8f1c7e04917ae6270f59ae30`、private事前登録SHAは`9c8c3b74a4474bfb906cdefd2db08d757c30e5e73b1af19f41b684b7755d2792`。元manifestと4入力・source provenanceをbyte入力moduleで検証し、specからDを導出した。全trainのnative-core material一致、source/exclusion receipts、実adapter・wrapperの検証とhash凍結はまだ必要。派生labels、学習モデル、正式測定はまだ生成していない。前回full学習907.622秒を参考にwall上限1,200秒、追加SSD 2 GiB以内を見積り、使用前に容量を再確認する。

学習・診断のPython実行処理は、取消時の子handle取得とcleanup/reapまでの実trainer lock保持を補強した。`train_cpu.py`は学習argv・環境・計算を維持し、PID/PGIDとcleanup状態を保存する。`scripts/diagnose_anchor.py`は元4入力とpure view/recipe、D/O、preregistered source、全3epochのmetadata/Adam/native、epoch3と最終weightを照合してから元teacher holdoutへ診断する。既存`diagnose_weights.py`のguardは変更しない。28件の診断fixtureと8件の新lifecycle fixture（既存初期化を含むtargeted 13件）が成功した。この補助の実装時点では実データのprobe・学習・診断は未実施だった。現在の実検証状況は下記を参照。cleanupの終了確認に失敗した場合は、記録したtrainer PID/PGIDが停止したことを確認するまで次の重い処理を開始しない。

v1の事前登録を保持し、実行補助のsource SHAだけを更新したv2を生成前に凍結した。v2 SHAは`febf8a220f0ad84b107a7aeaec6189b5fa93a5fa67cee256eef4e7009bbb4380`、`train_cpu.py` SHAは`e1e21e3c6ad251098c864345bd45af84d013144a403c7296af6f87c6780fdc07`。入力/spec/D・比率・init・学習argv・epoch3・export・正式採用条件はv1と同じで、旧登録や過去runを書き換えない。診断CLIは`--preregistration`と`--expected-preregistration-sha256`に最新v2を指定し、`--recipe`にはpure moduleのrecipeを追加改変せず渡す。実core/source/exclusion証拠は別のgeneration receiptへ記録する。

このspecの初回検査では、元SSDのtrain.labels SHAがmanifestの`aae8d2858f54129c52447e0df665bff0260604020cecc81a0dc84903f050aec4`に対し`a088d3ae7bec4194d7f616109e503d43907b48d624088f522d667181d4bdfe42`だったため、出力作成前に停止した。NASのIssue #15保管copyは固定SHAと一致した。差分は1行のteacher identityのみで、112,681行の順序・SFEN集合・cpは同じだった。変更原因は未特定で、元SSD・過去runは変更しない。

検証済みNASからmanifestと4ファイルを新private inputへ復元し、5 files / 64,444,603 bytesの集合・size・SHA、NASコピー前後、manifestとの一致を確認した。元SSDの全5hashは前後不変。復元receipt SHAは`8b9e42f4e5880af6cddef9373d61f09165be9426cedd36b4f1c8dc1c72c093b8`。新候補はこの復元copyを入力にし、元の不一致をmanifestの書換えや行除外で通さない。完了した正式比較のweight・runtime・attempt証拠は別に有効性を確認済みで、この入力復元をモデル採用の成功とは扱わない。


## functional-anchorの実生成前検証

固定audit Pythonで元データ、13 teacher pack、200局holdoutと400局stagingからの導出、全1000局の合法replayによる除外集合を再検証した。除外局面88,187を再現し、学習・検証局面との交差は0。入力1,056ファイル、固定source 526ファイル、build dependencies 63ファイルの集合とSHAは検証前後で一致した。source preflightのcanonical SHAは`59090622239769afe1f3a4eeba6b74cbd4f86cccd9eba24358cce416c7287e9e`。final splitと正式成績はこの検証で開いていない。

固定material initializerを両weight引数へ明示し、全train 112,681局面とholdout 5,895局面で既存native-core probeを新規実行した。全件でnative・nearest・material・量子化floatがPythonの固定駒得に一致し、FTの全加算prefixがi16範囲内だった。約11.872秒、exit 0、timeoutなし、process group cleanup済み。元の行順とstdin/raw SHA、MXCSR、固定core・build・入力617ファイルの検証前後の一致を記録した。core receipt SHAは`e2db237b0e37a018661753a81b8cbf8bd31c70cf282eae28204dc0f923ef7bd1`。これは静的forwardとrefreshの証拠で、探索中の全incremental経路を新たに検証した証拠とは区別する。

`scripts/prepare_anchor.py`はsource/coreのreceiptと両split全rawを再検証し、別の新規private出力へmanifest・4入力・pure recipe・generation receiptを保存する。元holdoutとtrain positionsのbytesは保持する。pure recipeの`real_generation_ready=False`はそのままにし、実出力の検証完了だけを別のgeneration receiptへ記録する。8件の合成fixtureで失敗の拒否、出力byte、0700/0600とpure recipe保持を確認した。実行するwrapperは入力読込から最終再照合まで共有lockを保持し、固定source/core snapshotとpack・除外raw・導出入力の現状を再照合する。比率・D・init・3epoch/epoch3固定・export・採用条件はv2事前登録から変更しない。実target生成はこの補助の実装時点では未実施だった。現在の実生成・学習状況は下記を参照。

取消・D/O診断のcommit `b63dc66`はCI 196テスト成功（125.218秒）。generation adapterはその後の追加8テストで検証した。fixtureの成功と、実データ生成・モデル採用の成功を分けて記録する。


## 派生データ生成と学習開始

新private outputへ学習用half-target viewを実生成し、generationと外側のoperational verificationがともにcompleteになった。source/coreの固定receipt、両splitの全raw、pure view再生成、保存後のbytes/SHA、source pack・除外raw・導出入力1,042ファイルの前後一致を共有lock内で確認した。6出力（manifest・4データファイル・pure recipe）は64,345,017 bytes、generation receipt SHAは`f85ab76abf50ae530f9618d25d35eda66565c79507d0d34b24fc924daf255c9a`。train positionsと元holdout2ファイルはbyte一致で保持し、train labelsのcpとteacher identityだけを変更した。49,115行がhalf-integerで、丸め誤差は各行最大0.5 cp、符号付き誤差の合計は-0.5 cpだった。pure recipeの未検証フラグを変更せず、実生成の証拠を別receiptに残した。

3epoch・selected epoch3・material init・fresh Adam・seed/shuffle42・LR0.0001・step-half horizon3・min-lr0・wall上限1,200秒の学習を新runで開始した。起動logで112,681 positionsと112,681 cache entries、全件cache-only保持、D target、absolute初期値とFTZ/DAZを確認した。開始時SSD空き約26.29 GiB。元teacher Oのholdout診断、fixed epoch3 nearest-even export、core bridgeと正式100万ノード比較、採否・保存はまだ残る。学習中のlossやこの疎通を採用成功と扱わず、最良fallbackを維持する。

generation adapterのcommit `97fe087`はCI 204テスト成功（124.048秒）。外側wrapper・source検証codeとそのhash、source/core/生成/学習のprivate receiptsを保存し、完了した次候補のNAS snapshotへ含める。


## functional-anchor epoch3の診断・exportと正式比較開始

固定3epoch学習は900.934秒、出力258,432,475 bytes、最大RSS約609 MiBで有効に完了した。全3epochの112,681 cache hits/misses=0・D/absolute/Adam metadataと入力/source前後一致を確認し、trainer process groupの停止も確認した。固定epoch3 native SHAは`2e1328ebd7cbab463d571a9f60e3071bf92bfa92b42b9da4457faf7396f71d18`。3つのcheckpointと最終重みを保持し、静的診断から別epochを選んでいない。

D checkpointを元教師Oの固定holdout 5,895局面へ診断し、約41.405秒でcleanup/input不変・native全byte再exportと全件出力を検証した。native静的MAEは659.883 cp、raw floatは659.548 cp、元materialは809.796 cp。これらは正式探索後の指標ではない。

既存nearest-even方式でFTとFT biasだけをexportし、native全byte再構成・非FT byte保持・D/FNV/absolute sidecarを照合した。nearest SHAは`54e388057db60cab61f7d4110565cf2456564aa28dea11af0456cf0c4fa6d111`、metadata SHAは`a4e4877d8a4ea058a11006f13dbcd5b1c73e7a1b4c77ba89b361ab7044b21128`。nearestの静的O holdout MAEは658.952 cp。export recipeの`engine_verified=false`は維持し、実core証拠を別receiptへ記録した。

native/nearest両重みを既存core probeへ渡し、元train112,681・holdout5,895と公開fixture15の計118,591局面を検証した。両NNUEのfinite-f32/core差は全件1.001 cp未満、FT全refresh prefixはi16範囲内、material列は固定Python駒得と一致した。holdoutのnative core・float32 bits・materialは完成O診断と全件一致し、native/Adam/metadataを同じ診断/export入力SHAへ束縛した。約11.639秒、exit0・cleanup成功・source/input前後不変。candidate core receipt SHAは`cc5467e63d9313f665061933fca5c27a9c302327f0368d0c5891e708143cc2fc`。全incremental探索経路の新規証明とは区別する。

2026-10-03 08:24:33 UTCに候補自身のMAE pilotを開始した。run prefixは`development-17-anchor-half-e3-v1`。続く正式MAE/Top3は固定development5局・100万ノードで、既定設定・採用条件を維持する。

開始前の記録生成で設定名を`benchmark.json`と誤認し、記録保存がFileNotFoundErrorで失敗した後もshellが続いたためpilotが開始された。この経緯をprivate erratumへ保持した。実測は既存helperが正しい既定`development-benchmark.json`を読んでおり、pilot中に実candidate configが既定とmodel/own pilot evidence以外完全一致すること、570局面・100万ノード・固定weight/source・事前登録不変を再確認した。訂正記録v2はpilot開始後の検証として明示し、事前記録へ読み替えない。measurementのpilot/formal gateは変更せず、再起動や既存runの上書きは行わない。以降の複数段階shellは`set -e`で途中失敗時に停止する。

正式比較・厳密採否・保存は進行中。最良はfallback、goalは継続中。

候補自身のMAE pilotは102/102 attemptを完了し、2026-10-03 08:28:27 UTCに正式MAEへ進んだ。独立監査は全raw SHA・position・USI lifecycle・option順・deadline・cleanup・supervisor・exit 0、両engineの17局面×3反復安定と51回ずつのweight読込を確認した。技術失敗・欠測・invalid node evidenceは0。最大報告はSekirei 1,000,001 / teacher 1,001,086で規定内、入力241ファイルの検証前後hashも一致した。fingerprintは`e5d41c0e6b6c107bba743137a4a781fb5402c853843da520ff49e4729f8da23b`、独立監査receipt SHAは`9f3099746275e388a8837f4d9b69a09b5f84ea602d300158e0d2180dc7c5e001`。訂正v2がpilot中の検証である点も独立照合し、開始前の記録へ読み替えていない。正式MAE・Top3の採否は未確定である。

## 条件付きbounded-material案の純粋検証

現在のfunctional-anchorが有効な正式比較で採用未達だった場合に限り、固定material seed42からFT全体と駒得経路を固定し、補助28ユニットだけを元教師Oへ直接学習する案を準備している。補助FTには駒の位置・種類・所有者・玉・持駒の固定random特徴を含み、駒得だけの入力ではない。fresh Adam・seed/shuffle42・3epoch/epoch3・LR0.0001・step-half3を候補条件とし、補助出力係数の実数上界99 cpを一つだけ固定する。100 cpの整数差に1 cpの丸め余裕を見込む仮説であり、探索Top3維持や全coreでの整数差上限を確認した段階ではない。条件案SHAは`70d86307d5a23319d70dd6fa8cedb472954f178c3663d4265a7c982a85befd5e`。現在の未完了formal結果やfinalから条件を選んでいない。

`scripts/bounded_material.py`はbyte入力だけを扱い、固定FT/FT bias、駒得L2の全512行×4列、駒得bias/out、出力biasと材入力から補助へのpositive-zeroを検証する。補助outの実binary32値をexact Fractionで集計し、`127/64 * Σabs <= 99`を判定する。13件の合成fixtureで座標改変・非有限値・サイズ/magic・符号付きzero・隣接f32予算境界などを確認した。固定referenceのFT `/64→×64`復元は純粋な算術確認であり、実Adamのfloat FT固定や実エンジン検証ではない。recipeのtrainer・optimizer・学習・実core整数差・採用の各フラグは未検証のまま保持する。moduleはファイル保存・学習・engine起動を行わない。

専用trainerの更新maskはparameterとAdam m/vの呼出しをskipし、補助outを各Adam更新後に一様縮小して保存f32の予算を再検査する設計とする。実装、専用source/build、metadata/source binding、保存後のnative/Adam再構成、全局面core・incremental/undo、候補自身の正式比較はまだ必要。既存prepare/exportの固定source guardは変更せず、専用経路で新patchを束縛する。次案の実学習・実probeは開始していない。

関連: [Issue #17](https://github.com/phni3j9a/sekirei-weight2/issues/17)、[PR #18](https://github.com/phni3j9a/sekirei-weight2/pull/18)（下書き・未マージ）、[前回の実験](WEIGHT_IMPROVEMENT.md)、[研究方針](RESEARCH.md)、[環境](ENVIRONMENT.md)。
