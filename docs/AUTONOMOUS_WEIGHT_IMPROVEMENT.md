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

## 最初の候補の正式結果

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

## functional-anchorの正式結果

候補自身の全4runは2026-10-03 08:24:33〜09:33:07 UTC、4,114.338秒（約68.6分）で完了した。MAE pilot102、正式MAE1,140、Top3 pilot36、正式Top3551の計1,829 attempt。厳密比較は8identity一致・Teacher-E266/266・Top3551/551・7,365入力fileのhashと集合不変を確認し、有効に完了した。

| 採用指標（5局等重み） | 現行fallback | functional-anchor half epoch3最近傍 |
| --- | ---: | ---: |
| MAE | 1084.478601 cp | 995.803800 cp |
| Top3入り率 | 55.388757% | 55.382847% |

候補の厳密有理数はMAE `176356853/177100`、Top3 `59346323/107156504`。MAEは約88.674800 cp改善したが、Top3は約0.005910ポイント低下し、維持条件を満たさない。全551点のhit合計は両者304だが、採用に使う固定指標は各局を等重みとした平均であり、表示丸めや全局面の直接平均で採用へ読み替えない。[正式比較とグラフ](validation/autonomous-weight-2026-10-03/functional-anchor-half-e3-nearest/comparison.md)を公開し、最良fallbackを維持する。formal launcherはexit0で終了し、goalは継続中。独立全run監査とNAS保存も完了した。

独立監査は全4runの1,829 attempt、raw/lifecycle/node/options・明示的NNUE読込1,208/1,208、合法なTop3・Teacher-E266・8identityを再検証し、同じ厳密値とhold判定を得た。技術失敗・timeout・cleanup失敗は0、最大nodesは候補1,000,003 / 教師1,001,086。監査前後の7,379 input filesと8 inventoriesが一致し、finalは未使用。独立receipt SHAは`7656a6d08d340b3cd9627ceab73fe7fdeb699a7cb1b2a19beb4b42af00f986ea`。

完了generation、3epochの全checkpoint/Adam/resume/metadata、O診断、native/nearest、二つの全row core証拠、evaluationと4run、凍結completion/codeをNASへ保管した。12 source roots、3,831 files・43 directories・0 symlinks・415,829,630 bytesで、コピー前後と保存先の集合・size・SHA-256等が全一致した。SSD原本は保持する。[公開保存集計](validation/autonomous-weight-2026-10-03/functional-anchor-half-e3-nearest/archive.json)を参照。元O入力・固定runtime・audit venv・baseline等の既存依存が必要であり、単体の全環境復元とは扱わない。snapshot作成でmetadata名を誤認した最初の試行は保存前に停止し、正しい名前でbyte確認して完成した経緯もprivate erratumへ残した。

## 条件付きbounded-material案の純粋検証

functional-anchorの有効な正式比較で採用未達という開始条件を満たしたため、次は固定material seed42からFT全体と駒得経路を固定し、補助28ユニットだけを元教師Oへ直接学習する案を事前登録した。補助FTには駒の位置・種類・所有者・玉・持駒の固定random特徴を含み、駒得だけの入力ではない。fresh Adam・seed/shuffle42・3epoch/epoch3・LR0.0001・step-half3を候補条件とし、補助出力係数の実数上界99 cpを一つだけ固定する。100 cpの整数差に1 cpの丸め余裕を見込む仮説であり、探索Top3維持や全coreでの整数差上限を確認した段階ではない。条件案SHAは`70d86307d5a23319d70dd6fa8cedb472954f178c3663d4265a7c982a85befd5e`。この条件はformal終了前に保存し、当時の未完了結果やfinalから選んでいない。

`scripts/bounded_material.py`はbyte入力だけを扱い、固定FT/FT bias、駒得L2の全512行×4列、駒得bias/out、出力biasと材入力から補助へのpositive-zeroを検証する。補助outの実binary32値をexact Fractionで集計し、`127/64 * Σabs <= 99`を判定する。13件の合成fixtureで座標改変・非有限値・サイズ/magic・符号付きzero・隣接f32予算境界などを確認した。固定referenceのFT `/64→×64`復元は純粋な算術確認であり、実Adamのfloat FT固定や実エンジン検証ではない。recipeのtrainer・optimizer・学習・実core整数差・採用の各フラグは未検証のまま保持する。moduleはファイル保存・学習・engine起動を行わない。

専用trainerの更新maskはparameterとAdam m/vの呼出しをskipし、補助outを各Adam更新後に一様縮小して保存f32の予算を再検査する設計とする。実装、専用source/build、metadata/source binding、保存後のnative/Adam再構成、全局面core・incremental/undo、候補自身の正式比較はまだ必要。既存prepare/exportの固定source guardは変更せず、専用経路で新patchを束縛する。この準備段階では実学習・実probeは開始していなかった。

pure contractのcommit `be7e0da`はCI217公開fixture tests成功（118.346秒）。型検査前の専用trainer追加patchもprivate stagingに準備したが、rustfmt parse/dry-applyを通した段階であり、8件のRust testや実build・学習の成功へ読み替えない。

## bounded-material専用経路の検証

`prepare_bounded.py` / `train_bounded.py` / `diagnose_bounded.py`とtrainer専用追加patchを準備した。既存external patch、上流commit、main/trainerだけの変更、全526 source files、8 workspace依存ファイル、Rust/Cargo実体、固定flagsとbinaryを新build manifestへ束縛する。既存prepare/export/diagnosticのsource guardは変更しない。専用source/buildで9件のRust fixturesとrelease buildが成功し、metadata・saveの失敗は非zeroになること、固定parameter/m/vはAdam更新をskipし補助部分が学習することを確認した。追加patch SHAは`6602c12ea83b602cb1a93fa7f16624d1c67e4f23a05aa7bd4d253b21426d48c8`、実trainer binary SHAは`71ab7126629cb2fa242b5290d3c11a0ea89eb272866d7d91146ab1ec1a15006b`。

最初の専用buildは、compilerのbefore/after辞書比較が不一致となりsource clone前に停止した。比較値を保存していなかったため原因は未特定で、現在の再照合は一致する。失敗runtimeを保持し、両値を不一致時に表示する診断を追加して、同じ厳密guardの新runtimeでbuildを完了した。原本・旧学習器・比較engineを上書きしていない。

元Oデータはmanifest SHA、全cache/position集合、depth0・全cp絶対値30000未満、112,681/5,895分離を再検証した。過去の全1000局legal replay/除外証拠を、全raw集合と1,042 source入力の前後hash一致で再利用し、final splitは開いていない。新build/sourceも照合したpreflight receipt SHAは`1540bdf1991a174bab774b99ee32c52ceee0eb1d6004557705917f477063dcdd`。

固定coreにリンクした技術probeはmaterial initializerで公開15fixture・16固定walk（うち8は探索用move API）を実行し、8,185観測・capture632・promotion186・drop577・undo3,837・null undo240でincremental/refresh accumulatorと評価値、undo後のSFEN/hash/手番/ply/acc復元、観測したf32中間値のfiniteを確認した。material差最大0 cp、FTZ/DAZ制御も維持、cleanupとsource/input前後不変だった。これは限定された技術観測であり、学習後のNNUE、全局面/全探索経路の100cp保証、採用の証明とは扱わない。

Pythonの新規fixtureはactual full-shape AdamのFT 1ULP改変（native/nearest byteが同じでも拒否）、固定moments、variance/step、全byte reexport、厳密な99cp予算・metadata型、異なる教師/epoch/cap、final sidecar形式、cleanup失敗receipt等を確認した。既存byte契約を含む37 testsが22.329秒で成功。共有supervisorの取消・reapを再利用し、その前後に外側termination guardとgroup停止確認を追加した。追加の隔離process fixtureも成功し、supervisorのhandler復元直後のSIGTERMで外側guardが両lockを保持して再cleanup・cancelled receiptまで到達することを確認した。今後の実学習は事前登録した3epoch/E3だけを採点し、lossからepoch・cap・LRを選び直さない。

## bounded-material実学習のmetadata停止

専用commit `fe4abb0`のCIは成功した。13 helpers・元O入力・専用build・固定3epoch/E3・99cp予算を事前登録し、同じ計算条件で実学習を開始したが、epoch1のNNUE/Adam/resume保存後、追加metadataの保存でexit1となった。30.125秒、cache112,681/112,681・misses0、group停止/cleanup成功。正式比較・採用には進んでいない。

原因は`save_checkpoint_meta`の引数が`weights.epoch1.meta.json`なのに、追加処理が単純な`with_extension("bin")`で`weights.epoch1.meta.bin`を参照したこと。実際のnativeは`weights.epoch1.bin`であり、NNUE保存は済んでいた。失敗runの6 files・89,422,337 bytes、元のsource/build・事前登録・logを保持し、private erratumへhashを凍結した。metadataの厳密なsuffix解釈と実保存の回帰fixtureを追加して新source/buildを作る。学習math・CLI・mask・99cp・LR・seed・3epoch/E3は変えず、初期materialとfresh Adamから新runを始める。失敗epoch1を採点候補やresumeに流用しない。

関連: [Issue #17](https://github.com/phni3j9a/sekirei-weight2/issues/17)、[PR #18](https://github.com/phni3j9a/sekirei-weight2/pull/18)（下書き・未マージ）、[前回の実験](WEIGHT_IMPROVEMENT.md)、[研究方針](RESEARCH.md)、[環境](ENVIRONMENT.md)。

metadataパスの修正版v3は、二つの正確なsuffixを確認して実`*.bin`を参照し、不正な名を拒否する。公開synthetic NNUEを実serializerで保存して`save_checkpoint_meta`全経路を通す回帰testを追加し、複数ドット・親dirのsuffix・FNV・教師identity・非zero補助予算・native bytes保持を確認した。新専用source/buildのRust 10 testsとrelease buildが成功した。追加patch SHAは`ee819bd80c41301ee78a0d3ce29efd976c3d762c1ddf07bf6cc259703d38933b`、binary SHAは`86a17aa6fd75ddf76f4679b05d072ca0d930662aa7fb6b9dadd2647e79ab7ee3`。v2のtrainer数学はbyte一致で保持し、元の失敗source/helper/事前登録も別snapshotへ凍結した。旧失敗runの再開や採用判定には使わず、新事前登録とfresh Adamで同じ3epoch/E3を再実行する。

## bounded-material修正版の固定E3と技術検証

修正版commit `c191670`のCIは242 tests（141.323秒）が成功した。新事前登録v2は元plan・O入力・教師・init・全hyperparametersを維持し、v3 buildと13 helpersのhashだけを新しい実体へ束縛した。初期material/fresh Adamから3epochを有効に完了した。trainer 88.014秒、外側の全source検証を含め101.792秒、出力203,715,187 bytes。全3epochのstep=112681/225362/338043、FT/biasの実binary32固定、保護したm/vのpositive-zero、native全byte再構成、nearest全byte一致、厳密99cp予算を検証し、group停止/cleanupと入力/source前後不変を確認した。選択は事前固定E3だけで、weight SHAは`edd072bc04f1f018245d90d563576f1a9df77884c5e0370277b3e498f111927a`。

元O holdout5,895局面の独立診断はcomplete/exit0、native再export・raw/native予測一致・整数駒得差最大99・finite bridge最大0.999970cp・全入力前後一致/cleanup成功。静的core MAEは799.814758cp、駒得809.796268cpで、正式探索の改善とは扱わない。学習metadataの補助28unitはepoch2/3で飽和率1.0・L2更新0となり、全正outの予算が約98.999908cpへ寄った。診断の駒得差も平均98.999861cp・標準偏差0.000099cpで、局面別の残差を学んだ証拠ではなく、ほぼ一定の正のSTM補正で説明できる。base LR表示と実StepHalfを区別し、LR単独が原因とは断定しない。独立静的reviewの16入力hashは前後一致し、正式結果/finalは使っていない。

固定coreで元train112,681・holdout5,895・公開fixture15の計118,591局面を照合し、native/nearest整数・float32 bitsが全一致、全float/core bridge<1.001cp、駒得列と固定Python Mが一致、整数駒得差<=100、FT refresh prefixがi16範囲内となった。holdoutは元O診断の全行と一致し、全checkpoint/Adam/metadata・617 fixed reference files・v3 source/buildを前後hashで束縛した。core receipt SHAは`8cc3768cef852b3f3f13b853b4a99cbe1f0e2786e7b89dca6e56bad4532b3449`。同じ候補の15 fixtures・16 walks・8 search move API walk計8,185観測もrefresh/incremental・parent/null/undo一致、finite intermediates、駒得差最大99で成功した。有限集合の証拠であり、普遍的整数上限や全探索経路・棋力・採用の証明にはしない。その後、同じE3のown pilotsと固定100万ノード正式比較を実施した。結果は次節に示す。

## bounded-material固定E3の正式結果

候補自身のMAE pilot 17局面×3反復×2エンジン（102 attempt）、正式MAE 570局面×2エンジン（1,140）、Top3 pilot 12局面×3反復（36）、正式Top3 551を完了した。2026-10-03 11:18:01–12:23:54 UTC、約65.9分。評価器の外側検証では1,204事前入力とworker/事前登録/起動前証拠のunion 1,205ファイルが前後一致し、専用v3 source/build・E3も不変だった。最初のlaunch preflightはuniverseのdictに`len`を使った局面数誤認で評価開始前に停止し、全5局のoccurrences合計570を検証する正しい経路へ修正した。設定・重み・対象を変えず、失敗と修正をprivate erratumへ記録した。

正式比較は有効だが**不採用**。MAEは`165665231/151800 = 1091.338808 cp`、Top3は`8344907/15455265 = 53.993943%`。固定fallbackのMAE1084.478601 cp / Top3 55.388757%に対し、MAEが6.860207 cp悪化しTop3が1.394814ポイント低下した。5局のTop3 hitは46/84・83/122・51/127・52/104・65/114。教師E 266点、Top3 551点、教師評価・bestmove・PV・合法手集合・モデル以外の実行条件の8 identityを固定した厳密有理数比較で判定し、許容差を追加していない。[公開結果とグラフ](validation/autonomous-weight-2026-10-03/bounded-material-residual-100cp-e3/comparison.md)を参照。

独立監査は候補・基準それぞれ1,829 attemptのraw SHA、USI options・型付き結果・deadline・supervisor/runner終了・明示weight読込・ノードgrammarを検証し、Top3を生ログから再count、MAEを整数cp誤差から再集計した。9,578入力と8run inventoryは前後一致し、technical failure/timeoutは0。観測最大nodesは教師1,001,086、候補1,000,004で固定上限1,010,000以内。MAE pilotのterminal bare-resign 3件は既存のno-score例外のまま扱った。独立監査SHAは`2a52e6929e51ffab326b8561a53aa7215dc19ba9b0c00628b5be40fefb013a66`。全協調lockの排他取得・全raw cleanup証拠・launcher exit0/reap・関連processの2回の不存在確認を別receiptへ残し、次の重い処理の開始条件を満たした。

静的O holdoutの約10cpの改善と、ほぼ一定の+99cp STM残差は正式探索での改善を示さなかった。探索実装や比較条件を変える理由とは扱わず、最良は駒得fallbackを維持する。finalは未使用、Plusは非採用候補の選択に使わない。

完成E1/E2/E3の全native/Adam/metadata、失敗v2のepoch1とその正確な旧source/helper、元Oの復元入力、静的・core・incremental証拠、evaluation/4run、凍結completionを既存NASへ保存した。15 source roots・3,899 files・71 directories・0 symlinks・428,614,972 bytesで、コピー前後と保存先のファイル集合・size・SHA-256、directory/symlink集合が一致した。SSD原本と旧runの参照パスは保持し、build/venvはコピーしない。[公開保存集計](validation/autonomous-weight-2026-10-03/bounded-material-residual-100cp-e3/archive.json)を参照。最初の保存呼出しはbuild lock名が既存helperの許容範囲外のためNAS作成前に停止した。helperのguardを変えず、build lockを外側で保持し、training/prepare/benchmark lockをhelperが保持する経路で保存を完了した。失敗と再試行・source snapshotもarchive-supportへ保持した。

## 次の固定案：補助L2更新のFanIn509正規化

現在候補の正式成績を見る前に、補助28unitのepoch2/3飽和・L2更新0という静的診断から次案を一つだけ固定した。可変補助L2 weightとbiasのAdam実効LRを`f32(epoch_lr)/f32(509)`とする。分母は508個の非material入力+1biasで、outのLR・勾配・moment計算・99cp縮小・保護maskを維持する。勾配だけの正規化はAdamで相殺されるため、更新幅を直接正規化する仮説である。分母・cap・epochのgridは試さない。plan SHAは`359cf9daa1a5e57be6c7a2d56b2ecfef5c9e24cfc7648e45f7930544b264ae6e`、記録時刻11:30:04 UTCは現在候補の正式終了より前。

元O train112,681/holdout5,895・original absolute teacher・material seed42/fresh Adam・seed/shuffle42・LR0.0001/step-half3・固定3epoch/E3・実数残差予算99cpを維持する。開始条件は前候補の有効な正式非採用・独立監査成功・全group停止で、actual receiptのpath/SHA/型・相互参照を新build/preregから検証する。別mode `bounded-material-fanin509-v1` と専用schema/source/metadataを使い、旧guardや`SOURCE_HASHES`を変更しない。

sourceだけの準備・独立レビューではL2/biasの2つのAdam LR operand以外の学習数学と旧10fixture本体を保持した。新6件を加えた16 Rust fixturesを宣言し、4つの専用Python helper・activation検証・17件の公開synthetic fixturesを追加した。前候補の正式非採用・独立監査・停止のreceipt chainと実weight/metadataを検証してから実行経路を有効化した。新buildは専用checkoutで上流・patch・source・compiler・全16テストを照合し、学習と診断は新schemaと17helperのSHAを固定する。この実装準備時点では新modeの実build/Rust tests/学習/probeは未実施だった。実行結果は次節に示す。

初期bias4では小さなdeltaがf32丸めで保存値に現れない可能性もある。飽和や一定STM補正を防ぐ保証、学習改善・普遍的整数100cp上限・採用の証明とは扱わず、実保存bytes・全epoch・全元局面core/undo・own pilots・固定100万ノード正式比較で確認する。

## FanIn509候補の正式結果と復旧

専用buildで16件のRust fixturesが成功し、事前固定の元O・seed42・fresh Adam・3epoch/E3で実学習を90.720秒で完了した。FT/駒得の固定、全3epochのAdam・保存bytes、99cpの補助出力予算を確認した。選択したE3の重みSHAは`2ec2eacb5f3cb9f72921efd75478f024614a3f944c3ee9a528df9898d6d93164`。元holdoutの整数core MAEは808.623070cp（駒得809.796268cp）。補助L2の飽和ログは3epochとも0だったが、補助float評価は平均5.894539cp・標準偏差0.011699cpで、ほぼ一定の正のSTM補正が残った。元train/holdout/公開fixtureの計118,591局面、8,185のincremental/refresh/undo観測も成功した。これらは静的・有限集合の技術検証で、正式採用の代替ではない。

初回評価は2026-10-03 14:16:55–15:09:50 UTCにMAE pilot102と正式MAE1,140を完了した後、rootがsystem Pythonで起動したため、Top3のcshogi読込前にexit1で停止した。Top3ディレクトリやattemptはまだ作成されていなかった。失敗状態・18ファイルのevaluation snapshot・元ログ・正常終了済みMAEを保持し、全process停止、元1,221入力・source/build・設定・重みの不変を確認した。これはモデル品質の否定結果へ読み替えていない。

4,206復旧入力を凍結し、正しいcshogi1.0.4/NumPy1.26.4のvenvから同じモデル・設定・run IDでTop3 pilot36と正式551だけを実行した。15:40:34–15:55:51 UTC、916.746秒、exit0で全段階を完了した。MAEは再実行せず、旧失敗logは保持し、新しい復旧logへ保存した。evaluationの元作成時刻・評価source SHAを維持し、失敗segmentと復旧segmentを別々に記録した。実行時間合計は4,091.287秒で、復旧準備待ちを含む壁時計の経過時間とは区別する。

| 採用指標（5局等重み） | 現行fallback | FanIn509 E3 |
| --- | ---: | ---: |
| MAE | 1084.478601 cp | 1084.349896 cp |
| Top3入り率 | 55.388757% | 55.010955% |

正式比較は有効だが**不採用**。MAEは`1728345299/1593900`で0.128705cp改善した一方、Top3は`884217247/1607347560`で0.377802ポイント低下した。既定の厳密な共同採用条件を維持し、最良モデルはfallbackのままとした。[比較集計とグラフ](validation/autonomous-weight-2026-10-03/bounded-material-fanin509-100cp-e3/comparison.md)を参照。

独立監査では候補・基準各1,829 attemptをrawから再解析し、USI lifecycle・明示weight読込・node grammar・deadline・supervisor/cleanup・8identity・厳密有理数採否を確認した。10,082入力の前後hashが一致し、技術失敗/timeoutは0、最大nodesは教師1,001,086/候補1,000,004で規定内。旧system-Python失敗と新venv成功、18ファイルの失敗snapshot、4,207復旧入力の別map、実interpreter・dependencyまで照合した。独立監査SHAは`44239bf763de9cf9cf7e2c65b5fd85e42ed875477054d700cf3290b41c1c3c26`。全協調lockを排他取得した停止receiptではlauncher exit0/reap、全raw終了証拠、関連processの2回の不存在を確認し、次の重い処理の開始条件を満たした。

集計の最初の呼出しではrootがディレクトリ引数へevaluation.jsonを渡したため起動前に停止した。無効receiptを別名で保持し、正しいディレクトリで新しい出力を作成した。入力・探索・採否条件は変えていない。独立監査sourceの暫定準備通知後の追加照合はrootのSHA guardで検出し、通知済みbytes・追加後bytesを双方保存して、新しく固定したv3 sourceで実監査・停止確認を行った。過去の失敗・準備版を成功版に書き換えない。

全3epochのnative/Adam/metadata、元Oの復元入力、静的/core/incremental証拠、evaluation/4run、失敗した初回起動と18ファイルのsnapshot、復旧と監査のsource/receiptをNASへ保存した。14 source roots・3,965 files・77 directories・0 symlinks・355,207,871 bytesで、コピー前後と保存先の集合・size・SHA-256を照合した。SSD原本・参照パスを保持し、build/venvは除外した。[公開保存集計](validation/autonomous-weight-2026-10-03/bounded-material-fanin509-100cp-e3/archive.json)を参照。

## 第六候補：対になった線形補助出力

第五候補の正式成績を見る前に、元train112,681局面だけで残差`d=T−M`を診断した。残差の平均は244.461116cp、中央値122cp、標準偏差1339.422567cpで、絶対値198cp超は81,456局面（72.289028%）だった。残差が大きい例は少数の外れ値だけではない。Huber幅99cpの単一STM定数に対する微分は補正99cpでも負であり、lossだけの置換で正の定数補正を防げるとは判断しなかった。これはonline勾配やAdamの因果の実測ではない。

次はseed42のFT全bytesと駒得の4unitを維持し、補助unit4/5へ固定bias64/64・out+64/−64を置く。非material FTの254チャネルについて、同じ保存f32係数を(+u,−u,−u,+u)のsign-bit一致で両手番へ結ぶ。unused補助座標はpositive-zero、出力biasもpositive-zeroとする。理想実数の補助評価は`2u·(acc_us−acc_them)/64`で、共有のSTM定数を除く。ただしempirical平均0、物理的な盤面の回転対称性、nativeのbit単位反対称性はこの構造だけでは保証しない。位置・駒種・玉・持駒thresholdの固定random特徴を線形に投影する容量に限られ、駒間の相互作用や利きの新特徴は追加しない。

元trainの整数cpを加工せず使用し、固定特徴`Z=(acc_us−acc_them)/2`（整数、各座標±40以内）から`X=Z/16`を作る。目的は未正規化の半二乗誤差和と`1/2||u||²`（ridge=1）、L1制約は`rho=39.5−2^-12`の一案だけ。zero初期値・固定Gershgorin step bound・目的増加時の固定restart付きprojected FISTA、最大20,000反復/全fit1,200秒を事前固定した。exact dyadicな実行可能性とFrank–Wolfe gap/N<=1e-6を要求し、保存f32はexact L1<=39.5とzero/materialに対するexact目的差<=0を確認する。失敗時にlambda・loss・solver・半径を選び直すfallbackは設けない。

実数残差の条件付き上界は98.75cpで、既存の観測integer-core駒得差100cp guardを維持する。従来の補助out L1契約は新しい固定out(+64,−64)には適用できないため、旧guardを変更せず専用の機能的契約を検証する。元inputの合法性・source・除外、保存bytes、全局面core、incremental/undo、候補自身のpilotと固定100万ノード比較を通すまで採用しない。最終5局は使わない。

条件案SHAは`dabad54e419237335fd1f370063a0c6e82ac7b58ac27f0d20fd91ca21d9dd900`で、第五候補の正式成績参照前に固定した。旧generic solverの有理数PSD検査は254次元で計算予算を圧迫し得るため、固定整数designから唯一のproducerが`G=ZᵀZ`を生成したことをsource・digest・originで束縛し、`vᵀGv=Σ(Zv)²>=0`による専用検証を準備した。旧任意GramのPSD guardは保持する。条件案を固定した時点では新構造の実fit・core・正式測定は未実施だった。以下に実行結果を記録する。

専用の`paired_linear.py`は254個の保存f32からnative全bytesを再構成し、FT/駒得、sign-bit、zero、exact L1を検査する。`fit_paired_linear.py`は元TRAINの位置順を保ってSFENで教師ラベルを結合し、整数designから唯一のGram producerと固定FISTAへ渡す。design・target・Gram・右辺・f64/f32係数・exact dyadic certificateを個別artifactとして保存する。実OpenBLASのsetter/getterとlibrary SHAで1threadを確認し、全fit1,200秒は外process supervisorでも制限する。子processは終了証明を主張せず、親がwait/reap・group停止・入力/sourceの再hashを確認した後にrunとabsolute sidecarを作る。

既存を含む288 testsが成功（標準環境では数値13件をskip）。固定NumPyで数値13件とnative/activation16件、追加fit-driver13件も成功した。第五候補の有効な不採用・独立監査・停止を46個の現在inputと照合し、次案へのactivationを確認した。これらは準備の検証であり、以下の実fit・core観測・正式比較と区別する。

### 実fit・技術検証

元train112,681局面に対するfitは32.573954秒・22反復で終了した。Adam、epoch追加、resume、別条件へのfallbackは使っていない。保存f32のexact L1は`161791/4096=39.499755859375`で、非zero係数は254個中1個だった。f64・solver f32・保存f32の三つのcertificateを別workerが整数・dyadic演算で再計算し、FW gap/N=0、zero/materialからのexact目的差が非正であることを確認した。この最適性は固定の学習目的に対する結果であり、探索後の改善を意味しない。

native SHA-256は`12cc820db432fd2677ffcb37d58bab1536f2b03471af0fe14ad26997e2841de3`。stock v0.3.39 coreの全118,591局面（train112,681・holdout5,895・fixture15）でself-load・material参照とのbridge・有限値・観測駒得差100cp以内を確認した。最大差はtrain60cp・holdout55cp・fixture9cp。incremental/undoの8,185観測でもrefreshとの差0、capture・promotion・drop・undo・null undoと親状態の復元を確認し、最大駒得差40cpだった。これは有限の観測範囲の検証である。

numeric監査の最初の起動は外部SHA指定の誤りで入力読取前に拒否され、正しい固定値で再実行した。coreの最初の起動も、専用出力のguardが保護入力との包含を過剰に判定してprobe前に拒否された。元sourceと失敗記録を保持し、全保護入力との重なりを明示拒否する別sourceで再実行した。いずれもモデルの再学習や採用条件の緩和は行っていない。

### 正式比較・無効な初回履歴

初回の正式測定はMAEの937試行目でsupervisorのterminal statusが欠落し、cleanup failureとして無効になった。旧937試行・own pilot102試行・評価状態・sourceとcontrolの原本およびmirrorを4,218ファイルのsnapshotへ凍結し、6排他lock・2回のprocess観測・3,967入力の前後照合で停止とモデル不変を確認した。同時刻付近のPython general protection faultは観測したが、対象PIDと役割の因果対応は確認できず、原因は未確定。失敗した試行の短いlog offsetを実際の終了所要時間とは扱わない。

同じfit/nativeを使い、別のmeasurement attemptでown MAE pilot・正式MAE・own Top3 pilot・正式Top3を全て新規実行した。旧途中結果は再利用していない。固定venv/NumPy/cshogiとlaunch preflightを外部SHAで束縛し、候補own MAE pilot102試行の独立監査を通した後、四段階は正常終了した。測定時のGit commitは`480bd14844242770a3faf82fea1b2707530fc7a5`。

| 指標 | 駒得fallback | 第六候補 | 差 |
| --- | ---: | ---: | ---: |
| MAE | 1084.478601 cp | 1087.385635 cp | +2.907034 cp |
| Top3入り率 | 55.388757% | 54.483043% | −0.905714ポイント |

候補のexact MAEは`13701059/12600`、Top3は`15363717/28199080`。独立監査で候補・基準それぞれ1,829試行のraw SHA、options、nodes、score、bestmove、runner/supervisor lifecycleを再解析し、8比較identity・有理数集計・共同採用判定の一致を確認した。**正式比較は有効だが不採用**で、最良モデルは駒得fallbackのまま。final 5局は使っていない。[公開比較とグラフ](validation/autonomous-weight-2026-10-04/paired-linear-constrained-ridge1-l1-39p5-v2/comparison.md)を参照。

停止確認sourceの旧版は比較checkの辞書をboolとして受け取る不整合があった。旧版を保持し、producerの5辞書/3list・string identityの厳密なshape、typed true、SHA一致、mismatch_countの非bool整数0、CP/AU全record一致を検証する別版へ修正した。実producerから作る合成正常controlで旧版の拒否と修正版の通過を再現し、13 testsと独立peerを通した後、実停止確認も成功した。8,194入力の前後一致、6排他lockと2回の空process観測を記録した。

単一fitのnative・元design/Gram/係数/certificate・元Oの復元5入力・core/incremental証拠・新四段階のevaluation/run・無効な初回測定・sourceとcontrol・凍結completionを既存NASへ保管した。初回履歴の原本とmirror全4,218ファイルも保存対象へ対応付けた。archive-supportを含む15 source roots・8,164 files・80 directories・0 symlinks・230,230,076 bytesについて、source-before/source-after/destinationの集合・size・SHA-256とdirectory/symlink集合が一致した。保存helperと外側workerは正常終了・wait/reap・group停止を確認し、SSD原本を保持した。[公開保存集計](validation/autonomous-weight-2026-10-04/paired-linear-constrained-ridge1-l1-39p5-v2/archive.json)には集約値とhashのみを置く。固定runtime・compiler/build・audit venv・教師・baseline等はSSDの既存依存を使い、このコピーだけで全環境を復元できるとは扱わない。

### 次の方針

固定の凸目的に対するexact gap=0が確認されたため、同じ目的の反復やepochを増やす案は選ばない。次はflat特徴の白視点を`80−square`へ回転し、持駒aux bankを`0=3 / 1=2`へ共有する評価特徴の一案を準備する。源コードだけの独立検証では盤面4,536通り・持駒threshold152通りの対応が一致した。静的forwardの対応を目指す変更であり、100万ノード探索のPVやscoreの物理対称性は保証しない。

探索・合法手・盤面・USIの源コードと100万ノード条件を維持し、評価特徴だけを専用compile featureと別native magicで扱う。旧stock loader/metadata/comparatorのguardは保持する。新binaryのfallbackをown pilotを含む全四段階で新規測定し、基準の不変性を確認してから同じbinaryの候補と比較する。専用build・新モデルfit・formalは未実施。現在のMac mini CPUと既存ストレージの範囲でgoalを継続する。
