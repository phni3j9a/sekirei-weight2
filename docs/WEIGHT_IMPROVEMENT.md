# CPUウェイト改善実験（Issue #15）

## 範囲と判定

初回候補の悪化を受け、保存済みcheckpointの診断から次の学習条件を選ぶ。ユーザー承認済みの手動goalで、2026-10-02 14:33:50 UTCから約8時間を目安とする。採用候補が得られても、残り時間で検証・記録まで完了できる改善実験を続ける。期間終了とモデル改善達成は区別する。

探索Sekirei v0.3.39、固定水匠11β、100万ノード、MAE/Top3の対象・欠測処理・採用基準は維持する。final 5局は使わない。最初はfallbackのMAE 1084.4786006022964 cp / Top3 55.3888%を比較基準とし、採用候補が得られた後はその最良候補と比較する。MAEが下がりTop3が下がらない場合に更新する。学習lossや静的評価の診断値は採用指標ではない。

## 最初の仮説

初回55,404局面・3エポックは実効学習率0.001→0.0005→0.00025で、候補の評価値の振幅が教師より小さかった。保存済み3 checkpointと隔離済み保留5,895局面を使い、量子化前float・量子化復元float・core整数推論の固定予測を比べ、学習不足と量子化誤差を切り分ける。

追加学習は同じseedからの初期化と新しいAdam状態を使う。外部教師用の完全resumeは今回は拡張しない。既存checkpointは学習率のスケジュール長などを含む再開条件を保存しており、単なるエポック延長を同一条件の再開とはみなさない。

初回checkpointのL2出力は各要素が0〜127に制限されるため、出力層の重みから静的STM評価の理論範囲を求められる。3エポック目の量子化前の範囲は約−170.40〜756.59 cpで、保留ラベル5,895件のうち2,610件がこの範囲外だった。これは量子化だけでは説明できない振幅不足の証拠であり、学習量・スケジュールを変える仮説を支持する。ただし、実際の固定局面でのforward誤差や探索後MAEとは区別する。

## 計算・保存

- 現在のMac mini CPU、32 GiB RAM、既存SSD/NASだけを使う。重い処理は直列、解析jobs=1/Threads=1、build jobs=2。
- 追加SSD領域は8 GiB以内。開始時空き約30 GiB。各候補前に容量と残り時間を確認する。
- 学習1回は最大3,600秒、最大24エポック。実測から時間内の規模に調整する。期間終了間際には長い処理を新規開始しない。
- 専用学習runtimeを使い、前回のruntime・checkpoint・比較エンジンを変更しない。
- 生局面、教師ラベル、局面別診断、weight、raw logはGit外へ保存する。公開文書には集計・手順・ハッシュだけを載せる。
- PRは作成・更新まで行い、マージしない。

## 学習スケジュール

`scripts/train_cpu.py`は従来の`step-half`を既定のまま保持し、`--lr-schedule constant|cosine`、`--min-lr`、`--lr-schedule-epochs`を明示指定できる。設定をrunと各epoch metadataで照合する。基準学習率のmetadataはRustのf32値なので同じ丸めで比較し、実効学習率は学習ログに残す。

## 初回checkpointの固定診断

`scripts/diagnose_weights.py`は`--split train|holdout`を明示し、入力manifest・教師identity・checkpoint・専用学習器のhashを照合する。専用の`diagnose-external`経路は学習や教師探索を行わず、raw Adamの再量子化結果と推論用binが一致することを確かめて、量子化前float・量子化復元float・core推論を同じSTM cpで比べる。出力はGit外の専用runtimeに限定する。

| epoch | train静的MAE | holdout静的MAE | holdout予測標準偏差 | holdout量子化差の平均絶対値 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 705.198 cp | 739.536 cp | 182.482 cp | 2.911 cp |
| 2 | 642.673 cp | 720.075 cp | 287.611 cp | 4.235 cp |
| 3 | 604.317 cp | 713.274 cp | 327.245 cp | 5.179 cp |

MAEと予測標準偏差はcore推論の値。量子化差はraw floatと量子化復元floatの差である。holdout教師の標準偏差は1,345.374 cp。全6診断で教師ラベルを新規生成せず、coreとの整数化差を検証した。3エポック目のholdoutにおける量子化差の最大値は57.333 cp、量子化復元floatとcore推論の最大差は0.999955 cpだった。

holdoutの静的誤差は減少しているが、trainとの差は広がっており、学習不足と汎化の両方を追う必要がある。この結果から、同一55,404局面・seed42のrandom initを使い、constant LR=0.001で12エポックの候補を事前に選んだ。scheduleと学習量を同時に変える試行であり、エポック数だけの因果比較とは扱わない。採用は別途正式100万ノードで判断する。

診断機能追加前後の256局面・1エポックconstant学習smokeは、同じweight SHA-256 `df318ff8a4d2ec106126f36b4a1f3a308e1385a49d5a08b61bc5319f1900b993` を生成した。診断器追加による既存学習経路の変化がないことを小標本で確認した。

## Constant 12エポック候補

同じ55,404局面をconstant LR=0.001、seed42、fresh Adamで12エポック学習した。学習時間1,767.53秒、最大RSS約330 MiB、出力約765 MiB。最終weight SHA-256は `495373f626f4bf1bc7bd994a4daf802fd5c092c0ca59c62cd06d999b4ee936aa`。初回epochのbinは前回と完全一致しており、以降の学習率と学習量を変更した試行である。

| checkpoint | train静的MAE | holdout静的MAE | holdout予測標準偏差 | holdout量子化差の平均絶対値 |
| --- | ---: | ---: | ---: | ---: |
| constant epoch6 | 未測定 | 668.295 cp | 607.573 cp | 8.513 cp |
| constant epoch9 | 未測定 | 666.963 cp | 632.107 cp | 7.186 cp |
| constant epoch12 | 205.609 cp | 665.273 cp | 704.293 cp | 6.488 cp |

事前に選んだepoch12を正式比較へ進めた。epoch6/9は学習曲線の診断用で、正式候補の事後選び直しには使っていない。trainとholdoutの差が大きく、単なる学習延長では汎化の改善が小さくなっている。量子化復元floatとcore推論の差は引き続き1 cp未満。ここまでの数値は静的診断であり、採用結果ではない。

教師の絶対値が2,000 cp以下に限っても、静的MAEはtrain 117.32 cp / holdout 457.95 cpだった。5,000 cp超の割合はtrain 0.93% / holdout 0.71%で、極端な値の頻度差だけでは汎化差を説明できない。holdoutの負ラベル2,287件では教師平均−755.23 cpに対し予測平均+0.47 cp、符号一致52.16%だった。これらはデータ多様化や基本的な駒価値を初期値へ組み込む仮説を支持するが、データ不足・過学習・表現力の制約をこの集計だけで切り分けることはできない。

| 教師の絶対値 | train NNUE静的MAE | holdout NNUE静的MAE | holdout駒得静的MAE |
| --- | ---: | ---: | ---: |
| 500 cp以下 | 88.14 | 255.38 | 507.08 |
| 500超〜1,000 cp | 156.44 | 630.24 | 1,004.75 |
| 1,000超〜2,000 cp | 200.16 | 1,089.98 | 1,229.96 |
| 2,000超〜5,000 cp | 517.58 | 1,976.63 | 1,390.21 |
| 5,000 cp超 | 5,035.42 | 5,908.69 | 4,316.43 |

## Constant 12エポック候補の正式評価

固定100万ノード比較はMAE・Top3とも有効に完了し、候補は**不採用**。教師の570局面の結果とexact-cp集合266点、Top3対象551局面、モデル以外の実行identityが基準と一致することを再検証した。

| 指標 | 現行の駒得fallback | constant epoch12 | 候補 − 基準 |
| --- | ---: | ---: | ---: |
| MAE | 1084.479 cp | 1201.084 cp | +116.605 cp |
| Top3入り率 | 55.3888% | 29.9042% | −25.4845ポイント |

旧3エポック候補に対してMAEは減少したが、Top3はさらに低下した。学習量を増やして静的holdout MAEを下げても、探索後の両指標は改善しなかった。静的診断は次の仮説とcheckpoint選択に用い、採用判断を代替しない。[正式比較と5局の集計](validation/weight-improvement-2026-10-03/constant-e12/comparison.md)を参照。

MAE完了後、実行補助スクリプトが候補configをキー順に保存し、再読込したTop3側でengine・setoptionの順序が変わったため、既存validatorが不一致を検出して停止した。実行済みMAEのimmutable manifestから元の順序を復元し、config値の同一性、MAE全attemptとraw log、再生成reportの一致を確認してからTop3だけを実行した。失敗したv1と修復したv2は別の記録として保持し、基準やvalidatorを緩めていない。以後の補助スクリプトは順序を保って保存し、再読込後にも順序を照合する。

## 駒得からの初期化

次の仮説として、既存の駒得fallbackをdefault flat NNUEの初期重み内で厳密に表し、そこから外部教師ラベルへ学習する方法を準備した。探索、NNUE構造、`NnueOutput=absolute`は変更しない。`scripts/material_init.py`は固定commitとcore 4ファイルのSHA-256を照合し、標準ライブラリだけでSEKIRW01と上流互換sidecar、recipe、Python参照検証結果をGit外へ生成する。

各視点の自分の駒を「歩・と金」と「その他の非玉駒」の2群に分ける。FTの2 channelへ格納値`駒価値/2`、bias 64を置くと、復元後は`1 + 群の駒価値/128`になる。L2の4 unitで両視点の2群をコピーし、出力係数`+8192,+8192,-8192,-8192`と既存の最終`/64`で、手番側の駒得差と一致する。持駒は固定上流の4 bank・枚数thresholdを使う。

標準の駒総数以内では群の最大値は10,800/14,980 cp、FT最大累積7,554はclip上限8,128以下。各計算は整数または2の冪の分数で、f32の厳密表現範囲内に収まる。残り254 FT channelと28 L2 unitにはseed付きの小さな非ゼロ接続を置き、出力係数を0にする。初期出力を変えず、ClippedReLUの内側から学習を開始できる。ただし学習後の駒得保持や改善を保証するものではない。

seed42の生成weightは1,305,356 bytes、SHA-256 `bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40`。公開fixture 15局面の全手順とSFENをcshogi 1.0.4で検証した。`scripts/material_init_probe.rs`を固定coreの既存rlibと同じCPU指定・`panic=abort`でリンクし、fixtureと合法手の遷移・undoを計7,705回評価した。捕獲632回、成り186回、打ち577回、undo 3,837回を含め、明示的再計算と増分accumulatorの双方がmaterialと誤差0 cpで一致した。この検証は初期値の機能確認であり、棋力評価ではない。Pythonの検証記録は`engine_verified=false`を維持し、実機検証を別の`engine-verification.json`に保存する。

`scripts/train_cpu.py --init-weights PATH`は、固定構造・全float値の有限性・absolute sidecar・FNV-1aとSHA-256を検証して上流の既存初期化機能を呼ぶ。入力推論パラメータを使い、Adamのmomentとstepは新しくする。入力weightとsidecarが学習中に変わらなかったことも確認する。省略時は従来どおりrandom seed42。epoch集計metadataは初期化sidecarと形式が異なるため、そのまま初期化入力には使わない。

大きな出力係数による勾配感度の違いがあるため、random初期化と同じLRが適切とは仮定しない。学習集合の先頭4,096局面・1エポックでLR 0.00001と0.0001を直列に試した。固定holdoutのcore MAEはそれぞれ812.780 / 771.748 cp、量子化差の平均絶対値は13.996 / 66.193 cp、最大値は45.192 / 232.036 cpだった。量子化復元floatとcore推論の最大差は両者とも1 cp未満で、検証を通過した。

事前に決めた「検証を通過した試行のうちholdout core MAEが小さいLR」に従い0.0001を選び、55,404局面・6エポック・constant LR・seed42・fresh Adamで学習した。学習時間885.62秒、最大RSS 342,468 KiB。checkpointは事前指定したepoch1/3/6のうち固定holdout core MAEが最小のものを選び、同値なら早いepochを採る規則とした。

| epoch | holdout core MAE | holdout raw float MAE | 量子化差の平均絶対値 | 量子化差の最大値 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 706.155 cp | 662.064 cp | 184.409 cp | 585.550 cp |
| 3 | 756.924 cp | 688.566 cp | 217.461 cp | 908.437 cp |
| 6 | 762.816 cp | 711.384 cp | 174.558 cp | 744.860 cp |

規則に従いepoch1を選択した。SHA-256は `8a3e1503b2e6af14f54bc200dd757a7bb9a2abdf754b932052770bf4ab520f79`。同候補のpilotが安定・有効であることを確認し、正式100万ノード比較を開始した。学習の長期化はraw floatの保留MAEにも悪化を生じており、量子化だけが問題とは結論できない。

一方、epoch1の平均予測はraw floatの268.231 cpからcoreの449.403 cpへ変わっていた。量子化復元floatとcore推論の最大差は3 checkpointとも1 cp未満であり、今回の大きな差はrawパラメータを推論用へ変換する過程で現れる。最終的な採用には正式MAE・Top3を要求する。標準総駒数を超える任意SFEN、別の特徴構造、学習後の重みは初期値の厳密一致保証の対象外である。

## 保留データを固定したデータ拡大

`scripts/freeze_holdout_dataset.py`は、同じpack・ゲーム単位split seed・独立1,000局除外条件で作成した拡張datasetからtrainを取り出し、以前のholdoutをそのまま保持する。両入力のmanifest・4ファイルのhash/サイズ・教師identity・ゲームの由来を照合し、旧holdoutゲームと盤面をtrainから除外する。盤面比較は手数を無視し、手番と持駒は区別する。拡張時に追加されたholdoutゲームはtrainへ加えず、未使用の保留として残す。新しい最終評価集合を作る処理ではない。

出力のholdout 2ファイルは旧ファイルとバイト単位で一致し、trainラベルは残ったtrain局面の集合に一致する。入力ファイルが処理中に変わらなかったことも確認する。新規のGit外ディレクトリにだけ出力し、現行の学習・診断ツールで使用できる4ファイルのmanifestと派生条件を記録する。5件のfixtureテストと、既存55,404/5,895局面・予約holdoutゲーム246件の読み取り検証を完了した。

同じ13 packについて各400ゲームのprefixを使い、train **112,681局面** / frozen holdout **5,895局面**を作成した。準備全体13.53秒、holdout固定処理4.08秒。元holdoutの2ファイルはバイト単位で一致し、旧holdout 246ゲームと新たに予約された283ゲームは学習へ入れない。入力も処理前後で不変だった。独立1,000局の全88,187固有盤面を機械的に除外する条件は同じで、final専用ファイルは開いていない。派生manifest SHA-256は `ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6`。このデータでの学習はまだ実施していない。

## 同一checkpointの量子化方法の比較

固定trainerの推論用exportはFT weightとFT biasを64倍し、`[-32767,32767]`に制限してゼロ方向へ切り捨てる。今回の初期material経路ではFTの1刻みが初期状態で約2 cp/featureに対応し、小さな係数差が予測値へ増幅されうる。実際の学習後の影響は固定局面で測り、初期状態の感度だけから一般的な誤差上限を主張しない。

`scripts/export_nearest.py`は元Adam JSONをf32へ復元し、従来の切り捨てexportが元binの**全バイトと一致**することを要求する。その後、FT weight/biasだけを「f32復元→64倍→clamp→最近傍・厳密なhalf tieは偶数側」へ変更する。例えばscaled `+1.5→2`、`+2.5→2`、`−1.5→−2`。L2と出力層はoffset 1,239,560以降の元バイト列をそのままコピーする。

派生binとabsolute sidecar、入力hash・固定source hash・丸め規則・FT変更数を新しいGit外ディレクトリへ保存する。元binやAdamを置換せず、派生binを元Adamのnative exportと偽るsidecarも作らない。既存の`diagnose_weights.py`の再export一致ゲートは維持する。

独立した`scripts/core_pair_probe.rs`は固定coreへリンクし、native/nearestの直接評価と、それぞれの量子化復元floatを比較する。FT加算の各prefixでi16飽和がなく、float/coreの差が1.001 cp未満であることを要求する。旧診断との比較では`x86-ftz-daz`を明示し、MXCSRを記録する。実行する際は同一holdoutの順序・ラベル・件数を固定し、native予測が既存診断を全件再現することを先に検証する。

公開可能な合成入力のテストは通過した。固定coreへのbuildと実checkpointの比較は正式探索の終了後に直列実行する予定であり、現時点では量子化方法の改善や正式採用を主張しない。

結果と再現手順は検証後に追記する。初回実験の基準は[初回weight](FIRST_WEIGHT.md)、全体の比較条件は[研究方針](RESEARCH.md)を参照。
