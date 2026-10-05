# Huber1000候補のソース準備

Issue [#19](https://github.com/phni3j9a/sekirei-weight2/issues/19)の次候補として、第1候補と同じ白視点・対非線形headで、CP誤差のMSEを **2倍のHuber損失**へ変更する。δは1000 cp、f32 bitsは `447a0000`。学習出力を64で割ったscoreと教師cpの差が `err` であり、δに追加の尺度変換はしない。

`|err| <= 1000` では旧MSEの `loss = err * err` と、勾配の `checked_product(checked_product(weight, 2), err) / 64` を同じ演算順序で保持する。超過域だけを `loss = 2δ|err| − δ²`、勾配を `checked_product(checked_product(weight, 2), sign(err) * δ) / 64` に変える。上限処理の前に非有限誤差を拒否し、tailのoverflow検出と失敗後のpoisonも保持する。小残差域の実更新が旧MSEとbit一致することは、今後の実コンパイル済みfixtureで確認する。

入力は第1候補と同じ凍結dataset（train 112,681局面、holdout 5,895局面）。seed42の新規初期化からshuffle/resumeなしで3epoch、338,043更新、epoch3選択を固定する。Adam・定数learning rate・初期化・保護materialのparameterとm/v・tie制約・nearest03量子化・checkpoint/native codec・固定White core・Sekireiの探索と100万ノード比較条件を保持する。学習lossは採用基準にせず、最新bestに対するMAEの厳密低下とTop3の非低下を要求し、finalを日々の選択に使わない。

専用modeは `white-view-diverse-games-seed42-huber1000-e3-v1`、objectiveは `absolute-cp-twice-huber`。新plan、source preparation、build、recipe、source binding、parent preflight、completion、own proofの専用schemaとmodeを厳密に照合し、入力・source・compiler・binary・モデルのpath/bytes/full SHA-256を束縛する。旧MSEのwrapperと凍結済みsourceは変更せず、過去のreceiptを新候補の実行証拠に読み替えない。

[public-source-manifest.json](public-source-manifest.json) は3ファイルのbefore/after pins、[source/](source/)、[candidate.patch](candidate.patch)、必須のRust test suffix 10件を記録する。変更はproductionのloss/gradientと専用variant宣言・検証・cfg(test)追加である。manifestの初期 `status: source-only`、`compiled: false`、`training_started: false` は、この静的なソース準備記録として固定する。後の実行結果は別receiptで記録する。

実行入口は次の5 CLIと共有検証moduleである。各CLIの引数は `python3 -B scripts/<名前>.py --help` で確認できる。

| 入口 | 役割 |
| --- | --- |
| [prepare_weekly_nonlinear_huber1000.py](../../scripts/prepare_weekly_nonlinear_huber1000.py) | 固定upstreamと凍結plan/profileから新しい専用sourceを準備 |
| [weekly_nonlinear_huber1000_build.py](../../scripts/weekly_nonlinear_huber1000_build.py) | 固定compiler/depsで実テスト・ビルドし、source-bound receiptを作成 |
| [weekly_nonlinear_huber1000_preflight.py](../../scripts/weekly_nonlinear_huber1000_preflight.py) | 新候補のfullrefsを再確認し、専用recipe/binding/parent preflightを作成 |
| [weekly_nonlinear_huber1000_run.py](../../scripts/weekly_nonlinear_huber1000_run.py) | initialized readerのpositive/negative検証後、専用fitを監督し完了stateを検証 |
| [weekly_nonlinear_huber1000_post_training.py](../../scripts/weekly_nonlinear_huber1000_post_training.py) | 親からpure validatorを受け取り、入力bytesと3epochの観測を検証する共有module（独立CLIなし） |
| [weekly_nonlinear_huber1000_proof.py](../../scripts/weekly_nonlinear_huber1000_proof.py) | この候補のcompletion/modelに束縛したcore/incremental/nativeの技術検証 |

再現には、原典 [Sekirei f09c130](https://github.com/kent-tokyo/sekirei/tree/f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9)、[既存MSE snapshot](../white-view-paired-nonlinear-rust-v1/README.md)、runtime修復ソース、凍結済み非公開data/plan/profileと固定White engine・compiler/depsが必要である。専用source/build/outputをSSD上に新設し、同じLinux x86_64 platform・CPU/浮動小数点条件を満たす。[環境・非公開成果物の扱い](../../docs/ENVIRONMENT.md)に従う。この3 postimagesと差分patchだけでは単独のbuild環境にならず、教師data・weight・raw label・private runは付属しない。

現時点では公開ソースの準備とPythonの構文・help確認までで、新専用sourceの実作成、実コンパイルは未実行である。実compiledログに束縛した10 suffix、calling-thread FTZ/DAZ・MXCSR同条件での小残差3更新の全parameter/m/v/step bit-controlsとinitialized/native codec照合、実position失敗後のpoison検証、initialized readerのpositiveとmissing-FTZ negative、preflight、3epoch fit、候補自身のown proof、MAE/Top3のpilot/formal 4段階比較は、すべて未実行のgateとして残る。小fixtureは完了checkpointの338,043更新条件を緩めず、synthetic Huber helper失敗と実position失敗を区別する。準備ソースやhelpの成功から学習改善・正式比較成功・モデル採用を推定しない。
