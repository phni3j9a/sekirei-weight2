# 1週間の継続ウェイト改善（Issue #19）

## 期間と起点

運転期間は **2026-10-05 18:52:22〜2026-10-12 18:52:22 JST**（UTCでは09:52:22、168時間）。ユーザーの許可によりgoalを開始した。採用が一件得られた後も、残り期間に次の改善を進める。期間終了とモデル改善の成功を区別し、改善未達も記録する。

起点は[第8候補の採用モデル](validation/autonomous-weight-improvement-2026-10-04/best-model.json)、白視点・対非線形固定E3。正式MAEは907.6716211807516 cp、Top3入り率は59.951319613787%。モデルSHA-256は `f647864fa17a7e9d06ed44aed6894527128c721208bac1f761741c3c5ec1e042`。既存重みの実体を照合した。新しい採用モデルが得られるまではこのモデルを最良として保持する。

## 採用と継続

規定のdevelopment 5局・100万ノード正式比較を維持する。採用には、最新bestに対するMAEの厳密低下とTop3の非低下を同時に要求する。保存した各局の整数から5局等重みの有理数で判定し、表示丸めや学習lossを採用理由にしない。候補自身のMAE pilot・正式MAE・Top3 pilot・正式Top3を検証し、重み読込・入力集合・教師・比較identity・全attempt・USI lifecycleを確認する。

モデル更新のたびに比較基準を新bestへ進める。旧best、技術失敗、不採用候補の条件と成果物を保存する。既存fallback専用consumerをそのまま学習済みbestとの比較へ流用しない。探索実装・教師・比較設定は固定し、重み・特徴量・NNUE構造・学習方法・学習データを改善対象とする。final 5局を日々の仮説選択や局面採掘に使わない。

原典と今回の実測から一仮説ずつ選ぶ。不採用が続いたときはデータ・表現・量子化・学習目的等を診断し、単純なepoch/LRの再試行に偏らない。同じdevelopmentと既存holdoutの反復使用を、新しい棋譜への汎化の証明とは扱わない。

## 最初の準備と仮説

開始時は次の準備を進めている。準備の検証をモデル改善の成功に読み替えない。

- 前回の学習後親処理に残った未定義変数と、未実行のRust native fixtureの型不整合を、過去snapshotを保持した新しい実行経路で修復する。
- 現採用E3を明示的なincumbentとして、候補対候補の正式証拠を比較するconsumerを追加する。
- 教師packの各400ゲームprefixという抽出規則を見直す。最初はpack全体のゲーム境界から固定hash順位でゲームを選び、モデル構造・学習量を維持してデータの多様化を検証する案を準備する。固定holdout、独立1000局の全盤面除外、ゲーム分離、盤面重複除外を維持する。候補の選択規則は生成と正式比較前に固定する。

学習済みbestを明示する `scripts/compare_incumbent.py` を追加し、公開16 fixtureと保存済みE3の4段階rawの再検証に成功した。E3対E3の自己比較は8 identityが一致し、入力前後不変、MAE `1446737797/1593900`、Top3 `321208691/535782520` を再確認した。同一モデルなのでadopt=falseであり、新しい改善や採用ではない。最初のconsumer実機確認はcanonical occurrence IDと教師metric keyの表記差で失敗し、回帰fixtureと対応付けを修正した。失敗原本を保持し、新しいv2出力で成功を確認した。[公開再検証集計](validation/weekly-weight-improvement-2026-10-05/incumbent-revalidation.json)にはrawやローカルパスを含めない。

教師データの抽出は実機で完了した。最初に固定した各pack500局は、1個のpackが全392局のため31.87秒で失敗し、学習を開始しなかった。全13packの境界索引を確認し、12個各500局・短い1個392局、計6,392局を明記した新profileと事前登録v3を生成前に固定した。選択数を暗黙で縮めず、全局数・選択数・pack集合・source closureを検証するv2 producerへ改めた。

再生成は53.85秒で成功し、112,681学習局面とbyte一致の5,895 holdout局面を維持した。実際に学習行へ使ったゲームは4,537局、旧学習盤面との重複は14,782局面。手数と評価値帯の分布は旧データと近く、pack内prefix限定による偏りの有無や改善効果は未検証である。[公開データ準備集計](validation/weekly-weight-improvement-2026-10-05/diverse-data-preparation.json)は失敗と成功のreceipt SHAを含み、生局面・ラベルを含めない。

固定upstreamと旧complete patchを全postimageで確認し、専用trainer sourceを新runtimeへ展開した。新Readerは実manifestのmetadata検査を通過した。数値更新・scalar Adam・export・initialized I/O本体と旧snapshotを維持し、Python親にはgateを明示注入する。新親の小実プロセスfixtureで正常終了、exit失敗、timeout、SIGTERM時のwait/reapとgroup停止を確認した。長時間学習の前に、この候補自身の読込・初期状態保存/読込とFTZ未設定時のI/O前拒否を実行する。

専用trainerのCargo testは33件（failure/ignoreとも0）、buildは成功した。test 43.39秒・build 40.69秒、両childのwait/reapとgroup空、538 source・compiler2・dependency2,268件の前後不変を確認した。[公開build集計](validation/weekly-weight-improvement-2026-10-05/trainer-build.json)を残した。別agentの読取レビューでも、旧snapshotのnumeric update/Adam/native/export/floatと固定white coreのbyte維持を確認した。学習とモデル採用の成功はまだ主張しない。

## 計算と保存

現在のMac mini CPU・32 GiB RAM・既存SSD/NASだけを使う。解析jobs=1、Threads=1、build jobs=2、重い実験は直列。開始時SSD空き約27 GiB、NAS空き約5.9 TiB。初期の追加SSD作業領域は8 GiBを目安とし、各候補前に空き容量と時間を再確認する。

前回の非線形3epoch学習は、保存済みchild outcomeとcompletionの一致するwall timeで13,200.321秒（約3時間40分）。旧sparse trainerの時間は流用しない。正式4段階の過去実測は約66〜72分だが、新候補の実測で見積りを更新する。

Git/worktree・build・venv・使用中データと重み・実行中出力はSSDに置く。今回新規の完了runは、停止、ファイル集合・size・SHA-256一致、参照依存の復元可能性、元パスからの参照を確認して `/mnt/storage/NAS/sekirei-weight2` に保管する。確認済みの今回のSSD重複は整理できる。過去原本と既存runの参照を保持し、単なるsymlink置換でvalidatorが通るとは仮定しない。詳細receipt、モデル、教師、生局面・ラベル・ログは非公開に保持する。

NASの小fixture保存は初回のpublishで失敗した。実mergerfsが `renameat2(RENAME_NOREPLACE)` をEINVALで拒否したため、SSD原本とNAS stageを保持し、archive成功として扱わなかった。既存先を上書きしない排他mkdir・全集合物理コピー・manifest最終公開の別方式を準備し、実機で再検証する。

## 記録と終了

Issue [#19](https://github.com/phni3j9a/sekirei-weight2/issues/19)を起点に専用worktreeを用いる。意味のある成果単位でcommit/push・PR作成または更新を行い、READMEを現状へ更新する。モデル採用とGit統合を区別する。PRマージと定期実行は今回の許可に含まれない。

期限前は、新しい長時間処理を控えて検証・保存・停止確認・報告の時間を確保する。期末に最新bestと更新履歴、不採用と技術失敗、残事項、未完了処理、再開状態を残す。利用上限、容量や技術障害が生じた場合も状態を保存し、期間内に達成した範囲を報告する。
