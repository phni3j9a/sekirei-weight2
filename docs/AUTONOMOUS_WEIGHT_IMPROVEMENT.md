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

探索実装・教師・100万ノード条件・二指標の採用基準を維持する。重み・特徴量・NNUE構造・学習方法・学習データは改善対象。現在のMac mini CPU・32 GiB RAM・既存SSD/NASだけを使い、重い実験は直列、解析jobs=1/Threads=1、build jobs=2とする。開始時SSD空き約27 GiB。初期の追加作業領域は8 GiB以内を目安にし、候補ごとに容量と時間見積りを確認する。既存原本を削除して容量を作らない。

今回のprivate campaignは`campaign-17-autonomous-v1`。前回の実行・比較補助を新campaignへコピーし、削除済みIssue #15 worktreeへの参照だけを今回のworktreeへ変更した。比較を担う既存6スクリプトの内容hashは前回の正式比較と同じで、固定cshogi 1.0.4 / NumPy 1.26.4のvenvで実行する。適応補助のsource・変更後hashと候補identityはprivate receiptに残す。途中停止時は既存の厳密resumeで欠落attemptだけを再開し、技術失敗・破損・identity不一致を削除や書換えで救済しない。

生局面・ラベル・weight・raw log・詳細receiptはGit外に保持する。完了成果物はSSD原本を維持し、ファイル集合・サイズ・SHA-256を照合して既存NASへ保管する。公開Git/Actionsにはコード・集計・条件・ハッシュだけを載せる。利用上限や技術的な障害は率直に報告し、未達のままgoal完了とは扱わない。

## 実行状態

最初の候補の入力hashとabsolute metadataを確認し、専用worktreeとprivate campaignを準備した。正式比較は実行前で、採用判断は保留。最良モデルはfallbackを維持する。

関連: [Issue #17](https://github.com/phni3j9a/sekirei-weight2/issues/17)、[前回の実験](WEIGHT_IMPROVEMENT.md)、[研究方針](RESEARCH.md)、[環境](ENVIRONMENT.md)。
