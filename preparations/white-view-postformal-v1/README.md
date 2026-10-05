# White-view postformal stop and NAS request source preparation

SOURCE ONLY。Root以外はactual comparison/control/model/raw/data/runtime/proc/NASを読んでいない。新候補選択・fit・build・engine・proof実行なし。公開R/Cは変更せず、この新/tmp packageのみを編集した。

最小の実装は専用`white_view_postformal_contract.py`とdisabled `white-view-postformal-stop-worker-v1.py`。Rootがactual四段階・三つの独立raw audit・fallback semantic bridge・same-new-runtime comparisonを完了し、比較toolのexit0/closed/reapedを外部観測してからだけ、別enabled sourceを凍結して使う。module/worker双方のPROTOTYPE_ONLYはTrue、mainはargparseやI/Oより先に拒否する。

## Request/terminalの提案契約

- request: C/white-view-postformal-stop-request-v1.json、schema `sekirei.white-view-postformal-request.v1`、status `frozen-after-reaped-comparison`。
- Root comparison terminal: C/white-view-comparison-executor-terminal-v1.json、schema `sekirei.white-view-comparison-terminal.v1`、status `observed-stopped`。
- STOP: C/white-view-formal-stopped-v1.json、schema `sekirei.white-view-formal-stopped.v1`、status `verified-stopped-under-locks`。
- common/worker canonical source: C/white_view_postformal_contract.py、C/white-view-postformal-stop-worker-v1.py。

comparison terminalは、comparison_session_id（正整数）、exit_code=0（bool不可）、session_closed/tool_observed_reaped/all_related_groups_observed_stopped=True、comparison/comparison_request各fullref、source_head、final_used=False、goal_complete=Falseを含むexact shape。これはRootが実tool completionを観測して新規保存するproposalであり、source側では実terminalの存在や成功を主張しない。candidate executor sessionとcomparison sessionを取り違えない。CLIは両expected sessionを別々に要求する。

requestはcomparison/比較request/current candidate PF/gate/bridge/old-fallback/new-fallback/candidate audit/candidate terminal/factory source/固定opaque allowlistの12refs、gate19refs、現在model、Git HEAD、4actual evaluation module fullrefs、全source/helpers/transitive immutable mapを束縛する。fullrefは{path,bytes,sha256} exact型。既存fixed factory source a6f835348716585925bce2396cf5b8021835d44031648b7971cefb2914dea556、measurement HEAD 6d099c79b4cac03390b9d861f11d05e083f0e66b、opaque allowlist 0cb4e02d…を維持する。実moduleのenabled bytes SHAはRootのrequestに要求し、disabled D4 prepをactual enabled sourceとして偽装しない。

## 停止workerの実装範囲

D4のstrict reader/typed PF/gate、read_audit_bundle/read_verified_bridge、compare_same_runtimeをreuseする。comparisonを三つの外SHA-bound audit payloadとbridgeから再構成し、exact typed full JSONの一致を要求する。採用は5ゲーム等重みexact FractionでMAE strict decrease AND Top3 nondecrease。valid negativeも正常停止として保存し、adoption_applied/best_model_updated/goal_completeはFalse。旧第6候補schemaへのflag合成は行わない。

Rootfactoryv2のproc_identity/stopped_processesはAST同一のsourceを複製した。7協調lock（B/N prepare・N buildはSH、他EX）とB/N benchmark EXを保持し、lock内でrequestをstrict再parse/外SHA再照合、前後source/head/hash不変、2回empty related-process scan、固定5opaque descriptors full証跡を要求してからSTOPをexclusiveに書く。未知PermissionErrorはfatal。全5サービスが毎scan同じPermissionErrorの分類に入ることを要求するため、退出・権限変更の場合はfailclosedとなる。新opaque例外は発明しない。

STOPは監査由来12run rootsと全raw file identitiesを記録する。原則SOURCEの/PROC関数を再利用するだけで、Root外部terminalから真実性を受ける。自身のtool exit/reapedは子が観測できず、このSTOP自身のreaped成功は主張しない。後続NAS用のRoot stop terminalが別途必要。

Root enabled-copy CLI案:

```text
AUD/python -B C/white-view-postformal-stop-worker-v1.py
 --request C/white-view-postformal-stop-request-v1.json
 --expected-request-sha256 EXTERNAL
 --expected-worker-sha256 EXTERNAL --expected-source-sha256 EXTERNAL
 --expected-executor-session-id ACTUAL --expected-comparison-session-id ACTUAL
```

source/helper/data/runtime入力がfrozen HEADの状態を保つ間に実行する。公開文書更新でmeasurement HEADを変える前に停止/closureを確定する。

## NASへ進む最小契約と再利用範囲

`validate_archive_request`は専用 `sekirei.white-view-archive-request.v1` / `frozen-after-stopped-before-nas-copy` を検証する。専用Root stop terminal `sekirei.white-view-postformal-stop-terminal.v1`（stop_receipt/request fullrefs、stop_session_id正整数、exit0/closed/reaped/groups stopped、HEAD、final/goal False）を要求する。

Rootが新SSD completion dirへcontrol/sourceのexclusive byte copiesを作った後、そのoriginal→copy fullref ledger・選定mapping・全files/dirs/size/SHA inventoryをrequestへ束縛する。全12raw run roots・whole Q fit・core/incrementalのraw treesを選定mappingへ必須にし、STOP/request/12controls/gate19refsもledger込みで到達性を確認する。共有C/runtime/build/venvをwhole rootでコピーしない。選定から外すcompiler/runtime/既存dataset等は全immutable input identityをexternal_dependenciesへ明示し、NASだけで全環境を復元できるとは言わない。二重copy/overlap mapping、原本とcopyの違い、未分類inputを拒否する。

変更しないgeneric archive-helper5520a647…の、mapping→fresh NAS pathへのコピー・source-before/source-after/destination file/dir/size/SHA全照合・SSD原本保持の範囲は再利用可能。旧paired completion/archive workerのcandidate/native/fit/CP/AU/schema検証は再利用不可。NAS launcherのholder/deferred signal/wait/reap/bounded group cleanup実装はsource移植の対象となるが、このpackageではactual launcherを実装/有効化していない。Rootが実closure/容量/mount/新destination/receipt未存在を検証し、新専用launcherをfreezeすることが残項である。

generic helperはT.training/B.prepare/B.benchmark/FAN.trainingの4EXlockを所有、親はQ.fit/FAN.build/N.build/N.prepare/N.benchmarkの5EXlockを所有する分割を提案する。親とhelperが同じlockを二重取得しない。両者の合計は停止対象9lock集合を全EXで閉じる。childが生きている間とcleanup完了まで親lockを保持する。許可されたgeneric helper本体は今回のread範囲外なので、既存launcher sourceが記録する5520…契約だけを根拠とし、helper自体の実互換性/変更有無/actualarchive成功は主張しない。

`validate_archive_equality`は全選定closureのsource-before/source-after/destinationについてexact fileset・directoryset・size・SHAを要求する。copy中断/エラー/partial destinationは成功にしない。freshNAS pathのみ、再利用なし、SSD削除なし。測定/adopt boolを記録することとbestモデル更新/goal完了とは別である。

## 検証範囲

14 tiny synthetic tests PASS。実pure D4 PF/gate envelope/bridge/compareへの接続、validnegative/positive、terminal型/session/exit/closed/reap、gate19/model/sourcemap、full comparison/rational/semantic改変拒否、5opaque full証跡、archive原本/copy provenance・全12raw/fit/proof必須・terminal失敗・overlap/root拒否、全inventory equality、factory2 process関数AST同一、before-I/O barrierを確認した。

gate fixtureは許可されたD4 pure consumerのenvelope expressionを合成したschema fixtureであり、actual native/fit/numeric/core証明の独立実行ではない。actual工学の成功/全runtime closure/プロセス不在/容量/マウントは未検証。actorはRootのみ。
