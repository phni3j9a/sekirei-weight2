# White-view NAS launcher — SOURCE preparation

SOURCE ONLY。実C/control/model/data/raw/proc/runtime/NASは読み書きしていません。実コピー、マウント検査、容量検査、エンジン・学習・build・rsyncは起動していません。新/tmp packageのみ作成し、公開Rは不変です。

`white-view-nas-archive-launch-worker-v1.py` と `white_view_archive_runtime_v1.py` の `PROTOTYPE_ONLY=True` により、actual entryはargparse・read・write・process・signal mask操作より前に停止します。Rootだけが後で別enabled bytesを作成し、全外部SHAを固定して実行できます。`white_view_postformal_contract.py` は既存凍結811e6c323c4a3dccfa5c1faa4a6d8aecb6c38ec67223b34327f73e8f133b7d16の同一copyで変更していません。旧第6候補のschemaや成功flagは合成しません。

## 接続を実装した範囲

専用 `sekirei.white-view-archive-request.v1` の22-key exact contractを検証します。statusは `frozen-after-stopped-before-nas-copy`。STOP・Rootのstop terminal・selected mapping/inventory・12raw run roots・whole Q fit/core/incremental trees・必須control/19gate refs・SSD original→control-copy ledger・全inputのarchive/external dependency分類を要求します。有効な採用・不採用のどちらも保存できます。archiveの成功は採用適用、best-model更新、goal完了を意味しません。

Root提供の `/tmp/sekirei-weight2-white-view-generic-archive-helper-source-v1.py` を読取り、既存helper `5520a647ed77f25e47cd7208fd13ee9b6ae10d619a836c8b36709d33308a2816`（16384B）のCLI・inventory・receipt・lockを確認しました。actual helperのpathはcontractの `campaign-15-20261002/archive-helper.py` に固定します。source-pinned helper moduleは `inventory` / `combined_inventory` / `load_mapping` だけを親の照合に再利用し、同じ実helperを独立childとして起動します。

helperは `--mapping` / `--destination` / `--receipt-name` / 4個の `--lock` を受けます。helperがmount/RW/private-directory/容量・fresh NAS destinationとreceiptを検査し、copyを実施します。親はそれを自己申告で成功へ変換せず、helper exit0/wait/reap/PG停止、専用STOP controls不変、status `sekirei.private-archive.v1` / `complete` / `verified`、helper source/mapping SHA、4lock集合、件数とbytesを再検証します。

Rootの選定inventoryは、helperの `combined_inventory` と同じ全file・directory集合で凍結する必要があります。targetのancestor directory、root `.`、空directoryも含め、directory listはsortedにします。親のsource-before/source-after、helperの `source-before.json` / `source-after.json` / `expected-destination.json` / `destination.json`、現在destinationをすべてselected inventoryへ一致させます。helper本体はsymlinkを保存できますが、white-view契約では**全symlinks={}を要求**します。

## Lockとプロセスの所有

親の5EXはQ.fit、old FAN.build、新N.build、新N.prepare、新N.benchmarkです。helperの4EXはT.training、old B.prepare、old B.benchmark、old FAN.trainingです。両集合は重複せず、合計9個をEXで閉じます。親のlock FDはchildへ渡しません。helperは既存lockをreadonly/no-followで開き、新lockを作りません。

親がholderを所有し、Popen direct assignmentの間だけ親Python signal exceptionをdeferします。Popen前にSIGINT/SIGTERM maskをblockしないため、child/helper/rsyncへの通常のsignal配送を妨げません。wait timeoutは固定3600秒です。失敗・取消時はTERM、bounded wait、KILL、再waitとPG消滅観測を行い、leaderが先に終了していても残るdescendantへKILLします。2回のempty own-PG scansと実wait/reapが成立した時だけcleanup成功を記録します。

親5lockは全cleanup試行と失敗記録まで保持します。cleanup未確認ならhandleを保持し、wait/reap/group-stoppedを成功とせず、元errorとcleanup_errorを別々に残します。最終再cleanupが失敗しても最初のerrorを隠しません。プロセス終了後のlock保持や関連process不在をこの子自身から主張できないため、Rootの外部tool terminal/再観測は別途必要です。partial NAS destination/receiptは保持してinvalidとし、resume・再利用・削除を行いません。

結果は新規C `white-view-nas-archive-launch-v1.json`、正常時 `white-view-nas-archive-result-v1.json`、失敗時 `white-view-nas-archive-failure-v1.json` に分けます。全部exclusive 0600です。全selected source/inputとcontrol出力の双方向overlapを拒否します。正常result書込みが途中失敗しても、そのpartialを上書きせず別failureへ保存します。これら後から作るlauncher/result/Root requestをselected inventoryへ自己参照させません。

## Rootが整えるactual入力・残事項

準備コードの作成者はactual request/stop terminal/mapping/closure inventoryを読取・検証していません。Rootの実作成・検証とは区別します。Rootが実四段階/3audits/bridge/comparison/STOPと、自分のstop executorのexit0/closed/reaped/groups stoppedを観測し、外SHAを凍結する必要があります。新runtime・model gate等のproof bodyはこのarchive launcherで再実行しません。STOPの外SHA-bound専用証拠と全input不変を受けます。

request.inputsには、newworker/runtime/contract/generic helperのfullrefs、AUDのresolved regular interpreterと`pyvenv.cfg`も含めます。selected closureへコピーするか、`external_dependencies`へ明示して分類します。actual commandはAUDのpath/prefixとこれらcurrent SHAを検査します。Rootが現NAS mount・容量と3600秒のコピー予算・全writers停止・新destination/receipt未存在・SSD closure freezeを確認することは残項です。helper自身もmount/capacityを検査しますが、準備段階の成功証拠ではありません。

Root enabled-copy CLI案（現在実行不可）:

```text
AUD/python -B C/white-view-nas-archive-launch-worker-v1.py
  --request C/white-view-archive-request-v1.json
  --expected-request-bytes ACTUAL --expected-request-sha256 EXTERNAL
  --expected-worker-sha256 EXTERNAL --expected-runtime-source-sha256 EXTERNAL
  --expected-contract-sha256 EXTERNAL --expected-stop-session-id ACTUAL
```

コピー元の原本はSSDに保持します。NASは実験証拠の保管で、build/venv/compiler/private教師ファイル等の外部依存を含む全実行環境の単独restore証明ではありません。公開projection/size-SHA manifest、Gitの統合、best-model/goal、Root terminalの作成はこのlauncherの範囲外です。

## 合成検証

18 tiny fixturesを実施しました。normal valid positive/negative request、旧schema・typed terminal拒否、全file/dir/size/SHA差異・source変更・symlink拒否、helper status/source/mapping/lock/count差異、before-I/O barrier、lock split、Popen中取消/非継承mask、leader終了後descendant KILL、wait/scan/cleanup失敗とhandle保持、実launcher bodyの5lock保持・元error保持・未完了helper receipt拒否を確認しています。

process/lock/signalはfakeです。reader/writer/helper inventoryの実I/Oは新規/tmpの公開toy textだけで、mount/rsync/NASは実施しません。helper inventory sourceは同SHAの許可コピーで、実helperや実保存先を読んだ証拠ではありません。public CIへ移す場合はsource-only helper fixtureのpathをportableの固定source fixtureへ対応させ、actual helper CLIへの変換は行わないでください。

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover
  -s /tmp/sekirei-weight2-white-view-nas-launcher-source-v1 -p test_white_view_archive.py
```
