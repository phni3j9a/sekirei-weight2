# White-view 評価 runtime 接続の SOURCE 準備 v3

SOURCE のみ、全 `PROTOTYPE_ONLY=True`。実 control・model・棋譜・raw・成績の読取、engine/probe/build/fit、四段階測定は実施していません。Root が別の enabled bytes を固定し、外 SHA 付き入力を作成してから使用します。旧 public helper、guard、DEFAULT_RUNTIME、config、旧 native loader は変更していません。

`/tmp/sekirei-weight2-white-view-evaluation-source-v1` と runtime contract v1 の凍結 bytes は保持しています。D2 の凍結 bytes も保持し、本 D3 は cleanup と正式 gate 接続を別 variant として修正しました。

## 接続した source

- `white_view_evaluation.py`: explicit-runtime spec、role config、四段階順序、typed terminal/complete descriptor、固定 semantic projection、各局の整数からの exact Fraction、old/new fallback bridge と same-newbinary 8 identity 比較。
- `white_view_runtime_io.py`: canonical regular file/full raw SHA、typed pin の上書き拒否、fresh private output、immutable exclusive write、自作 state の atomic replace、全 raw directory membership inventory、既存 lock、signal を子へ mask 継承しない親 process holder と bounded wait/reap/group cleanup。
- `white_view_production_ports.py`: AUD interpreter/decoder/Git/helper map、専用 build verifier と actual input inventory、Cargo2 の両端の fullsource 重複 binding、旧 helper API の explicit-runtime 呼出し、全1829 strict raw/options/nodes/USI/lifecycle 再検証、payload/receipt writer、全 inventory 再確認による receipt reader。
- `white_view_operations.py`: `launch`、FD 継承した `child`、Root tool terminal 後の `complete`、`audit`、`bridge`、`compare` の source。すべて disabled で、`--help` も I/O 前に停止します。

旧四段階の呼出しと raw parser 本体は変更していません。clock/NPS/info 更新頻度、depth、実 reported node 数は bridge の意味一致に含めません。両側で全 node 文法と 1M/1.01M cap を再検証し、実最大 nodes と binary/build/runtime の差を別に残します。

## 実行準備の順序と外 binding

1. Root が source/build/runtime を検証し、`make_spec()` と `validate_launch_preflight()` に適合する `sekirei.white-view-evaluation-launch-preflight.v1` を SOURCE 外の新 private control として固定します。config_source、new_build_binding、clean repo_identity、numpy/cshogi versions、全 module/legacy helper・source・compiler・binary・teacher・Python を含む `spec.inputs` が必要です。
2. unloaded fallback の新 runtime 四段階を先に測ります。102 MAE pilot → candidate 自身の pilot gate を config へ固定 →1140 MAE →36 Top3 pilot →551 Top3 です。root/4run IDs は全部 absent、新規で、resume/reuse は拒否します。
3. `launch` は T training EX、旧B prepare SH、Q fit EX、旧Fan build/training EX、新N build/prepare SH を開始から cleanup/保存/rehash 終了まで保持します。親の flock open descriptions を `pass_fds` で継承し、子は path/inode/型/全 inventory を照合します。inner benchmark は旧 API の prepare SH/benchmark EX を自身で取得します。
4. 子は draft のみを出し、自身の reaped/cleanup complete を主張しません。親は実 wait/reap/子 PG 停止を確認しても `executor_tool_reaped=False` の parent outcome を出します。Root の wait tool が parent executor exit0/reaped を観測し、全 related groups の停止を別に確認した external normal terminal の全文 SHA を与えた後、`complete` が OP と typed state を出します。
5. old stock fallback と新 fallback の全1829 rawをそれぞれ `audit` で再parseします。各 receipt は別 payload fullref を参照する一方向です。payload に自身の receipt SHA は入りません。
6. `bridge` は両 receipt/payload/terminal/raw inventory を現物 SHA と membership で確認し、固定 semantic projection と各局の整数 metric が完全一致することを要求します。旧 execution と新 execution の差は保持します。旧 results を新 baseline として測定したことにはしません。
7. 同じ新 binary/build の candidate 全四段階が complete になった後、candidate の raw audit と `compare` を実行します。8 legacy identities は従来の normalized execution を含め完全一致を要求し、等局重み Fraction で MAE strict lower AND Top3 >= を判定します。

Root の外 terminal は `sekirei.white-view-evaluation-terminal.v1`、exit_code=0、session_closed/tool_observed_reaped/all_related_groups_observed_stopped=True、actual integer executor_session_id、role/run_ids/model が必要です。成功だけから terminal を自動生成しません。

親の bounded cleanup は自分の子 PG を対象とします。各 engine/supervisor の isolation/cleanup/reap は変更しない benchmark の attempt 証拠に依存します。取消・startup・cleanup・kernel fault 等の失敗は必ず invalid で、Root の全 related process scan と停止確認を別に要求します。parent PG が消えた事実だけを全 engine group 停止へ読み替えません。

## 外 JSON request と CLI

Root が固定する原典 request schema は以下です。具体 fullref は未知の実 runtime/model/results が揃うまで SOURCE fixture から生成しません。

- launch: `sekirei.white-view-evaluation-launch-preflight.v1`, status `frozen-before-first-write`。
- full audit: `sekirei.white-view-full-role-audit-request.v1`, status `frozen-before-raw-reparse`。role/model/runtime/descriptor/run_ids/config/build_manifest/terminal/source_inputs/source_helpers/repo_identity/decoder_versions/new_build_binding/output を exact keys で渡します。stock は旧固定Bの unloaded fallback に限ります。新Nは complete typed evaluation と new build verifier を必要とします。
- bridge/compare: `sekirei.white-view-comparison-request.v1`, status `frozen-before-comparison`。action と full source_inputs/source_helpers/output、bridge は old/new audit/build/runtime、compare は baseline/candidate audit/bridge/model を exact keys で渡します。全 transitively consumed raw/payload/control identity が入力 map にあることを照合します。

enabled copy の CLI 形（いずれも必須 expected SHA は実 completed input を Root が取得）：

```text
AUD -B white_view_operations.py launch --preflight PF --expected-preflight-sha256 SHA --expected-worker-sha256 SHA --parent-output NEW_PARENT
AUD -B white_view_operations.py complete --preflight PF --expected-preflight-sha256 SHA --expected-worker-sha256 SHA --parent-outcome RECORD --expected-parent-outcome-sha256 SHA --terminal TERM --expected-terminal-sha256 SHA
AUD -B white_view_operations.py audit --audit-request REQUEST --expected-audit-request-sha256 SHA --expected-worker-sha256 SHA
AUD -B white_view_operations.py bridge --comparison-request REQUEST --expected-comparison-request-sha256 SHA --expected-worker-sha256 SHA
AUD -B white_view_operations.py compare --comparison-request REQUEST --expected-comparison-request-sha256 SHA --expected-worker-sha256 SHA
```

実操作はこの SOURCE で行っていません。デフォルト formal timeout は20000秒上限、各 engine timeout と1M条件は旧 config のままです。失敗・completed ファイルの overwrite/retry/resume をしません。Root が closure と新4stage run を決定します。

## 正式専用 model gate の source 接続

`sekirei.white-view-model-technical-gate.v1` と producer の正式 exact keys に合わせました。`verify_candidate_inputs(reader,binding,spec)` の API を変更せず、外部の `sekirei.white-view-candidate-gate-binding.v1` 全19 fullrefs（common/controls/native/coefs/certs/proofs/gate/sources）から reference03・5 proofs・worker/common/control SHA を導出し、自己申告の refs を期待値へ流用しません。gate の全 transitive inputs は `spec.inputs` に型・bytes・SHA が一致して包含される必要があります。

専用 gate worker の実 source と private proof common は外部 SHA/fullref に束縛します。固有 module 名で common の exact bytes を load し、worker exec 中だけ canonical `white_view_proof_common` の import cache をその実 object へ一時束縛し、正常・例外とも元の cache を復元します。producer の canonical source/guard/API を改変しません。private 固有 alias の既存 module 再利用には loader が記録した exact source identity を要求します。legacy public helper の推移 import/cache は従来の canonical path/current SHA guard を保持します。

技術 gate は native03 再構成、reference03 全 byte transform、3 numerical certificates、118591 core observations、8185 incremental observations、観測100cp cap・finite/float bridge を専用 consumer で再検証します。read-only rustc/cargo/git 確認子コマンドは起動し得るため、`no_child_process_started=False`、`no_engine_or_fit_process_started=True`、`runtime_validator_may_invoke_readonly_source_compiler_checks=True` と正確に記録します。universal integer bound、native bitexact covariance、Python/native forward exactness、search/adoption を未証明のまま保持します。

この SOURCE 準備で実 model・receipt・raw を読み、上記の実成功を確認した主張はしません。全 actual entry は disabled です。Root が別 enabled source SHA・actual refs・same Git source freeze を確定し、producer/consumer の runtime 成功を確認してから使用します。旧 Adam/E3/absolute native guard の緩和・流用はありません。

## D3 の bounded group cleanup

TERM 後の直接子 wait/reap だけで終了しません。leader が即 wait 完了しても残る group を KILL し、別の最大5秒 deadline 内で group の消滅を観測します。wait timeout は KILL 後にもう一度最大5秒 wait します。確認できなければ fail closed で holder を保持し、親 lock の内側で失敗証拠を残します。直接子の reap・自分の group 停止は engine/supervisor の他 isolation groups 全体の停止証明にはなりません。Root の external related-group terminal と各 attempt cleanup/reap gate は従来どおり必須です。

## 合成検証

42 stdlib synthetic tests。固定 API/全段階順序、equal-game Fraction/8 check shapes、旧 source semantic field set、作者提供の正常 builder full inventory shape、Cargo2 duplicate endpoint と compiler 不一致否定、AUD/PROTO barrier、typed bytes/pin/symlink、fresh output/atomic state、actual flock FD 継承、spawn cancel/no child signal mask、wait+cleanup+save 失敗の元エラー維持、正常 parent→draft→wait/reap→external terminal→complete、one-way payload/receipt と後からの raw file 追加拒否に加え、immediate-reap leader/remaining descendant の KILL と未消滅 holder 保持、正式 gate_document source の純粋 constructor 正常/型・外refs・inputmap 否定、private common cache の正常/error 復元を確認しました。

```text
python3 -B -m unittest -q test_white_view_evaluation test_white_view_runtime_ports
```

normal lifecycle/legacy parser は synthetic mocks を使った境界確認です。実 binary/棋力/原本 source/current process/actual raw を検証した主張はしません。standalone restore、universal bound、search covariance、model adoption、goal complete は未確認です。
