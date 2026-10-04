# 非線形 nearest native03 の core/incremental proof SOURCE 準備 v2

SOURCE only、未compile/未model-read/未probe/未gate。v1 packageは不変。v1 core_pair probe/nativecontract2Rustはbyte同一で再利用し、専用incremental probeとpure parserを追加した。22小Python SOURCE/toy検査、Rustfmt構文3file、追加patch dry-runまでが準備の到達点。根拠は公開/frozen/tmp sourceのみで、C/runtime/data/models/process/NASは読んでいない。

Rootのmodel plan4bits/seed42/Adam3epoch/E3/noResume/noShuffle/constantLRは変えない。Rootはproductionに原典scalar Adamを選択済みと通知した。この版ではcache equivalenceをtraining successの必須証拠にしない。cache3/4のedgecase failure/未採用履歴を保存し、cacheへの切替を黙って許容しない。SOURCEからactual provenance/model outcomesを生成しない。

## v1 core契約と新incremental

v1同一core probeの説明はREADME-core-v1-preserved.mdに履歴として保持する。そのREADMEで当時必須候補としたcached-update実証は、Rootのoriginal-scalar選択により不要になった。旧linear <=100cp/noAdam/epochs0/FISTA/cert/固定boardFT flagsを新型へ流用しない。

`paired_nonlinear_incremental_probe.rs` のargvは `CANDIDATE.nearest03 REFERENCE03 FIXTURE_TSV`。旧argv `[100]` は受理しない。candidateとreferenceの1305356B/native03 headerを確認し、compiled reference reader/objectの全byte再構成、global candidate loader/objectの全byte再構成を原rawと照合する。nativecontractでseed42全reference再構成、保護M・FTbias・hand全256 tie・補助797と41-prefix・14paired head/signbit・Q32768とhead/output有限上界を検査する。modelhash/source/build/recipe/fullstate証拠の外側bindingはRootのworkerの責任。

元snapshot/null/do_move/do_move_for_search/undo両経路/16walk・8search_moveAPI/seed42 RNG/fixtureチェックの順は不変。毎観測でincremental accumulator==refresh、explicit core==global eval、finite diagnostic intcast==coreを要求する。finite forwardは各L2product/add、L2 clamp前、output product/addを確認し、双方のnormal<899000とstored-nearest f32/core bridge<1.001を要求する。実operation countsは全8185に対してL2各8185*16384/output各8185*32が一致する。FT float→nearest round差をこのbridge成功へ読み替えない。

旧100capを除くのはこの新variantだけ。`max_material_difference_cp` は非負の観測統計として保持するが合格capにしない。`material_bound_enabled` はnewparserでtyped Falseを要求し、旧100guardを成功したという主張をしない。new `normal_score_domain_verified`/`all_preclamp_intermediates_finite`/`native_structure_verified`/`native_loaded_fullbytes_equal`、max ordinary cp、max stored bridge、実operation counts、native functional upper boundsを専用29key parserで検証する。各maximum/count/booleanはraw stdoutから再計算/型検査し、実model/source/probe/binary/input/walk fixtureへ外SHA束縛する。

41-prefixは原nnue sourceのundo-capture一時+1と797quantizerからの条件であり、新probeが全intermediate undo saturating_addを個々にtraceしたという主張はしない。実accumulator/refresh/parent restorationの8185観測と別のsource/math条件を合わせる。物理的全legal局面や全search tree、bitexact physical covarianceを証明したと主張しない。

## 新typed gate proposal（actual producer/consumer未実装）

追加receipt層は設けず、Rootのtraining/export/core/incremental実記録を一つの専用technical gateへ束縛する。schema案は `sekirei.white-view-paired-nonlinear-model-technical-gate.v1`（未activation、Rootが実fieldをfreezeするまで旧gateへ渡さない）。旧モデルのsolver/math audit schemaや3certificateを要求しない。

ゲートがactual evidenceから再構成して確認する項目：
- externally pinned plan/activation/preregistration/source-preflight/source-helper/compiler/trainer/core/probe/build/原O manifests・reference03・native・metadata/fullstate/export/run/operational refs。
- selected4f32bits、seed42、original scalar Adam、3epoch112681/step338043/E3、noShuffle/noResume、absolute output。child宣言だけでなく3EpochTokens/原典CP loss/backward/source固定/親wait-reapと全input集合をbindする。
- 15Vec/3scalar/step全statefinite・nonnegative-v・保護param/m/v・material/fullFTbias/reference・全handとpair/moment ties・aux797projection/nearest-even・native全byte checkpoint reexportとfile readback。
- core118591 split counts112681/5895/15、全node scorerow、双方operation counts/finite-preclamp/FT prefix/reference==PythonM/storednative float bridge/ordinary domain。
- newincremental29key、8185/15/16/8、transition counts、global loaded nativefullbytes/refresh/undo/parent preservation、finite operation counts、normal score domain/budget。
- raw SHA/canonical membership/order・前後input/source/build/compiler/native等不変・親wait/reap/二PG停止・serial locks・timeout/cleanup失敗拒否。

補助boardFTは学習可能。旧 `board_ft_bytes_preserved=True` を移植せず、`whole_board_ft_preservation_claimed=False` と `protected_material_ft_bits_preserved=True` を区別する。oldfixedFT float/round bridgeや3certificate/FISTA flagsは存在しない。native ABI03/feature schemaは共通、同newbinaryの新fallback/候補100万node正式比較は既存厳密条件を保持する。

`gate-properties-proposal.json` はSOURCE要件だけで、statuscomplete実receiptではない。未知のfullstate/sidecar/run/source/proc証拠を成功へ合成しない。gateはsearch/adoption/final used/universal_integer_bound/native_bitexact_covarianceをclaimしない。Rootが実gate/ref型を決めてsource consumerへ接続する残事項である。

Rootだけが専用source inventory/compile receiptへこのsourceを追加し、private enabled別byteをexternSHA凍結してlaunchする。PROTOTYPE_ONLY=Trueはargs/ファイル/MXCSR操作より先に失敗する。既存workerのcanonical/SHA/原O/native/fixture/cleanup/locks beforeafter保護を保った新variantで利用し、旧white-view linear commonをmodeだけ書換えて使わない。
