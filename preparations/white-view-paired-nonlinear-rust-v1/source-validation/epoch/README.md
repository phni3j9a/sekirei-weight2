# Root 専用の add-only 技術テスト SOURCE

Root activation-v2 の `paired_nonlinear_positions.rs` 末尾に Rust addon をそのまま追加する。元位置更新・progress・zero-byte SOURCE guard・native/checkpoint/update の各本文は変更しない。新 API、reader、serializer、recipe defaults は追加しない。

Root が `SEKIREI_TRAIN_FTZ_DAZ=1` を明示し、新 white feature の実 crate で cargo test を実行する。`--nocapture` は使わない。fixture は既存 main の `configure_training_floats` を各テストの実スレッドで呼び、readback-only `runtime_ready` の actual control 0x9fc0 を確認する。環境変数は設定しない。こちらは Rust compile/test/MXCSR 実読戻しを実行していない。

Rust テストは2件のみ。公開の 7g7f 後 SFEN、ordinary keyed teacher500、固定 plan4bits の fullshape fresh trainer を使う。正常系は1局面×3epoch/step1,2,3を検証し、全状態 validation と step3 のメモリ native03 nearest/reexport を通す。正式 checkpoint/native の end step338043は保持され、step3の正式 export は拒否される。checkpoint context はメモリ上の shape fixture に過ぎず、実 provenance・実ファイル成功は主張しない。

失敗系は1局面 epoch1 の後、missing key、+30000、-30000、i32::MIN のいずれかを epoch2で拒否する。Context poison、修復した cacheでも次 epoch不可、全15Vec/3scalar/stepの bit不変、initialized/complete export不可、completed reference不可を検証する。下位 `nearest_native03` の明示 step付きメモリ診断を包括的に禁止する guard だとは主張しない。

実行した Python 検証は SOURCE 構造・固定値・接続 API のみ。Rust型検査、実 fullshape数値更新、実 MXCSR、native coreロード/save、実データ provenance、正式112681 fitは Root後続。全600kモデルをこちらで生成していない。
