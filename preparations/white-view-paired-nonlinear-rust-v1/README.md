# 第8候補で使用した非線形トレーナーのソース

`compiled-source/` は、学習ビルドで束縛されたソースからRootが保存した21ファイルを、そのまま保持するスナップショットです。新規Rust 10モジュール、変更されたmain/trainer・Cargo・NNUE、完全パッチ、原典ライセンス3ファイル、相対パスの公開manifestを含みます。実際の重み・教師キャッシュ・棋譜・checkpoint・実行ログ・private receiptの本文は含みません。

原典は [Sekirei f09c130](https://github.com/kent-tokyo/sekirei/tree/f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9) です。MIT OR Apache-2.0の著作表示とライセンス本文は `compiled-source/LICENSE`、`LICENSE-MIT`、`LICENSE-APACHE` に保持しています。`public-source-manifest.json` が20ファイルのsize/SHA-256と原典commit、測定時の公開source commit、非公開training-build receiptのSHAを記録します。manifest自身を自己ハッシュの対象にはしていません。

この保存集合は、white視点の180度回転とhand FTの0=3/1=2 tieを持つ専用 `SEKIRW03`、固定material座標、学習可能254 FT、14 paired nonlinear headを含みます。master勾配の集約後に原典Adamを1回だけ呼び、符号付きparameter/moment mirror、非活性masterのmoment更新、FT保存値797と一時41featureの範囲、nearest-even FT量子化、全state bit readback、失敗後のpoisonを専用経路で検査します。選択条件はseed42、元教師112681局面、3epoch/E3、shuffle/resumeなし、LRのf32 bits `3a83126f`、head幅 `3b800000`、bias `40800000`、output budget `47000000` です。

ソース内には準備段階のコメントや、一部のpureモジュールの閉じたstandalone入口も残っています。実際のmain/専用Readerは別のruntime境界で接続され、`SEKIREI_TRAIN_FTZ_DAZ=1` とMXCSR control `0x9fc0` の読戻しを要求します。snapshot内のソースを公開時にretag・整形・修正して、元のcompiled SHAと同一と扱うことはしません。ソースの運用パス定数は保存内容の一部であり、それだけで参照先データや環境が付属することを意味しません。

標準ライブラリだけのSOURCE検査は、このREADMEと同じディレクトリで実行できます。

```sh
python3 -B -m unittest -v test_public_source
```

リポジトリのどのディレクトリからでも、配置先を相対指定して実行できます。

```sh
python3 -B -m unittest discover -s preparations/white-view-paired-nonlinear-rust-v1 -p test_public_source.py
```

9件の検査は、全21ファイルのmembership・20原本のハッシュ、16パッチ対象の各postimage、新規10モジュールの全bytes包含、JSON重複/型/path/改変拒否、および小さな整数・control-word fixtureを確認します。privateパスのimport、NumPy、cshogi、Rustビルド、engine、教師データ、モデル生成は使いません。パッチを原典clean treeへ適用したという確認はRootの別記録です。適用手順は、完全な原典f09c130 checkoutのrootで `git apply --check` に続いて `git apply` を使用します。このsnapshotは全upstreamソースや依存を含む単独build環境ではありません。

SOURCE検査のPASSは、Rustの型検査、実MXCSR、実学習、core/incremental照合、100万node正式比較、停止、NAS保存、モデル採用それぞれの成功を代替しません。特に別のproof-sourceにある `native_contract` の `paired_zero` cfg(test) fixtureについては、reference/native型のClone接続が未確認という残事項があります。このsnapshotの10モジュールやパッチへ、その修正を無断で混ぜていません。公開default CIには、そのprivate依存のfixtureを追加しません。

private recipe/sourcebinding、Root parent全文、process authority、失敗履歴全文、NAS inventory等は別保管です。公開記録ではsource・技術検証・正式結果・保存状態を個別の根拠として示し、best更新やgoal完了をSOURCE検査から推定しません。
