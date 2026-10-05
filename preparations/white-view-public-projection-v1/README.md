# White-view public projection — SOURCE prototype

実験結果を読まずに作成した準備コードです。`PROTOTYPE_ONLY=True`、全actual入口は `runtime_guard()` により read/import/write より前に停止します。実 control、model、棋譜、raw、runtime の読取、エンジン／学習／build、公開統合は実施していません。旧 publisher／guard／R/C/runtime は変更していません。

`publish_white_view_comparison.py` は `sekirei.white-view-same-runtime-comparison.v1` を旧schemaへ変換せず、専用 `sekirei.white-view-public-comparison.v1` に allowlist projection します。公開候補は次の3ファイルだけです。

- `comparison.json`: G1..5 の整数 error sum/count・hit/denominator、5局等重みの Fraction、条件、採否、identity/receipt/source/binary/model SHA、検証 scope。
- `comparison.md`: 集計表、正確な採否規則、JSON/SVGへの相対リンク、限界。
- `comparison.svg`: MAE と Top3 の5局集計 bar。手数、局面、教師評価値の系列はありません。

この writer の公開 projection は3ファイルで、Rootが保存・公開時に各ファイルのsize/SHA manifestを別途付けます。既存のbenchmark public exportはhash manifestを含む4ファイルですが、その旧schemaの成功flagを合成しません。新しいaggregate SVGはモデル比較のための図であり、既存run側の5パネル `report.svg` の生成・保存を置き換えません。

private fullref の path、SFEN、棋譜moves、labels、個別 game/source IDs、raw、input map、run filenames、PV、合法手、teacher/model/archive contents は公開しません。元receiptをdeep-copyしてprivate fieldを削る方式ではなく、固定の公開キーだけを新規作成します。

## 既存証拠の再利用範囲

外部 request SHA/size、publisher source SHA、W contract/producer source fullrefs、comparison、bridge、旧fallback・新fallback・候補の3 audit receipt/payload、全推移 input map を固定します。runtime reader は parsed raw bytes/current SHAを照合し、全入力を1MiB単位でstream hashし、raw directory membershipも再確認します。

既存監査の `raw_reparsed=True`／1829 attempt／node cap／cleanup／normal tool-reaped terminalをtyped envelopeとして要求します。W source-pinned pure contractを使って旧→新fallback bridgeの7共通identity・semantic・exact整数集計と、same-newbinaryの8 identity・Fraction joint criteriaを再構成し、保存された comparison 全体へdeep typed一致を要求します。publisher自身はUSI rawを再parseせず、runtime/compiler/buildのfresh実検証も行いません。その範囲をpublic scopeに明示しています。

旧 `publish_comparison.py` には接続しません。旧 `benchmark_report.render_svg(public=True)` は各plyの評価値を描くため流用しません。旧 helper のguard/source hashesは不変です。

## Request契約と循環回避

`validate_request()` にexact keysを定義しています。schemaは `sekirei.white-view-public-projection-request.v1`、statusは `frozen-before-public-projection`。comparison/bridge/old_audit/baseline_audit/candidate_audit/contract_source/comparison_producer/worker_source は全部 fullref、`source_inputs` は各 audit の推移 input mapと各receipt/payload/terminal/build/binary/sourceを含む externally frozen `{absolute_path:{bytes,sha256}}` です。各refが同じmapに含まれる必要があります。request自体は外側 CLI SHA/bytesから別にpinし、自身のhashを入力mapに入れる循環は作りません。

`supplemental_evidence` のexact keysは numeric/core/incremental/model_gate/stop/nas。未提供は `None` としpublicでは `provided=False`／`not_provided` です。提供済みのfullrefも、ここでは `sha_reference_only`／`proof_body_independently_verified_by_publisher=False` と記録します。unknownを成功とせず、Rootが各専用proof/stop/NASの actual verification を済ませた後で独立refsを渡す必要があります。追加のbody validatorやactivation契約は、このprototypeに未実装です。

出力は新規directoryで、全3auditのraw rootを含む入力集合と双方向のoverlapをfirst write前に拒否します。3ファイルをfirst writeより前に全レビューし、O_EXCL/0600で保存します。出力前後の全private input SHAとmembershipを確認し、書込みの途中失敗は完成publicationの成功結果を出しません。親側のserial coordination/locks/STOP確認、public Git統合、READMEのリンク追加、NAS保存、best-model適用、goal完了はRootの後続手順です。このprototypeはこれらの完了を主張せず、自動activationやguard接続を持ちません。

## 合成検証のみ

`test_publish_white_view_comparison.py` は凍結D4の明示pure関数だけをAST抽出し、実producer shapeのmemory fixturesを作成します。actual artifactは使いません。tmpのtoy textを使うreader/writer fixtures以外は全てmemory-onlyです。mode/native等のmetadata fixtureは実modelの生成／読取ではありません。

検証対象: valid採用・joint条件未達、exact Fraction/表示丸め、canonical ratio/bool拒否、8dict checks、件数/ref/build/sourceHead不一致、改変SHA、pending proof、private余分fieldの非投影、公開3file検査、symlink/new raw file/既存出力/入力重複拒否、actual before-I/O barrier。

準備テストコマンド（Rootの実artifactは参照しない）:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s /tmp/sekirei-weight2-white-view-public-projection-source-v1 -p test_publish_white_view_comparison.py
```

未有効のactual CLI契約（実行禁止）:

```text
publish_white_view_comparison.py --request <private-request> --expected-request-sha256 <sha> --expected-request-bytes <bytes> --expected-worker-sha256 <source-sha>
```

Rootが実証拠／producer-source fullrefs／外側停止・coordination guardを整えて別enabled copyを記録するまでは、actual CLIは停止したままです。
