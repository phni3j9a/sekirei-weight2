# White-view append-only pure CI port

新しい preparation とその公開 fixture だけを追加する移植です。既存の
NUM1/proof1/evaluation-v3 source・test・固定SHAは変更しません。

新しい source の production body は凍結した SOURCE ONLY preparation と
byte 同一です。全 actual entry の PROTOTYPE_ONLY=True を保持しています。
各 suite は独立した Python process で実行するため、旧 suite の canonical
module cache を新しい version の検証へ流用しません。6 wrapper が内部の
125 tests（18+14+14+23+13+43）を実行し、件数も厳密に確認します。

`white-view-source-fixtures-v1/*.py.txt` は immutable な原典 source text
です。実際に enabled だった NUM1/proof source や import 時 private path を
読む Root factory を含むため、**import/CLI 実行は禁止**です。NUM1 の既知の
PROTOTYPE_ONLY=False と新NUM2の True はそれぞれ exact assert し、この
既知 flag 差分だけを assignment 比較から分けています。全 function/class
AST比較は保持します。generic helper の元SHAを確認したうえで、inventoryの
3関数とstdlib importsだけをASTから抽出し、公開tmp textへ適用します。
元helperのmount/rsync/mainはCIで一切実行しません。

portable に変更した箇所は test の source 選択だけです。gate constructor
ASTは新proof2 preparation、publisher/stopのpure producerは新evaluation-v4
preparation、numericの原典ASTはsource text fixturesへ接続しています。
source内の private absolute path はdisabled契約の宣言値として残しています。
typed negative・lifecycle・binding・source hash・exact rational比較のassertions
を削除していません。before-I/O拒否のnegative testsは維持しています。

公開synthetic build JSONのpathは架空です。fixture testsはその参照先を
追いません。public material seed42はメモリ内fixtureだけに使います。
actualC/model/data/runtime/NAS/proc、fit/build/engine/probe、採用の確認は行いません。

実行例：`python3 -B -m unittest discover -s tests -p test_white_view_append_preparations.py`
通常の公開tests discoveryでもこの6 wrappersが見つかります。過去のsuiteの
代わりではなく、appendしたversionの追加検証です。


初回移植ではE4 field-policyテストの比較元sourceがホスト絶対pathのまま残っていた。修正はそのselector一行だけで、`Path(__file__).resolve().parents[2] / 'scripts' / 'compare_candidates.py'`へ接続した。production sourceとassertionsは不変。元40file treeの凍結を保持し、この修正を別commitに記録する。移設した公開overlayで実ホストとNASの読取りを拒否し、旧selectorのPermissionError再現と新125件の成功を独立に確認した。
