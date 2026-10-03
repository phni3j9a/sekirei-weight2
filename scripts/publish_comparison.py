#!/usr/bin/env python3
"""Create a new redacted report from a completed strict comparison receipt."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frac(value):
    require(type(value['numerator']) is int and type(value['denominator']) is int
            and value['denominator'] > 0, 'invalid rational metric')
    return Fraction(value['numerator'], value['denominator'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--candidate-label', required=True)
    p.add_argument('--baseline-label', default='駒得fallback')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    comparison = json.loads(args.comparison.read_text())
    require(comparison['status'] == 'complete' and comparison['comparison_valid'] is True
            and comparison['inputs_unchanged'] is True and comparison['final_used'] is False,
            'comparison not complete/valid/immutable/development-only')
    require(len(comparison['checks']) == 8 and all(v['equal'] for v in comparison['checks'].values()),
            'reference identity mismatch')
    baseline, candidate = comparison['baseline'], comparison['candidate']
    bmae, cmae, btop, ctop = [frac(x) for x in
                            (baseline['mae'], candidate['mae'], baseline['top3'], candidate['top3'])]
    require(comparison['mae_improved'] == (cmae < bmae)
            and comparison['top3_preserved'] == (ctop >= btop)
            and comparison['adopt'] == (cmae < bmae and ctop >= btop), 'adoption arithmetic differs')
    for value in (baseline, candidate):
        require(value['mae_report_valid'] and value['top3_report_valid']
                and value['teacher_count'] == 570 and value['teacher_E_count'] == 266
                and value['top3_count'] == 551, 'invalid universe or report')
    model = candidate['model_identity']
    require(model['kind'] == 'nnue' and len(model['sha256']) == 64, 'expected identified NNUE candidate')
    source_evaluation = Path(comparison['sources']['candidate']['evaluation'])
    saved_inputs = {entry['path']: entry for entry in comparison['input_receipt']['files']}
    for field in ('evaluation', 'config'):
        path = Path(comparison['sources']['candidate'][field])
        require(str(path) in saved_inputs and sha(path) == saved_inputs[str(path)]['sha256'],
                'comparison input changed after validation')
    evaluation = json.loads(source_evaluation.read_text())
    require(evaluation['status'] == 'complete' and evaluation['weight_sha256'] == model['sha256'],
            'completed evaluation identity differs')
    config = json.loads(Path(comparison['sources']['candidate']['config']).read_text())
    game_ids = list(config['universe']['plies'])
    require(len(game_ids) == 5 and set(game_ids) == set(baseline['per_game']) == set(candidate['per_game']),
            'game order/set mismatch')
    rows = []
    for ordinal, game in enumerate(game_ids, 1):
        old, new = baseline['per_game'][game], candidate['per_game'][game]
        require(old['e_count'] == new['e_count']
                and old['top3']['denominator'] == new['top3']['denominator'], 'per-game denominator differs')
        rows.append({'ordinal': ordinal, 'teacher_exact_count': old['e_count'],
                     'baseline_mae_cp': float(frac(old['mae'])), 'candidate_mae_cp': float(frac(new['mae'])),
                     'mae_delta_cp': float(frac(new['mae'])-frac(old['mae'])),
                     'top3_denominator': old['top3']['denominator'],
                     'baseline_top3_hits': old['top3']['hits'], 'candidate_top3_hits': new['top3']['hits'],
                     'baseline_top3_rate': float(Fraction(old['top3']['hits'], old['top3']['denominator'])),
                     'candidate_top3_rate': float(Fraction(new['top3']['hits'], new['top3']['denominator']))})
    ratios = {name: {metric: {key: source[metric][key] for key in ('numerator', 'denominator')}
                     for metric in ('mae', 'top3')}
              for name, source in [('baseline', baseline), ('candidate', candidate)]}
    public = {'schema_version': 1, 'report': 'fixed-development-weight-comparison',
              'candidate_label': args.candidate_label, 'baseline_label': args.baseline_label,
              'candidate_weight_sha256': model['sha256'], 'candidate_weight_bytes': model['bytes'],
              'conditions': {'sekirei': 'v0.3.39', 'teacher': 'Suisho Concerto 202512 (Suisho11beta)',
                             'requested_nodes': 1000000, 'threads': 1, 'hash_mib': 128,
                             'mae_multipv': 1, 'candidate_top3_multipv': 3, 'games': 5,
                             'occurrences': 570, 'teacher_exact_count': 266, 'top3_eligible_count': 551,
                             'aggregation': 'equal mean of five per-game metrics',
                             'nnue_output': 'absolute', 'final_used': False},
              'decision': {key: comparison[key] for key in
                           ('comparison_valid', 'adopt', 'decision', 'mae_improved', 'top3_preserved')},
              'headline': {'baseline_mae_cp': baseline['reported_mae_cp'],
                           'candidate_mae_cp': candidate['reported_mae_cp'],
                           'mae_delta_cp': float(cmae-bmae), 'baseline_top3_rate': baseline['reported_top3_rate'],
                           'candidate_top3_rate': candidate['reported_top3_rate'],
                           'top3_delta_percentage_points': float(100*(ctop-btop)), 'exact_ratios': ratios},
              'per_game': rows,
              'evidence': {'mae_formal_valid': True, 'top3_formal_valid': True,
                           'mae_attempts': 1140, 'top3_attempts': 551,
                           'reference_checks': {k: v['equal'] for k, v in comparison['checks'].items()},
                           'comparison_inputs_unchanged': True,
                           'mae_fingerprint': candidate['mae_fingerprint'],
                           'top3_fingerprint': candidate['top3_fingerprint'],
                           'baseline_mae_fingerprint': baseline['mae_fingerprint'],
                           'baseline_top3_fingerprint': baseline['top3_fingerprint']},
              'evidence_sha256': {'private_comparison': sha(args.comparison),
                                  'completed_evaluation': sha(source_evaluation)},
              'limitations': ['Fixed development five-game result; no claim about final data or general playing strength.',
                              'Static holdout diagnostics are distinct from these 1000000-node adoption metrics.']}
    public['decision']['rule'] = 'MAE strictly lower and Top3 non-decreasing; exact rational comparison'
    encoded = json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    forbidden = game_ids + ['/home/', '/mnt/', 'startpos moves', 'sfen ']
    require(not any(token in encoded for token in forbidden), 'private data in redacted comparison')
    source = source_evaluation.parent / 'public-mae'
    names = {'validation.md', 'validation.json', 'reviewed.svg', 'manifest.json'}
    require(source.is_dir() and {x.name for x in source.iterdir()} == names, 'public export file matrix differs')
    source_before = {name: sha(source/name) for name in sorted(names)}
    export_manifest = json.loads((source/'manifest.json').read_text())
    require(export_manifest['files'] == {name: source_before[name] for name in names-{'manifest.json'}},
            'public export manifest hash mismatch')
    export_report = json.loads((source/'validation.json').read_text())
    require(export_report['headline']['valid'] is True and export_report['headline']['formal'] is True
            and export_report['headline']['mae_cp'] == candidate['reported_mae_cp']
            and export_report['game_count'] == 5 and export_report['occurrence_count'] == 570,
            'public export report differs from compared measurement')
    for name in names:
        require(not any(token in (source/name).read_text() for token in forbidden), 'private data in public MAE export')
    output = args.output.resolve()
    require(not output.exists() and output.parent.is_dir(), 'new report with existing parent required')
    output.mkdir()
    shutil.copytree(source, output/'mae')
    require({name: sha(output/'mae'/name) for name in names} == source_before, 'copied export differs')
    require({name: sha(source/name) for name in names} == source_before, 'source export changed')
    (output/'comparison.json').write_text(encoded)
    verdict = '採用条件を満たした。' if comparison['adopt'] else '採用条件を満たさなかった。'
    lines = [f'# {args.title}', '', f'**{verdict}** 正式MAEとTop3の証拠は有効で、固定教師・対象集合・モデル以外の実行条件が比較基準と一致した。', '',
             f'| 採用指標 | {args.baseline_label} | 候補 | 候補 − 基準 |', '| --- | ---: | ---: | ---: |',
             f'| MAE | {float(bmae):.3f} cp | {float(cmae):.3f} cp | {float(cmae-bmae):+.3f} cp |',
             f'| Top3入り率 | {float(btop)*100:.4f}% | {float(ctop)*100:.4f}% | {float(100*(ctop-btop)):+.4f}ポイント |', '',
             '採用条件は「MAEが厳密に改善し、Top3入り率が下がらないこと」。各局の整数cp誤差・hit数から5局等重みの有理数で判定した。未丸め値と分数は comparison.json に保存する。', '',
             '両者100万ノード指定、Threads=1、Hash=128 MiB。MAEはMultiPV=1、Top3はSekireiの別runでMultiPV=3。最終評価用データは使っていない。', '',
             '| 局 | 教師exact点数 | 基準MAE | 候補MAE | 基準Top3 hit / 対象 | 候補Top3 hit / 対象 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['ordinal']} | {row['teacher_exact_count']} | {row['baseline_mae_cp']:.3f} | {row['candidate_mae_cp']:.3f} | {row['baseline_top3_hits']} / {row['top3_denominator']} | {row['candidate_top3_hits']} / {row['top3_denominator']} |")
    lines += ['', 'MAEは570局面・1,140 attemptを完了し、固定Teacher-E 266点を全て採点した。Top3は対象551点を全て測定した。比較検証では教師の型付き評価・bestmove・PV、Teacher-Eのcp、Top3の対象・合法手・分母・参照bestmoveを照合した。', '',
              f"候補：`{args.candidate_label}`。重みSHA-256：`{model['sha256']}`。", '',
              'この結果は固定development 5局の測定であり、最終評価や一般的な棋力の改善は示さない。今回の条件と次の判断は[自律改善](../../../AUTONOMOUS_WEIGHT_IMPROVEMENT.md)、前回の学習・診断条件は[改善実験](../../../WEIGHT_IMPROVEMENT.md)を参照。', '',
              '- [候補の評価値集計とグラフ](mae/validation.md)',
              '- [比較値とidentity](comparison.json)', '',
              'MAE公開exportの4ファイルは元のまま保持した。生成見出しは共通の「baseline validation」だが、fingerprintと数値はこの候補のもの。', '']
    markdown = '\n'.join(lines)
    require(not any(token in markdown for token in forbidden), 'private data in Markdown')
    (output/'comparison.md').write_text(markdown)
    print(json.dumps({'status': 'complete', 'files': 6, 'adopt': comparison['adopt'],
                      'mae_cp': float(cmae), 'top3_rate': float(ctop)}))


if __name__ == '__main__':
    main()
