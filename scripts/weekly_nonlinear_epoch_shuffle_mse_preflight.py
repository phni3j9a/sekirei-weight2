#!/usr/bin/env python3
"""Author and revalidate fresh weekly training predecessors, without fitting.

The selected-game legal replay belongs to the pinned dataset producer. This
standard-library consumer rechecks pack boundaries/ranking, complete raw
provenance, independent-pool board exclusions and dataset joins. It never
reuses the old origin certificate or starts a compiler, trainer or engine.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import time

import benchmark as benchmark
import diverse_pack_dataset as diverse
import freeze_holdout_dataset as freeze
import prepare_weekly_nonlinear_epoch_shuffle_mse as preparation
import weekly_nonlinear_epoch_shuffle_mse_build as builder
import weekly_nonlinear_epoch_shuffle_mse_run as parent_run
from acquire_quest import parse_csa
from pack_dataset import split_for_game
from weekly_nonlinear_epoch_shuffle_mse_post_training import require, small

REPO = preparation.REPO
DATA_FILES = set(preparation.DATASET_FILES)
N, H = 112681, 5895
PACK_COUNT, POOL_COUNT = 13, 1000
ENGINE_SHA = 'acdf54127ecea9d6d3317490f1547a13c43f65db749a48115c055ebc7d9e5c30'
ENGINE_BINARY_SHA = 'c8818d71c3c3b54684f76ef5c204d56b1f1445e12cf81d22820c69a8c8b0c1c3'
ORIGIN_CHECKS = {'selection_profile_hash_body_valid', 'selected_replay_producer_source_bound',
                 'whole_pack_index_membership_valid', 'frozen_holdout_bytes_preserved',
                 'complete_pool_excluded', 'split_and_boards_disjoint',
                 'raw_input_hashes_unchanged', 'exact_train_label_join_valid'}
PARENT_CHECKS = {'fresh_dataset_origin_raw_and_transitive_refs_valid',
                 'fresh_dataset_schema_semantics_and_join_valid',
                 'frozen_holdout_and_pool_exclusion_verified', 'reference03_fullbytes_reconstructed',
                 'new_training_source_build_compiler_abi_bound', 'fixed_white_view_engine_unchanged',
                 'inputs_before_after_equal', 'source_membership_equal', 'no_conflicting_heavy_process'}
ORIGIN_KEYS = {'schema', 'status', 'producer_source', 'dataset_inputs', 'frozen_dataset_inputs',
               'profile', 'pack_manifest', 'independent_pool_manifest', 'input_hashes', 'checks',
               'counts', 'inputs_before', 'inputs_after', 'actual_replay_repeated_by_verifier', 'final_used'}
PARENT_KEYS = {'schema', 'status', 'mode', 'selected_plan', 'training_build', 'engine_build',
               'dataset_inputs', 'reference03', 'dataset_origin_proof', 'origin_proof_policy',
               'origin_counts', 'source_inputs', 'inputs_before', 'inputs_after', 'checks',
               'process_evidence', 'source_head'}
BINDING_KEYS = {'schema', 'status', 'mode', 'selected_plan', 'parent_preflight', 'training_build',
                'training_binary', 'training_source_files', 'compiler_files', 'engine_build',
                'dataset_inputs', 'reference03', 'float_policy'}
RECIPE_KEYS = {'schema', 'mode', 'feature_schema', 'seed', 'epochs', 'train_count', 'objective',
               'optimizer', 'learning_rate_f32_bits', 'head_init_width_f32_bits',
               'head_bias_init_f32_bits', 'output_native_l1_budget_f32_bits', 'shuffle_seed', 'row_order',
               'teacher_identity', 'manifest', 'reference03', 'positions', 'labels',
               'source_binding', 'resume_allowed', 'ft_saved_q_max', 'ft_bias_q'}


def exact(actual, expected, message='typed values differ'):
    require(preparation.typed_equal(actual, expected), message)


def obj(value, keys):
    require(type(value) is dict and set(value) == set(keys), 'unknown or missing JSON field')
    return value


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'strict SHA256 required')
    return value


def integer(value, minimum=0):
    require(type(value) is int and minimum <= value <= (1 << 64) - 1, 'strict unsigned integer required')
    return value


def fullref(value):
    obj(value, {'path', 'bytes', 'sha256'})
    integer(value['bytes']); sha(value['sha256'])
    require(type(value['path']) is str and Path(value['path']).is_absolute()
            and str(Path(value['path'])) == value['path'] and '..' not in Path(value['path']).parts,
            'canonical fullref path required')
    return value


def identity_map(value):
    require(type(value) is dict and value, 'nonempty identity map required')
    for path, identity in value.items():
        obj(identity, {'bytes', 'sha256'}); fullref({'path': path, **identity})
    return value


class Reader:
    def __init__(self):
        self.files = {}

    def pin(self, reference):
        fullref(reference)
        exact(parent_run.physical_ref(Path(reference['path'])), reference, 'changed physical fullref')
        parent_run.merge(self.files, {reference['path']: small(reference)})
        return reference

    def read(self, reference):
        self.pin(reference)
        return parent_run.read_ref(reference)

    def json(self, reference):
        return parent_run.strict_json(self.read(reference))

    def map(self, values):
        identity_map(values)
        for path, identity in values.items():
            self.pin({'path': path, **identity})

    def current(self):
        return parent_run.verify_map(self.files)


def pinned(path, expected_sha):
    reference = parent_run.physical_ref(path)
    exact(reference['sha256'], sha(expected_sha), 'external fullref SHA differs')
    return reference


def dataset_five(root):
    return {name: parent_run.physical_ref(root / name) for name in sorted(DATA_FILES)}


def check_five(reader, references):
    obj(references, DATA_FILES)
    root = Path(references['manifest.json']['path']).parent
    for name, reference in references.items():
        exact(Path(reference['path']), root / name, 'dataset fullrefs must be five siblings')
        reader.pin(reference)
    return reader.json(references['manifest.json'])


def public_consumer_sources():
    """Pin the conservative closure of local imports used by this reader."""
    import ast
    pending, visited = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                source = REPO / 'scripts' / (module.split('.')[0] + '.py')
                if source.is_file() and source not in visited:
                    pending.append(source)
    visited.add(REPO / 'config/quest-corpus.json')
    return {str(path): parent_run.info(path) for path in sorted(visited)}


def dataset_join(reader, references, manifest, expected_counts):
    """Require full SFEN/label joins and complete game/board separation."""
    obj(manifest['files'], DATA_FILES - {'manifest.json'})
    exact(manifest['positions'], expected_counts, 'fixed dataset row count differs')
    games = {}
    locations = set()
    for game in manifest['games']:
        require(set(game) in ({'game_id', 'split', 'pack_sha256', 'game_index'},
                              {'game_id', 'split', 'pack_sha256', 'game_index', 'selection_rank'}),
                'unknown game provenance fields')
        gid = sha(game['game_id']); sha(game['pack_sha256']); integer(game['game_index'])
        if 'selection_rank' in game: sha(game['selection_rank'])
        exact(game['split'], split_for_game(gid, manifest['seed']), 'misassigned complete-game split')
        location = (game['pack_sha256'], game['game_index'])
        require(gid not in games and location not in locations, 'duplicate game identity or location')
        games[gid] = game; locations.add(location)
    splits = {}
    for split, count in expected_counts.items():
        rows = {}
        keys = set()
        game_ids = set()
        for kind in ('positions', 'labels'):
            name = split + '.' + kind + '.jsonl'
            exact(small(references[name]), manifest['files'][name], 'dataset manifest file identity differs')
            lines = reader.read(references[name]).decode().splitlines()
            require(len(lines) == count, 'dataset line count differs')
            parsed = [parent_run.strict_json(line) for line in lines]
            if kind == 'positions':
                for row in parsed:
                    require(set(row) in ({'schema_version', 'sfen', 'source'},
                                         {'schema_version', 'sfen', 'source', 'tags'}),
                            'unknown position row fields')
                    exact(row['schema_version'], 1)
                    obj(row['source'], {'kind', 'path', 'ply'})
                    exact(row['source']['kind'], 'gensfen-pack'); integer(row['source']['ply'], 1)
                    sfen = row['sfen']; key = freeze.board_key(sfen); gid = row['source']['path']
                    require(sfen not in rows and key not in keys and gid in games
                            and games[gid]['split'] == split, 'duplicate board or misassigned row game')
                    if 'tags' in row:
                        obj(row['tags'], {'side_to_move', 'phase'})
                        require(row['tags']['side_to_move'] in ('black', 'white')
                                and row['tags']['phase'] == 'middlegame', 'invalid position tags')
                        exact(row['tags']['side_to_move'], 'black' if sfen.split()[1] == 'b' else 'white',
                              'position side tag differs from SFEN')
                    rows[sfen] = row; keys.add(key); game_ids.add(gid)
            else:
                labels = {}
                for row in parsed:
                    obj(row, {'sfen', 'score_cp', 'teacher_identity', 'label_depth'})
                    require(type(row['score_cp']) is int and abs(row['score_cp']) < 30000,
                            'integer finite-scale label required')
                    exact(row['label_depth'], 0); exact(row['teacher_identity'], manifest['teacher_identity'])
                    require(row['sfen'] not in labels, 'duplicate label SFEN')
                    labels[row['sfen']] = row
                require(set(labels) == set(rows), 'complete train/holdout SFEN label join differs')
        splits[split] = {'keys': keys, 'game_ids': game_ids}
    require(not (splits['train']['keys'] & splits['holdout']['keys'])
            and not (splits['train']['game_ids'] & splits['holdout']['game_ids']),
            'training and holdout are not disjoint')
    return splits


def tracker_key(board):
    symbols = {'FU': 'P', 'KY': 'L', 'KE': 'N', 'GI': 'S', 'KI': 'G', 'KA': 'B', 'HI': 'R', 'OU': 'K',
               'TO': '+P', 'NY': '+L', 'NK': '+N', 'NG': '+S', 'UM': '+B', 'RY': '+R'}
    ranks = []
    for rank in range(1, 10):
        parts, empty = [], 0
        for file in range(9, 0, -1):
            piece = board.board.get((file, rank))
            if piece is None:
                empty += 1
            else:
                if empty: parts.append(str(empty)); empty = 0
                symbol = symbols[piece[1]]
                parts.append(symbol if piece[0] == 1 else symbol.lower())
        if empty: parts.append(str(empty))
        ranks.append(''.join(parts))
    hands = []
    for owner in (1, -1):
        for piece in ('HI', 'KA', 'KI', 'GI', 'KE', 'KY', 'FU'):
            count = board.hands[owner][piece]
            if count:
                symbol = symbols[piece]
                hands.append((str(count) if count > 1 else '') + (symbol if owner == 1 else symbol.lower()))
    return '/'.join(ranks) + (' b ' if board.side == 1 else ' w ') + (''.join(hands) or '-')


def pool_exclusions(reader, pool_ref, expected):
    """Reconstruct exclusion boards from all raw CSA; never open split files."""
    manifest = reader.json(pool_ref)
    exact(manifest['accepted_games'], POOL_COUNT)
    root = Path(pool_ref['path']).parent
    paths = sorted((root / 'games').glob('*.csa'))
    require(len(paths) == POOL_COUNT, 'complete raw independent pool membership required')
    keys, canonical = set(), []
    for path in paths:
        reference = parent_run.physical_ref(path)
        exact(reference['sha256'], path.stem, 'raw pool file name/hash differs')
        text = reader.read(reference).decode()
        canonical.append(parse_csa(text)['canonical_sha256'])
        replay = benchmark.parse_csa_text(text)
        keys.add(tracker_key(benchmark.BoardTracker.standard()))
        keys.update(tracker_key(board) for board in replay['boards_after'])
    aggregate = hashlib.sha256(''.join(sorted(canonical)).encode()).hexdigest()
    actual = {'games': POOL_COUNT, 'unique_positions': len(keys), 'corpus_canonical_sha256': aggregate,
              'source_manifest_sha256': pool_ref['sha256'], 'policy': freeze.EXCLUSION_POLICY}
    exact(actual, expected, 'complete independent pool exclusion identity differs')
    exact(manifest['corpus_canonical_sha256'], aggregate)
    require(paths == sorted((root / 'games').glob('*.csa')), 'raw pool membership changed')
    return keys


def verify_origin_inputs(reader, dataset, frozen, profile_ref, pack_ref, pool_ref):
    manifest = check_five(reader, dataset); old = check_five(reader, frozen)
    obj(manifest, {'schema_version', 'teacher_identity', 'source_corpus_manifest_sha256', 'dependencies',
        'sampling', 'selection_seed', 'games_per_pack', 'train_budget', 'seed', 'split', 'games',
        'counts', 'positions', 'independent_exclusions', 'label_depth', 'files', 'derivation', 'limitations'})
    exact(manifest['schema_version'], 1)
    exact(frozen['manifest.json']['sha256'], preparation.FROZEN_MANIFEST_SHA, 'original frozen O manifest required')
    profile = reader.json(profile_ref)
    require(profile['kind'] == diverse.PROFILE_KIND and manifest['derivation']['kind'] ==
            'whole-pack-hash-ranked-frozen-holdout-v2', 'fresh weekly dataset/profile required')
    derivation = manifest['derivation']
    obj(derivation, {'kind', 'profile', 'profile_sha256', 'producer_sources', 'input_sha256', 'script_sha256',
        'ordering', 'indexes', 'frozen_dataset_manifest_sha256', 'train_histogram', 'old_train_histogram',
        'old_train_board_overlap', 'input_unchanged', 'wall_seconds'})
    exact(derivation['input_unchanged'], True)
    exact(derivation['frozen_dataset_manifest_sha256'], frozen['manifest.json']['sha256'])
    expected = diverse.selection_profile(argparse.Namespace(
        selection_seed=profile['selection_seed'], games_per_pack=profile['games_per_pack'],
        train_count=N, expected_frozen_manifest_sha256=frozen['manifest.json']['sha256']),
        {'manifest': old}, reader.json(pack_ref), pack_ref['sha256'],
        {p['sha256']: {'games': p['games'], 'selected_games': p['selected_games']}
         for p in profile['packs']})
    exact(profile, expected, 'fresh selection profile actual/source body differs')
    for key in ('sampling', 'selection_seed', 'games_per_pack', 'train_budget'):
        expected_value = profile['train_budget'] if key == 'train_budget' else profile[key]
        exact(manifest[key], expected_value, 'manifest selection declaration differs')
    exact(derivation['ordering'], profile['ordering'])
    exact(derivation['profile'], profile); exact(derivation['profile_sha256'], profile_ref['sha256'])
    exact(derivation['producer_sources'], profile['producer_sources'])
    hashes = derivation['input_sha256']
    require(type(hashes) is dict and hashes, 'complete raw derivation hashes required')
    for path, digest in hashes.items():
        reference = parent_run.physical_ref(Path(path)); exact(reference['sha256'], sha(digest))
        reader.pin(reference)
    for name, identity in profile['producer_sources'].items():
        matches = [path for path in hashes if path.endswith('/' + name)]
        require(len(matches) == 1, 'producer source closure membership differs')
        exact(reader.files[matches[0]], identity, 'producer source closure size/SHA differs')
    producer = parent_run.physical_ref(REPO / 'scripts/diverse_pack_dataset.py')
    reader.pin(producer); exact(derivation['script_sha256'], producer['sha256'])
    for reference in [*frozen.values(), profile_ref, pack_ref, pool_ref]:
        exact(hashes.get(reference['path']), reference['sha256'], 'required original input missing from derivation')
    new_splits = dataset_join(reader, dataset, manifest, {'train': N, 'holdout': H})
    dataset_join(reader, frozen, old, {'train': N, 'holdout': H})
    for field in ('teacher_identity', 'source_corpus_manifest_sha256', 'dependencies', 'seed', 'split',
                  'positions', 'independent_exclusions', 'label_depth'):
        exact(manifest[field], old[field], 'fresh/frozen semantics differ: ' + field)
    exact(manifest['teacher_identity'], preparation.TEACHER)
    exact(manifest['source_corpus_manifest_sha256'], pack_ref['sha256'])
    for name in ('holdout.positions.jsonl', 'holdout.labels.jsonl'):
        exact(reader.read(dataset[name]), reader.read(frozen[name]), 'holdout bytes changed')
    excluded = pool_exclusions(reader, pool_ref, manifest['independent_exclusions'])
    require(not ((new_splits['train']['keys'] | new_splits['holdout']['keys']) & excluded),
            'dataset contains an independent-pool board')
    pack_manifest = reader.json(pack_ref)
    pack_root = Path(pack_ref['path']).parent
    declared = {item['sha256']: item for item in pack_manifest['unique_files']}
    require(len(declared) == len(pack_manifest['unique_files']) == len(profile['packs']) == PACK_COUNT,
            'complete pack inventory required')
    indexes = {item['pack_sha256']: item for item in derivation['indexes']}
    require(len(indexes) == len(derivation['indexes']) == PACK_COUNT and set(indexes) == set(declared),
            'whole-pack index membership differs')
    selected_locations = {}
    for item in profile['packs']:
        obj(indexes[item['sha256']], {'pack_sha256', 'selected_spans', 'census', 'temporary_selected_bytes_sha256'})
        packed = declared[item['sha256']]
        path = pack_root / packed['path']; reference = parent_run.physical_ref(path)
        exact(reference['sha256'], item['sha256']); exact(reference['bytes'], item['bytes'])
        reader.pin(reference)
        with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            spans, census = diverse.select_spans(data, item['sha256'], profile['selection_seed'], item['selected_games'])
            exact(indexes[item['sha256']]['selected_spans'], spans, 'actual selected span ranking differs')
            exact(indexes[item['sha256']]['census'], census, 'actual whole-pack census differs')
            exact(census['games'], item['games'])
            subset = hashlib.sha256()
            for span in spans: subset.update(data[span['start']:span['end']])
            exact(indexes[item['sha256']]['temporary_selected_bytes_sha256'], subset.hexdigest())
            selected_locations.update({(item['sha256'], span['game_index']): span['rank'] for span in spans})
    old_holdout = {game['game_id']: game for game in old['games'] if game['split'] == 'holdout'}
    games = {game['game_id']: game for game in manifest['games']}
    for gid, game in old_holdout.items():
        exact(games.get(gid), game, 'frozen holdout game provenance changed')
    for game in manifest['games']:
        if game['game_id'] not in old_holdout:
            location = (game['pack_sha256'], game['game_index'])
            require(location in selected_locations and 'selection_rank' in game,
                    'candidate game lacks preregistered selected pack provenance')
            exact(game['selection_rank'], selected_locations[location],
                  'candidate game is not in the preregistered selected pack spans')
    origin_inputs = {path: identity for path, identity in reader.files.items()
                     if path not in {reference['path'] for reference in dataset.values()}}
    counts = {'train': N, 'holdout': H, 'acquired_pool_games': POOL_COUNT,
              'selected_pack_games': sum(item['selected_games'] for item in profile['packs'])}
    return manifest, profile, producer, hashes, origin_inputs, counts


def verify_engine(reader, reference):
    exact(reference['sha256'], ENGINE_SHA, 'fixed white-view build manifest differs')
    engine = reader.json(reference)
    exact(engine['status'], 'complete'); exact(engine['base_commit'], preparation.UPSTREAM)
    reader.map(engine['compiler_files'])
    for value in engine['binaries'].values(): reader.pin(value)
    exact(engine['binaries']['sekirei']['sha256'], ENGINE_BINARY_SHA)
    root = Path(engine['runtime']) / 'source'
    sources = {str(root / name): identity for name, identity in engine['source_files_after'].items()}
    reader.map(sources)
    exact(parent_run.source_map(root), sources, 'fixed engine source membership differs')
    for path, identity in engine['inputs_after'].items():
        actual = Path(path)
        if not actual.exists():
            old = Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
            require(actual.is_relative_to(old), 'missing fixed engine raw input')
            actual = REPO / actual.relative_to(old)
        reader.pin({'path': str(actual), **identity})
    return engine


def verify_prepared_source(reader, prep, build, plan_ref, plan):
    obj(prep, {'schema', 'status', 'upstream_commit', 'source_root', 'plan', 'manifest', 'selection_profile',
               'base_patch', 'source_files', 'numeric_update_export_modules_preserved', 'compiled',
               'training_started', 'model_adoption_claimed', 'final_used', 'row_order_variant'})
    for key, value in {'schema': 'sekirei.weekly-nonlinear-epoch-shuffle-mse-source-preparation.v1', 'status': 'source-only',
        'upstream_commit': preparation.UPSTREAM, 'source_root': build['source_root'], 'plan': plan_ref,
        'source_files': build['source_files'], 'manifest': plan['dataset_inputs']['manifest.json'],
        'selection_profile': plan['selection_profile'], 'numeric_update_export_modules_preserved': False,
        'compiled': False, 'training_started': False, 'model_adoption_claimed': False, 'final_used': False}.items():
        exact(prep[key], value, 'Shuffle source preparation body differs: ' + key)
    exact(plan_ref['bytes'], preparation.SHUFFLE_PLAN_PIN['bytes'])
    exact(plan_ref['sha256'], preparation.SHUFFLE_PLAN_PIN['sha256'])
    variant = preparation.shuffle_variant_sources()
    exact(prep['row_order_variant'], variant, 'Shuffle source declaration differs')
    for reference in [variant['manifest'], variant['patch'], *variant['postimages'].values()]:
        reader.pin(reference)
        exact(build['inputs_before'].get(reference['path']), small(reference), 'Shuffle source input absent before actual compile')
    declared = reader.json(variant['manifest'])
    patch_ref = parent_run.physical_ref(preparation.BASE / 'nonlinear-complete.patch')
    exact(prep['base_patch'], patch_ref); reader.pin(patch_ref)
    published_ref = parent_run.physical_ref(preparation.BASE / 'public-source-manifest.json')
    published = reader.json(published_ref)
    exact(published['upstream_commit'], preparation.UPSTREAM)
    root = Path(build['source_root'])
    revised = {'paired_nonlinear_actual.rs', 'paired_nonlinear_parent_binding.rs', 'weekly_nonlinear_manifest_tests.rs'}
    changed = set(preparation.SHUFFLE_CHANGED_FILES)
    for name, info in published['files'].items():
        reader.pin({'path': str(preparation.BASE / name), **info})
        if name.startswith('source/'):
            relative = name[len('source/'):]
            if relative not in changed and Path(relative).name not in revised | {'main.rs', 'paired_nonlinear_cli.rs'}:
                exact(build['source_files'].get(str(root / relative)), info,
                      'preserved non-objective numeric/core source differs from complete snapshot')
    train = root / 'crates/sekirei-train/src'
    for name in revised:
        reference = parent_run.physical_ref(preparation.REPAIR / name); reader.pin(reference)
        if name == 'paired_nonlinear_parent_binding.rs':
            exact(small(reference), declared['changed_files']['crates/sekirei-train/src/' + name]['before'],
                  'Shuffle parent-binding preimage differs from frozen runtime-v2')
        else:
            exact(reader.read(reference), (train / name).read_bytes(), 'unchanged runtime-v2 reader/tests differ')
    main = (preparation.BASE / 'source/crates/sekirei-train/src/main.rs').read_text()
    needle = '#[cfg(feature = "nnue_white_view_aux_tied")]\nmod paired_nonlinear_sha256;'
    require(main.count(needle) == 1, 'published main registration differs')
    exact((train / 'main.rs').read_text(), main.replace(needle,
        '#[cfg(feature = "nnue_white_view_aux_tied")]\nmod weekly_nonlinear_profile;\n' + needle))
    cli = (preparation.BASE / 'source/crates/sekirei-train/src/paired_nonlinear_cli.rs').read_text()
    needle = 'pub const MODE:&str="white-view-paired-nonlinear-adam-e3-v1";'
    require(cli.count(needle) == 1, 'published mode declaration differs')
    before_cli = cli.replace(needle, 'pub const MODE:&str=crate::weekly_nonlinear_profile::MODE;').encode()
    exact(preparation.digest(before_cli), declared['changed_files']['crates/sekirei-train/src/paired_nonlinear_cli.rs']['before'])
    before_positions = published['files']['source/crates/sekirei-train/src/paired_nonlinear_positions.rs']
    exact(before_positions, declared['changed_files']['crates/sekirei-train/src/paired_nonlinear_positions.rs']['before'])
    for name, reference in variant['postimages'].items():
        expected = declared['changed_files'][name]['after']
        exact(build['source_files'].get(str(root / name)), expected, 'Shuffle source identity differs')
        exact(reader.read(reference), (root / name).read_bytes(), 'Shuffle physical postimage differs')


def verify_dependencies(build):
    dependencies = build['dependency_files']
    actual = builder.registry_dependencies(Path(build['environment']['CARGO_TARGET_DIR']) / 'release/deps')
    require(all(dependencies.get(path) == identity for path, identity in actual.items()),
            'actual compiled registry dependencies are absent or changed')
    registry = Path.home() / '.cargo/registry/src'
    packages = set()
    for name in dependencies:
        path = Path(name)
        require(path.is_relative_to(registry) and len(path.relative_to(registry).parts) >= 3,
                'registry package closure path required')
        parts = path.relative_to(registry).parts
        packages.add(registry / parts[0] / parts[1])
    inventory = {str(path): parent_run.info(path) for package in sorted(packages)
                 for path in sorted(package.rglob('*')) if path.is_file()}
    exact(inventory, dependencies, 'registry dependency package membership differs')


def verify_build(reader, reference, plan_ref, plan):
    build = reader.json(reference)
    obj(build, {'schema', 'status', 'mode', 'producer_source', 'source_root', 'upstream_commit', 'source_head',
                'source_files', 'compiler_files', 'dependency_files', 'engine_build', 'training_binary', 'commands',
                'environment', 'tests', 'process_outcomes', 'inputs_before', 'inputs_after', 'scalar_power_cache_used'})
    for key, expected in {'schema': 'sekirei.weekly-nonlinear-epoch-shuffle-mse-training-build.v1', 'status': 'complete',
                          'mode': plan['mode'], 'upstream_commit': preparation.UPSTREAM,
                          'scalar_power_cache_used': False}.items(): exact(build[key], expected)
    require(type(build['source_head']) is str and re.fullmatch('[0-9a-f]{40}', build['source_head']), 'build source SHA40 required')
    exact(build['inputs_before'], build['inputs_after']); reader.map(build['inputs_before'])
    for key in ('source_files', 'compiler_files', 'dependency_files'):
        reader.map(build[key])
        for path, identity in build[key].items(): exact(build['inputs_before'].get(path), identity)
    exact(parent_run.source_map(Path(build['source_root'])), build['source_files'], 'training source membership differs')
    reader.pin(build['producer_source']); exact(build['producer_source'], parent_run.physical_ref(REPO / 'scripts/weekly_nonlinear_epoch_shuffle_mse_build.py'))
    for bound in (build['producer_source'], build['engine_build'], plan_ref):
        fullref(bound)
        exact(build['inputs_before'].get(bound['path']), small(bound), 'required build predecessor not frozen before compile')
    reader.pin(build['training_binary']); require(os.access(build['training_binary']['path'], os.X_OK), 'executable training binary required')
    engine = verify_engine(reader, build['engine_build']); exact(build['compiler_files'], engine['compiler_files'])
    for name, identity in engine['source_files_after'].items():
        if not name.startswith('crates/sekirei-train/'):
            exact(build['source_files'].get(str(Path(build['source_root']) / name)), identity,
                  'fixed search/core/ABI source differs')
    prep_ref = parent_run.physical_ref(Path(build['source_root']).parent / 'source-preparation.json')
    exact(build['inputs_before'].get(prep_ref['path']), small(prep_ref), 'source preparation not build-bound')
    prep = reader.json(prep_ref)
    verify_prepared_source(reader, prep, build, plan_ref, plan)
    generated = preparation.profile_source(plan, small(plan_ref), reader.json(prep['manifest']),
        small(prep['manifest']), reader.json(prep['selection_profile']), small(prep['selection_profile']))
    source_profile = Path(build['source_root']) / 'crates/sekirei-train/src/weekly_nonlinear_profile.rs'
    exact(source_profile.read_bytes(), generated, 'compiled identities differ from actual selected plan/profile')
    env = obj(build['environment'], {'CARGO_TARGET_DIR', 'RUSTC', 'RUSTFLAGS', 'CARGO_BUILD_JOBS'})
    exact(env['RUSTFLAGS'], '-C target-cpu=x86-64-v3'); exact(env['CARGO_BUILD_JOBS'], '2')
    require(env['RUSTC'] in build['compiler_files'], 'actual compiler is not bound')
    require(Path(env['CARGO_TARGET_DIR']) == Path(build['training_binary']['path']).parents[1], 'build target differs')
    cargo = [path for path in build['compiler_files'] if Path(path).name == 'cargo']
    require(len(cargo) == 1, 'one fixed Cargo binary required')
    common = ['--package', 'sekirei-train', '--bin', 'train', '--release', '--offline', '--locked',
              '--jobs', '2', '--features', 'nnue_white_view_aux_tied']
    exact(build['commands'], [[cargo[0], 'test', *common, 'paired_nonlinear', '--', '--test-threads=1'],
                              [cargo[0], 'build', *common]], 'actual Cargo option vector differs')
    require(len(build['commands']) == len(build['process_outcomes']) == 2, 'two actual build stages required')
    for command, outcome, action in zip(build['commands'], build['process_outcomes'], ('test', 'build')):
        require(type(command) is list and len(command) > 3 and all(type(x) is str and x for x in command), 'literal command required')
        require(Path(command[0]).name == 'cargo' and command[0] in build['compiler_files'] and command[1] == action,
                'fixed compiler/test/build command required')
        for flag in ('--release', '--offline', '--locked'): require(flag in command, 'fixed cargo flag missing')
        require(any(command[i:i+2] == ['--jobs', '2'] for i in range(len(command))) and
                any(command[i:i+2] == ['--features', 'nnue_white_view_aux_tied'] for i in range(len(command))), 'jobs2/feature required')
        obj(outcome, {'command', 'pid', 'pgid', 'returncode', 'waited', 'reaped', 'timed_out', 'group_empty_scans', 'wall_seconds', 'log'})
        exact(outcome['command'], command); integer(outcome['pid'], 1); exact(outcome['pgid'], outcome['pid'])
        for key, value in {'returncode': 0, 'waited': True, 'reaped': True, 'timed_out': False,
                           'group_empty_scans': [True, True]}.items(): exact(outcome[key], value)
        require(type(outcome['wall_seconds']) in (float, int) and math.isfinite(outcome['wall_seconds'])
                and outcome['wall_seconds'] >= 0, 'finite lifecycle wall seconds required')
        reader.pin(outcome['log'])
    parsed_tests = builder.parse_test_log(Path(build['tests']['log']['path']))
    obj(build['tests'], {'log', 'passed', 'failed', 'ignored', 'required_test_names'})
    exact(parsed_tests, build['tests'], 'actual complete Rust test log differs')
    preparation.require_shuffle_tests(parsed_tests)
    exact(build['tests']['log'], build['process_outcomes'][0]['log'])
    reader.pin(build['tests']['log'])
    verify_dependencies(build)
    return build


def service_map(allowlist):
    obj(allowlist, {'schema', 'status', 'services'})
    exact(allowlist['schema'], 'sekirei.weekly-nonlinear-epoch-shuffle-mse-heavy-process-allowlist.v1')
    exact(allowlist['status'], 'frozen-before-preflight')
    require(type(allowlist['services']) is list, 'strict service array required')
    services = {}
    for item in allowlist['services']:
        obj(item, {'pid', 'uid', 'comm', 'starttime_ticks', 'cmdline_sha256'})
        integer(item['pid'], 1); integer(item['uid']); integer(item['starttime_ticks'], 1)
        require(type(item['comm']) is str and item['comm'], 'service comm required')
        if item['cmdline_sha256'] is not None: sha(item['cmdline_sha256'])
        require(item['pid'] not in services, 'duplicate allowlisted PID'); services[item['pid']] = item
    return services


def observe_processes(allowlist, proc_root=Path('/proc'), own_pid=None):
    """Augment the bound scanner: unreadable same-user commands fail closed.

    A pinned, harmless preexisting service may cover unreadable cmdline only
    when PID, UID, comm and kernel start time all still match. Known heavy
    comm names always fail, including processes belonging to other users.
    """
    services = service_map(allowlist)
    excluded, conflicts = [], parent_run.scan_heavy_processes()
    own_pid = os.getpid() if own_pid is None else own_pid
    names = {'train', 'sekirei', 'sekirei-train', 'yaneuraou', 'shogiesa', 'cargo', 'rustc'}
    for root in sorted(proc_root.iterdir()):
        if not root.name.isdigit() or int(root.name) == own_pid:
            continue
        pid = int(root.name)
        try:
            uid = root.stat().st_uid
            comm = (root / 'comm').read_text().strip()
            require(comm not in names, 'another heavy experiment process is running')
            try:
                command = (root / 'cmdline').read_bytes()
            except PermissionError:
                command = None
                require(uid != os.getuid() or pid in services, 'unobserved same-user process is not allowlisted')
            if pid in services:
                actual = {'pid': pid, 'uid': uid, 'comm': comm,
                          'starttime_ticks': int((root / 'stat').read_text().rsplit(')', 1)[1].split()[19]),
                          'cmdline_sha256': None if command is None else hashlib.sha256(command).hexdigest()}
                exact(actual, services[pid], 'preexisting service identity changed')
                excluded.append(actual)
        except (FileNotFoundError, ProcessLookupError):
            continue
    require(not conflicts, 'another heavy experiment process is running')
    return {'conflicts': [], 'excluded_preexisting_services': excluded}


def recipe_body(plan, binding_ref, reference03):
    keys = {'seed', 'epochs', 'train_count', 'objective', 'optimizer', 'learning_rate_f32_bits',
            'head_init_width_f32_bits', 'head_bias_init_f32_bits', 'output_native_l1_budget_f32_bits',
            'shuffle_seed', 'row_order', 'resume_allowed', 'ft_saved_q_max', 'ft_bias_q'}
    return {'schema': 'sekirei.white-view-paired-nonlinear-epoch-shuffle-mse-recipe.v1', 'mode': plan['mode'],
            'feature_schema': plan['architecture']['feature_schema'],
            **{key: plan['training'][key] for key in keys}, 'teacher_identity': plan['teacher_identity'],
            'manifest': plan['dataset_inputs']['manifest.json'], 'reference03': reference03,
            'positions': plan['dataset_inputs']['train.positions.jsonl'],
            'labels': plan['dataset_inputs']['train.labels.jsonl'], 'source_binding': binding_ref}


def validate_source_binding(recipe_ref):
    """Read-only launch consumer. Caller owns all coordination locks."""
    reader = Reader(); recipe = reader.json(recipe_ref); obj(recipe, RECIPE_KEYS)
    sb = reader.json(recipe['source_binding']); obj(sb, BINDING_KEYS)
    parent = reader.json(sb['parent_preflight']); obj(parent, PARENT_KEYS)
    plan = reader.json(sb['selected_plan']); preparation.validate_plan(plan)
    exact(small(sb['selected_plan']), preparation.SHUFFLE_PLAN_PIN)
    reader.pin(plan['incumbent_model'])
    exact(plan['schema'], 'sekirei.weekly-nonlinear-epoch-shuffle-mse-selected-plan.v1'); exact(plan['status'], 'frozen-before-build')
    exact(recipe, recipe_body(plan, recipe['source_binding'], sb['reference03']))
    for key, value in {'schema': 'sekirei.weekly-nonlinear-epoch-shuffle-mse-source-binding.v1', 'status': 'ready',
                       'mode': plan['mode'], 'float_policy': 'x86-ftz-daz'}.items(): exact(sb[key], value)
    for key, value in {'schema': 'sekirei.weekly-nonlinear-epoch-shuffle-mse-parent-preflight.v1', 'status': 'complete',
                       'mode': plan['mode'], 'origin_proof_policy': 'fresh-whole-pack-ranked-dataset-with-frozen-holdout'}.items(): exact(parent[key], value)
    exact(parent['checks'], {key: True for key in PARENT_CHECKS})
    for key in ('selected_plan', 'training_build', 'engine_build', 'dataset_inputs', 'reference03'):
        exact(parent[key], sb[key])
    exact(sb['dataset_inputs'], plan['dataset_inputs'])
    forbidden = {recipe_ref['path'], recipe['source_binding']['path'], sb['parent_preflight']['path']}
    require(not (forbidden & set(parent['source_inputs'])), 'backward/self reference in parent DAG')
    exact(parent['inputs_before'], parent['source_inputs']); exact(parent['inputs_after'], parent['source_inputs'])
    reader.map(parent['source_inputs'])
    reader.map(public_consumer_sources())
    build = verify_build(reader, sb['training_build'], sb['selected_plan'], plan)
    exact(build['source_head'], parent['source_head']); exact(sb['training_binary'], build['training_binary'])
    exact(sb['training_source_files'], build['source_files']); exact(sb['compiler_files'], build['compiler_files'])
    origin = reader.json(parent['dataset_origin_proof']); obj(origin, ORIGIN_KEYS)
    exact(origin['schema'], 'sekirei.weekly-nonlinear-epoch-shuffle-mse-dataset-origin.v1'); exact(origin['status'], 'verified')
    exact(origin['checks'], {key: True for key in ORIGIN_CHECKS})
    exact(origin['actual_replay_repeated_by_verifier'], False); exact(origin['final_used'], False)
    exact(origin['dataset_inputs'], sb['dataset_inputs']); exact(origin['profile'], plan['selection_profile'])
    _, _, producer, hashes, inputs, counts = verify_origin_inputs(Reader(), origin['dataset_inputs'],
        origin['frozen_dataset_inputs'], origin['profile'], origin['pack_manifest'], origin['independent_pool_manifest'])
    exact(origin['producer_source'], producer); exact(origin['input_hashes'], hashes)
    exact(origin['inputs_before'], inputs); exact(origin['inputs_after'], inputs)
    exact(origin['counts'], counts); exact(parent['origin_counts'], counts)
    reader.map(inputs)
    evidence = parent['process_evidence']
    obj(evidence, {'schema', 'status', 'producer_source', 'process_reader', 'allowlist', 'scan_records', 'lock_records', 'source_head'})
    exact(evidence['schema'], 'sekirei.weekly-nonlinear-epoch-shuffle-mse-preflight-process-evidence.v1'); exact(evidence['status'], 'observed-clear')
    exact(evidence['source_head'], parent['source_head'])
    for key in ('producer_source', 'process_reader', 'allowlist'): reader.pin(evidence[key])
    services = service_map(reader.json(evidence['allowlist']))
    exact(evidence['producer_source'], parent_run.physical_ref(Path(__file__).resolve()))
    exact(evidence['process_reader'], parent_run.physical_ref(REPO / 'scripts/weekly_nonlinear_epoch_shuffle_mse_run.py'))
    require(type(evidence['scan_records']) is list and len(evidence['scan_records']) == 2, 'two actual process observations required')
    for scan in evidence['scan_records']:
        obj(scan, {'conflicts', 'excluded_preexisting_services'}); exact(scan['conflicts'], [])
        require(type(scan['excluded_preexisting_services']) is list, 'service observation array required')
        observed = set()
        for service in scan['excluded_preexisting_services']:
            require(type(service) is dict and service.get('pid') not in observed, 'duplicate observed service')
            exact(service, services.get(service.get('pid')), 'unallowlisted or changed observed service')
            observed.add(service['pid'])
    lock_paths = set()
    for lock in evidence['lock_records']:
        obj(lock, {'path', 'exclusive', 'nonblocking', 'acquired'})
        require(Path(lock['path']).is_absolute() and lock['path'] not in lock_paths and type(lock['exclusive']) is bool,
                'unique canonical typed lock record required')
        exact(lock['nonblocking'], True); exact(lock['acquired'], True); lock_paths.add(lock['path'])
        exact(lock['exclusive'], True)
    require(lock_paths, 'actual acquired locks required')
    numeric_ref = next({'path': path, **identity} for path, identity in parent['source_inputs'].items()
                       if Path(path).name == 'white_view_build_contract.py' and identity['sha256'] == parent_run.NUMERIC_SOURCE_SHA[2])
    gate, gate_sources = parent_run.load_numeric_gate(Path(numeric_ref['path']))
    reader.map(gate_sources)
    exact(reader.read(sb['reference03']), gate.encode_native(gate.canonical_reference()), 'seed42 reference03 fullbytes differ')
    # All directly and transitively consumed refs must be external predecessors.
    for path, identity in reader.files.items():
        if path not in forbidden:
            exact(parent['source_inputs'].get(path), identity, 'consumed predecessor absent from parent closure')
    reader.current()
    exact(parent_run.source_map(Path(build['source_root'])), build['source_files'],
          'training source membership changed while validating')
    return {'recipe': recipe, 'source_binding': sb, 'parent': parent, 'plan': plan, 'build': build}


def fresh_private_output(output, protected=()):
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists(), 'fresh canonical private output required')
    ancestors = (output.parent, *output.parent.parents)
    require(not any((path / '.git').exists() for path in ancestors), 'preflight output must be outside Git')
    require(any(path.stat().st_uid == os.getuid() and path.stat().st_mode & 0o777 == 0o700
                for path in ancestors), 'owned private 0700 output ancestor required')
    require(not any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected),
            'preflight output overlaps a protected input tree')


def create(args):
    reader = Reader()
    plan_ref = pinned(args.plan, args.expected_plan_sha256); plan = reader.json(plan_ref)
    preparation.validate_plan(plan)
    exact(small(plan_ref), preparation.SHUFFLE_PLAN_PIN)
    require(plan['schema'] == 'sekirei.weekly-nonlinear-epoch-shuffle-mse-selected-plan.v1' and plan['status'] == 'frozen-before-build', 'new frozen plan required')
    build_ref = pinned(args.training_build, args.expected_training_build_sha256)
    reference03 = pinned(args.reference03, args.expected_reference03_sha256)
    pack_ref = pinned(args.pack_manifest, args.expected_pack_manifest_sha256)
    pool_ref = pinned(args.pool_manifest, args.expected_pool_manifest_sha256)
    frozen = dataset_five(args.frozen_dataset)
    exact(frozen['manifest.json']['sha256'], args.expected_frozen_manifest_sha256)
    allow_ref = pinned(args.allowlist, args.expected_allowlist_sha256)
    output = args.output
    fresh_private_output(output, (REPO, args.frozen_dataset,
        Path(plan['dataset_inputs']['manifest.json']['path']).parent, args.pack_manifest.parent, args.pool_manifest.parent))
    require(args.lock and len(set(args.lock)) == len(args.lock), 'unique coordination lock paths required')
    with ExitStack() as locks:
        records = []
        for path in sorted(args.lock):
            require(path.is_absolute() and path.resolve() == path, 'canonical lock path required')
            locks.enter_context(benchmark.nonblocking_lock(path, exclusive=True))
            records.append({'path': str(path), 'exclusive': True, 'nonblocking': True, 'acquired': True})
        reader.map(public_consumer_sources())
        scans = observe_processes(reader.json(allow_ref))
        build = verify_build(reader, build_ref, plan_ref, plan)
        require(not output.is_relative_to(Path(build['source_root']).parent), 'output overlaps training runtime')
        manifest, profile, producer, hashes, origin_inputs, counts = verify_origin_inputs(Reader(),
            plan['dataset_inputs'], frozen, plan['selection_profile'], pack_ref, pool_ref)
        reader.map(origin_inputs)
        for reference in plan['dataset_inputs'].values(): reader.pin(reference)
        reader.pin(plan['incumbent_model']); reader.pin(reference03)
        gate, gate_sources = parent_run.load_numeric_gate(args.numeric_build_contract)
        exact(reader.read(reference03), gate.encode_native(gate.canonical_reference()), 'seed42 full reference03 differs')
        reader.map(gate_sources)
        producer_source = parent_run.physical_ref(Path(__file__).resolve()); reader.pin(producer_source)
        process_reader = parent_run.physical_ref(REPO / 'scripts/weekly_nonlinear_epoch_shuffle_mse_run.py'); reader.pin(process_reader)
        scans_after = observe_processes(reader.json(allow_ref))
        process_evidence = {'schema': 'sekirei.weekly-nonlinear-epoch-shuffle-mse-preflight-process-evidence.v1', 'status': 'observed-clear',
            'producer_source': producer_source, 'process_reader': process_reader, 'allowlist': allow_ref,
            'scan_records': [scans, scans_after], 'lock_records': records, 'source_head': build['source_head']}
        reader.current()
        exact(parent_run.source_map(Path(build['source_root'])), build['source_files'],
              'training source membership changed while authoring')
        output.mkdir(mode=0o700)
        origin = {'schema': 'sekirei.weekly-nonlinear-epoch-shuffle-mse-dataset-origin.v1', 'status': 'verified',
            'producer_source': producer, 'dataset_inputs': plan['dataset_inputs'], 'frozen_dataset_inputs': frozen,
            'profile': plan['selection_profile'], 'pack_manifest': pack_ref, 'independent_pool_manifest': pool_ref,
            'input_hashes': hashes, 'checks': {key: True for key in ORIGIN_CHECKS}, 'counts': counts,
            'inputs_before': origin_inputs, 'inputs_after': origin_inputs,
            'actual_replay_repeated_by_verifier': False, 'final_used': False}
        origin_ref = parent_run.write_new(output / 'dataset-origin.json', origin); reader.pin(origin_ref)
        before = reader.current()
        parent = {'schema': 'sekirei.weekly-nonlinear-epoch-shuffle-mse-parent-preflight.v1', 'status': 'complete', 'mode': plan['mode'],
            'selected_plan': plan_ref, 'training_build': build_ref, 'engine_build': build['engine_build'],
            'dataset_inputs': plan['dataset_inputs'], 'reference03': reference03, 'dataset_origin_proof': origin_ref,
            'origin_proof_policy': 'fresh-whole-pack-ranked-dataset-with-frozen-holdout', 'origin_counts': counts,
            'source_inputs': before, 'inputs_before': before, 'inputs_after': before,
            'checks': {key: True for key in PARENT_CHECKS}, 'process_evidence': process_evidence, 'source_head': build['source_head']}
        parent_ref = parent_run.write_new(output / 'parent-preflight.json', parent)
        sb = {'schema': 'sekirei.weekly-nonlinear-epoch-shuffle-mse-source-binding.v1', 'status': 'ready', 'mode': plan['mode'],
            'selected_plan': plan_ref, 'parent_preflight': parent_ref, 'training_build': build_ref,
            'training_binary': build['training_binary'], 'training_source_files': build['source_files'],
            'compiler_files': build['compiler_files'], 'engine_build': build['engine_build'],
            'dataset_inputs': plan['dataset_inputs'], 'reference03': reference03, 'float_policy': 'x86-ftz-daz'}
        binding_ref = parent_run.write_new(output / 'source-binding.json', sb)
        recipe_ref = parent_run.write_new(output / 'recipe.json', recipe_body(plan, binding_ref, reference03))
        reader.current()
        print(json.dumps({'status': 'preflight-authored', 'recipe': recipe_ref, 'counts': counts}, indent=2))
        return recipe_ref


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'training-build', 'reference03', 'pack-manifest', 'pool-manifest', 'allowlist'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    parser.add_argument('--frozen-dataset', type=Path, required=True)
    parser.add_argument('--expected-frozen-manifest-sha256', required=True)
    parser.add_argument('--numeric-build-contract', type=Path, required=True)
    parser.add_argument('--lock', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    create(parser.parse_args())


if __name__ == '__main__':
    main()
