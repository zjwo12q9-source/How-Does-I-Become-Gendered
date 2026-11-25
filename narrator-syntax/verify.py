import hashlib
import json

import numpy as np
import pandas as pd


def check(name, computed, expected):
    if list(computed) != list(expected):
        raise ValueError(f'{name}: {computed} != {expected}')


def verify(output, reference):
    target = json.loads(reference.read_text(encoding='utf-8'))

    def read(name, keys=None):
        table = pd.read_csv(output / f'{name}.csv', keep_default_na=False)
        return table.set_index(keys, verify_integrity=True) if keys else table

    corpus = read('panel_03', 'group')
    check('03', [corpus.at['all','works'], corpus.at['female','works'],
                 corpus.at['male','works'], corpus.at['all','tokens']], target['corpus'])
    grid = read('conditions_510', ['arm','mask']).sort_index()
    cells = grid[['tn','fp','fn','tp']].to_numpy(dtype='<i4')
    if len(grid) != 510 or hashlib.sha256(cells.tobytes()).hexdigest() != target['condition_count_digest']:
        raise ValueError('The 510 condition counts do not match.')
    models = read('panel_05', 'model')
    for name, values in target['models'].items():
        check('05', models.loc[name].round(2), values)
    check('06', read('panel_06').iloc[0].round(2), target['features'])
    pos = read('panel_06_1', ['scope','pos','arm'])
    for scope, tag, *values in target['pos']:
        row = []
        for arm in ['raw', 'clean']:
            item = pos.loc[(scope, tag, arm)]
            row.extend([item['count'], round(item.percent, 2)])
        check('06-1', row, values)
    ranking = read('panel_06_2', ['category','group','rank'])
    for group, entries in target['gi'].items():
        for rank, expected in enumerate(entries, 1):
            row = []
            for gender in ('female', 'male'):
                item = ranking.loc[(group, gender, rank)]
                row.extend([item.surface, item['count']])
            check('06-2', row, expected)
    bases = read('verb_baselines', 'verb')
    for verb, values in target['baselines'].items():
        check('06-3', bases.loc[verb, ['N_f','N_m']], target['verb_denominators'])
        check('06-3', bases.loc[verb, ['V_f','V_m','female_percent','male_percent','female_to_male']].round(3), values)
    pairs = read('panel_06_3', ['verb','higher_group','rank'])
    for verb, group, rank, adverb, *values in target['adverbs']:
        item = pairs.loc[(verb,group,rank)]
        check('06-3', [item.adverb], [adverb])
        check('06-3', np.round(item[['C_f','C_m','female_percent','male_percent','display_ratio']].astype(float), 3), values)
    totals = read('adverb_totals','verb')
    for verb, values in target['adverb_totals'].items():
        check('06-3', totals.loc[verb, ['combinations','occurrences']], values)
    check('06-3', read('multiple_testing').iloc[0], target['tests'])
    tests = read('all_adverbs', ['verb','adverb'])
    for verb, adverb, p, q in target['examples']:
        item = tests.loc[(verb,adverb)]
        check('06-3', [round(item.fisher_p, 5), round(item.bh_q, 4)], [p, q])
    examples = read('panel_07_pairs', ['verb','higher_group'])
    for verb, group, rank, adverb, *_ in target['adverbs']:
        if rank == 1:
            check('07', [examples.loc[(verb, group), 'adverb']], [adverb])
