import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact


def feature_counts(data, output):
    counts = pd.read_csv(data / 'pos_counts.csv', keep_default_na=False)
    counts['total'] = counts.groupby(['arm', 'scope'])['count'].transform('sum')
    counts['percent'] = 100 * counts['count'] / counts.total
    counts['rank'] = counts.groupby(['arm', 'scope'])['count'].rank(method='min', ascending=False).astype(int)
    selected = counts[counts.scope.isin(['B', 'B_VERB→E']) & (counts['rank'] <= 3)]
    selected.sort_values(['scope', 'rank', 'arm']).to_csv(output / 'panel_06_1.csv', index=False)
    clean = counts[counts.arm == 'clean'].set_index(['scope', 'pos'])
    base = int(clean.loc[('B', 'VERB'), 'total'])
    extension = int(clean.loc[('E', 'VERB'), 'total'])
    combined = int(clean.loc[('B+E', 'VERB'), 'total'])
    if base + extension != combined:
        raise ValueError('Inconsistent feature totals.')
    pd.DataFrame([{'baseline': base, 'expanded': combined, 'added': extension,
                   'increase_pct': 100 * extension / base,
                   'added_adv': clean.loc[('E', 'ADV'), 'count'],
                   'added_verb': clean.loc[('E', 'VERB'), 'count']}]).to_csv(output / 'panel_06.csv', index=False)


def gi_rankings(data, output):
    counts = pd.read_csv(data / 'gi_verb_counts.csv')
    groups = pd.read_csv(data / 'gi_categories.csv')
    long = counts.melt(id_vars=['lemma', 'surface'], value_vars=['female', 'male'],
                       var_name='group', value_name='count')
    ranked = long.merge(groups, on='lemma').sort_values(
        ['category', 'group', 'count', 'surface'], ascending=[True, True, False, True])
    ranked['rank'] = ranked.groupby(['category','group']).cumcount() + 1
    ranked[ranked['rank'] <= 3].to_csv(output / 'panel_06_2.csv', index=False)


def verb_adverbs(data, output):
    bases = pd.read_csv(data / 'verb_totals.csv')
    bases['female_percent'] = 100 * bases.V_f / bases.N_f
    bases['male_percent'] = 100 * bases.V_m / bases.N_m
    bases['female_to_male'] = bases.female_percent / bases.male_percent
    bases.to_csv(output / 'verb_baselines.csv', index=False)
    pairs = pd.read_csv(data / 'adverb_counts.csv').merge(bases[['verb','V_f','V_m','N_f','N_m']],
                                                          on='verb', validate='many_to_one')
    pairs['female_percent'] = 100 * pairs.C_f / pairs.V_f
    pairs['male_percent'] = 100 * pairs.C_m / pairs.V_m
    pairs['conditional_ratio'] = (pairs.C_f / pairs.V_f) / (pairs.C_m / pairs.V_m)
    ratios = (pairs.C_f / pairs.N_f) / (pairs.C_m / pairs.N_m)
    pairs['higher_group'] = np.select([ratios > 1, ratios < 1], ['female', 'male'], default='tie')
    pairs['selection_ratio'] = np.maximum(ratios, 1 / ratios)
    pairs['display_ratio'] = np.where(pairs.higher_group == 'male',
                                      1 / pairs.conditional_ratio, pairs.conditional_ratio)
    pairs['fisher_p'] = [fisher_exact([[r.C_f, r.V_f-r.C_f], [r.C_m, r.V_m-r.C_m]],
                                      alternative='two-sided').pvalue for r in pairs.itertuples()]
    pairs['bh_q'] = false_discovery_control(pairs.fisher_p)
    pairs = pairs.sort_values(['verb','higher_group','selection_ratio','adverb'],
                              ascending=[True,True,False,True])
    pairs['rank'] = pairs.groupby(['verb','higher_group']).cumcount() + 1
    pairs.to_csv(output / 'all_adverbs.csv', index=False)
    ranked = pairs[(pairs.higher_group != 'tie') & (pairs['rank'] <= 4)]
    ranked.to_csv(output / 'panel_06_3.csv', index=False)
    ranked[ranked['rank'] == 1][['verb','adverb','higher_group']].to_csv(output / 'panel_07_pairs.csv', index=False)
    sets = pairs.groupby('verb').agg(combinations=('adverb','size'),female=('C_f','sum'),male=('C_m','sum'))
    sets['occurrences'] = sets.female + sets.male
    sets.to_csv(output / 'adverb_totals.csv')
    pd.DataFrame([{'pairs': len(pairs), 'fisher_below_05': (pairs.fisher_p < .05).sum(),
                   'bh_below_05': (pairs.bh_q < .05).sum()}]).to_csv(output / 'multiple_testing.csv', index=False)


def corpus(data, output):
    counts = pd.read_csv(data / 'corpus_counts.csv').set_index('group')
    counts.loc['all'] = counts.sum()
    counts.to_csv(output / 'panel_03.csv')
