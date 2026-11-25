
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


FAMILIES = ('agent', 'patient', 'modifier', 'possessive', 'ccomp', 'advmod', 'amod', 'nmod')


def coordinates(products, n_train, n_test):
    triangle = n_train * (n_train + 1) // 2
    cross_end = triangle + n_test * n_train
    gram = np.empty((n_train, n_train), dtype=float)
    lower = np.tril_indices(n_train)
    gram[lower] = products[:triangle]
    gram[(lower[1], lower[0])] = products[:triangle]
    cross = products[triangle:cross_end].reshape(n_test, n_train)
    train_norms = np.sqrt(gram.diagonal())
    test_norms = np.sqrt(products[cross_end:])
    train_norms[train_norms == 0] = 1
    test_norms[test_norms == 0] = 1
    gram /= train_norms[:, None] * train_norms[None, :]
    cross = cross / (test_norms[:, None] * train_norms[None, :])
    values, vectors = np.linalg.eigh(gram)
    keep = values > len(values) * np.finfo(float).eps * values[-1]
    scale = np.sqrt(values[keep])
    basis = vectors[:, keep]
    return basis * scale, (cross @ basis) / scale


def fit_fold(path):
    cells = np.zeros((2, 255, 4), dtype=np.int64)
    with np.load(path, allow_pickle=False) as fold:
        train_y, test_y = fold['train_y'], fold['test_y']
        retained, removed = fold['products']
    for arm, components in enumerate((retained + removed, retained)):
        for mask in range(1, 256):
            chosen = [i for i in range(len(FAMILIES)) if mask & (1 << i)]
            train, test = coordinates(components[chosen].sum(axis=0), len(train_y), len(test_y))
            model = LogisticRegression(C=1, solver='liblinear', class_weight='balanced',
                                       max_iter=1500, tol=1e-4, random_state=42)
            model.fit(train, train_y)
            prediction = model.predict_proba(test)[:, 1] >= 0.5
            cells[arm, mask - 1] = np.bincount(2 * test_y + prediction, minlength=4)
    return cells


def classification(data, output):
    folds = [data / 'folds' / f'fold_{i}.npz' for i in range(1, 6)]
    counts = np.zeros((2, 255, 4), dtype=np.int64)
    with threadpool_limits(limits=1):
        for i, path in enumerate(folds, 1):
            counts += fit_fold(path)
            print(f'{i}/{len(folds)} folds complete', flush=True)
    rows = []
    for arm, subset in zip(('raw', 'clean'), counts):
        for mask, (tn, fp, fn, tp) in enumerate(subset, 1):
            members = '+'.join(name for j, name in enumerate(FAMILIES) if mask & (1 << j))
            f1 = (2 * tn / (2 * tn + fp + fn) + 2 * tp / (2 * tp + fp + fn)) / 2
            recall = tp / (tp + fn)
            balanced = (recall + tn / (tn + fp)) / 2
            rows.append((arm, mask, members, tn, fp, fn, tp, f1, balanced, recall))
    results = pd.DataFrame(rows, columns=['arm','mask','families','tn','fp','fn','tp',
                                          'macro_f1','balanced_accuracy','female_recall'])
    results.to_csv(output / 'conditions_510.csv', index=False)
    clean = results[results.arm == 'clean'].set_index('mask')
    metrics = ['macro_f1', 'balanced_accuracy', 'female_recall']
    mixed = clean[(clean.index & 15 != 0) & (clean.index & 240 != 0)]
    if len(mixed) != 225:
        raise ValueError('Expected 225 mixed feature combinations.')
    summary = pd.DataFrame([clean.loc[15, metrics], clean.loc[255, metrics], mixed[metrics].mean()],
                           index=['B4','B4+E4','mixed_225_mean'], columns=metrics).astype(float)
    summary.mul(100).rename_axis('model').to_csv(output / 'panel_05.csv')
