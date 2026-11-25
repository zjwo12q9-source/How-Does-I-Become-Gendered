import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


ROLES = {'agent': {'ccomp', 'advmod'}, 'patient': {'ccomp', 'advmod'},
         'mod': {'ccomp', 'advmod'}, 'poss': {'amod', 'nmod'}}
BITS = {s: i for i, s in enumerate((*ROLES, 'ccomp', 'advmod', 'amod', 'nmod'))}


def booknlp_counts(book, tokens):
    with book.open(encoding='utf-8-sig') as stream:
        characters = json.load(stream)['characters']
    narrator, = (row for row in characters if str(row['id']) == '0')
    features = Counter()
    anchors = defaultdict(list)
    for role in ROLES:
        for item in narrator[role]:
            word = str(item['w']).strip().lower()
            if not word or '|' in word:
                raise ValueError('Invalid anchor word.')
            features[f'B|{role}|{word}'] += 1
            anchors[int(item['i'])].append((role, word))
    prepositions = {}
    objects = []
    with tokens.open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream, delimiter='\t', quoting=csv.QUOTE_NONE)
        for expected, row in enumerate(rows):
            token = int(row['token_ID_within_document'])
            if token != expected or None in row or None in row.values():
                raise ValueError('Invalid token record.')
            word = row['word'].strip().lower()
            if not word or '|' in word:
                continue
            head = int(row['syntactic_head_ID'])
            relation = row['dependency_relation'].strip().lower()
            if relation == 'prep':
                prepositions[token] = head
            elif relation in ('pobj', 'pcomp'):
                objects.append((head, word))
            for role, anchor in anchors.get(head, ()):
                if relation in ROLES[role]:
                    features[f'E|{role}|{relation}|{anchor}|{word}'] += 1
    for head, word in objects:
        for role, anchor in anchors.get(prepositions.get(head), ()):
            if role == 'poss':
                features[f'E|poss|nmod|{anchor}|{word}'] += 1
    return features


def marker_key(word):
    return re.sub(r'^[^\w]+|[^\w]+$', '', word.casefold())


def feature_index(names, markers):
    bits, retained = [], []
    for name in names:
        parts = name.split('|')
        if len(parts) == 3 and parts[0] == 'B' and parts[1] in ROLES:
            family, words = parts[1], parts[2:]
        elif len(parts) == 5 and parts[0] == 'E' and parts[2] in ROLES.get(parts[1], ()):
            family, words = parts[2], parts[3:]
        else:
            raise ValueError('Unknown feature format.')
        bits.append(BITS[family])
        retained.append(all(marker_key(word) not in markers for word in words))
    return np.array(bits), np.array(retained)


def pack_folds(docs, labels, folds, markers, output, rng):
    encoder = DictVectorizer(sort=True)
    counts = encoder.fit_transform(docs).tocsr()
    bits, retained = feature_index(encoder.get_feature_names_out(), markers)
    if set(folds) != set(range(1, 6)) or set(labels) != {0, 1}:
        raise ValueError('Expected five folds and two labels.')
    output.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        train = rng.permutation(np.flatnonzero(folds != fold))
        test = rng.permutation(np.flatnonzero(folds == fold))
        n = len(train)
        df = counts[train].getnnz(axis=0)
        idf = np.log((1 + n) / (1 + df)) + 1
        weighted = counts[np.r_[train,test]].multiply(idf).tocsc()
        blocks = []
        for keep in (True, False):
            for family in range(8):
                mask = (bits == family) & (retained == keep) & (df > 0)
                features = weighted[:, mask].tocsr()
                product = (features @ features.T).toarray()
                blocks.append(np.r_[product[:n, :n][np.tril_indices(n)],
                                    product[n:, :n].ravel(), np.diag(product)[n:]])
        np.savez_compressed(output / f'fold_{fold}.npz', products=np.array(blocks).reshape(2, 8, -1),
                            train_y=labels[train], test_y=labels[test])


def prepare(directory, output):
    index = pd.read_csv(directory / 'index.csv', dtype={'item':str,'group':str}, keep_default_na=False)
    if not index.item.is_unique or index.groupby('group').fold.nunique().max() != 1:
        raise ValueError('Each item must be unique and each author group must occupy one fold.')
    if not index.item.str.fullmatch(r'[\w-]+').all() or (index.group == '').any():
        raise ValueError('Invalid item or group identifier.')
    if set(index.label) != {0, 1}:
        raise ValueError('Expected labels 0 and 1.')
    source = directory / 'features.jsonl.gz'
    if source.is_file():
        records = {}
        with gzip.open(source, 'rt', encoding='utf-8') as stream:
            for row in map(json.loads, stream):
                item = str(row['item'])
                if item in records:
                    raise ValueError('Duplicate feature record.')
                records[item] = row['features']
        docs = [records[item] for item in index.item]
    else:
        docs = [booknlp_counts(directory / f'{item}.book', directory / f'{item}.tokens') for item in index.item]
    markers = {marker_key(word) for word in pd.read_csv(Path(__file__).parent / 'data' / 'markers.csv').word}
    pack_folds(docs, index.label.to_numpy(np.uint8), index.fold.to_numpy(), markers,
               output, np.random.default_rng())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.directory, args.output)
