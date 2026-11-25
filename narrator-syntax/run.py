import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from classification import classification
from language import corpus, feature_counts, gi_rankings, verb_adverbs
from verify import verify


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Reproduce the narrator syntax experiments.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    output = args.output.resolve()
    if any(output == p or p in output.parents for p in (ROOT / 'data', ROOT / 'reference', ROOT / '.git')):
        parser.error('Choose an output directory outside the experiment inputs.')
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        stage = Path(temp)
        data = ROOT / 'data'
        corpus(data, stage)
        feature_counts(data, stage)
        gi_rankings(data, stage)
        verb_adverbs(data, stage)
        classification(data, stage)
        verify(stage, ROOT / 'reference' / 'poster.json')
        for result in stage.glob('*.csv'):
            result.replace(output / result.name)
    print(f'510 conditions and poster tables verified. Results: {output}')


if __name__ == '__main__':
    main()
