# How Does “I” Become Gendered?
*Lexical and Syntactic Features of Gender in First-Person Narration in English-Language Fiction*

This repository reproduces the quantitative results of a DH2027 poster submission examining narrator gender and syntactic features in English-language first-person fiction. It uses BookNLP-derived features from 340 novels (48 with female narrators and 292 with male narrators) for narrator gender classification and verb–adverb pair analysis.

The repository is designed to recalculate the numerical results and tables presented in the poster using the included model inputs. It is not a package for collecting the original novels or running the full BookNLP pipeline from scratch.

## 1. Requirements and Usage

From the repository root, run the following commands using Python 3.12:

```bash
python -m pip install -r requirements.txt
python run.py
```

By default, results are saved in `results/`. To use a different output folder, run:

```bash
python run.py --output ./reproduced_results
```

`run.py` calculates corpus statistics, feature statistics, vocabulary rankings, verb–adverb analysis, and classification results in order. It then uses `verify.py` to compare the results with the poster reference values in `reference/poster.json`.

If the values do not match, an error is raised. Only result CSV files that pass verification are saved in the specified output folder. Successful execution of the full workflow must be checked in the user's environment.

## 2. Main Files and Folders

```text
narrator-syntax/
├── README.md
├── README.txt
├── .gitignore
├── requirements.txt
├── run.py                  # Run all reproduction steps
├── classification.py       # Gender classification: 510 experimental conditions
├── language.py             # Corpus, feature, vocabulary, and verb–adverb statistics
├── verify.py               # Compare results with poster reference values
├── prepare.py              # Separate tool for preparing fold inputs from original features
├── data/
│   ├── folds/
│   │   ├── fold_1.npz
│   │   ├── fold_2.npz
│   │   ├── fold_3.npz
│   │   ├── fold_4.npz
│   │   └── fold_5.npz
│   ├── corpus_counts.csv
│   ├── pos_counts.csv
│   ├── gi_categories.csv
│   ├── gi_verb_counts.csv
│   ├── verb_totals.csv
│   ├── adverb_counts.csv
│   └── markers.csv
├── reference/
│   └── poster.json         # Reference values for poster results and verification
└── results/                # Created when run.py is executed
```

The files `data/folds/fold_1.npz` through `fold_5.npz` contain numerical labels and TF–IDF inner-product data for each category within each fold. These inputs are used to reproduce the classification results. They do not contain original sentences or BookNLP `.book` and `.tokens` files.

## 3. Methods

**Classification experiments.** We examine four BookNLP-derived categories (agent, patient, possessive, and modifier) and four added dependency relations (`advmod`, `ccomp`, `amod`, and `nmod`). Document-level features are represented using TF–IDF.

We evaluate all 255 nonempty combinations of the eight categories, with explicit gender markers (`raw`) and after removing 147 explicit gender markers (`clean`), yielding 510 experimental conditions.

Stratified five-fold cross-validation is used to assess macro-F1, balanced accuracy, and female-class recall. This gives 2 × 255 × 5 = 2,550 classifier fits. `classification.py` reconstructs coordinates from the included inner-product matrices and fits L2-regularized logistic regression models (`liblinear`, `C=1`, balanced class weights, classification threshold of 0.5).

This package reproduces the five-fold evaluation of 510 fixed experimental conditions. It does not reproduce the separate inner four-fold model selection process of nested cross-validation. The included anonymized numerical inputs also do not allow an independent audit of how the original author-based folds were created.

**Vocabulary and collocation analysis.** `language.py` calculates feature frequencies, part-of-speech ratios, vocabulary rankings by General Inquirer (GI) category, and verb–adverb pairs involving `felt` and `thought`.

We focus on `felt` and `thought` as examples of feeling and cognition, without restricting feature extraction to these verbs.

We selected 47 verb–adverb pairs occurring at least 20 times overall and at least five times in each narrator-gender group before deduplication.

Conditional collocation rates are calculated against narrator-linked tokens of the corresponding verb within each gender group, not all corpus words. This denominator is distinct from the frequency criteria used for pair selection.

Gender differences are tested using two-sided Fisher's exact tests on raw verb-token counts, with Benjamini–Hochberg correction for multiple comparisons.

## 4. Main Results

### 4.1. Feature Extraction

Adding four dependency relations to the original BookNLP categories increased extracted feature occurrences by 49.57%, yielding 561,262 additional occurrences.

The `advmod` relation alone contributed 229,319 occurrences, accounting for 40.86% of the added material.

### 4.2. Narrator Gender Classification

The following table presents the fixed comparison under the `clean` condition, after removing 147 explicit gender markers.

| Feature configuration | Macro-F1 | Balanced accuracy | Female-class recall |
| --- | ---: | ---: | ---: |
| B4: Four original BookNLP categories | 74.58% | 75.23% | 58.33% |
| B4+E4: Original categories and four added dependency relations | 73.24% | 70.69% | 45.83% |

Adding all four dependency relations did not improve classification performance over the baseline model. However, the extension revealed lexical combinations unavailable in the original categories. Lower classification performance alone does not imply that the additional features have no linguistic value.

### 4.3. Verb–Adverb Collocation

The following pairs showed differences in conditional collocation rates between narrator-gender groups.

| Verb | Adverb | Group with higher rate | Rate ratio |
| --- | --- | --- | ---: |
| felt | instinctively | Female narrators | 3.32 |
| felt | suddenly | Male narrators | 2.10 |
| thought | only | Female narrators | 3.16 |
| thought | however | Male narrators | 1.74 |

The unadjusted Fisher p-values for `felt–instinctively` and `thought–only` were approximately 0.00223 and 0.00319, respectively. Both pairs had Benjamini–Hochberg-adjusted p-values of approximately 0.0750.

None of the 47 verb–adverb pairs met the 0.05 significance threshold after correction for multiple comparisons.

## 5. Generated Result Files

| Poster panel | Generated files | Description |
| --- | --- | --- |
| 03 | `panel_03.csv` | Number of novels and tokens by narrator gender in the 340-novel corpus |
| 05 | `conditions_510.csv`, `panel_05.csv` | Confusion matrices and classification performance for 510 experimental conditions, fixed B4 vs. B4+E4 comparison, and mean results for 225 mixed conditions |
| 06 / 6-1 | `panel_06.csv`, `panel_06_1.csv` | Growth in feature occurrences and part-of-speech frequencies and ratios |
| 6-2 | `panel_06_2.csv` | Verb rankings by GI category |
| 6-3 | `verb_baselines.csv`, `adverb_totals.csv`, `all_adverbs.csv`, `panel_06_3.csv`, `multiple_testing.csv` | Baseline verb frequencies, 47 verb–adverb pairs, statistical tests, and rankings for display |
| 07 | `panel_07_pairs.csv` | Four top-ranked verb–adverb pairs, one for each verb–gender group |

`panel_07_pairs.csv` automatically extracts the top-ranked verb–adverb pair from each of the four groups. It does not reproduce or verify the literary close readings themselves.

## 6. Scope of Reproduction

The default workflow starts with precomputed inputs in `data/folds/`. It does not rerun BookNLP on the original novels.

To use `prepare.py` separately, you need `index.csv` and feature inputs (`features.jsonl.gz` or the original `.book` and `.tokens` files).

`verify.py` checks whether the recalculated results match the reference values in `reference/poster.json`. This verification is distinct from independent validation using an external corpus.

## 7. Software and Data Sources

- [BookNLP — Official Software Repository](https://github.com/booknlp/booknlp)
- [Project Gutenberg — Online eBook Catalog](https://www.gutenberg.org/ebooks/)

## 8. References

Bamman, David, Ted Underwood, and Noah A. Smith. 2014. “A Bayesian Mixed Effects Model of Literary Character.” *Proceedings of ACL 2014*, 370–379. https://aclanthology.org/P14-1035/

Benjamini, Yoav, and Yosef Hochberg. 1995. “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.” *Journal of the Royal Statistical Society: Series B* 57 (1): 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Underwood, Ted. 2019. *Distant Horizons: Digital Evidence and Literary Change*. University of Chicago Press. https://doi.org/10.7208/chicago/9780226612973.001.0001

Underwood, Ted, David Bamman, and Sabrina Lee. 2018. “The Transformation of Gender in English-Language Fiction.” *Journal of Cultural Analytics*. https://doi.org/10.22148/16.019
