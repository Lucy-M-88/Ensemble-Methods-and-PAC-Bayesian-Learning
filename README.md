# Ensemble Methods and PAC-Bayesian Learning

**Master MALIA — Machine Learning Project**

This project studies ensemble learning methods and PAC-Bayesian generalization bounds. It is divided into two parts:

- **Part 1 — Ensemble Methods:** implementation and experimental comparison of Bagging, Discrete AdaBoost and Stacking.
- **Part 2 — PAC-Bayesian Learning:** study of PAC-Bayesian bounds using distributions over decision stumps and linear classifiers.

---

## 1. Project Structure

```text
Ensemble-Methods-and-PAC-Bayesian-Learning/
│
├── code/
│   ├── benchkit.py
│   ├── ensemble_methods.py
│   ├── benchmark_part1.py
│   ├── compare_with_sklearn.py
│   ├── analyse_results.py
│   ├── analyse_imbalance.py
│   ├── make_figures.py
│   ├── dataset_analysis.py
│   └── pacbayes_tools.py
│
├── data/
│   └── ...
│
├── results/
│   ├── baseline_results.csv
│   ├── failed_datasets.csv
│   ├── sklearn_comparison.csv
│   ├── mean_performance.csv
│   ├── friedman_results.csv
│   ├── wilcoxon_holm_results.csv
│   └── figures/
│
├── requirements.txt
└── README.md
```

*Note: the structure above describes the main files. Additional files may be present in the repository.*

---

## 2. Requirements and Installation

Python 3.10 or later is recommended.

Clone or download the project repository, then navigate to its root directory.

Create and activate a virtual environment:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Part 1 — Ensemble Methods

### Implemented methods

Three ensemble learning methods are implemented:

- **Bagging:** bootstrap aggregation of decision trees, including out-of-bag (OOB) evaluation.
- **Discrete AdaBoost:** sequential combination of weak decision-tree learners using sample weights.
- **Stacking:** combination of heterogeneous base classifiers using out-of-fold predictions and a logistic regression meta-classifier.

These methods are compared with the following reference classifiers:

- Decision Tree
- Logistic Regression
- k-Nearest Neighbours
- Random Forest
- Gradient Boosting
- HistGradientBoosting

### Experimental protocol

The experiments are conducted on 22 binary classification datasets.

The evaluation protocol uses:

- Stratified 5-fold cross-validation.
- Fixed random seed (`42`) for reproducibility.
- Consistent data splits across classifiers.
- Preprocessing fitted on training folds only, where applicable.

The following metrics are reported:

| Metric | Description |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Balanced Accuracy | Average recall across classes |
| Minority-class F1 | F1-score for the less frequent class |
| ROC AUC | Area under the ROC curve |

In addition, the following computational cost metrics are recorded per fold, in order to address the *"at what cost?"* question:

| Metric | Description |
|---|---|
| `fit_time` | Wall-clock time of `fit` on the training fold (seconds) |
| `predict_time` | Wall-clock time of `predict` on the test fold (seconds) |
| `predict_proba_time` | Wall-clock time of `predict_proba` on the test fold (seconds) |

### Run the benchmark

From the project root:

```bash
python code/benchmark_part1.py
```

The benchmark results are saved to:

```text
results/baseline_results.csv
```

If any dataset fails to load or be evaluated, the script does **not** crash: the failing datasets and their error messages are written to:

```text
results/failed_datasets.csv
```

### Comparison with scikit-learn

To check that the three hand-written implementations (`BaggingOOBClassifier`, `AdaBoostDiscreteClassifier`, `StackingClassifierCustom`) match their scikit-learn counterparts, run:

```bash
python code/compare_with_sklearn.py
```

The script trains, on a representative subset of 8 datasets and using the same 5-fold stratified splits, both the hand-written model and the equivalent scikit-learn model with strictly identical hyperparameters:

- `BaggingOOBClassifier` ↔ `sklearn.ensemble.BaggingClassifier` (with `oob_score=True`)
- `AdaBoostDiscreteClassifier` ↔ `sklearn.ensemble.AdaBoostClassifier(algorithm="SAMME")`
- `StackingClassifierCustom` ↔ `sklearn.ensemble.StackingClassifier`

The per-dataset scores and the absolute differences are saved to:

```text
results/sklearn_comparison.csv
```

and a summary (mean and max absolute difference on `accuracy`, `balanced_accuracy` and `auc`) is printed to the console.

### Statistical analysis

To compute average ranks, the Friedman test and pairwise Wilcoxon signed-rank tests with Holm correction, run:

```bash
python code/analyse_results.py
```

The generated files include:

- `average_ranks_accuracy.csv`
- `average_ranks_balanced_accuracy.csv`
- `average_ranks_minority_f1.csv`
- `average_ranks_auc.csv`
- `friedman_results.csv`
- `wilcoxon_holm_results.csv`
- `mean_performance.csv`

### Class imbalance analysis

To analyze performance according to class imbalance, run:

```bash
python code/analyse_imbalance.py
```

Datasets are divided into two groups according to the proportion of the minority class:

- **Imbalanced:** minority-class proportion strictly below 30%.
- **Other:** minority-class proportion greater than or equal to 30%.

The results are saved in:

- `dataset_imbalance.csv`
- `imbalance_performance.csv`
- Rank comparison CSV files for each group.

### Dataset descriptive table

To build the descriptive table of the datasets (number of examples, dimension, number of classes, minority proportion, imbalance flag), run:

```bash
python code/dataset_analysis.py
```

Results are saved to:

```text
results/tables/dataset_summary.csv
```

### Generate figures

To generate the figures used in the report, run:

```bash
python code/make_figures.py
```

The figures are saved in:

```text
results/figures/
```

They include:

- Overall model performance comparison.
- Average model ranks.
- Minority-class proportions across datasets.
- Performance comparison by imbalance group.

### Complementary analyses (bagging-vs-boosting, stacking-vs-members, CD diagrams)

```bash
python code/analyse_extra.py --bvb        # Bagging vs AdaBoost, per dataset
python code/analyse_extra.py --cd         # Demšar CD diagrams (4 metrics)
python code/analyse_extra.py --stacking   # Stacking vs its members (~20-30 min)
python code/analyse_extra.py --all --no-stacking   # everything except the slow group

---

## 4. Part 2 — PAC-Bayesian Learning

The second part investigates PAC-Bayesian generalization bounds.

The experiments consider:

- Categorical distributions over decision stumps.
- Gaussian distributions over linear classifiers.
- Prior and posterior distributions.
- Empirical risks and generalization certificates.

The provided `pacbayes_tools.py` module contains tools used to compute and evaluate PAC-Bayesian bounds.

The corresponding experiments and results are documented in the project report and associated code.

---

## 5. Reproducibility

To reproduce the Part 1 experiments, execute the following commands from the project root, in this order:

```bash
python code/benchmark_part1.py          # produces baseline_results.csv (+ failed_datasets.csv)
python code/compare_with_sklearn.py     # produces sklearn_comparison.csv
python code/analyse_results.py          # reads baseline_results.csv
python code/analyse_imbalance.py        # reads baseline_results.csv
python code/dataset_analysis.py         # produces dataset_summary.csv
python code/make_figures.py             # reads baseline_results.csv
python code/analyse_extra.py --all --no-stacking
```

The ordering matters: `analyse_results.py`, `analyse_imbalance.py` and `make_figures.py` all read `results/baseline_results.csv` produced by `benchmark_part1.py`, so the benchmark must be run first.

The benchmark uses fixed random seeds and stratified cross-validation to ensure consistent data splits and reproducible results.

All preprocessing steps that require fitting are performed within the corresponding training folds to avoid data leakage.

The generated CSV files contain the numerical results, while the figures are saved in `results/figures/`.

For exact dependency versions, the project can also provide a `requirements-lock.txt` file generated from the working Python environment.

---

## 6. Results

The experimental results show that ensemble methods generally outperform both
the standalone Decision Tree baseline and the non-tree baselines (Logistic
Regression, k-NN).

### Global ranking (mean over 22 datasets)

| Model | Accuracy | Balanced Acc. | Minority F1 | ROC AUC | Fit time (s) |
|---|---|---|---|---|---|
| Stacking              | **0.9029** | 0.8393 | 0.7709 | **0.9359** | 4.70 |
| RandomForest          | 0.9018 | 0.8378 | 0.7716 | 0.9341 | 0.18 |
| HistGradientBoosting  | 0.8986 | **0.8436** | **0.7806** | 0.9291 | 0.14 |
| AdaBoost              | 0.8970 | 0.8328 | 0.7551 | 0.9241 | 0.20 |
| GradientBoosting      | 0.8956 | 0.8382 | 0.7715 | 0.9283 | 0.78 |
| Bagging               | 0.8893 | 0.8183 | 0.7364 | 0.9224 | 1.28 |
| LogisticRegression    | 0.8792 | 0.7949 | 0.6961 | 0.8969 | 0.006 |
| KNN                   | 0.8652 | 0.8020 | 0.7277 | 0.8879 | 0.002 |
| DecisionTree          | 0.8602 | 0.7968 | 0.7041 | 0.8530 | 0.010 |

### Cost–performance trade-off

Stacking achieves the best raw scores, but at a cost ~27× higher than
RandomForest and ~750× higher than LogisticRegression (fit time), which must
be taken into account when choosing a model in practice. HistGradientBoosting
is the best compromise: best balanced_accuracy and minority F1, and one of the
lowest fit times among ensembles.

### Effect of class imbalance

Splitting the datasets into *Imbalanced* (< 30% minority, 8 datasets) and
*Other* (≥ 30%, 14 datasets) reveals a reordering:

- On *balanced* datasets, **Stacking** dominates all metrics (avg. rank ≈ 2.8–3.2).
- On *imbalanced* datasets, **HistGradientBoosting** and **RandomForest** take
  the lead on balanced_accuracy / minority F1, and Stacking drops to 4th place
  on accuracy (avg. rank 4.06). This suggests that the stacking meta-model,
  trained on OOF probabilities biased toward the majority class, does not
  fully transfer to imbalanced regimes.

### Statistical tests

Friedman tests are highly significant for all four metrics (p < 1e-6).
Pairwise Wilcoxon with Holm correction shows that:

- All ensembles significantly beat the Decision Tree.
- No ensemble is **statistically** distinguishable from the best ensemble on
  accuracy (Stacking vs RF: Holm p = 1.00; vs HistGB: p = 1.00).
- Stacking significantly outperforms AdaBoost on AUC (Holm p = 0.001) and
  Bagging on accuracy / balanced_accuracy / minority_f1 (Holm p < 0.01).

### scikit-learn agreement

The hand-written ensembles reproduce their scikit-learn counterparts with a
maximum absolute difference of 0.019 (accuracy), 0.021 (balanced accuracy) and
0.006 (AUC) across the 8 compared datasets — confirming a correct
implementation of Bagging-OOB, Discrete AdaBoost and Stacking.

---

## 7. Authors

Lucy MARTIN et Clément CORREIA--PELTIER - 22/11/2026
Master MALIA — Ensemble Methods and PAC-Bayesian Learning project.