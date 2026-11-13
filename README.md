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

### Run the benchmark

From the project root:

```bash
python code/benchmark_part1.py
```

The benchmark results are saved to:

```text
results/baseline_results.csv
```

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

To reproduce the Part 1 experiments, execute the following commands from the project root:

```bash
python code/benchmark_part1.py
python code/analyse_results.py
python code/analyse_imbalance.py
python code/make_figures.py
```

The benchmark uses fixed random seeds and stratified cross-validation to ensure consistent data splits and reproducible results.

All preprocessing steps that require fitting are performed within the corresponding training folds to avoid data leakage.

The generated CSV files contain the numerical results, while the figures are saved in `results/figures/`.

For exact dependency versions, the project can also provide a `requirements-lock.txt` file generated from the working Python environment.

---

## 6. Results

The experimental results show that ensemble methods generally outperform the standalone Decision Tree baseline.

Across the evaluated datasets:

- Stacking achieves the highest average accuracy and ROC AUC.
- HistGradientBoosting achieves the highest average balanced accuracy and minority-class F1-score.
- Random Forest remains competitive across all metrics.
- HistGradientBoosting obtains the best average ranks for balanced accuracy and minority-class F1 on the selected imbalanced datasets.

Statistical tests are used to assess whether observed differences between classifiers are significant.

Detailed results, figures and interpretations are provided in the project report.

---

## 7. Authors

Lucy MARTIN et Clément CORREIA--PELTIER - 22/11/2026
Master MALIA — Ensemble Methods and PAC-Bayesian Learning project.

