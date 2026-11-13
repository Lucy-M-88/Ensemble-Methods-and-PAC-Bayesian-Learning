
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
INPUT_FILE = RESULTS_DIR / "baseline_results.csv"

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "minority_f1",
    "auc",
]

df = pd.read_csv(INPUT_FILE)

# One score per dataset and model:
# average across the 5 cross-validation folds.
aggregated = (
    df.groupby(["dataset", "model"])[METRICS]
    .mean()
    .reset_index()
)

models = sorted(aggregated["model"].unique())
datasets = sorted(aggregated["dataset"].unique())

print(f"Datasets: {len(datasets)}")
print(f"Models: {len(models)}")

# Pivot tables: rows = datasets, columns = models
tables = {
    metric: aggregated.pivot(
        index="dataset",
        columns="model",
        values=metric,
    )
    for metric in METRICS
}

# Check for missing results
for metric, table in tables.items():
    if table.isna().any().any():
        raise ValueError(f"Missing values for metric: {metric}")


rank_tables = {}

for metric, table in tables.items():
    ranks = table.rank(
        axis=1,
        ascending=False,
        method="average",
    )
    rank_tables[metric] = ranks

    average_ranks = ranks.mean().sort_values()

    print(f"\n=== Average ranks: {metric} ===")
    print(average_ranks.round(3))

    average_ranks.to_csv(
        RESULTS_DIR / f"average_ranks_{metric}.csv"
    )


friedman_results = []

for metric, table in tables.items():
    statistic, p_value = friedmanchisquare(
        *[table[model].values for model in models]
    )

    friedman_results.append({
        "metric": metric,
        "friedman_statistic": statistic,
        "p_value": p_value,
        "significant_0.05": p_value < 0.05,
    })

    print(f"\n=== Friedman: {metric} ===")
    print(f"Statistic = {statistic:.4f}")
    print(f"p-value   = {p_value:.6g}")

friedman_df = pd.DataFrame(friedman_results)
friedman_df.to_csv(
    RESULTS_DIR / "friedman_results.csv",
    index=False,
)

def holm_correction(p_values):
    """Holm-Bonferroni adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(n)

    running_max = 0.0

    for rank, index in enumerate(order):
        corrected = (n - rank) * p_values[index]
        running_max = max(running_max, corrected)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


pairwise_results = []

for metric, table in tables.items():

    # Only run post-hoc tests if Friedman is significant
    friedman_p = friedman_df.loc[
        friedman_df["metric"] == metric, "p_value"
    ].iloc[0]

    if friedman_p >= 0.05:
        print(f"\nNo significant Friedman result for {metric}: "
              "post-hoc tests skipped.")
        continue

    comparisons = []
    raw_p_values = []

    for model_a, model_b in combinations(models, 2):
        scores_a = table[model_a].values
        scores_b = table[model_b].values

        differences = scores_a - scores_b

        if np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(
                scores_a,
                scores_b,
                alternative="two-sided",
            )

        comparisons.append({
            "metric": metric,
            "model_a": model_a,
            "model_b": model_b,
            "mean_a": scores_a.mean(),
            "mean_b": scores_b.mean(),
            "wilcoxon_statistic": statistic,
        })

        raw_p_values.append(p_value)

    adjusted_p_values = holm_correction(raw_p_values)

    for result, raw_p, adjusted_p in zip(
        comparisons, raw_p_values, adjusted_p_values
    ):
        result["p_value"] = raw_p
        result["holm_p_value"] = adjusted_p
        result["significant_0.05"] = adjusted_p < 0.05
        pairwise_results.append(result)

pairwise_df = pd.DataFrame(pairwise_results)

pairwise_df.to_csv(
    RESULTS_DIR / "wilcoxon_holm_results.csv",
    index=False,
)

print("\n=== Pairwise Wilcoxon + Holm ===")
if not pairwise_df.empty:
    print(
        pairwise_df[
            [
                "metric",
                "model_a",
                "model_b",
                "holm_p_value",
                "significant_0.05",
            ]
        ].to_string(index=False)
    )
else:
    print("No post-hoc comparisons were necessary.")

summary = (
    aggregated.groupby("model")[METRICS]
    .mean()
    .sort_values("balanced_accuracy", ascending=False)
)

summary.to_csv(RESULTS_DIR / "mean_performance.csv")

print("\n=== Mean performance ===")
print(summary.round(4))

print("\nAnalysis complete. Files saved in:", RESULTS_DIR)
