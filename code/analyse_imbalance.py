
from pathlib import Path

import numpy as np
import pandas as pd
import benchkit as bk

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
INPUT = RESULTS / "baseline_results.csv"

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "minority_f1",
    "auc",
]

THRESHOLD = 0.30

# Load cross-validation results and average folds
df = pd.read_csv(INPUT)

aggregated = (
    df.groupby(["dataset", "model"])[METRICS]
    .mean()
    .reset_index()
)

# Compute minority-class proportion for every dataset
imbalance_rows = []

for dataset in sorted(df["dataset"].unique()):
    X, y = bk.real_dataset(dataset)
    y = np.asarray(y)

    _, counts = np.unique(y, return_counts=True)
    minority_proportion = counts.min() / counts.sum()

    imbalance_rows.append({
        "dataset": dataset,
        "minority_proportion": minority_proportion,
        "is_imbalanced": minority_proportion < THRESHOLD,
    })

imbalance = pd.DataFrame(imbalance_rows)

imbalance["group"] = np.where(
    imbalance["is_imbalanced"],
    "Imbalanced (<30%)",
    "Other (>=30%)",
)

imbalance.to_csv(
    RESULTS / "dataset_imbalance.csv",
    index=False,
)

# Join imbalance information to model results
results = aggregated.merge(
    imbalance[["dataset", "minority_proportion", "group"]],
    on="dataset",
)

print("\n=== Dataset imbalance ===")
print(
    imbalance.sort_values("minority_proportion").to_string(
        index=False
    )
)

print("\nNumber of datasets per group:")
print(imbalance["group"].value_counts())

# Mean performance by group and model
group_summary = (
    results.groupby(["group", "model"])[METRICS]
    .mean()
    .round(4)
)

print("\n=== Mean performance by imbalance group ===")
print(group_summary)

group_summary.to_csv(
    RESULTS / "imbalance_performance.csv"
)

# Average ranks within each group
for group in results["group"].unique():
    subset = results[results["group"] == group]

    print(f"\n=== Average ranks: {group} ===")

    for metric in METRICS:
        table = subset.pivot(
            index="dataset",
            columns="model",
            values=metric,
        )

        ranks = table.rank(
            axis=1,
            ascending=False,
            method="average",
        )

        print(f"\n{metric}:")
        print(ranks.mean().sort_values().round(3))

        ranks.mean().to_csv(
            RESULTS / f"ranks_{group.split()[0]}_{metric}.csv"
        )

print("\nAnalysis saved in:", RESULTS)
