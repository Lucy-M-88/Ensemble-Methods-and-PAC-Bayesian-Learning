
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE_DIR))

import benchkit as bk

INPUT = RESULTS_DIR / "baseline_results.csv"

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "minority_f1",
    "auc",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "minority_f1": "Minority F1",
    "auc": "ROC AUC",
}

THRESHOLD = 0.30

df = pd.read_csv(INPUT)

# Average the folds: one result per dataset and model
aggregated = (
    df.groupby(["dataset", "model"])[METRICS]
    .mean()
    .reset_index()
)

models = sorted(aggregated["model"].unique())

# 1. Global performance comparison

summary = aggregated.groupby("model")[METRICS].mean()

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.ravel()

for ax, metric in zip(axes, METRICS):
    values = summary[metric].sort_values(ascending=False)

    ax.bar(values.index, values.values)
    ax.set_title(METRIC_LABELS[metric])
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)

fig.suptitle(
    "Average performance across {aggregated['dataset'].nunique()} datasets",
    fontsize=15,
)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "performance_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# 2. Average ranks

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.ravel()

for ax, metric in zip(axes, METRICS):
    table = aggregated.pivot(
        index="dataset",
        columns="model",
        values=metric,
    )

    ranks = table.rank(
        axis=1,
        ascending=False,
        method="average",
    )

    average_ranks = ranks.mean().sort_values()

    ax.barh(
        average_ranks.index[::-1],
        average_ranks.values[::-1],
    )
    ax.set_title(METRIC_LABELS[metric])
    ax.set_xlabel("Average rank (lower is better)")
    ax.grid(axis="x", alpha=0.25)

n_datasets = aggregated["dataset"].nunique()
fig.suptitle(
    f"Average performance across {n_datasets} datasets",
    fontsize=15,
)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "average_ranks.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# 3. Dataset imbalance

imbalance_rows = []

for dataset in sorted(df["dataset"].unique()):
    X, y = bk.real_dataset(dataset)
    y = np.asarray(y)

    _, counts = np.unique(y, return_counts=True)
    proportion = counts.min() / counts.sum()

    imbalance_rows.append({
        "dataset": dataset,
        "minority_proportion": proportion,
        "imbalanced": proportion < THRESHOLD,
    })

imbalance = pd.DataFrame(imbalance_rows)
imbalance = imbalance.sort_values("minority_proportion")

fig, ax = plt.subplots(figsize=(12, 7))

bar_colors = [
    "tab:red" if value < THRESHOLD else "tab:blue"
    for value in imbalance["minority_proportion"]
]

ax.barh(
    imbalance["dataset"],
    imbalance["minority_proportion"],
    color=bar_colors,
)

ax.axvline(
    THRESHOLD,
    linestyle="--",
    color="black",
    label="30% threshold",
)

ax.set_xlabel("Minority class proportion")
ax.set_title("Class imbalance across datasets")
ax.legend()
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "dataset_imbalance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# 4. Performance by imbalance group

results = aggregated.merge(
    imbalance[["dataset", "imbalanced"]],
    on="dataset",
)

results["group"] = np.where(
    results["imbalanced"],
    "Imbalanced (<30%)",
    "Other (>=30%)",
)

group_summary = (
    results.groupby(["group", "model"])[METRICS]
    .mean()
)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.ravel()

for ax, metric in zip(axes, METRICS):
    pivot = group_summary[metric].unstack("group")
    pivot = pivot.reindex(models)

    pivot.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(METRIC_LABELS[metric])
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

fig.suptitle(
    "Performance by class imbalance group",
    fontsize=15,
)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "imbalance_performance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("Figures successfully generated in:", FIGURES_DIR)
