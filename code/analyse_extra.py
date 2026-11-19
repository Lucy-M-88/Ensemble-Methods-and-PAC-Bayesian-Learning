"""All complementary analyses for Part 1, in a single file.

Groups
------
  --bvb        Bagging vs AdaBoost, dataset by dataset (balanced_accuracy + minority_f1).
  --stacking   Stacking vs its internal members (tree/forest/svc), with fit-time overhead.
               WARNING: this group re-trains 3 models on every dataset (~20-30 min).
  --cd         Demšar (2006) Critical Difference diagrams, Nemenyi post-hoc, alpha=0.05.

Usage
-----
  python analyse_extra.py --all                  # run all three groups
  python analyse_extra.py --bvb                  # just one
  python analyse_extra.py --bvb --cd             # several
  python analyse_extra.py --all --no-stacking    # skip the slow group

Requires
--------
  results/baseline_results.csv   (produced by benchmark_part1.py)
  results/dataset_imbalance.csv  (produced by analyse_imbalance.py)
  results/tables/dataset_summary.csv  (produced by dataset_analysis.py)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

# paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchkit as bk

RANDOM_STATE = 42
N_SPLITS = 5
SVM_MAX_SAMPLES = 8000  # subsample for the SVM member on very large datasets

METRICS = ["accuracy", "balanced_accuracy", "minority_f1", "auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "minority_f1": "Minority F1",
    "auc": "ROC AUC",
}



# Shared helpers

def _load_baseline() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "baseline_results.csv")
    return (
        df.groupby(["dataset", "model"])[METRICS + ["fit_time"]]
        .mean()
        .reset_index()
    )


def _pivot(agg: pd.DataFrame, metric: str) -> pd.DataFrame:
    return agg.pivot(index="dataset", columns="model", values=metric)



# 1. Bagging vs AdaBoost

def analyse_bagging_vs_boosting() -> None:
    print("\n=== [bvb] Bagging vs AdaBoost ===")

    agg = _load_baseline()
    piv_bal = _pivot(agg, "balanced_accuracy")
    piv_f1 = _pivot(agg, "minority_f1")
    piv_acc = _pivot(agg, "accuracy")
    piv_auc = _pivot(agg, "auc")

    comp = pd.DataFrame({
        "Bagging_bal": piv_bal["Bagging"],
        "AdaBoost_bal": piv_bal["AdaBoost"],
        "Bagging_f1": piv_f1["Bagging"],
        "AdaBoost_f1": piv_f1["AdaBoost"],
        "Bagging_acc": piv_acc["Bagging"],
        "AdaBoost_acc": piv_acc["AdaBoost"],
        "Bagging_auc": piv_auc["Bagging"],
        "AdaBoost_auc": piv_auc["AdaBoost"],
    })
    comp["diff_bal"] = comp["AdaBoost_bal"] - comp["Bagging_bal"]
    comp["diff_f1"] = comp["AdaBoost_f1"] - comp["Bagging_f1"]
    comp["winner_bal"] = np.where(comp["diff_bal"] > 0, "AdaBoost", "Bagging")
    comp["winner_f1"] = np.where(comp["diff_f1"] > 0, "AdaBoost", "Bagging")

    # context (m, d, minority proportion)
    imb_path = RESULTS / "dataset_imbalance.csv"
    sum_path = RESULTS / "tables" / "dataset_summary.csv"
    if imb_path.exists():
        imb = pd.read_csv(imb_path).set_index("dataset")
        comp = comp.join(imb[["minority_proportion", "group"]])
    if sum_path.exists():
        summary = pd.read_csv(sum_path).set_index("Dataset")
        comp = comp.join(summary[["Samples (m)", "Features (d)"]])

    comp = comp.sort_values("diff_bal")
    comp.to_csv(RESULTS / "bagging_vs_boosting.csv")

    # console
    print(f"AdaBoost wins (balanced_accuracy): {(comp['diff_bal'] > 0).sum()}/{len(comp)}")
    print(f"Bagging  wins (balanced_accuracy): {(comp['diff_bal'] < 0).sum()}/{len(comp)}")
    print(f"Mean diff (AdaBoost − Bagging)   : {comp['diff_bal'].mean():+.4f}")
    print(f"AdaBoost wins (minority_f1)      : {(comp['diff_f1'] > 0).sum()}/{len(comp)}")

    # figure 1 — side-by-side
    y = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(y - 0.2, comp["Bagging_bal"], height=0.4, label="Bagging", color="tab:blue")
    ax.barh(y + 0.2, comp["AdaBoost_bal"], height=0.4, label="AdaBoost", color="tab:orange")
    ax.set_yticks(y); ax.set_yticklabels(comp.index)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("Bagging vs AdaBoost — balanced accuracy per dataset")
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "bagging_vs_boosting_balanced_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # figure 2 — signed diff
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["tab:green" if d > 0 else "tab:red" for d in comp["diff_bal"]]
    ax.barh(y, comp["diff_bal"], color=colors)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(comp.index)
    ax.set_xlabel("AdaBoost − Bagging  (balanced accuracy)")
    ax.set_title("Signed difference per dataset  (green = AdaBoost wins)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "bagging_vs_boosting_diff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[bvb] saved bagging_vs_boosting.csv + 2 figures")



# 2. Stacking vs its internal members

def _make_members():
    return {
        "member_tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE,
        ),
        "member_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "member_svm": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(random_state=RANDOM_STATE),
                method="sigmoid", cv=3,
            ),
        ),
    }


def _eval_member(model, X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    acc, bal, auc, ft = [], [], [], []
    for tr, te in cv.split(X, y):
        t0 = time.perf_counter()
        m = clone(model); m.fit(X[tr], y[tr])
        ft.append(time.perf_counter() - t0)
        pred = m.predict(X[te]); proba = m.predict_proba(X[te])
        acc.append(accuracy_score(y[te], pred))
        bal.append(balanced_accuracy_score(y[te], pred))
        try:
            auc.append(roc_auc_score(y[te], proba[:, 1]))
        except ValueError:
            auc.append(np.nan)
    return float(np.mean(acc)), float(np.mean(bal)), float(np.nanmean(auc)), float(np.mean(ft))


def analyse_stacking_members() -> None:
    print("\n=== [stacking] Stacking vs its internal members ===")
    agg = _load_baseline()
    stack = agg[agg["model"] == "Stacking"].set_index("dataset")
    members = _make_members()

    rows = []
    for name in sorted(stack.index):
        try:
            X, y = bk.real_dataset(name)
            X = np.asarray(X, dtype=float); y = np.asarray(y)
        except Exception as exc:
            print(f"[SKIP] {name}: {exc}"); continue
        print(f"[{name}]")

        row = {
            "dataset": name,
            "stacking_bal": float(stack.loc[name, "balanced_accuracy"]),
            "stacking_auc": float(stack.loc[name, "auc"]),
            "stacking_fit": float(stack.loc[name, "fit_time"]),
        }
        for mname, model in members.items():
            is_svm = mname == "member_svm"
            if is_svm and len(X) > SVM_MAX_SAMPLES:
                rng = np.random.default_rng(RANDOM_STATE)
                idx = rng.choice(len(X), SVM_MAX_SAMPLES, replace=False)
                Xm, ym = X[idx], y[idx]
            else:
                Xm, ym = X, y
            _, bal, auc, ft = _eval_member(model, Xm, ym)
            row[f"{mname}_bal"] = bal
            row[f"{mname}_auc"] = auc
            row[f"{mname}_fit"] = ft

        bals = {n: row[f"{n}_bal"] for n in members}
        best = max(bals, key=bals.get)
        row["best_member"] = best
        row["best_member_bal"] = bals[best]
        row["stacking_gain"] = row["stacking_bal"] - row["best_member_bal"]
        row["stacking_overhead"] = row["stacking_fit"] / row[f"{best}_fit"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "stacking_vs_members.csv", index=False)

    print(f"Stacking wins: {(df['stacking_gain'] > 0).sum()}/{len(df)} datasets")
    print(f"Mean gain over best member: {df['stacking_gain'].mean():+.4f}")
    print(f"Mean overhead              : {df['stacking_overhead'].mean():.1f}x")

    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(df))
    colors = ["tab:green" if g > 0 else "tab:red" for g in df["stacking_gain"]]
    ax.barh(y, df["stacking_gain"], color=colors)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(df["dataset"])
    ax.set_xlabel("Stacking − best member  (balanced accuracy)")
    ax.set_title("Does stacking beat the best of its members?")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "stacking_vs_members.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[stacking] saved stacking_vs_members.csv + 1 figure")



# 3. CD diagrams (Demšar 2006, Nemenyi alpha=0.05)

Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
              7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def _cd_diagram(metric: str, agg: pd.DataFrame) -> None:
    pivot = _pivot(agg, metric)
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    avg = ranks.mean().sort_values()
    k, N = len(avg), len(ranks)
    q = Q_ALPHA_05[k]
    CD = q * np.sqrt(k * (k + 1) / (6 * N))

    fig, ax = plt.subplots(figsize=(11, 0.6 * k + 2))
    y_top = np.arange(k)[::-1] * 0.5
    for i, (model, rank) in enumerate(avg.items()):
        y = y_top[i]
        ax.hlines(y, 1, k, color="grey", linewidth=0.4, alpha=0.35)
        ax.plot(rank, y, "o", color="black", markersize=7)
        ax.text(k + 0.5, y, f"{model}  ({rank:.2f})",
                va="center", ha="left", fontsize=9)

    items = list(avg.items())
    clique_y = -0.6
    for i in range(k):
        j_max = i
        for j in range(i + 1, k):
            if items[j][1] - items[i][1] < CD:
                j_max = j
            else:
                break
        if j_max > i:
            ax.plot([items[i][1], items[j_max][1]], [clique_y, clique_y],
                    color="tab:blue", linewidth=3.5, solid_capstyle="butt")
            clique_y -= 0.35

    y_cd = clique_y - 0.5
    ax.plot([1, 1 + CD], [y_cd, y_cd], color="tab:red", linewidth=2.5)
    ax.plot([1, 1], [y_cd - 0.15, y_cd + 0.15], color="tab:red")
    ax.plot([1 + CD, 1 + CD], [y_cd - 0.15, y_cd + 0.15], color="tab:red")
    ax.text(1 + CD / 2, y_cd - 0.35, f"CD = {CD:.2f}",
            ha="center", va="top", fontsize=9, color="tab:red")

    ax.set_xlim(0.5, k + 3.5)
    ax.set_ylim(min(clique_y, y_cd) - 0.6, max(y_top) + 0.5)
    ax.set_xticks(range(1, k + 1)); ax.set_yticks([])
    ax.invert_xaxis()
    ax.set_xlabel(f"Average rank — {METRIC_LABELS[metric]} (lower is better)")
    ax.set_title(f"Critical Difference diagram — {METRIC_LABELS[metric]}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / f"cd_diagram_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[cd] {metric}: CD={CD:.3f}, k={k}, N={N}")


def plot_cd_diagrams() -> None:
    print("\n=== [cd] Demšar CD diagrams ===")
    agg = _load_baseline()
    for metric in METRICS:
        _cd_diagram(metric, agg)
    print("[cd] saved 4 figures in results/figures/")



# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bvb", action="store_true",
                        help="Bagging vs AdaBoost per dataset")
    parser.add_argument("--stacking", action="store_true",
                        help="Stacking vs members (slow, ~20-30 min)")
    parser.add_argument("--cd", action="store_true",
                        help="Demšar CD diagrams")
    parser.add_argument("--all", action="store_true",
                        help="Run every group (including --stacking)")
    parser.add_argument("--no-stacking", action="store_true",
                        help="With --all, skip the slow --stacking group")
    args = parser.parse_args()

    run_bvb = args.bvb or args.all
    run_stack = (args.stacking or args.all) and not args.no_stacking
    run_cd = args.cd or args.all

    if not (run_bvb or run_stack or run_cd):
        parser.error("nothing to do — pass --bvb, --stacking, --cd, or --all")

    if run_bvb:
        analyse_bagging_vs_boosting()
    if run_stack:
        analyse_stacking_members()
    if run_cd:
        plot_cd_diagrams()

    print("\nDone.")


if __name__ == "__main__":
    main()