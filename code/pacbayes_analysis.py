"""PAC-Bayesian analysis (Part 2), all five questions in a single file.

Sub-commands
  --q1   Bound curves: A, B and Seeger as functions of m and KL; price of the grid.
  --q2   A vs B on wdbc + 5 other datasets x 5 splits; lambda-trace on wdbc.
  --q3   Impact of the prior (uniform vs learned; N(0,I) vs N(w0,I)).
  --q4   Family comparison (stumps vs linear) + extra posteriors (Dirac, top-k).
  --q5   Classical ensembles vs bound minimization on 6 datasets x 5 splits.

Every sub-command writes one or two CSVs into results/ and one or two PNGs into
results/figures/. All primitives come from the provided pacbayes_tools.py.

Usage
  python code/pacbayes_analysis.py --all
  python code/pacbayes_analysis.py --q1 --q2
  python code/pacbayes_analysis.py --q5          # just the classical-vs-bound comparison
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, avoids Tkinter crash on Windows
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchkit as bk
import pacbayes_tools as pb
from pacbayes_tools import (
    bound_linear, bound_mcallester, bound_seeger,
    run_categorical, run_gaussian,
    learn_linear_categorical,
    gibbs_risk, vote_risk, kl_categorical,
    GRID, DELTA,
)
from ensemble_methods import (
    AdaBoostDiscreteClassifier,
    StackingClassifierCustom,
)

# paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# 6 datasets (the minimum required by the subject is 5 for Q2 and 6 for Q5)
DATASETS = ["wdbc", "heart", "iono", "sonar", "vehicle", "pima"]

# Number of random splits (train/test + prior/S) per dataset
N_SEEDS = 5

# Colors
C_A = "tab:blue"
C_B = "tab:orange"
C_S = "tab:green"


# Q1 — Bound curves
def q1_bounds_curves() -> None:
    print("\n=== [q1] Bound curves (A, B, Seeger) ===")
    emp = 0.10
    kls = [1.0, 5.0, 20.0, 50.0]

    # (a) vs m, for several KL
    m_grid = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, kl_val in zip(axes.ravel(), kls):
        a = [bound_linear(emp, kl_val, int(m), lam=256.0) for m in m_grid]
        b = [bound_mcallester(emp, kl_val, int(m)) for m in m_grid]
        s = [bound_seeger(emp, kl_val, int(m)) for m in m_grid]
        ax.plot(m_grid, a, "o-", color=C_A, label="A (λ=256)")
        ax.plot(m_grid, b, "s-", color=C_B, label="B (McAllester)")
        ax.plot(m_grid, s, "^-", color=C_S, label="Seeger")
        ax.set_xscale("log")
        ax.set_xlabel("m")
        ax.set_ylabel("bound value")
        ax.set_title(f"KL = {kl_val}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Bounds vs m (emp = {emp})", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "pacbayes_q1_vs_m.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) vs KL, for several m
    kl_grid = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ms = [100, 500, 2000, 10000]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, m_val in zip(axes.ravel(), ms):
        a = [bound_linear(emp, float(k), m_val, lam=256.0) for k in kl_grid]
        b = [bound_mcallester(emp, float(k), m_val) for k in kl_grid]
        s = [bound_seeger(emp, float(k), m_val) for k in kl_grid]
        ax.plot(kl_grid, a, "o-", color=C_A, label="A (λ=256)")
        ax.plot(kl_grid, b, "s-", color=C_B, label="B")
        ax.plot(kl_grid, s, "^-", color=C_S, label="Seeger")
        ax.set_xscale("log")
        ax.set_xlabel("KL")
        ax.set_ylabel("bound value")
        ax.set_title(f"m = {m_val}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Bounds vs KL (emp = {emp})", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "pacbayes_q1_vs_kl.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (c) bound A vs λ, and the price of the grid
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lams = np.array(GRID, dtype=float)
    for kl_val in kls:
        a = [bound_linear(emp, kl_val, 1000, lam=float(l)) for l in lams]
        axes[0].plot(lams, a, "o-", label=f"KL={kl_val}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("λ")
    axes[0].set_ylabel("bound A")
    axes[0].set_title(f"Bound A vs λ (emp={emp}, m=1000)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    grid_cost = np.log(len(GRID)) / lams
    axes[1].plot(lams, grid_cost, "o-", color="tab:red")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("log(|Λ|) / λ   (grid penalty)")
    axes[1].set_title(f"Price of the grid (|Λ|={len(GRID)})")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "pacbayes_q1_vs_lambda.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[q1] saved 3 figures")
    print(f"  bound_linear(0.10, 5, 1000, 256) = {bound_linear(0.10, 5, 1000, 256.0):.4f}")
    print(f"  bound_mcallester(0.10, 5, 1000)  = {bound_mcallester(0.10, 5, 1000):.4f}")
    print(f"  bound_seeger(0.10, 5, 1000)       = {bound_seeger(0.10, 5, 1000):.4f}")


# Q2 — Compare A vs B (multi-dataset + lambda-trace)

def _flatten_run(run_dict, dataset, seed, setting):
    rows = []
    for bound_name, r in run_dict.items():
        rows.append({
            "dataset": dataset, "seed": seed,
            "setting": setting, "bound_name": bound_name,
            **{k: v for k, v in r.items() if k != "Q"},
        })
    return rows


def q2_compare() -> None:
    print("\n=== [q2] A vs B across datasets and splits ===")
    rows, trace_rows = [], []

    for dataset in DATASETS:
        X, y = bk.real_dataset(dataset)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        for seed in range(N_SEEDS):
            rows += _flatten_run(run_categorical(X, y, seed=seed),
                                 dataset, seed, "categorical")
            rows += _flatten_run(run_gaussian(X, y, seed=seed),
                                 dataset, seed, "gaussian")

            # lambda-trace on wdbc, categorical, uniform prior
            if dataset == "wdbc":
                Xp, yp, Xs, ys, Xte, yte = pb.split(X, y, seed)
                H = pb.make_stumps(Xp)
                V, Vte = pb.stump_votes(H, Xs), pb.stump_votes(H, Xte)
                P = np.full(len(H), 1 / len(H))
                L, m = pb.voter_risks(V, ys), len(ys)
                _, lam_rows = learn_linear_categorical(P, L, m)
                for r in lam_rows:
                    Q = r["Q"]
                    trace_rows.append({
                        "seed": seed, "lam": r["lam"],
                        "emp": r["emp"], "KL": r["KL"],
                        "bound_A": r["bound"],
                        "seeger": r["seeger"],
                        "test_gibbs": gibbs_risk(Q, Vte, yte),
                        "test_vote": vote_risk(Q, Vte, yte),
                        "neff": 1.0 / np.sum(Q ** 2),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "pacbayes_q2.csv", index=False)
    trace_df = pd.DataFrame(trace_rows)
    trace_df.to_csv(RESULTS / "pacbayes_q2_lambda_trace.csv", index=False)

    # figure (a): bound vs test_gibbs, A and B
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, setting in zip(axes, ["categorical", "gaussian"]):
        sub = df[df["setting"] == setting]
        for bn, marker, color in [("A (linear)", "o", C_A),
                                  ("B (McAllester)", "s", C_B)]:
            s = sub[sub["bound_name"] == bn]
            ax.scatter(s["bound"], s["test_gibbs"],
                       marker=marker, color=color, label=bn, alpha=0.7)
        lim = float(max(sub["bound"].max(), sub["test_gibbs"].max())) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4)
        ax.set_xlabel("Bound (certificate)")
        ax.set_ylabel("Test Gibbs risk")
        ax.set_title(setting)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Q2 — Certificate vs test Gibbs risk (A vs B)")
    fig.tight_layout()
    fig.savefig(FIGURES / "pacbayes_q2_A_vs_B.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # figure (b): lambda-trace on wdbc
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for seed in range(N_SEEDS):
        sub = trace_df[trace_df["seed"] == seed]
        axes[0].plot(sub["lam"], sub["bound_A"], alpha=0.6,
                     label="bound A" if seed == 0 else None)
        axes[1].plot(sub["lam"], sub["KL"], alpha=0.6,
                     label="KL(Q||P)" if seed == 0 else None)
        axes[2].plot(sub["lam"], sub["test_vote"], alpha=0.6,
                     label="test vote risk" if seed == 0 else None)
    for ax, ylabel in zip(axes, ["bound A", "KL(Q||P)", "test vote risk"]):
        ax.set_xscale("log")
        ax.set_xlabel("λ")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle("Q2 — λ-trace on wdbc (categorical, uniform prior)")
    fig.tight_layout()
    fig.savefig(FIGURES / "pacbayes_q2_lambda_trace.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(df.groupby(["setting", "bound_name"])[
        ["bound", "seeger", "KL", "test_gibbs", "test_vote"]].mean().round(4))
    print("[q2] saved 2 CSVs + 2 figures")


# Q3 — Impact of the prior

def q3_prior() -> None:
    print("\n=== [q3] Impact of the prior ===")
    rows = []

    for dataset in DATASETS:
        X, y = bk.real_dataset(dataset)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        for seed in range(N_SEEDS):
            for prior in ["uniform", "learned"]:
                out = run_categorical(X, y, seed=seed, prior=prior)
                for bn, r in out.items():
                    rows.append({
                        "dataset": dataset, "seed": seed,
                        "setting": "categorical", "prior": prior,
                        "bound_name": bn,
                        **{k: v for k, v in r.items() if k != "Q"},
                    })
            for prior in ["zero", "learned"]:
                out = run_gaussian(X, y, seed=seed, prior=prior)
                for bn, r in out.items():
                    rows.append({
                        "dataset": dataset, "seed": seed,
                        "setting": "gaussian", "prior": prior,
                        "bound_name": bn,
                        **{k: v for k, v in r.items() if k != "Q"},
                    })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "pacbayes_q3.csv", index=False)

    # Figure: KL, bound, test_gibbs, test_vote per (prior, bound_name)
    metrics = ["KL", "bound", "test_gibbs", "test_vote"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for i, setting in enumerate(["categorical", "gaussian"]):
        sub = df[df["setting"] == setting]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            agg = sub.groupby(["prior", "bound_name"])[metric].mean().unstack()
            agg.plot(kind="bar", ax=ax)
            ax.set_title(f"{setting} — {metric}")
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Q3 — Impact of the prior")
    fig.tight_layout()
    fig.savefig(FIGURES / "pacbayes_q3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(df.groupby(["setting", "prior", "bound_name"])[
        ["KL", "bound", "seeger", "test_gibbs", "test_vote"]
    ].mean().round(4))
    print("[q3] saved 1 CSV + 1 figure")


# Q4 — Family comparison + extra posteriors (Dirac, top-k)
def q4_families() -> None:
    print("\n=== [q4] Family comparison and extra posteriors ===")
    rows = []

    for dataset in DATASETS:
        X, y = bk.real_dataset(dataset)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        for seed in range(N_SEEDS):
            # A / B on stumps
            for bn, r in run_categorical(X, y, seed=seed).items():
                rows.append({
                    "dataset": dataset, "seed": seed,
                    "family": "stumps", "bound_name": bn,
                    **{k: v for k, v in r.items() if k != "Q"},
                })
            # A / B on linear classifiers
            for bn, r in run_gaussian(X, y, seed=seed).items():
                rows.append({
                    "dataset": dataset, "seed": seed,
                    "family": "linear", "bound_name": bn,
                    **{k: v for k, v in r.items() if k != "Q"},
                })

            # Extra posteriors: Dirac on best stump, uniform on top-k
            Xp, yp, Xs, ys, Xte, yte = pb.split(X, y, seed)
            H = pb.make_stumps(Xp)
            V, Vte = pb.stump_votes(H, Xs), pb.stump_votes(H, Xte)
            P = np.full(len(H), 1 / len(H))
            L, m = pb.voter_risks(V, ys), len(ys)

            for k_val, label in [(1, "dirac_best_stump"),
                                 (5, "uniform_top5"),
                                 (20, "uniform_top20")]:
                order = np.argsort(L)[:k_val]
                Q = np.zeros(len(H))
                Q[order] = 1.0 / k_val
                emp = float(Q @ L)
                KL = kl_categorical(Q, P)
                rows.append({
                    "dataset": dataset, "seed": seed,
                    "family": "stumps", "bound_name": label,
                    "bound": bound_mcallester(emp, KL, m),
                    "seeger": bound_seeger(emp, KL, m),
                    "KL": KL, "emp": emp,
                    "neff": 1.0 / np.sum(Q ** 2),
                    "test_gibbs": gibbs_risk(Q, Vte, yte),
                    "test_vote": vote_risk(Q, Vte, yte),
                })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "pacbayes_q4.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["neff", "test_gibbs", "test_vote"]):
        agg = df.groupby(["family", "bound_name"])[metric].mean().unstack(0)
        agg.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Q4 — Family comparison + extra posteriors")
    fig.tight_layout()
    fig.savefig(FIGURES / "pacbayes_q4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(df.groupby(["family", "bound_name"])[
        ["bound", "seeger", "KL", "neff", "test_gibbs", "test_vote"]
    ].mean().round(4))
    print("[q4] saved 1 CSV + 1 figure")


# Q5 — Classical ensembles vs bound minimization
def q5_classical() -> None:
    print("\n=== [q5] Classical ensembles vs bound minimization ===")
    rows = []

    for dataset in DATASETS:
        X, y = bk.real_dataset(dataset)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        for seed in range(N_SEEDS):
            Xp, yp, Xs, ys, Xte, yte = pb.split(X, y, seed)
            Xtr = np.vstack([Xp, Xs])
            ytr = np.concatenate([yp, ys])

            # Classical ensembles, trained on the WHOLE training part
            classical = {
                "classical_AdaBoost_stumps": AdaBoostDiscreteClassifier(
                    n_estimators=100, max_depth=1,
                    learning_rate=1.0, random_state=42),
                "classical_RandomForest": RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1),
                "classical_LogReg": LogisticRegression(
                    max_iter=5000, random_state=42),
                "classical_Stacking": StackingClassifierCustom(
                    n_splits=5, random_state=42),
            }
            for name, mdl in classical.items():
                try:
                    m_fitted = clone(mdl)
                    m_fitted.fit(Xtr, ytr)
                    acc = accuracy_score(yte, m_fitted.predict(Xte))
                    rows.append({
                        "dataset": dataset, "seed": seed,
                        "method": name, "source": "classical",
                        "test_risk": float(1.0 - acc),
                        "certificate": np.nan,
                    })
                except Exception as exc:
                    print(f"  [WARN] {dataset}/{seed}/{name}: {exc}")

            # Bound-minimized classifiers (A and B, both families)
            for setting, runner in [("stumps", run_categorical),
                                    ("linear", run_gaussian)]:
                for bn, r in runner(X, y, seed=seed).items():
                    rows.append({
                        "dataset": dataset, "seed": seed,
                        "method": f"bound_{setting}_{bn}",
                        "source": "bound",
                        "test_risk": float(r["test_vote"]),
                        "certificate": float(r["seeger"]),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "pacbayes_q5.csv", index=False)

    agg = df.groupby("method")[["test_risk", "certificate"]].mean()
    agg = agg.sort_values("test_risk")

    # Figure: bar chart of mean test risk, classical in orange, bound in blue
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["tab:orange" if idx.startswith("classical_") else "tab:blue"
              for idx in agg.index]
    ax.barh(agg.index, agg["test_risk"], color=colors)
    for i, (_, row) in enumerate(agg.iterrows()):
        if not np.isnan(row["certificate"]):
            ax.text(row["test_risk"] + 0.005, i,
                    f"cert={row['certificate']:.2f}",
                    va="center", fontsize=8, color="black")
    ax.set_xlabel("Mean test risk (lower is better)")
    ax.set_title("Q5 — Classical vs bound-minimized classifiers "
                 "(certificates shown for bound methods)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "pacbayes_q5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(agg.round(4))
    print("[q5] saved 1 CSV + 1 figure")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--q1", action="store_true",
                        help="Bound curves vs m, KL and lambda")
    parser.add_argument("--q2", action="store_true",
                        help="A vs B, multi-dataset + lambda-trace")
    parser.add_argument("--q3", action="store_true",
                        help="Impact of the prior")
    parser.add_argument("--q4", action="store_true",
                        help="Family comparison + extra posteriors")
    parser.add_argument("--q5", action="store_true",
                        help="Classical vs bound minimization")
    parser.add_argument("--all", action="store_true",
                        help="Run every sub-command")
    args = parser.parse_args()

    tasks = {
        "q1": args.q1 or args.all,
        "q2": args.q2 or args.all,
        "q3": args.q3 or args.all,
        "q4": args.q4 or args.all,
        "q5": args.q5 or args.all,
    }
    if not any(tasks.values()):
        parser.error("nothing to do — pass --q1, --q2, --q3, --q4, --q5, or --all")

    t0 = time.perf_counter()
    if tasks["q1"]: q1_bounds_curves()
    if tasks["q2"]: q2_compare()
    if tasks["q3"]: q3_prior()
    if tasks["q4"]: q4_families()
    if tasks["q5"]: q5_classical()
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f}s")
    print(f"Outputs saved in {RESULTS} and {FIGURES}")


if __name__ == "__main__":
    main()