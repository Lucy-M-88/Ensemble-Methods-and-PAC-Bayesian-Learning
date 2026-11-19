"""Compare les implémentations maison (ensemble_methods.py) avec les
équivalents scikit-learn sur un sous-ensemble de datasets.

Vérifie que les 3 méthodes imposées par le sujet
(bagging OOB, AdaBoost discret, stacking) sont fonctionnellement
équivalentes à leurs contreparties scikit-learn.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchkit as bk
from ensemble_methods import (
    BaggingOOBClassifier,
    AdaBoostDiscreteClassifier,
    StackingClassifierCustom,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

# Sous-ensemble de datasets (compromis temps/representativite)
DATASETS = [
    "wdbc", "heart", "pima", "iono", "sonar",
    "australian", "german", "vehicle",
]


# Construction des équivalents scikit-learn

def build_sklearn_bagging():
    base = DecisionTreeClassifier(
        max_depth=5, random_state=RANDOM_STATE
    )
    try:
        return BaggingClassifier(
            estimator=base, n_estimators=200,
            oob_score=True, random_state=RANDOM_STATE,
        )
    except TypeError:  # sklearn < 1.2
        return BaggingClassifier(
            base_estimator=base, n_estimators=200,
            oob_score=True, random_state=RANDOM_STATE,
        )


def build_sklearn_adaboost():
    """Construit l'équivalent discret (SAMME) de notre AdaBoostDiscrete."""
    base = DecisionTreeClassifier(
        max_depth=1, random_state=RANDOM_STATE
    )
    candidates = [
        # sklearn 1.4–1.5 : estimator + algorithm="SAMME"
        dict(estimator=base, n_estimators=100,
             algorithm="SAMME", random_state=RANDOM_STATE),
        # sklearn < 1.4 : base_estimator + algorithm="SAMME"
        dict(base_estimator=base, n_estimators=100,
             algorithm="SAMME", random_state=RANDOM_STATE),
        # sklearn >= 1.6 : algorithm retire, SAMME par defaut
        dict(n_estimators=100, random_state=RANDOM_STATE),
    ]
    for kwargs in candidates:
        try:
            return AdaBoostClassifier(**kwargs)
        except TypeError:
            continue
    raise RuntimeError("Impossible de construire un AdaBoost sklearn.")


def build_sklearn_stacking():
    bases = [
        ("tree", DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE)),
        ("forest", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        ("svm", make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(random_state=RANDOM_STATE),
                method="sigmoid", cv=3,
            ),
        )),
    ]
    return StackingClassifier(
        estimators=bases,
        final_estimator=LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )


# Évaluation sur une CV

def evaluate_model(model, X, y, cv):
    scores = {"accuracy": [], "balanced_accuracy": [], "auc": []}

    for tr, te in cv.split(X, y):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = clone(model)
                clf.fit(X[tr], y[tr])

            pred = clf.predict(X[te])
            proba = clf.predict_proba(X[te])
            classes = clf.classes_
            pos_idx = 1 if len(classes) == 2 else 0

            scores["accuracy"].append(accuracy_score(y[te], pred))
            scores["balanced_accuracy"].append(
                balanced_accuracy_score(y[te], pred))
            try:
                scores["auc"].append(
                    roc_auc_score(y[te], proba[:, pos_idx]))
            except ValueError:
                scores["auc"].append(np.nan)
        except Exception as exc:
            print(f"  [WARN] {exc}")
            for k in scores:
                scores[k].append(np.nan)

    return {k: float(np.nanmean(v)) for k, v in scores.items()}


# Main

def main():
    pairs = {
        "Bagging": (
            BaggingOOBClassifier(n_estimators=200, max_depth=5,
                                 random_state=RANDOM_STATE),
            build_sklearn_bagging(),
        ),
        "AdaBoost": (
            AdaBoostDiscreteClassifier(
                n_estimators=100, max_depth=1,
                learning_rate=1.0, random_state=RANDOM_STATE),
            build_sklearn_adaboost(),
        ),
        "Stacking": (
            StackingClassifierCustom(
                n_splits=5, random_state=RANDOM_STATE),
            build_sklearn_stacking(),
        ),
    }

    rows = []

    for name in DATASETS:
        try:
            X, y = bk.real_dataset(name)
            X = np.asarray(X, dtype=float)
            y = np.asarray(y)
        except Exception as exc:
            print(f"[SKIP] {name}: {exc}")
            continue

        cv = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True,
            random_state=RANDOM_STATE,
        )

        for method, (ours, theirs) in pairs.items():
            print(f"[{name}] {method} ...")
            res_ours = evaluate_model(ours, X, y, cv)
            res_theirs = evaluate_model(theirs, X, y, cv)

            row = {"dataset": name, "method": method}
            for metric in ("accuracy", "balanced_accuracy", "auc"):
                row[f"ours_{metric}"] = res_ours[metric]
                row[f"sklearn_{metric}"] = res_theirs[metric]
                row[f"abs_diff_{metric}"] = abs(
                    res_ours[metric] - res_theirs[metric])
            rows.append(row)

    df = pd.DataFrame(rows)
    out_file = OUTPUT_DIR / "sklearn_comparison.csv"
    df.to_csv(out_file, index=False)

    print("\n=== Écart absolu moyen (nôtre vs scikit-learn) ===")
    for metric in ("accuracy", "balanced_accuracy", "auc"):
        stats = (
            df.groupby("method")[f"abs_diff_{metric}"]
            .agg(["mean", "max"])
            .round(4)
        )
        print(f"\n{metric}:")
        print(stats)

    print(f"\nRésultats sauvegardés dans : {out_file}")


if __name__ == "__main__":
    main()