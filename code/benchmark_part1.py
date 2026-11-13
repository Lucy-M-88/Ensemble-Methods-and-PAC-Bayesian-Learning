
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchkit as bk
from ensemble_methods import (
    BaggingOOBClassifier,
    AdaBoostDiscreteClassifier,
    StackingClassifierCustom,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

DATASETS = list(dict.fromkeys(bk.REAL_SMALL + bk.REAL_LARGE))

MODELS = {
    "DecisionTree": DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_STATE,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Bagging": BaggingOOBClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=RANDOM_STATE,
    ),
    "AdaBoost": AdaBoostDiscreteClassifier(
        n_estimators=100,
        max_depth=1,
        learning_rate=1.0,
        random_state=RANDOM_STATE,
    ),
    "Stacking": StackingClassifierCustom(
        n_splits=5,
        random_state=RANDOM_STATE,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=15,
        random_state=RANDOM_STATE,
    ),
}


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_dataset(name, X, y):

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        classes = np.unique(y_train)

        if len(classes) != 2:
            raise ValueError(
                f"{name}: classification binaire attendue."
            )

        minority_class = min(
            classes,
            key=lambda c: np.mean(y_train == c)
        )

        positive_class = classes[1]

        for model_name, model_template in MODELS.items():

            # Nouvelle instance pour chaque fold
            model = clone(model_template)

            # Stacking gère son propre prétraitement interne.
            # Les autres modèles utilisent un scaler ajusté
            # uniquement sur les données d'entraînement.
            if model_name == "Stacking":
                clf = model
            else:
                clf = make_pipeline(
                    StandardScaler(),
                    model,
                )

            # Entraînement
            clf.fit(X_train, y_train)

            # Récupération du modèle final pour les scores OOB
            fitted_model = (
                clf.steps[-1][1]
                if hasattr(clf, "steps")
                else clf
            )

            oob_accuracy = getattr(
                fitted_model, "oob_score_", np.nan
            )

            oob_balanced_accuracy = getattr(
                fitted_model,
                "oob_balanced_accuracy_",
                np.nan,
            )

            # Prédictions
            pred = clf.predict(X_test)

            # Probabilité associée à la classe positive
            probabilities = clf.predict_proba(X_test)

            fitted_classes = (
                clf.classes_
                if hasattr(clf, "classes_")
                else clf[-1].classes_
            )

            positive_idx = np.where(
                fitted_classes == positive_class
            )[0][0]

            scores = probabilities[:, positive_idx]

            # Métriques
            rows.append({
                "dataset": name,
                "fold": fold,
                "model": model_name,

                "accuracy": accuracy_score(
                    y_test, pred
                ),

                "balanced_accuracy": balanced_accuracy_score(
                    y_test, pred
                ),

                "minority_f1": f1_score(
                    y_test,
                    pred,
                    pos_label=minority_class,
                    zero_division=0,
                ),

                "auc": roc_auc_score(
                    y_test,
                    scores,
                ),

                "oob_accuracy": oob_accuracy,

                "oob_balanced_accuracy": oob_balanced_accuracy,
            })

    return rows


# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

all_results = []

for name in DATASETS:

    try:
        X, y = bk.real_dataset(name)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        results = evaluate_dataset(name, X, y)

        all_results.extend(results)

        print(f"[OK] {name}")

    except Exception as exc:
        print(f"[ERREUR] {name}: {exc}")


# --------------------------------------------------
# Save results
# --------------------------------------------------

df = pd.DataFrame(all_results)

if df.empty:
    raise RuntimeError(
        "Aucun résultat obtenu. Vérifie les erreurs affichées."
    )

results_file = OUTPUT_DIR / "baseline_results.csv"

df.to_csv(results_file, index=False)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nNombre de datasets réussis :",
      df["dataset"].nunique(), "/", len(DATASETS))

print("\nNombre d'évaluations :",
      len(df))

print("\nRésultats moyens par modèle :")

summary = df.groupby("model")[
    [
        "accuracy",
        "balanced_accuracy",
        "minority_f1",
        "auc",
    ]
].mean()

print(summary.round(4))

print("\nRésultats OOB :")

oob_summary = df.groupby("model")[
    [
        "oob_accuracy",
        "oob_balanced_accuracy",
    ]
].mean()

print(oob_summary.round(4))

print(f"\nRésultats sauvegardés dans : {results_file}")
