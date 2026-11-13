
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


class BaggingOOBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=200,
        max_depth=5,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = len(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        oob_sum = np.zeros((n_samples, n_classes))
        oob_count = np.zeros(n_samples, dtype=int)

        for _ in range(self.n_estimators):
            # Tirage bootstrap : n_samples tirages avec remise
            bootstrap_idx = rng.integers(0, n_samples, n_samples)

            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[bootstrap_idx] = True
            oob_idx = np.flatnonzero(~in_bag)

            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[bootstrap_idx], y[bootstrap_idx])
            self.estimators_.append(tree)

            # Prédictions uniquement sur les observations OOB
            if len(oob_idx) > 0:
                probabilities = tree.predict_proba(X[oob_idx])

                # Aligner les colonnes sur les classes globales
                aligned = np.zeros((len(oob_idx), n_classes))
                for j, cls in enumerate(tree.classes_):
                    global_j = np.where(self.classes_ == cls)[0][0]
                    aligned[:, global_j] = probabilities[:, j]

                oob_sum[oob_idx] += aligned
                oob_count[oob_idx] += 1

        valid = oob_count > 0
        self.oob_valid_ = valid
        self.oob_counts_ = oob_count

        self.oob_decision_function_ = np.full(
            (n_samples, n_classes), np.nan
        )
        self.oob_decision_function_[valid] = (
            oob_sum[valid] / oob_count[valid, None]
        )

        oob_pred = self.classes_[
            np.argmax(self.oob_decision_function_[valid], axis=1)
        ]

        self.oob_score_ = accuracy_score(y[valid], oob_pred)
        self.oob_balanced_accuracy_ = balanced_accuracy_score(
            y[valid], oob_pred
        )

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        total = np.zeros((len(X), len(self.classes_)))

        for tree in self.estimators_:
            probabilities = tree.predict_proba(X)

            for j, cls in enumerate(tree.classes_):
                global_j = np.where(self.classes_ == cls)[0][0]
                total[:, global_j] += probabilities[:, j]

        return total / len(self.estimators_)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class AdaBoostDiscreteClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=100,
        max_depth=1,
        learning_rate=1.0,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = len(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("AdaBoost nécessite au moins deux classes.")

        weights = np.full(n_samples, 1.0 / n_samples)
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )

            tree.fit(X, y, sample_weight=weights)
            pred = tree.predict(X)

            incorrect = pred != y
            error = np.sum(weights * incorrect)

            # Un classifieur parfait termine l'apprentissage
            if error <= 1e-12:
                self.estimators_.append(tree)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(error)
                break

            # Un classifieur moins bon que le hasard n'est pas retenu
            if error >= 1.0 - 1.0 / n_classes:
                if len(self.estimators_) == 0:
                    raise ValueError(
                        "Aucun classifieur faible meilleur que le hasard."
                    )
                break

            alpha = self.learning_rate * (
                np.log((1.0 - error) / error)
                + np.log(n_classes - 1)
            )

            self.estimators_.append(tree)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)

            # Mise à jour des poids des observations
            weights *= np.exp(alpha * incorrect)
            weights /= weights.sum()

        self.estimator_weights_ = np.asarray(
            self.estimator_weights_
        )
        self.estimator_errors_ = np.asarray(
            self.estimator_errors_
        )

        if len(self.estimators_) == 0:
            raise ValueError("Aucun estimateur n'a été entraîné.")

        return self

    def decision_function(self, X):
        X = np.asarray(X)
        scores = np.zeros((len(X), len(self.classes_)))

        for tree, alpha in zip(
            self.estimators_, self.estimator_weights_
        ):
            pred = tree.predict(X)

            for j, cls in enumerate(self.classes_):
                scores[:, j] += alpha * (pred == cls)

        return scores

    def predict_proba(self, X):
        return softmax(self.decision_function(X), axis=1)

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


class AdaBoostDiscreteClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        n_estimators=100,
        max_depth=1,
        learning_rate=1.0,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("AdaBoost discret nécessite deux classes.")

        # Conversion des labels en {-1, +1}
        y_signed = np.where(y == self.classes_[1], 1, -1)

        n_samples = len(y)
        weights = np.ones(n_samples) / n_samples

        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        self.estimator_weights_ = []
        self.errors_ = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )

            tree.fit(X, y_signed, sample_weight=weights)

            predictions = tree.predict(X)
            incorrect = predictions != y_signed

            error = np.sum(weights * incorrect)

            # Un apprenant au hasard ou pire n'apporte rien
            if error >= 0.5 - 1e-12:
                break

            error = max(error, 1e-12)

            alpha = self.learning_rate * (
                0.5 * np.log((1 - error) / error)
            )

            self.estimators_.append(tree)
            self.estimator_weights_.append(alpha)
            self.errors_.append(error)

            weights *= np.exp(-alpha * y_signed * predictions)
            weights /= weights.sum()

            # Arrêt si classification parfaite
            if error <= 1e-12:
                break

        if not self.estimators_:
            raise ValueError(
                "Aucun apprenant faible avec une erreur inférieure à 0.5."
            )

        self.estimator_weights_ = np.asarray(
            self.estimator_weights_
        )

        return self

    def decision_function(self, X):
        X = np.asarray(X)

        scores = np.zeros(len(X))

        for tree, alpha in zip(
            self.estimators_, self.estimator_weights_
        ):
            scores += alpha * tree.predict(X)

        return scores

    def predict(self, X):
        scores = self.decision_function(X)

        predictions = np.where(
            scores >= 0, self.classes_[1], self.classes_[0]
        )

        return predictions

    def predict_proba(self, X):
        scores = self.decision_function(X)

        # Transformation monotone des scores en probabilités
        margin = np.clip(2 * scores / self.estimator_weights_.sum(),
                         -30, 30)
        p_positive = 1 / (1 + np.exp(-margin))

        return np.column_stack((1 - p_positive, p_positive))


class StackingClassifierCustom(ClassifierMixin, BaseEstimator):

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def _positive_proba(self, model, X):
        proba = model.predict_proba(X)
        positive_idx = np.where(model.classes_ == self.classes_[1])[0][0]
        return proba[:, positive_idx]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("StackingClassifierCustom nécessite deux classes.")

        self.base_estimators_ = [
            ("tree", DecisionTreeClassifier(
                max_depth=5, random_state=self.random_state
            )),
            ("forest", RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
            )),
            ("svm", make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(
                    SVC(random_state=self.random_state),
                    method="sigmoid",
                    cv=3,
                )
            )),
        ]

        n_samples = len(y)
        n_models = len(self.base_estimators_)
        oof_features = np.zeros((n_samples, n_models))

        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # Construire les prédictions out-of-fold
        for model_idx, (_, estimator) in enumerate(self.base_estimators_):
            for train_idx, valid_idx in cv.split(X, y):
                model = clone(estimator)
                model.fit(X[train_idx], y[train_idx])

                # Extraire les probabilités de la classe positive
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X[valid_idx])
                    classes = (
                        model.classes_ if hasattr(model, "classes_")
                        else model[-1].classes_
                    )
                    pos_col = np.where(classes == self.classes_[1])[0][0]
                    oof_features[valid_idx, model_idx] = probabilities[:, pos_col]
                else:
                    raise ValueError("Chaque estimateur doit fournir predict_proba.")

        self.meta_model_ = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
        )
        self.meta_model_.fit(oof_features, y)

        # Réentraîner les modèles de base sur toutes les données
        self.estimators_ = []
        for _, estimator in self.base_estimators_:
            fitted = clone(estimator)
            fitted.fit(X, y)
            self.estimators_.append(fitted)

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        meta_features = np.column_stack([
            self._positive_proba(model, X)
            for model in self.estimators_
        ])
        return self.meta_model_.predict_proba(meta_features)

    def predict(self, X):
        return self.classes_[
            np.argmax(self.predict_proba(X), axis=1)
        ]
