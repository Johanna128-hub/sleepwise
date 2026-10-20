"""Model training, comparison, explanation and what-if analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .data import CATEGORICAL, CLASSES, FEATURES, MODIFIABLE, NUMERIC, TARGET

RANDOM_STATE = 42


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def candidate_models() -> dict[str, Pipeline]:
    models = {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "SVM (RBF)": CalibratedClassifierCV(SVC(C=2.0, class_weight="balanced"), cv=3, ensemble=False),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.08, max_iter=200, random_state=RANDOM_STATE),
    }
    return {name: Pipeline([("prep", _preprocessor()), ("clf", clf)]) for name, clf in models.items()}


@dataclass
class TrainingResult:
    cv_table: pd.DataFrame
    best_name: str
    model: Pipeline
    test_metrics: dict
    confusion: pd.DataFrame
    report: pd.DataFrame
    importance: pd.DataFrame
    n_train: int
    n_test: int
    classes: list[str]


def train_and_compare(df: pd.DataFrame, test_size: float = 0.2) -> TrainingResult:
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    min_class = int(y_tr.value_counts().min())
    cv = StratifiedKFold(n_splits=max(2, min(5, min_class)), shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in candidate_models().items():
        s = cross_validate(pipe, X_tr, y_tr, cv=cv,
                           scoring=["accuracy", "f1_macro", "balanced_accuracy"])
        rows.append({
            "Model": name,
            "Accuracy": s["test_accuracy"].mean(),
            "Macro F1": s["test_f1_macro"].mean(),
            "Macro F1 ±": s["test_f1_macro"].std(),
            "Balanced Acc.": s["test_balanced_accuracy"].mean(),
            "Fit time (s)": s["fit_time"].mean(),
        })
    cv_table = pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)

    # Pick the best real model by cross-validated macro F1 (robust to class imbalance).
    best_name = cv_table.loc[~cv_table["Model"].str.startswith("Baseline"), "Model"].iloc[0]
    model = candidate_models()[best_name].fit(X_tr, y_tr)
    pred = model.predict(X_te)

    classes = [c for c in CLASSES if c in set(y)]
    test_metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "Macro F1": f1_score(y_te, pred, average="macro"),
        "Balanced Acc.": balanced_accuracy_score(y_te, pred),
    }
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=classes), index=classes, columns=classes)
    rep = pd.DataFrame(classification_report(y_te, pred, labels=classes, output_dict=True,
                                             zero_division=0)).T.loc[classes]

    pi = permutation_importance(model, X_te, y_te, scoring="f1_macro",
                                n_repeats=15, random_state=RANDOM_STATE)
    importance = (pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean,
                                "Std": pi.importances_std})
                  .sort_values("Importance", ascending=False).reset_index(drop=True))

    # Refit on all data for serving predictions.
    model = candidate_models()[best_name].fit(X, y)

    return TrainingResult(cv_table, best_name, model, test_metrics, cm, rep, importance,
                          len(X_tr), len(X_te), classes)


# ---------------------------------------------------------------- prediction

def predict_profile(model: Pipeline, profile: dict) -> dict[str, float]:
    X = pd.DataFrame([profile])[FEATURES]
    proba = model.predict_proba(X)[0]
    return dict(zip(model.classes_, proba))


def sleep_score(proba: dict[str, float]) -> float:
    """0–100 score: P(Good) counts fully, P(Fair) counts half."""
    return 100 * (proba.get("Good", 0) + 0.5 * proba.get("Fair", 0))


def sweep(model: Pipeline, profile: dict, feature: str, n: int = 40) -> pd.DataFrame:
    lo, hi, step, *_ = MODIFIABLE[feature]
    values = np.arange(lo, hi + step / 2, step)
    if len(values) > n:
        values = np.linspace(lo, hi, n)
    X = pd.DataFrame([profile] * len(values))[FEATURES]
    X[feature] = values
    proba = model.predict_proba(X)
    out = pd.DataFrame(proba, columns=model.classes_)
    out[feature] = values
    out["Score"] = 100 * (out.get("Good", 0) + 0.5 * out.get("Fair", 0))
    return out


def what_if(model: Pipeline, profile: dict) -> pd.DataFrame:
    """Effect of one realistic change to each modifiable habit, holding the rest fixed."""
    base = sleep_score(predict_profile(model, profile))
    rows = []
    for feat, (lo, hi, step, unit, nudge) in MODIFIABLE.items():
        new_val = float(np.clip(profile[feat] + nudge, lo, hi))
        if new_val == profile[feat]:
            continue
        changed = {**profile, feat: new_val}
        # Steps and activity move together in real life.
        if feat == "Physical Activity Level":
            changed["Daily Steps"] = float(np.clip(profile["Daily Steps"] + 60 * (new_val - profile[feat]), 1000, 15000))
        score = sleep_score(predict_profile(model, changed))
        rows.append({"Habit": feat, "From": profile[feat], "To": new_val, "Unit": unit,
                     "Score": score, "Change": score - base})
    return pd.DataFrame(rows).sort_values("Change", ascending=False).reset_index(drop=True)
