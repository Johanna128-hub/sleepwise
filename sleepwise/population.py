"""General ("population") model trained on the Sleep Health & Lifestyle dataset.

It predicts the probability of each sleep class (Poor / Fair / Good) and turns that into an
*expected quality score* on the 1-10 scale:  E[q] = sum_c P(c) * mean_quality(c).
Because E[q] is linear in the class probabilities, TreeSHAP values for each class combine
exactly into SHAP values for E[q].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, ROOT, TARGET, TARGET_SCORE, load_dataset

MODEL_PATH = ROOT / "models" / "population.joblib"


@dataclass
class PopulationModel:
    prep: ColumnTransformer
    rf: RandomForestClassifier
    class_quality: dict          # class -> mean quality score in training data
    medians: dict                # fallbacks for optional profile fields
    source: str
    is_sample: bool
    feature_map: list            # transformed column -> original feature
    ranges: dict                 # numeric feature -> (min, max) seen in training

    # ------------------------------------------------------------ build
    @classmethod
    def train(cls, source=None) -> "PopulationModel":
        df, report = load_dataset(source)
        prep = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
        X = prep.fit_transform(df[FEATURES])
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                    random_state=42).fit(X, df[TARGET])
        class_quality = df.groupby(TARGET)[TARGET_SCORE].mean().to_dict()
        fmap = []
        for name in prep.get_feature_names_out():
            kind, rest = name.split("__", 1)
            fmap.append(rest if kind == "num" else next(c for c in CATEGORICAL if rest.startswith(c + "_")))
        medians = df[NUMERIC].median().to_dict()
        ranges = {c: (float(df[c].min()), float(df[c].max())) for c in NUMERIC}
        return cls(prep, rf, class_quality, medians, report.source, report.is_sample, fmap, ranges)

    @classmethod
    def load_or_train(cls, path: Path = MODEL_PATH) -> "PopulationModel":
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                pass
        m = cls.train()
        path.parent.mkdir(exist_ok=True)
        joblib.dump(m, path)
        return m

    # ------------------------------------------------------------ inputs
    def frame(self, user: dict, nights: pd.DataFrame) -> pd.DataFrame:
        """Combine a user profile with diary-style rows into model features."""
        g = user.get("gender") if user.get("gender") in ("Male", "Female") else "Male"
        out = pd.DataFrame({
            "Age": float(user.get("age") or self.medians["Age"]),
            "Sleep Duration": nights["sleep_hours"].astype(float).values,
            "Physical Activity Level": nights["activity_min"].astype(float).values,
            "Stress Level": nights["stress"].astype(float).values,
            "Heart Rate": float(user.get("heart_rate") or self.medians["Heart Rate"]),
            "Daily Steps": nights["steps"].astype(float).values,
            "Systolic BP": float(user.get("systolic_bp") or self.medians["Systolic BP"]),
            "Diastolic BP": float(user.get("diastolic_bp") or self.medians["Diastolic BP"]),
            "Gender": g,
            "BMI Category": user.get("bmi_category") or "Normal",
        }, index=nights.index)
        return out[FEATURES]

    def _genders(self, user: dict) -> list[dict]:
        # If gender isn't Male/Female, average over both so no group is assumed.
        if user.get("gender") in ("Male", "Female"):
            return [user]
        return [{**user, "gender": "Male"}, {**user, "gender": "Female"}]

    def out_of_range(self, user: dict, night: dict) -> list[str]:
        """Friendly names of inputs outside what the general model has seen."""
        row = self.frame(user, pd.DataFrame([night])).iloc[0]
        names = {"Sleep Duration": "sleep duration", "Stress Level": "stress", "Daily Steps": "steps",
                 "Physical Activity Level": "activity", "Age": "age", "Heart Rate": "heart rate"}
        return [n for c, n in names.items() if not (self.ranges[c][0] <= row[c] <= self.ranges[c][1])]

    # ------------------------------------------------------------ predict
    def expected_quality(self, user: dict, nights: pd.DataFrame) -> np.ndarray:
        q = np.array([self.class_quality[c] for c in self.rf.classes_])
        preds = [self.rf.predict_proba(self.prep.transform(self.frame(u, nights))) @ q
                 for u in self._genders(user)]
        return np.mean(preds, axis=0)

    def explain(self, user: dict, nights: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """SHAP values on the expected-quality scale, aggregated to original features.

        Returns (base_value, contributions[n_rows x features]).
        """
        q = np.array([self.class_quality[c] for c in self.rf.classes_])
        expl = _explainer(self)
        bases, contribs = [], []
        for u in self._genders(user):
            X = self.prep.transform(self.frame(u, nights))
            sv = expl.shap_values(X, check_additivity=False)
            sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv   # (n, f, classes)
            per_col = sv @ q                                             # (n, f)
            agg = pd.DataFrame(per_col, columns=self.feature_map, index=nights.index)
            contribs.append(agg.T.groupby(level=0).sum().T)
            bases.append(float(np.asarray(expl.expected_value) @ q))
        return float(np.mean(bases)), sum(contribs) / len(contribs)


_EXPLAINERS: dict[int, shap.TreeExplainer] = {}


def _explainer(m: PopulationModel) -> shap.TreeExplainer:
    key = id(m.rf)
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(m.rf)
    return _EXPLAINERS[key]
