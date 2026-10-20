"""Personal model: learns how *this* user's sleep differs from the general model.

    final prediction = population expected quality  +  personal correction

The personal correction is a ridge regression fitted on the user's own diary to the residuals
(actual rating - population prediction). Ridge shrinks the correction towards zero when there
are few entries, so a new user gets the general model and the app becomes more personal as
the diary grows. It also uses habits the public dataset doesn't have (caffeine, screen time,
bedtime). The algorithm stays the same; only the data it learns from is personal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .population import PopulationModel

PERSONAL_FEATURES = ["sleep_hours", "stress", "activity_min", "steps", "caffeine", "screen_min", "bed_offset"]
MIN_ENTRIES = 7        # personal model switches on
FULL_ENTRIES = 30      # shown as "fully personalised" in the UI

LABELS = {
    # personal feature / population feature -> friendly name
    "sleep_hours": "Sleep duration", "Sleep Duration": "Sleep duration",
    "stress": "Stress", "Stress Level": "Stress",
    "activity_min": "Physical activity", "Physical Activity Level": "Physical activity",
    "steps": "Steps", "Daily Steps": "Steps",
    "caffeine": "Caffeine after noon", "screen_min": "Screen time before bed",
    "bed_offset": "Bedtime",
    "Age": "Age", "Gender": "Gender", "BMI Category": "BMI",
    "Heart Rate": "Resting heart rate", "Systolic BP": "Blood pressure", "Diastolic BP": "Blood pressure",
}
MODIFIABLE_LABELS = ["Sleep duration", "Stress", "Physical activity", "Steps",
                     "Caffeine after noon", "Screen time before bed", "Bedtime"]


def quality_label(q: float) -> str:
    return "Poor" if q < 6.5 else ("Fair" if q < 7.5 else "Good")


@dataclass
class Prediction:
    population: float
    correction: float
    final: float
    label: str
    personalised: bool


@dataclass
class PersonalModel:
    pop: PopulationModel
    user: dict
    ridge: Pipeline | None
    n: int
    mae_population: float | None = None
    mae_personal: float | None = None

    # ------------------------------------------------------------ fit
    @classmethod
    def fit(cls, pop: PopulationModel, user: dict, diary: pd.DataFrame) -> "PersonalModel":
        n = len(diary)
        if n < MIN_ENTRIES:
            return cls(pop, user, None, n)
        X = diary[PERSONAL_FEATURES].astype(float)
        base = pop.expected_quality(user, diary)
        resid = diary["quality"].astype(float).values - base

        def make():
            return Pipeline([("scale", StandardScaler()),
                             ("ridge", RidgeCV(alphas=np.logspace(0, 3, 13)))])

        ridge = make().fit(X, resid)
        mae_pop = mae_pers = None
        if n >= 10:
            cv = KFold(n_splits=min(5, n // 2), shuffle=True, random_state=0)
            resid_cv = cross_val_predict(make(), X, resid, cv=cv)
            y = diary["quality"].astype(float).values
            mae_pop = float(np.mean(np.abs(y - base)))
            mae_pers = float(np.mean(np.abs(y - np.clip(base + resid_cv, 1, 10))))
        return cls(pop, user, ridge, n, mae_pop, mae_pers)

    @property
    def personalised(self) -> bool:
        return self.ridge is not None

    # ------------------------------------------------------------ predict
    def predict(self, night: dict) -> Prediction:
        df = pd.DataFrame([night])
        p = float(self.pop.expected_quality(self.user, df)[0])
        c = float(self.ridge.predict(df[PERSONAL_FEATURES].astype(float))[0]) if self.ridge else 0.0
        f = float(np.clip(p + c, 1, 10))
        return Prediction(p, c, f, quality_label(f), self.personalised)

    def predict_many(self, nights: pd.DataFrame) -> np.ndarray:
        p = self.pop.expected_quality(self.user, nights)
        if self.ridge:
            p = p + self.ridge.predict(nights[PERSONAL_FEATURES].astype(float))
        return np.clip(p, 1, 10)

    # ------------------------------------------------------------ explain (SHAP)
    def explain(self, nights: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """Exact additive explanation: base + sum(contributions) = prediction (before clipping).

        Population part: TreeSHAP on the random forest.
        Personal part: linear SHAP for ridge, coef * (x - mean(x)), i.e. relative to your average night.
        """
        base, pop_c = self.pop.explain(self.user, nights)
        contrib = pop_c.T.groupby(lambda c: LABELS[c]).sum().T
        if self.ridge:
            sc, rg = self.ridge.named_steps["scale"], self.ridge.named_steps["ridge"]
            Z = (nights[PERSONAL_FEATURES].astype(float).values - sc.mean_) / sc.scale_
            pc = pd.DataFrame(Z * rg.coef_, columns=[LABELS[f] for f in PERSONAL_FEATURES], index=nights.index)
            contrib = contrib.add(pc, fill_value=0.0)
            base += float(rg.intercept_)
        return base, contrib

    def importance(self, diary: pd.DataFrame) -> pd.Series:
        """Mean |SHAP| over the user's diary — which factors move *your* sleep most."""
        if diary.empty:
            return pd.Series(dtype=float)
        _, c = self.explain(diary)
        return c.abs().mean().sort_values(ascending=False)

    # ------------------------------------------------------------ what-if
    def what_if(self, night: dict) -> pd.DataFrame:
        base = self.predict(night).final
        changes = [
            ("Sleep duration", "sleep_hours", +1.0, lambda v: f"{v:g} h"),
            ("Stress", "stress", -2, lambda v: f"{v:.0f}/10"),
            ("Screen time before bed", "screen_min", -30, lambda v: f"{v:.0f} min"),
            ("Caffeine after noon", "caffeine", -1, lambda v: f"{v:.0f} cups"),
            ("Bedtime", "bed_offset", -0.5, _fmt_offset),
            ("Physical activity", "activity_min", +30, lambda v: f"{v:.0f} min"),
        ]
        lims = {"sleep_hours": (3, 11), "stress": (1, 10), "screen_min": (0, 240), "caffeine": (0, 10),
                "bed_offset": (-4, 4), "activity_min": (0, 240)}
        rows = []
        for label, key, delta, fmt in changes:
            if key in ("screen_min", "caffeine", "bed_offset") and not self.personalised:
                continue   # the general model doesn't know these habits
            new = float(np.clip(night[key] + delta, *lims[key]))
            if new == night[key]:
                continue
            changed = {**night, key: new}
            if key == "activity_min":
                changed["steps"] = night["steps"] + 60 * (new - night[key])
            q = self.predict(changed).final
            rows.append({"Habit": label, "From": fmt(night[key]), "To": fmt(new), "Change": q - base})
        return pd.DataFrame(rows).sort_values("Change", ascending=False).reset_index(drop=True)


def _fmt_offset(x: float) -> str:
    h = (x + 24) % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


# ------------------------------------------------------------------ descriptive patterns

def patterns(diary: pd.DataFrame, goal: float, min_group: int = 3) -> list[dict]:
    """Simple, honest comparisons from the user's own diary (no model involved)."""
    if len(diary) < 2 * min_group:
        return []
    q = diary["quality"]
    tests = [
        ("Sleep duration", diary["sleep_hours"] >= goal - 0.5,
         f"you slept at least {goal - 0.5:g} h", f"you slept less than {goal - 0.5:g} h"),
        ("Screen time before bed", diary["screen_min"] <= 20,
         "screen time before bed was 20 min or less", "screen time before bed was over 20 min"),
        ("Caffeine after noon", diary["caffeine"] == 0,
         "you had no caffeine after noon", "you had caffeine after noon"),
        ("Bedtime", diary["bed_offset"] <= diary["bed_offset"].median(),
         f"you went to bed by {_fmt_offset(diary['bed_offset'].median())}",
         f"you went to bed after {_fmt_offset(diary['bed_offset'].median())}"),
        ("Stress", diary["stress"] <= diary["stress"].median(),
         f"stress was {int(diary['stress'].median())}/10 or lower", f"stress was above {int(diary['stress'].median())}/10"),
        ("Steps", diary["steps"] >= diary["steps"].median(),
         f"you walked {int(diary['steps'].median()):,}+ steps", f"you walked under {int(diary['steps'].median()):,} steps"),
    ]
    out = []
    for habit, mask, good_txt, bad_txt in tests:
        a, b = q[mask], q[~mask]
        if len(a) < min_group or len(b) < min_group:
            continue
        out.append({"habit": habit, "diff": float(a.mean() - b.mean()), "good": good_txt, "bad": bad_txt,
                    "mean_a": float(a.mean()), "mean_b": float(b.mean()), "n_a": len(a), "n_b": len(b)})
    return sorted(out, key=lambda d: -abs(d["diff"]))
