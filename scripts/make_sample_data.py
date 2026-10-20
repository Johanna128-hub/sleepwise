"""Generate a synthetic sample dataset with the same schema as the Kaggle
"Sleep Health and Lifestyle Dataset" so the app runs out of the box.

The numbers are SIMULATED. Replace data/sample_sleep_health.csv with the real
Kaggle CSV (save it as data/Sleep_health_and_lifestyle_dataset.csv) for your
actual experiments — the app picks the real file automatically if present.

Usage:  python scripts/make_sample_data.py [n_rows]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OCCUPATIONS = {
    # occupation: (stress shift, activity shift)
    "Nurse": (1.0, 15), "Doctor": (1.2, -5), "Engineer": (-0.6, 0),
    "Lawyer": (0.3, 10), "Teacher": (0.2, -10), "Accountant": (-0.3, -5),
    "Salesperson": (1.3, -15), "Software Engineer": (0.5, -10),
    "Scientist": (0.6, -10), "Manager": (0.4, 0),
}


def generate(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    occ = rng.choice(list(OCCUPATIONS), size=n)
    s_shift = np.array([OCCUPATIONS[o][0] for o in occ])
    a_shift = np.array([OCCUPATIONS[o][1] for o in occ])

    gender = rng.choice(["Male", "Female"], size=n)
    age = rng.integers(27, 60, size=n)

    stress = np.clip(np.round(5.3 + s_shift + rng.normal(0, 1.4, n)), 3, 8).astype(int)
    activity = np.clip(np.round(58 + a_shift + rng.normal(0, 17, n) - 2 * (stress - 5)), 30, 90).astype(int)
    steps = np.clip(np.round((2600 + 62 * activity + rng.normal(0, 900, n)) / 100) * 100, 3000, 10000).astype(int)
    sleep = np.clip(7.95 - 0.28 * (stress - 5) + 0.006 * (activity - 60) + rng.normal(0, 0.42, n), 5.8, 8.5).round(1)

    bmi_raw = rng.normal(0, 1, n) + 0.025 * (age - 40) - 0.02 * (activity - 60)
    bmi = np.where(bmi_raw > 1.25, "Obese", np.where(bmi_raw > 0.35, "Overweight", "Normal"))
    # Kaggle file mixes "Normal" and "Normal Weight" — reproduce that quirk.
    bmi = np.where((bmi == "Normal") & (rng.random(n) < 0.08), "Normal Weight", bmi)

    bmi_bump = pd.Series(bmi).map({"Normal": 0, "Normal Weight": 0, "Overweight": 8, "Obese": 14}).to_numpy()
    systolic = np.round(117 + 0.3 * (age - 40) + bmi_bump + 1.2 * (stress - 5) + rng.normal(0, 4, n)).astype(int)
    diastolic = np.round(systolic * 0.65 + rng.normal(0, 3, n)).astype(int)
    heart = np.clip(np.round(68 + 1.6 * (stress - 5) + 0.25 * bmi_bump - 0.04 * (activity - 60) + rng.normal(0, 2.5, n)), 60, 90).astype(int)

    quality_cont = (
        7.25
        + 1.35 * (sleep - 7.2)
        - 0.42 * (stress - 5.3)
        + 0.012 * (activity - 58)
        + 0.00008 * (steps - 6200)
        - 0.06 * (heart - 69)
        + 0.01 * (age - 42)
        + rng.normal(0, 0.45, n)
    )
    quality = np.clip(np.round(quality_cont), 4, 9).astype(int)

    p_apnea = 1 / (1 + np.exp(-(-3.2 + 0.18 * bmi_bump + 0.05 * (systolic - 125))))
    p_insom = 1 / (1 + np.exp(-(-2.3 + 0.55 * (stress - 5) - 0.9 * (sleep - 7))))
    u = rng.random(n)
    disorder = np.where(u < p_apnea, "Sleep Apnea", np.where(u < p_apnea + p_insom * (1 - p_apnea), "Insomnia", "None"))

    return pd.DataFrame({
        "Person ID": np.arange(1, n + 1),
        "Gender": gender,
        "Age": age,
        "Occupation": occ,
        "Sleep Duration": sleep,
        "Quality of Sleep": quality,
        "Physical Activity Level": activity,
        "Stress Level": stress,
        "BMI Category": bmi,
        "Blood Pressure": [f"{s}/{d}" for s, d in zip(systolic, diastolic)],
        "Heart Rate": heart,
        "Daily Steps": steps,
        "Sleep Disorder": disorder,
    })


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = Path(__file__).resolve().parents[1] / "data" / "sample_sleep_health.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate(n).to_csv(out, index=False)
    print(f"Wrote {n} rows to {out}")
