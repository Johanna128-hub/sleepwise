"""Loading, cleaning and labelling the Sleep Health & Lifestyle data."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = ROOT / "data" / "Sleep_health_and_lifestyle_dataset.csv"
SAMPLE_DATA = ROOT / "data" / "sample_sleep_health.csv"

TARGET_SCORE = "Quality of Sleep"
TARGET = "Sleep Class"
CLASSES = ["Poor", "Fair", "Good"]

NUMERIC = [
    "Age", "Sleep Duration", "Physical Activity Level", "Stress Level",
    "Heart Rate", "Daily Steps", "Systolic BP", "Diastolic BP",
]
# Occupation is left out: the dataset's job list (nurse, lawyer, ...) doesn't cover the
# app's real users (students and others), so it can't generalise to them.
CATEGORICAL = ["Gender", "BMI Category"]
FEATURES = NUMERIC + CATEGORICAL

# Habits a person can actually change — used for what-if analysis & advice.
MODIFIABLE = {
    # feature: (min, max, step, unit, realistic change per "nudge")
    "Sleep Duration": (5.0, 9.0, 0.25, "h", 1.0),
    "Stress Level": (1, 10, 1, "/10", -2),
    "Physical Activity Level": (0, 120, 5, "min", 30),
    "Daily Steps": (1000, 15000, 500, "steps", 3000),
}

# Excluded on purpose:
#  - Person ID: an identifier, not a signal.
#  - Sleep Disorder: a diagnosis that is itself an outcome of poor sleep (target leakage),
#    and something a typical user of the app would not know.
EXCLUDED = ["Person ID", "Sleep Disorder"]

REQUIRED_RAW = [
    "Gender", "Age", "Occupation", "Sleep Duration", TARGET_SCORE,
    "Physical Activity Level", "Stress Level", "BMI Category",
    "Blood Pressure", "Heart Rate", "Daily Steps",
]


@dataclass
class DataReport:
    source: str
    is_sample: bool
    raw_rows: int
    duplicate_rows: int
    missing_cells: int
    clean_rows: int
    notes: list[str] = field(default_factory=list)


def score_to_class(score: float, poor_max: int = 6, good_min: int = 8) -> str:
    if score <= poor_max:
        return "Poor"
    if score >= good_min:
        return "Good"
    return "Fair"


def default_source() -> tuple[Path, bool]:
    if REAL_DATA.exists():
        return REAL_DATA, False
    return SAMPLE_DATA, True


def load_raw(source=None) -> tuple[pd.DataFrame, str, bool]:
    if source is None:
        path, is_sample = default_source()
        return pd.read_csv(path), path.name, is_sample
    name = getattr(source, "name", str(source))
    return pd.read_csv(source), Path(name).name, False


def clean(raw: pd.DataFrame, poor_max: int = 6, good_min: int = 8,
          drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_RAW if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    info: dict = {"notes": []}

    # "Normal Weight" and "Normal" mean the same thing in the Kaggle file.
    bmi = df["BMI Category"].astype(str).str.strip()
    n_norm = int((bmi == "Normal Weight").sum())
    df["BMI Category"] = bmi.replace({"Normal Weight": "Normal"})
    if n_norm:
        info["notes"].append(f"Merged {n_norm} 'Normal Weight' labels into 'Normal'.")

    # "126/83" -> two numeric columns.
    bp = df["Blood Pressure"].astype(str).str.split("/", expand=True)
    df["Systolic BP"] = pd.to_numeric(bp[0], errors="coerce")
    df["Diastolic BP"] = pd.to_numeric(bp[1], errors="coerce")
    df = df.drop(columns=["Blood Pressure"])
    info["notes"].append("Split 'Blood Pressure' into Systolic BP and Diastolic BP.")

    if "Sleep Disorder" in df.columns:
        df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

    info["missing_cells"] = int(df[FEATURES + [TARGET_SCORE]].isna().sum().sum())
    df = df.dropna(subset=FEATURES + [TARGET_SCORE])

    # Duplicates: identical people (ignoring the ID) leak between train and test
    # and inflate accuracy. Remove them before any split.
    key_cols = [c for c in df.columns if c != "Person ID"]
    dup_mask = df.duplicated(subset=key_cols)
    info["duplicate_rows"] = int(dup_mask.sum())
    if drop_duplicates and dup_mask.any():
        df = df[~dup_mask]
        info["notes"].append(
            f"Removed {info['duplicate_rows']} duplicate rows so the same person "
            "can't appear in both training and test sets."
        )

    df[TARGET] = df[TARGET_SCORE].apply(lambda s: score_to_class(s, poor_max, good_min))
    df = df.reset_index(drop=True)
    return df, info


def load_dataset(source=None, poor_max: int = 6, good_min: int = 8,
                 drop_duplicates: bool = True) -> tuple[pd.DataFrame, DataReport]:
    raw, name, is_sample = load_raw(source)
    df, info = clean(raw, poor_max, good_min, drop_duplicates)
    report = DataReport(
        source=name, is_sample=is_sample, raw_rows=len(raw),
        duplicate_rows=info["duplicate_rows"], missing_cells=info["missing_cells"],
        clean_rows=len(df), notes=info["notes"],
    )
    return df, report
