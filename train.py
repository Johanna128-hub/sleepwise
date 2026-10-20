"""Train and compare models from the command line (no UI), and save the best one.

Usage:  python train.py [path/to/data.csv]
"""
import sys
from pathlib import Path

import joblib

from sleepwise.data import load_dataset
from sleepwise.model import train_and_compare

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    df, report = load_dataset(src)
    print(f"Data: {report.source}  rows={report.raw_rows}  duplicates={report.duplicate_rows}  used={report.clean_rows}")
    for note in report.notes:
        print("  -", note)
    print("\nClass balance:\n", df["Sleep Class"].value_counts().to_string(), "\n")

    res = train_and_compare(df)
    print("Cross-validation (training split):")
    print(res.cv_table.round(3).to_string(index=False))
    print(f"\nBest model: {res.best_name}")
    print("Held-out test:", {k: round(v, 3) for k, v in res.test_metrics.items()})
    print("\nConfusion matrix (rows = actual):\n", res.confusion.to_string())
    print("\nPermutation importance:\n", res.importance.round(3).to_string(index=False))

    out = Path(__file__).parent / "models" / "sleepwise_model.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(res.model, out)
    print(f"\nSaved model to {out}")
