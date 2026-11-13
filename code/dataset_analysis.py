
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchkit as bk


OUTPUT_DIR = Path("../results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = bk.REAL_SMALL + bk.REAL_LARGE

# On part du principe ici qu'un classe déséquilibré est - de 30% du dataset
IMBALANCE_THRESHOLD = 0.30


def analyze_dataset(name):
    """Load and analyze one dataset."""

    X, y = bk.real_dataset(name)

    classes, counts = np.unique(y, return_counts=True)

    minority_proportion = counts.min() / counts.sum()
    majority_proportion = counts.max() / counts.sum()

    return {
        "Dataset": name,
        "Samples (m)": X.shape[0],
        "Features (d)": X.shape[1],
        "Classes": len(classes),
        "Minority proportion": minority_proportion,
        "Majority proportion": majority_proportion,
        "Imbalanced": minority_proportion < IMBALANCE_THRESHOLD,
    }


def main():
    results = []

    for name in DATASETS:
        try:
            results.append(analyze_dataset(name))
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    df = pd.DataFrame(results)

    if df.empty:
        print("No dataset could be loaded.")
        return

    df = df.sort_values("Minority proportion")

    output_file = OUTPUT_DIR / "dataset_summary.csv"
    df.to_csv(output_file, index=False)

    print("\n========== DATASET SUMMARY ==========")
    print(df.to_string(index=False))

    print(f"\nDatasets analyzed: {len(df)}")
    print(f"Imbalanced datasets: {df['Imbalanced'].sum()}")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
