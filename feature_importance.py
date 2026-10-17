
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "students.csv"
OUTPUT_CSV = BASE_DIR / "feature_importance.csv"
OUTPUT_PNG = BASE_DIR / "feature_importance.png"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
]
TARGET = "result"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]

    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data[required].apply(pd.to_numeric, errors="raise")
    data = data.dropna(subset=[TARGET])

    if data.empty:
        raise ValueError("No usable records found.")

    if data[TARGET].nunique() != 2 or not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The result column must contain both 0 (Fail) and 1 (Pass).")

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    model.fit(X, y)

    classifier = model.named_steps["classifier"]

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(OUTPUT_CSV, index=False)

    plt.figure(figsize=(9, 5))
    plt.barh(importance["feature"], importance["importance"])
    plt.xlabel("Random Forest Feature Importance")
    plt.ylabel("Feature")
    plt.title("Student Performance — Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()

    print("\nFEATURE IMPORTANCE")
    print(importance.to_string(index=False))
    print(f"\nSaved table: {OUTPUT_CSV}")
    print(f"Saved chart: {OUTPUT_PNG}")
    print(
        "\nInterpretation note: feature importance measures how the fitted "
        "Random Forest uses features; it does not prove causation. "
        "Results are based on a small synthetic dataset."
    )


if __name__ == "__main__":
    main()
