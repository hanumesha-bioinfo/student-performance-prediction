
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "students.csv"
OUTPUT_PATH = BASE_DIR / "model_comparison.csv"

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

    data = data[required].apply(pd.to_numeric, errors="raise").dropna()

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 (Fail) and 1 (Pass).")

    if data[TARGET].value_counts().min() < 5:
        raise ValueError("At least 5 examples of each class are required.")

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    rows = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            error_score="raise",
        )

        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nMODEL COMPARISON — 5-FOLD CROSS-VALIDATION")
    print(results.round(3).to_string(index=False))
    print(f"\nSaved comparison results to: {OUTPUT_PATH}")
    print(
        "\nWarning: the dataset is small and synthetic. "
        "These scores do not establish real-world performance."
    )


if __name__ == "__main__":
    main()
