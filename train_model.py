
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "students.csv"
MODEL_PATH = BASE_DIR / "student_model.pkl"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
]
TARGET = "result"


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data = data[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data = data.drop_duplicates().reset_index(drop=True)

    if data.empty:
        raise ValueError("The dataset contains no usable records.")

    if data.isnull().all(axis=0).any():
        raise ValueError("A required column contains only missing values.")

    if data[TARGET].isnull().any():
        raise ValueError("The target column contains missing values.")

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The result column must contain only 0 (Fail) or 1 (Pass).")

    class_counts = data[TARGET].value_counts()

    if len(class_counts) != 2 or class_counts.min() < 5:
        raise ValueError(
            "Five-fold stratified evaluation requires at least "
            "five records in each target class."
        )

    return data


def create_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42)),
    ])


def main():
    data = load_dataset()
    X = data[FEATURES]
    y = data[TARGET].astype(int)

    print("\nDataset shape:", data.shape)
    print("\nMissing values:")
    print(data.isnull().sum())
    print("\nTarget counts (0 = Fail, 1 = Pass):")
    print(y.value_counts().sort_index())

    # Each record is evaluated using a model that did not train on it.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    evaluation_model = create_model()
    y_pred = cross_val_predict(
        evaluation_model,
        X,
        y,
        cv=cv,
        method="predict",
    )

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print("\n===== 5-FOLD CROSS-VALIDATION RESULTS =====")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 score : {f1:.3f}")

    print("\nOut-of-fold classification report:")
    print(
        classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=["Fail", "Pass"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"],
    )
    disp.plot()
    plt.title("Student Performance - 5-Fold Cross-Validation")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    counts = y.value_counts().reindex([0, 1], fill_value=0)
    plt.figure()
    plt.bar(["Fail", "Pass"], counts.values)
    plt.xlabel("Result")
    plt.ylabel("Number of Records")
    plt.title("Student Pass/Fail Distribution")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "pass_fail_distribution.png", dpi=150)
    plt.close()

    # Evaluate first, then fit the final prediction model on all available data.
    final_model = create_model()
    final_model.fit(X, y)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(final_model, file)

    print(f"\nFinal model trained on all {len(data)} records.")
    print(f"Saved model: {MODEL_PATH}")
    print("Evaluation and plots completed.")
    print(
        "\nImportant: This is a small synthetic dataset. "
        "These metrics do not establish real-world predictive performance."
    )


if __name__ == "__main__":
    main()
