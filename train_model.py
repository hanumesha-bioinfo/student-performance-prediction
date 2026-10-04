import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# 1. Load dataset
data = pd.read_csv("students.csv")

print("\nDataset:")
print(data)

# 2. Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# 3. Remove duplicate rows
data = data.drop_duplicates()

# 4. Separate features and target
X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_score"
    ]
]

y = data["result"]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 6. Create ML pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

# 7. Train model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fail", "Pass"],
        zero_division=0
    )
)

# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)

disp.plot()
plt.title("Student Performance - Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# 11. Pass/Fail graph
result_counts = data["result"].value_counts().sort_index()

plt.figure()

plt.bar(
    ["Fail", "Pass"],
    [
        result_counts.get(0, 0),
        result_counts.get(1, 0)
    ]
)

plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.title("Student Pass/Fail Distribution")

plt.savefig("pass_fail_distribution.png")
plt.close()

# 12. Save trained model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nTrained model saved as student_model.pkl")
