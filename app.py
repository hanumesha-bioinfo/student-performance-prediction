
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_model.pkl"
DATA_PATH = BASE_DIR / "students.csv"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
]

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Prediction")
st.write(
    "An interactive machine-learning dashboard to explore "
    "student performance and predict a PASS or FAIL result."
)

st.warning(
    "Educational demonstration only: this project uses a small "
    "synthetic dataset, not real student records. Predictions are "
    "not validated for real-world academic decisions."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python train_model.py")
    st.stop()

try:
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
except (OSError, pickle.UnpicklingError, EOFError) as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

if not DATA_PATH.exists():
    st.error("students.csv was not found.")
    st.stop()

try:
    data = pd.read_csv(DATA_PATH)
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not read the dataset: {error}")
    st.stop()

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(data))
col2.metric("Features", len(FEATURES))
col3.metric("Pass Rate", f"{(data['result'].eq(1).mean() * 100):.1f}%")

with st.expander("View dataset"):
    st.dataframe(data, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("Pass vs Fail Distribution")
    counts = data["result"].map({0: "Fail", 1: "Pass"}).value_counts()
    st.bar_chart(counts)

with right:
    st.subheader("Feature Relationships")
    st.write("Correlation between numerical features:")
    st.dataframe(
        data[FEATURES + ["result"]].corr(numeric_only=True).round(2),
        width="stretch",
    )

st.divider()
st.header("🔮 Predict Student Performance")
st.write("Enter values below to generate a model prediction.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "Study hours per week",
            min_value=0.0,
            max_value=168.0,
            value=10.0,
            step=1.0,
        )
        attendance = st.slider(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=80,
        )

    with col2:
        previous_score = st.slider(
            "Previous score (%)",
            min_value=0,
            max_value=100,
            value=70,
        )
        assignment_score = st.slider(
            "Assignment score (%)",
            min_value=0,
            max_value=100,
            value=75,
        )

    submitted = st.form_submit_button(
        "Predict Performance",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_data = pd.DataFrame(
        [[study_hours, attendance, previous_score, assignment_score]],
        columns=FEATURES,
    )

    try:
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        fail_probability = float(probabilities[classes.index(0)]) * 100
        pass_probability = float(probabilities[classes.index(1)]) * 100

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Predicted Result: PASS ✅")
        else:
            st.error("Predicted Result: FAIL ⚠️")

        col1, col2 = st.columns(2)
        col1.metric("Model's PASS probability", f"{pass_probability:.1f}%")
        col2.metric("Model's FAIL probability", f"{fail_probability:.1f}%")

        st.caption(
            "These percentages are model estimates, not calibrated "
            "probabilities or guarantees of actual student performance."
        )

    except (ValueError, KeyError, AttributeError) as error:
        st.error(f"Prediction failed: {error}")

st.divider()

st.header("🤖 Model Comparison")

comparison_path = BASE_DIR / "model_comparison.csv"

if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)

    st.write(
        "Compare Logistic Regression and Random Forest using "
        "5-fold cross-validation."
    )

    metric_options = {
        "Accuracy": "accuracy_mean",
        "Precision": "precision_mean",
        "Recall": "recall_mean",
        "F1-score": "f1_mean",
    }

    selected_metric = st.selectbox(
        "Select evaluation metric",
        list(metric_options.keys()),
    )

    metric_column = metric_options[selected_metric]

    st.bar_chart(
        comparison.set_index("model")[[metric_column]]
    )

    st.subheader("Detailed Comparison")
    st.dataframe(
        comparison.round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Values show mean cross-validation scores. Standard deviation "
        "columns show variation between folds."
    )

    st.warning(
        "Both models scored 1.00 on this small synthetic dataset. "
        "This does not establish real-world accuracy or prove that "
        "the models will generalize to unseen real students."
    )

    csv_data = comparison.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Model Comparison CSV",
        data=csv_data,
        file_name="model_comparison.csv",
        mime="text/csv",
    )
else:
    st.info(
        "Comparison results not found. Run "
        "`python compare_models.py` to generate them."
    )

st.divider()
st.header("🧪 Model Evaluation")

for filename, title in [
    ("confusion_matrix.png", "Confusion Matrix"),
    ("pass_fail_distribution.png", "Pass/Fail Distribution Plot"),
]:
    image_path = BASE_DIR / filename
    if image_path.exists():
        st.subheader(title)
        st.image(str(image_path), use_container_width=True)

st.write(
    "Run `python train_model.py` to evaluate the model and regenerate "
    "the evaluation plots. Cross-validation scores from this synthetic "
    "dataset should not be interpreted as real-world accuracy."
)

for filename, title in [
    ("confusion_matrix.png", "Confusion Matrix"),
    ("pass_fail_distribution.png", "Pass/Fail Distribution Plot"),
]:
    image_path = BASE_DIR / filename
    if image_path.exists():
        st.subheader(title)
        st.image(str(image_path), use_container_width=True)


st.divider()
st.header("🔍 Feature Importance")

importance_path = BASE_DIR / "feature_importance.csv"
importance_chart = BASE_DIR / "feature_importance.png"

if importance_path.exists():
    importance_data = pd.read_csv(importance_path)
    st.write(
        "These scores show how the fitted Random Forest uses each feature "
        "when making predictions."
    )
    st.bar_chart(
        importance_data.set_index("feature")[["importance"]]
    )

    st.download_button(
        "Download Feature Importance CSV",
        data=importance_data.to_csv(index=False).encode("utf-8"),
        file_name="feature_importance.csv",
        mime="text/csv",
    )

    st.caption(
        "Importance is not proof that a feature causes academic success. "
        "These results use synthetic data and are for demonstration only."
    )
else:
    st.info("Run `python feature_importance.py` to generate feature importance.")
