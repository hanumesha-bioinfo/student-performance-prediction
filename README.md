# Student Performance Prediction Using Machine Learning

An educational machine-learning project that predicts student pass/fail outcomes using study hours, attendance, previous scores, and assignment scores.

## Features

- Interactive Streamlit dashboard
- Student performance prediction using Logistic Regression
- Logistic Regression vs Random Forest comparison
- Five-fold stratified cross-validation
- Feature importance analysis using Random Forest
- Dataset visualization and downloadable CSV reports
- Input validation and automated tests

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Pytest

## Dataset

The project currently uses a small **synthetic dataset of 30 records**. It is intended for learning and demonstrating an ML workflow, not for making real academic decisions.

Input features:
- Study hours
- Attendance percentage
- Previous score
- Assignment score

Target:
- `0` — Fail
- `1` — Pass

## Setup

```bash
git clone https://github.com/Hanumesh-gif/student-performance-prediction.git
cd student-performance-prediction

python3 -m venv venv
source venv/bin/activate
python -m pip install pandas scikit-learn matplotlib streamlit pytest
```

## Run the project

Train the model and generate evaluation plots:

```bash
python train_model.py
```

Compare models:

```bash
python compare_models.py
```

Generate feature importance:

```bash
python feature_importance.py
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

Run automated tests:

```bash
python -m pytest -v
```

## Evaluation

The current synthetic dataset produced 1.00 mean accuracy, precision, recall, and F1-score for both models in five-fold cross-validation.

These results do **not** establish real-world performance. The dataset is small and synthetic, and independent evaluation on appropriate real-world data would be needed to assess generalization.

## Limitations

- Small synthetic dataset
- No independent real-world test dataset
- Feature importance does not demonstrate causation
- Predicted probabilities are not necessarily calibrated

## Future Improvements

- Evaluate on a suitable independent dataset
- Add more robust data validation and testing
- Improve model evaluation and calibration
- Deploy the dashboard for portfolio demonstration

## Author

Hanumesh

GitHub: https://github.com/Hanumesh-gif