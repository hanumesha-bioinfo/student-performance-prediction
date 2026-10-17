
import pickle
import pandas as pd

MODEL_PATH = "student_model.pkl"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
]


def get_valid_number(prompt, minimum, maximum):
    """Read a number and ensure it falls within the allowed range."""
    while True:
        try:
            value = float(input(prompt))

            if not minimum <= value <= maximum:
                print(f"Enter a value between {minimum} and {maximum}.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
    except (OSError, pickle.UnpicklingError) as exc:
        print(f"Could not load the trained model: {exc}")
        return

    print("=" * 40)
    print(" Student Performance Prediction")
    print("=" * 40)
    print("Educational demonstration using synthetic data.\n")

    study_hours = get_valid_number("Study hours (0-24): ", 0, 24)
    attendance = get_valid_number("Attendance percentage (0-100): ", 0, 100)
    previous_score = get_valid_number("Previous exam score (0-100): ", 0, 100)
    assignment_score = get_valid_number("Assignment score (0-100): ", 0, 100)

    student = pd.DataFrame(
        [[study_hours, attendance, previous_score, assignment_score]],
        columns=FEATURES,
    )

    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]
    class_to_probability = dict(zip(model.classes_, probabilities))

    result = "PASS" if prediction == 1 else "FAIL"

    print(f"\nPredicted result: {result}")
    print(f"Estimated pass probability: {class_to_probability.get(1, 0) * 100:.2f}%")
    print(f"Estimated fail probability: {class_to_probability.get(0, 0) * 100:.2f}%")
    print("\nNote: These estimates are based on a small synthetic dataset.")


if __name__ == "__main__":
    main()
