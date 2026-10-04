import pickle

# Load trained model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)

print("===================================")
print(" Student Performance Prediction")
print("===================================")

# Get student information
study_hours = float(input("Study hours: "))
attendance = float(input("Attendance percentage: "))
previous_score = float(input("Previous exam score: "))
assignment_score = float(input("Assignment score: "))

# Prepare student data
student = [[
    study_hours,
    attendance,
    previous_score,
    assignment_score
]]

# Make prediction
prediction = model.predict(student)

# Get probability
probability = model.predict_proba(student)

# Display prediction
if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")

print(f"Pass probability: {probability[0][1] * 100:.2f}%")
print(f"Fail probability: {probability[0][0] * 100:.2f}%")
