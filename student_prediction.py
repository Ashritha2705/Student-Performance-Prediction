import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
data = pd.read_csv("student_performance.csv")

print("\nDataset:\n")
print(data.head())

# Features
X = data[['StudyHours', 'Attendance', 'PreviousMarks', 'Assignments']]

# Target
y = data['Performance']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("MAE:", round(mae,2))
print("R2 Score:", round(r2,2))

# Comparison
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print("\nResults")
print(result)

# Plot
plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions)

plt.xlabel("Actual Performance")
plt.ylabel("Predicted Performance")
plt.title("Actual vs Predicted Scores")

plt.show()

# User Prediction
print("\nEnter Student Details")

study = float(input("Study Hours: "))
attendance = float(input("Attendance: "))
previous = float(input("Previous Marks: "))
assignments = float(input("Assignments Score: "))

new_data = [[study, attendance, previous, assignments]]

prediction = model.predict(new_data)

print(f"\nPredicted Performance: {prediction[0]:.2f}")