import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, r2_score

# Load data
data = pd.read_csv("Data/training_data.csv")

# Convert Date
data["Date"] = pd.to_datetime(data["Date"])

# Time features
data["Month"] = data["Date"].dt.month
data["Day of Year"] = data["Date"].dt.dayofyear

# Features
features = [
    "Max Temperature",
    "Min Temperature",
    "Rain",
    "Wind Speed",
    "Humidity",
    "Pressure",
    "PM2.5",
    "PM10",
    "CO",
    "NO2",
    "SO2",
    "O3",
    "Month",
    "Day of Year"
]

X = data[features]
y = data["AQI"]

# Chronological split
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Load the trained model
model = joblib.load("models/aqi_model.pkl")

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nChronological Validation")
print("------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 3))