import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load training data
data = pd.read_csv("Data/training_data.csv")

# Convert Date to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Create time-based features
data["Month"] = data["Date"].dt.month
data["Day of Year"] = data["Date"].dt.dayofyear

# Features
X = data[[
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
]]

# Target
y = data["AQI"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Make predictions
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 3))

# Save model
joblib.dump(model, "models/aqi_model.pkl")

print("Model saved successfully!")

# Feature importance
importance = model.feature_importances_

for feature, score in zip(X.columns, importance):
    print(feature, ":", round(score, 3))