import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------
# 1. Load forecasting dataset
# --------------------------------

data = pd.read_csv("Data/forecast_training_data_v2.csv")

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values("Date").reset_index(drop=True)


# --------------------------------
# 2. Define input features
# --------------------------------

features = [
    "AQI_Lag_1",
    "AQI_Lag_2",
    "AQI_Lag_3",

    "PM2.5_Lag_1",
    "PM2.5_Lag_2",

    "PM10",

    "Temperature_Lag_1",
    "Humidity_Lag_1",

    "Max Temperature",
    "Min Temperature",
    "Rain",
    "Wind Speed",
    "Humidity",
    "Pressure",

    "CO",
    "NO2",
    "SO2",
    "O3",

    "Month",
    "Day of Year"
]


# --------------------------------
# 3. X = information available today
#    y = tomorrow's AQI
# --------------------------------

X = data[features]

y = data["Target_AQI"]


# --------------------------------
# 4. Chronological train/test split
# --------------------------------

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------
# 5. Train Random Forest
# --------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print()
print("Forecast Model V2 trained successfully!")


# --------------------------------
# 6. Test predictions
# --------------------------------

predictions = model.predict(X_test)


# --------------------------------
# 7. Evaluate
# --------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)


print()
print("## Forecast Model V2 Performance")
print()
print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 3))


# --------------------------------
# 8. Save model
# --------------------------------

joblib.dump(
    model,
    "models/aqi_forecast_model_v2.pkl"
)

print()
print("Forecast Model V2 saved successfully!")


# --------------------------------
# 9. Feature importance
# --------------------------------

print()
print("## Feature Importance")

importance = model.feature_importances_

for feature, score in sorted(
    zip(features, importance),
    key=lambda x: x[1],
    reverse=True
):

    print(feature, ":", round(score, 3))

print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R² Score:", round(r2, 3))