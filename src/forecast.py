import pandas as pd
import joblib


# ---------------------------------------
# 1. Load historical forecasting data
# ---------------------------------------

data = pd.read_csv("Data/forecast_training_data.csv")

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values("Date").reset_index(drop=True)


# ---------------------------------------
# 2. Load trained forecasting model
# ---------------------------------------

model = joblib.load("models/aqi_forecast_model.pkl")


# ---------------------------------------
# 3. Get the latest available day
# ---------------------------------------

latest = data.iloc[-1]

print("Latest available date:", latest["Date"].date())


# ---------------------------------------
# 4. Define model features
# ---------------------------------------

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


# ---------------------------------------
# 5. Create prediction input
# ---------------------------------------

X_latest = latest[features].to_frame().T


# ---------------------------------------
# 6. Predict AQI
# ---------------------------------------

prediction = model.predict(X_latest)[0]


print()
print("AQI Forecast")
print("------------")
print("Next available forecast AQI:", round(prediction, 2))