import pandas as pd
import joblib


# -----------------------------------------
# 1. Load historical forecasting data
# -----------------------------------------

data = pd.read_csv("Data/forecast_training_data_v2.csv")

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values("Date").reset_index(drop=True)


# -----------------------------------------
# 2. Load Forecast Model V2
# -----------------------------------------

model = joblib.load(
    "models/aqi_forecast_model_v2.pkl"
)


# -----------------------------------------
# 3. Features used by the model
# -----------------------------------------

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


# -----------------------------------------
# 4. Start with latest available data
# -----------------------------------------

latest_row = data.iloc[-1].copy()

print()
print("Latest available date:",
      latest_row["Date"].date())


# -----------------------------------------
# 5. Forecast Day +1
# -----------------------------------------

X = latest_row[features].to_frame().T

prediction_1 = model.predict(X)[0]

prediction_1 = round(prediction_1, 2)


# -----------------------------------------
# 6. Forecast Day +2
# -----------------------------------------

# Shift the AQI history forward

latest_row["AQI_Lag_3"] = latest_row["AQI_Lag_2"]

latest_row["AQI_Lag_2"] = latest_row["AQI_Lag_1"]

latest_row["AQI_Lag_1"] = prediction_1


X = latest_row[features].to_frame().T

prediction_2 = model.predict(X)[0]

prediction_2 = round(prediction_2, 2)


# -----------------------------------------
# 7. Forecast Day +3
# -----------------------------------------

latest_row["AQI_Lag_3"] = latest_row["AQI_Lag_2"]

latest_row["AQI_Lag_2"] = latest_row["AQI_Lag_1"]

latest_row["AQI_Lag_1"] = prediction_2


X = latest_row[features].to_frame().T

prediction_3 = model.predict(X)[0]

prediction_3 = round(prediction_3, 2)


# -----------------------------------------
# 8. Display forecast
# -----------------------------------------

print()
print("===================================")
print("       3-DAY AQI FORECAST")
print("===================================")

print()

print("Tomorrow :", prediction_1)
print("Day +2   :", prediction_2)
print("Day +3   :", prediction_3)

print()
print("Forecast completed successfully!")