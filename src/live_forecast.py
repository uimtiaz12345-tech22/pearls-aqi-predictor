import requests
import pandas as pd
import joblib


# ==================================================
# 1. LOCATION
# ==================================================

LATITUDE = 24.8607
LONGITUDE = 67.0011


# ==================================================
# 2. LOAD TRAINED MODEL
# ==================================================

model = joblib.load(
    "models/aqi_forecast_model_v2.pkl"
)


# ==================================================
# 3. GET WEATHER FORECAST
# ==================================================

weather_url = (
    "https://api.open-meteo.com/v1/forecast?"
    f"latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    "&daily="
    "temperature_2m_max,"
    "temperature_2m_min,"
    "precipitation_sum,"
    "wind_speed_10m_max"
    "&hourly="
    "relative_humidity_2m,"
    "surface_pressure"
    "&forecast_days=4"
    "&timezone=Asia%2FKarachi"
)

weather_response = requests.get(weather_url)

print("Weather API Status:", weather_response.status_code)

weather_data = weather_response.json()


# ==================================================
# 4. GET AIR QUALITY FORECAST
# ==================================================

air_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    "&hourly="
    "pm2_5,"
    "pm10,"
    "carbon_monoxide,"
    "nitrogen_dioxide,"
    "sulphur_dioxide,"
    "ozone"
    "&forecast_days=4"
    "&timezone=Asia%2FKarachi"
)

air_response = requests.get(air_url)

print("Air Quality API Status:", air_response.status_code)

air_data = air_response.json()


# ==================================================
# 5. PREPARE DAILY WEATHER DATA
# ==================================================

daily_weather = pd.DataFrame({
    "Date": weather_data["daily"]["time"],
    "Max Temperature": weather_data["daily"]["temperature_2m_max"],
    "Min Temperature": weather_data["daily"]["temperature_2m_min"],
    "Rain": weather_data["daily"]["precipitation_sum"],
    "Wind Speed": weather_data["daily"]["wind_speed_10m_max"]
})


# ==================================================
# 6. PREPARE HOURLY WEATHER VARIABLES
# ==================================================

hourly_weather = pd.DataFrame({
    "DateTime": weather_data["hourly"]["time"],
    "Humidity": weather_data["hourly"]["relative_humidity_2m"],
    "Pressure": weather_data["hourly"]["surface_pressure"]
})

hourly_weather["Date"] = (
    pd.to_datetime(hourly_weather["DateTime"])
    .dt.date
)

daily_weather_extra = (
    hourly_weather
    .groupby("Date")
    .agg({
        "Humidity": "mean",
        "Pressure": "mean"
    })
    .reset_index()
)

daily_weather_extra["Date"] = (
    daily_weather_extra["Date"]
    .astype(str)
)


# Merge weather information

daily_weather = daily_weather.merge(
    daily_weather_extra,
    on="Date",
    how="left"
)


# ==================================================
# 7. PREPARE DAILY AIR QUALITY DATA
# ==================================================

hourly_air = pd.DataFrame({
    "DateTime": air_data["hourly"]["time"],
    "PM2.5": air_data["hourly"]["pm2_5"],
    "PM10": air_data["hourly"]["pm10"],
    "CO": air_data["hourly"]["carbon_monoxide"],
    "NO2": air_data["hourly"]["nitrogen_dioxide"],
    "SO2": air_data["hourly"]["sulphur_dioxide"],
    "O3": air_data["hourly"]["ozone"]
})

hourly_air["Date"] = (
    pd.to_datetime(hourly_air["DateTime"])
    .dt.date
)

hourly_air["Date"] = (
    hourly_air["Date"]
    .astype(str)
)


daily_air = (
    hourly_air
    .groupby("Date")
    .agg({
        "PM2.5": "mean",
        "PM10": "mean",
        "CO": "mean",
        "NO2": "mean",
        "SO2": "mean",
        "O3": "mean"
    })
    .reset_index()
)


# ==================================================
# 8. COMBINE WEATHER + AIR QUALITY
# ==================================================

forecast = daily_weather.merge(
    daily_air,
    on="Date",
    how="inner"
)

# ==================================================
# VERIFY FORECAST DATES
# ==================================================

forecast["Date"] = pd.to_datetime(
    forecast["Date"]
)

today = pd.Timestamp.now(
    tz="Asia/Karachi"
).normalize().tz_localize(None)

print()
print("======================================")
print("        FORECAST DATE CHECK")
print("======================================")
print()

print("System date:", today.date())

print(
    "API forecast starts:",
    forecast["Date"].min().date()
)

print(
    "API forecast ends:",
    forecast["Date"].max().date()
)

print()
# ==================================================
# 9. LOAD HISTORICAL DATA FOR LAGS
# ==================================================

historical = pd.read_csv(
    "Data/forecast_training_data_v2.csv"
)

historical["Date"] = pd.to_datetime(
    historical["Date"]
)

historical = historical.sort_values(
    "Date"
)


latest = historical.iloc[-1]


# ==================================================
# 10. INITIAL LAG VALUES
# ==================================================

aqi_lag_1 = latest["AQI"]
aqi_lag_2 = latest["AQI_Lag_1"]
aqi_lag_3 = latest["AQI_Lag_2"]

pm25_lag_1 = latest["PM2.5_Lag_1"]
pm25_lag_2 = latest["PM2.5_Lag_2"]

temperature_lag_1 = latest["Temperature_Lag_1"]
humidity_lag_1 = latest["Humidity_Lag_1"]


# ==================================================
# 11. FORECAST NEXT 3 DAYS
# ==================================================

results = []


# ==================================================
# SELECT NEXT 3 DAYS
# ==================================================

forecast = forecast.sort_values(
    "Date"
).reset_index(drop=True)

future_dates = forecast[
    forecast["Date"] > today
].head(3)

if len(future_dates) < 3:

    raise ValueError(
        "API did not provide enough future days "
        "for a 3-day forecast."
    )


# ==================================================
# FORECAST NEXT 3 DAYS
# ==================================================

results = []

for _, row in future_dates.iterrows():

    date = pd.to_datetime(
        row["Date"]
    )
    date = pd.to_datetime(
        row["Date"]
    )

    features = pd.DataFrame([{

        "AQI_Lag_1": aqi_lag_1,
        "AQI_Lag_2": aqi_lag_2,
        "AQI_Lag_3": aqi_lag_3,

        "PM2.5_Lag_1": pm25_lag_1,
        "PM2.5_Lag_2": pm25_lag_2,

        "PM10": row["PM10"],

        "Temperature_Lag_1":
            temperature_lag_1,

        "Humidity_Lag_1":
            humidity_lag_1,

        "Max Temperature":
            row["Max Temperature"],

        "Min Temperature":
            row["Min Temperature"],

        "Rain":
            row["Rain"],

        "Wind Speed":
            row["Wind Speed"],

        "Humidity":
            row["Humidity"],

        "Pressure":
            row["Pressure"],

        "CO":
            row["CO"],

        "NO2":
            row["NO2"],

        "SO2":
            row["SO2"],

        "O3":
            row["O3"],

        "Month":
            date.month,

        "Day of Year":
            date.dayofyear

    }])


    prediction = model.predict(
        features
    )[0]

    prediction = round(
        prediction,
        2
    )


    # ------------------------------------------
    # Save prediction
    # ------------------------------------------

    results.append({
        "Date": date.strftime("%Y-%m-%d"),
        "Predicted AQI": prediction
    })


    # ------------------------------------------
    # Update recursive variables
    # ------------------------------------------

    aqi_lag_3 = aqi_lag_2
    aqi_lag_2 = aqi_lag_1
    aqi_lag_1 = prediction

    pm25_lag_2 = pm25_lag_1
    pm25_lag_1 = row["PM2.5"]

    temperature_lag_1 = row[
        "Max Temperature"
    ]

    humidity_lag_1 = row[
        "Humidity"
    ]


# ==================================================
# 12. DISPLAY FINAL FORECAST
# ==================================================

result_df = pd.DataFrame(results)
result_df.to_csv(
    "Data/forecast_results.csv",
    index=False
)

print()
print("Forecast results saved successfully!")


print()
print("======================================")
print("       AUTOMATIC 3-DAY AQI FORECAST")
print("======================================")
print()

print(result_df.to_string(index=False))

print()
print("Forecast generated automatically!")