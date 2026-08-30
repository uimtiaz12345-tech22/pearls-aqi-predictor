import requests
import csv

LATITUDE = 24.8607
LONGITUDE = 67.0011

START_DATE = "2022-08-05"
END_DATE = "2025-12-31"

URL = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start_date={START_DATE}&end_date={END_DATE}"
    f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
    f"wind_speed_10m_max,relative_humidity_2m_mean,surface_pressure_mean"
    f"&timezone=auto"
)

response = requests.get(URL)
weather_data = response.json()

daily = weather_data["daily"]

with open("Data/historical_weather.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Date",
        "Max Temperature",
        "Min Temperature",
        "Rain",
        "Wind Speed",
        "Humidity",
        "Pressure"
    ])

    for i in range(len(daily["time"])):

        writer.writerow([
            daily["time"][i],
            daily["temperature_2m_max"][i],
            daily["temperature_2m_min"][i],
            daily["precipitation_sum"][i],
            daily["wind_speed_10m_max"][i],
            daily["relative_humidity_2m_mean"][i],
            daily["surface_pressure_mean"][i]
        ])

print("Historical weather CSV created successfully!")