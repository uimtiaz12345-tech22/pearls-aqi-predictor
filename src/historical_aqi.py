import requests
import csv

LATITUDE = 24.8607
LONGITUDE = 67.0011

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

URL = (
    f"https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start_date={START_DATE}"
    f"&end_date={END_DATE}"
    f"&hourly=us_aqi"
    f"&timezone=auto"
)

response = requests.get(URL)

aqi_data = response.json()

print(aqi_data.keys())

print(aqi_data)

exit()
times = hourly["time"]
aqi_values = hourly["us_aqi"]

with open("Data/historical_aqi.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["Date", "Average AQI"])

    for i in range(0, len(aqi_values), 24):

        date = times[i][:10]

        daily_values = aqi_values[i:i+24]

        # Remove missing values
        daily_values = [value for value in daily_values if value is not None]

        # Skip days with no AQI data
        if len(daily_values) == 0:
            continue

        average_aqi = sum(daily_values) / len(daily_values)

        writer.writerow([date, round(average_aqi, 2)])

print("Historical AQI CSV created successfully!")