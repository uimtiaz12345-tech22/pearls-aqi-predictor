import requests
import csv

LATITUDE = 24.8607
LONGITUDE = 67.0011

START_DATE = "2022-08-05"
END_DATE = "2025-12-31"

URL = (
    f"https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start_date={START_DATE}"
    f"&end_date={END_DATE}"
    f"&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,"
    f"sulphur_dioxide,ozone"
    f"&timezone=auto"
)

response = requests.get(URL)

print("API Status:", response.status_code)

pollution_data = response.json()

hourly = pollution_data["hourly"]

times = hourly["time"]
pm25 = hourly["pm2_5"]
pm10 = hourly["pm10"]
co = hourly["carbon_monoxide"]
no2 = hourly["nitrogen_dioxide"]
so2 = hourly["sulphur_dioxide"]
o3 = hourly["ozone"]

daily_data = {}

for i in range(len(times)):

    date = times[i][:10]

    if date not in daily_data:
        daily_data[date] = {
            "pm25": [],
            "pm10": [],
            "co": [],
            "no2": [],
            "so2": [],
            "o3": []
        }

    if pm25[i] is not None:
        daily_data[date]["pm25"].append(pm25[i])

    if pm10[i] is not None:
        daily_data[date]["pm10"].append(pm10[i])

    if co[i] is not None:
        daily_data[date]["co"].append(co[i])

    if no2[i] is not None:
        daily_data[date]["no2"].append(no2[i])

    if so2[i] is not None:
        daily_data[date]["so2"].append(so2[i])

    if o3[i] is not None:
        daily_data[date]["o3"].append(o3[i])


with open("Data/historical_pollution.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Date",
        "PM2.5",
        "PM10",
        "CO",
        "NO2",
        "SO2",
        "O3"
    ])

    for date in sorted(daily_data.keys()):

        data = daily_data[date]

        # Only create a row when all pollutant measurements exist
        if all([
            data["pm25"],
            data["pm10"],
            data["co"],
            data["no2"],
            data["so2"],
            data["o3"]
        ]):

            writer.writerow([
                date,
                round(sum(data["pm25"]) / len(data["pm25"]), 2),
                round(sum(data["pm10"]) / len(data["pm10"]), 2),
                round(sum(data["co"]) / len(data["co"]), 2),
                round(sum(data["no2"]) / len(data["no2"]), 2),
                round(sum(data["so2"]) / len(data["so2"]), 2),
                round(sum(data["o3"]) / len(data["o3"]), 2)
            ])

print("Historical pollution CSV created successfully!")
print("Days available:", len(daily_data))