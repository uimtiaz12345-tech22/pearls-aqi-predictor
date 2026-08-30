import requests
import csv
LATITUDE = 24.8607
LONGITUDE = 67.0011

START_DATE = "2024-01-01"
END_DATE = "2024-01-10"
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

hourly = aqi_data["hourly"]

print(hourly["time"][:5])

print(hourly["us_aqi"][:5])