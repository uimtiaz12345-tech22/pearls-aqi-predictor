import requests
import csv
import os
from datetime import datetime

API_KEY = "e5e0c03a2b47de2811e46326531a7565"

CITY = "Karachi"
COUNTRY = "PK"

URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric"

response = requests.get(URL)
weather_data = response.json()

current_time = datetime.now()

file_exists = os.path.isfile("Data/weather_data.csv")

with open("Data/weather_data.csv", "a", newline="") as file:

    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "Timestamp",
            "City",
            "Temperature",
            "Humidity",
            "Pressure",
            "Wind Speed"
        ])

    writer.writerow([
        current_time,
        weather_data["name"],
        weather_data["main"]["temp"],
        weather_data["main"]["humidity"],
        weather_data["main"]["pressure"],
        weather_data["wind"]["speed"]
    ])

print("City:", weather_data["name"])
print("Temperature:", weather_data["main"]["temp"], "°C")
print("Humidity:", weather_data["main"]["humidity"], "%")
print("Pressure:", weather_data["main"]["pressure"], "hPa")
print("Wind Speed:", weather_data["wind"]["speed"], "m/s")