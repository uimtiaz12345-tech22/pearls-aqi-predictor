import csv

# -----------------------------
# Load weather data
# -----------------------------

weather = {}

with open("Data/historical_weather.csv", "r") as file:

    reader = csv.DictReader(file)

    for row in reader:
        weather[row["Date"]] = row


# -----------------------------
# Load pollution data
# -----------------------------

pollution = {}

with open("Data/historical_pollution.csv", "r") as file:

    reader = csv.DictReader(file)

    for row in reader:
        pollution[row["Date"]] = row


# -----------------------------
# Create combined dataset
# -----------------------------

training_data = []

with open("Data/historical_aqi.csv", "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        date = row["Date"]

        if date in weather and date in pollution:

            weather_row = weather[date]
            pollution_row = pollution[date]

            training_data.append([
                date,

                # Weather
                weather_row["Max Temperature"],
                weather_row["Min Temperature"],
                weather_row["Rain"],
                weather_row["Wind Speed"],
                weather_row["Humidity"],
                weather_row["Pressure"],

                # Pollution
                pollution_row["PM2.5"],
                pollution_row["PM10"],
                pollution_row["CO"],
                pollution_row["NO2"],
                pollution_row["SO2"],
                pollution_row["O3"],

                # Target
                row["Average AQI"]
            ])


# -----------------------------
# Save final training dataset
# -----------------------------

with open("Data/training_data.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Date",
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
        "AQI"
    ])

    writer.writerows(training_data)


print("Training dataset created successfully!")
print("Training rows:", len(training_data))