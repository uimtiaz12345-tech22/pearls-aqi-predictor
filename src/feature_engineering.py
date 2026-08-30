import csv
from datetime import datetime

training_data = []

with open("Data/training_data.csv", "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        date = datetime.strptime(row["Date"], "%Y-%m-%d")

        month = date.month

        day = date.day

        day_of_week = date.strftime("%A")

        training_data.append([
            row["Date"],
            month,
            day,
            day_of_week,
            row["Max Temperature"],
            row["Min Temperature"],
            row["Rain"],
            row["Wind Speed"],
            row["AQI"]
        ])

with open("Data/features.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Date",
        "Month",
        "Day",
        "DayOfWeek",
        "Max Temperature",
        "Min Temperature",
        "Rain",
        "Wind Speed",
        "AQI"
    ])

    writer.writerows(training_data)

print("Feature engineering completed successfully!")