import pandas as pd


# --------------------------------
# 1. Load our existing dataset
# --------------------------------

data = pd.read_csv("Data/forecast_training_data.csv")


# --------------------------------
# 2. Convert Date to datetime
# --------------------------------

data["Date"] = pd.to_datetime(data["Date"])


# --------------------------------
# 3. Sort chronologically
# --------------------------------

data = data.sort_values("Date").reset_index(drop=True)


# --------------------------------
# 4. Create tomorrow's AQI
#    as our prediction target
# --------------------------------

data["Target_AQI"] = data["AQI"].shift(-1)


# --------------------------------
# 5. Remove final row
#    because it has no tomorrow
# --------------------------------

data = data.dropna(subset=["Target_AQI"])


# --------------------------------
# 6. Save the new dataset
# --------------------------------

data.to_csv(
    "Data/forecast_training_data_v2.csv",
    index=False
)


# --------------------------------
# 7. Display results
# --------------------------------

print("3-day forecasting dataset created successfully!")

print("Rows:", len(data))
print("Columns:", len(data.columns))

print()

print(
    data[
        ["Date", "AQI", "Target_AQI"]
    ].head()
)