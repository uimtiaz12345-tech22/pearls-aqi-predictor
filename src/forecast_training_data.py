import pandas as pd

# Load our existing aligned dataset
data = pd.read_csv("Data/training_data.csv")

# Convert date
data["Date"] = pd.to_datetime(data["Date"])

# Sort chronologically
data = data.sort_values("Date").reset_index(drop=True)

# -----------------------------
# Create lag features
# -----------------------------

# Previous day's information
data["AQI_Lag_1"] = data["AQI"].shift(1)

data["PM2.5_Lag_1"] = data["PM2.5"].shift(1)
data["PM10_Lag_1"] = data["PM10"].shift(1)

data["Temperature_Lag_1"] = data["Max Temperature"].shift(1)
data["Humidity_Lag_1"] = data["Humidity"].shift(1)

# Information from two days ago
data["AQI_Lag_2"] = data["AQI"].shift(2)
data["PM2.5_Lag_2"] = data["PM2.5"].shift(2)

# Information from three days ago
data["AQI_Lag_3"] = data["AQI"].shift(3)
data["PM2.5_Lag_3"] = data["PM2.5"].shift(3)

# -----------------------------
# Create calendar features
# -----------------------------

data["Month"] = data["Date"].dt.month
data["Day of Year"] = data["Date"].dt.dayofyear

# Remove rows where lag information isn't available
data = data.dropna()

# Save forecasting dataset
data.to_csv(
    "Data/forecast_training_data.csv",
    index=False
)

print("Forecasting dataset created successfully!")
print("Rows:", len(data))
print("Columns:", len(data.columns))