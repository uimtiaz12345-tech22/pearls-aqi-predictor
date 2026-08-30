import json
import os
from datetime import datetime


# ============================================================
# AIRSENSE MODEL REGISTRY
# ============================================================

MODEL_PATH = "models/aqi_forecast_model_v2.pkl"
REGISTRY_PATH = "models/model_registry.json"


# ============================================================
# MODEL INFORMATION
# ============================================================

model_information = {
    "model_name": "AirSense AQI Forecast Model",

    "version": "v2",

    "algorithm": "Random Forest Regressor",

    "model_file": MODEL_PATH,

    "training_samples": 992,

    "testing_samples": 249,

    "MAE": 7.57,

    "RMSE": 10.47,

    "R2": 0.729,

    "forecast_horizon": "3 Days",

    "registered_at": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
}


# ============================================================
# CREATE REGISTRY FOLDER IF NEEDED
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

with open(
    REGISTRY_PATH,
    "w"
) as file:

    json.dump(
        model_information,
        file,
        indent=4
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("======================================")
print("       MODEL REGISTRY")
print("======================================")
print()

for key, value in model_information.items():

    print(
        f"{key}: {value}"
    )

print()
print(
    "Model successfully registered!"
)

print(
    f"Registry saved to: {REGISTRY_PATH}"
)