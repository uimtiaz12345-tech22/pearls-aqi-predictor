import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# =====================================================
# 1. LOAD HISTORICAL DATA
# =====================================================

data = pd.read_csv(
    "Data/forecast_training_data_v2.csv"
)

data["Date"] = pd.to_datetime(
    data["Date"]
)

data = data.sort_values(
    "Date"
).reset_index(drop=True)


# =====================================================
# 2. LOAD TRAINED RANDOM FOREST MODEL
# =====================================================

model = joblib.load(
    "models/aqi_forecast_model_v2.pkl"
)


# =====================================================
# 3. DEFINE MODEL FEATURES
# =====================================================

features = [
    "AQI_Lag_1",
    "AQI_Lag_2",
    "AQI_Lag_3",
    "PM2.5_Lag_1",
    "PM2.5_Lag_2",
    "PM10",
    "Temperature_Lag_1",
    "Humidity_Lag_1",
    "Max Temperature",
    "Min Temperature",
    "Rain",
    "Wind Speed",
    "Humidity",
    "Pressure",
    "CO",
    "NO2",
    "SO2",
    "O3",
    "Month",
    "Day of Year"
]


X = data[features]


# =====================================================
# 4. CREATE SHAP EXPLAINER
# =====================================================

explainer = shap.TreeExplainer(
    model
)


# =====================================================
# 5. CALCULATE SHAP VALUES
# =====================================================

shap_values = explainer(
    X
)


# =====================================================
# 6. DISPLAY GLOBAL FEATURE IMPORTANCE
# =====================================================

print()
print("======================================")
print("        SHAP FEATURE IMPORTANCE")
print("======================================")
print()

shap_importance = pd.DataFrame({

    "Feature": features,

    "Importance": abs(
        shap_values.values
    ).mean(axis=0)

})

shap_importance = (
    shap_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print(
    shap_importance.to_string(
        index=False
    )
)


# =====================================================
# 7. CREATE SHAP BAR CHART
# =====================================================

plt.figure()

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.title(
    "SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "Data/shap_feature_importance.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print()
print(
    "SHAP analysis completed successfully!"
)

print(
    "Chart saved to:"
)

print(
    "Data/shap_feature_importance.png"
)