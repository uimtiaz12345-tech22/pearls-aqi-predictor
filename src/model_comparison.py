import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================================
# 1. LOAD DATA
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
# 2. FEATURES
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

y = data["Target_AQI"]


# =====================================================
# 3. CHRONOLOGICAL SPLIT
# =====================================================

split_index = int(
    len(data) * 0.8
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# =====================================================
# 4. RANDOM FOREST
# =====================================================

random_forest = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(
    X_train,
    y_train
)

rf_predictions = random_forest.predict(
    X_test
)


# =====================================================
# 5. RIDGE REGRESSION
# =====================================================

ridge = Ridge(
    alpha=1.0
)

ridge.fit(
    X_train,
    y_train
)

ridge_predictions = ridge.predict(
    X_test
)


# =====================================================
# 6. EVALUATION FUNCTION
# =====================================================

def evaluate_model(
    name,
    actual,
    predictions
):

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = mean_squared_error(
        actual,
        predictions
    ) ** 0.5

    r2 = r2_score(
        actual,
        predictions
    )

    print()
    print(
        "================================"
    )

    print(
        name
    )

    print(
        "================================"
    )

    print(
        "MAE:",
        round(mae, 2)
    )

    print(
        "RMSE:",
        round(rmse, 2)
    )

    print(
        "R²:",
        round(r2, 3)
    )

    return mae, rmse, r2


# =====================================================
# 7. COMPARE MODELS
# =====================================================

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)

ridge_results = evaluate_model(
    "Ridge Regression",
    y_test,
    ridge_predictions
)


# =====================================================
# 8. FINAL COMPARISON
# =====================================================

comparison = pd.DataFrame({

    "Model": [
        "Random Forest",
        "Ridge Regression"
    ],

    "MAE": [
        rf_results[0],
        ridge_results[0]
    ],

    "RMSE": [
        rf_results[1],
        ridge_results[1]
    ],

    "R2": [
        rf_results[2],
        ridge_results[2]
    ]

})


print()
print(
    "MODEL COMPARISON"
)

print()

print(
    comparison.to_string(
        index=False
    )
)