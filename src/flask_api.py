from flask import Flask, jsonify
import pandas as pd
import os


# ============================================================
# AIRSENSE FLASK API
# ============================================================

app = Flask(__name__)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "AirSense Karachi AQI Prediction API"
    })


# ============================================================
# FORECAST ENDPOINT
# ============================================================

@app.route("/forecast")
def forecast():

    file_path = "Data/forecast_results.csv"

    # Check whether forecast exists
    if not os.path.exists(file_path):

        return jsonify({
            "error": "Forecast data not found. Run live_forecast.py first."
        }), 404


    # Load forecast
    data = pd.read_csv(file_path)


    # Convert dataframe into records
    results = data.to_dict(
        orient="records"
    )


    return jsonify({
        "location": "Karachi, Pakistan",
        "forecast_days": len(results),
        "forecast": results
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("       AIRSENSE FLASK API")
    print("======================================")
    print()

    print(
        "API running at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    app.run(
        debug=True,
        port=5000
    )