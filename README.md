# AirSense Karachi — AI-Powered AQI Predictor

AirSense Karachi is an end-to-end machine learning system designed to forecast the Air Quality Index (AQI) in Karachi for the next three days.

## Project Objective

The objective of this project is to build an automated machine learning pipeline that collects environmental data, performs feature engineering, trains forecasting models, generates future AQI predictions, explains model predictions using SHAP, and presents the results through a web dashboard.

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Random Forest
- Ridge Regression
- SHAP
- Streamlit
- Flask
- Git
- GitHub Actions
- Open-Meteo Weather API
- Open-Meteo Air Quality API

## System Pipeline

External APIs
↓
Weather and Air Quality Data
↓
Historical Dataset
↓
Feature Engineering
↓
Machine Learning Training
↓
Model Evaluation
↓
Random Forest Model
↓
SHAP Explainability
↓
3-Day AQI Forecast
↓
Flask API
↓
Streamlit Dashboard
↓
GitHub Actions Automation

## Machine Learning Models

Two models were evaluated:

1. Random Forest Regressor
2. Ridge Regression

### Final Model

Random Forest Regressor was selected because it achieved better performance on the test dataset.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 7.57 | 10.47 | 0.729 |
| Ridge Regression | 7.92 | 10.61 | 0.722 |

The Random Forest model was trained using 992 training observations and evaluated using 249 testing observations.

## Model Features

The forecasting model uses:

- AQI lag features
- PM2.5 lag features
- PM10
- Carbon Monoxide (CO)
- Nitrogen Dioxide (NO2)
- Sulphur Dioxide (SO2)
- Ozone (O3)
- Temperature
- Humidity
- Atmospheric Pressure
- Wind Speed
- Rain
- Month
- Day of Year

## SHAP Explainability

SHAP is used to understand which variables contribute most strongly to the model's predictions.

The most influential features identified were:

1. CO
2. PM10
3. PM2.5 previous-day value
4. SO2
5. O3

The SHAP analysis improves transparency by showing which variables the model relies on most heavily.

## AQI Health Alerts

The dashboard classifies predicted AQI into health-related categories:

- Good
- Moderate
- Unhealthy for Sensitive Groups
- Unhealthy
- Very Unhealthy
- Hazardous

The dashboard displays an alert based on the predicted AQI level.

## Dashboard

The Streamlit dashboard provides:

- Current forecast information
- Three-day AQI forecast
- AQI health alerts
- Forecast trend
- Forecast details
- Model performance
- SHAP explainability
- Project architecture
- Model information

## Flask API

A Flask API is included to expose the AQI forecasting functionality through an HTTP endpoint.

## Automation

GitHub Actions is used to automate the forecasting pipeline.

The workflow is configured to:

1. Check out the repository
2. Set up Python
3. Install project dependencies
4. Run the AQI forecasting script
5. Update forecast results

The workflow can also be triggered manually.

## Project Structure

```text
AQI-Predictor/
│
├── .github/
│   └── workflows/
│       └── forecast.yml
│
├── Data/
│   ├── historical_aqi.csv
│   ├── historical_pollution.csv
│   ├── historical_weather.csv
│   ├── forecast_training_data.csv
│   ├── forecast_training_data_v2.csv
│   ├── forecast_results.csv
│   └── shap_feature_importance.png
│
├── models/
│   ├── aqi_forecast_model.pkl
│   ├── aqi_forecast_model_v2.pkl
│   ├── aqi_model.pkl
│   └── model_registry.json
│
├── reports/
│
├── src/
│   ├── app.py
│   ├── live_forecast.py
│   ├── flask_api.py
│   ├── model_comparison.py
│   ├── model_registry.py
│   ├── shap_analysis.py
│   ├── train_forecast_model_v2.py
│   └── other pipeline modules
│
├── requirements.txt
└── README.md