import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import sys

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

FORECAST_FILE = os.path.join(DATA_DIR, "forecast_results.csv")
HISTORICAL_FILE = os.path.join(DATA_DIR, "historical_aqi.csv")
SHAP_FILE = os.path.join(DATA_DIR, "shap_feature_importance.png")
LIVE_FORECAST_FILE = os.path.join(SRC_DIR, "live_forecast.py")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AirSense Karachi",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.main-title {
    font-size: 42px;
    font-weight: 750;
    margin-bottom: 0px;
}

.main-subtitle {
    font-size: 17px;
    color: #6b7280;
    margin-bottom: 25px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}

.small-text {
    color: #6b7280;
    font-size: 14px;
}

.forecast-card {
    padding: 20px;
    border: 1px solid #d9dee7;
    border-radius: 12px;
    background-color: #ffffff;
    min-height: 190px;
}

.forecast-day {
    font-size: 20px;
    font-weight: 650;
}

.forecast-date {
    color: #6b7280;
    font-size: 14px;
}

.forecast-aqi {
    font-size: 38px;
    font-weight: 750;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# AQI FUNCTIONS
# ============================================================

def aqi_category(aqi):

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def aqi_emoji(aqi):

    if aqi <= 50:
        return "🟢"
    elif aqi <= 100:
        return "🟡"
    elif aqi <= 150:
        return "🟠"
    elif aqi <= 200:
        return "🔴"
    elif aqi <= 300:
        return "🟣"
    else:
        return "⚫"


def health_message(aqi):

    if aqi <= 50:
        return (
            "Air quality is good. Air pollution poses little "
            "or no risk."
        )

    elif aqi <= 100:
        return (
            "Air quality is acceptable. Sensitive individuals "
            "may experience minor effects."
        )

    elif aqi <= 150:
        return (
            "Sensitive groups should consider reducing prolonged "
            "or heavy outdoor activity."
        )

    elif aqi <= 200:
        return (
            "Everyone may begin to experience health effects. "
            "Sensitive groups may experience more serious effects."
        )

    elif aqi <= 300:
        return (
            "Health alert: the risk of health effects is increased "
            "for everyone."
        )

    else:
        return (
            "Health emergency: hazardous air quality conditions "
            "are expected."
        )


def show_aqi_alert(aqi):

    category = aqi_category(aqi)
    emoji = aqi_emoji(aqi)
    message = health_message(aqi)

    text = (
        f"**{emoji} {category} — Forecast AQI: {aqi:.0f}**\n\n"
        f"{message}"
    )

    if aqi <= 100:
        st.success(text)
    elif aqi <= 150:
        st.warning(text)
    else:
        st.error(text)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌤️ AirSense Karachi</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-powered 3-day Air Quality Index forecasting system'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌤️ AirSense Karachi")
    st.markdown("---")

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "Dashboard",
            "Model Performance",
            "Model Explainability",
            "Project Information"
        ]
    )

    st.markdown("---")

    st.markdown("### System")

    st.write("📍 Location: Karachi, Pakistan")
    st.write("🔮 Forecast: Next 3 Days")
    st.write("🤖 Model: Random Forest")
    st.write("🧠 Explainability: SHAP")

    st.markdown("---")

    # ========================================================
    # LIVE REFRESH
    # ========================================================

    st.markdown("### 🔄 Live Forecast")

    if st.button(
        "Refresh Forecast",
        use_container_width=True
    ):

        with st.spinner(
            "Collecting latest weather and air-quality data..."
        ):

            try:

                result = subprocess.run(
                    [
                        sys.executable,
                        LIVE_FORECAST_FILE
                    ],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:

                    st.success(
                        "Forecast updated successfully."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Forecast update failed."
                    )

                    st.code(
                        result.stderr
                    )

            except Exception as e:

                st.error(
                    f"Could not run live forecast: {e}"
                )


# ============================================================
# LOAD FORECAST
# ============================================================

if not os.path.exists(FORECAST_FILE):

    st.error(
        "Forecast results were not found."
    )

    st.info(
        "Run live_forecast.py first or use the Refresh Forecast "
        "button in the sidebar."
    )

    st.stop()


forecast = pd.read_csv(
    FORECAST_FILE
)

forecast["Date"] = pd.to_datetime(
    forecast["Date"],
    errors="coerce"
)

forecast["Predicted AQI"] = pd.to_numeric(
    forecast["Predicted AQI"],
    errors="coerce"
)

forecast = forecast.dropna(
    subset=[
        "Date",
        "Predicted AQI"
    ]
)

forecast = forecast.sort_values(
    "Date"
).reset_index(drop=True)

# ============================================================
# CRITICAL DATE LOGIC
# ============================================================
# Never hardcode dates.
# Always use the next three dates contained in forecast_results.

today = pd.Timestamp.now(
    tz="Asia/Karachi"
).normalize().tz_localize(None)

future_forecast = forecast[
    forecast["Date"] > today
].copy()

future_forecast = future_forecast.head(3)

# ============================================================
# SAFETY FALLBACK
# ============================================================

if len(future_forecast) < 3:

    # If the CSV already contains exactly 3 forecast rows,
    # use them rather than showing an incomplete dashboard.

    if len(forecast) >= 3:

        future_forecast = forecast.head(3).copy()

    else:

        st.error(
            "Less than three forecast days are available."
        )

        st.stop()


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">'
        '🌫️ Air Quality Overview'
        '</div>',
        unsafe_allow_html=True
    )

    tomorrow_aqi = future_forecast.iloc[0]["Predicted AQI"]
    tomorrow_date = future_forecast.iloc[0]["Date"]

    worst_aqi = future_forecast[
        "Predicted AQI"
    ].max()

    worst_row = future_forecast.loc[
        future_forecast["Predicted AQI"].idxmax()
    ]

    latest_forecast_date = future_forecast[
        "Date"
    ].max()

    # ========================================================
    # TOP METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Tomorrow's AQI",
            f"{tomorrow_aqi:.0f}"
        )

        st.caption(
            f"{aqi_emoji(tomorrow_aqi)} "
            f"{aqi_category(tomorrow_aqi)}"
        )

    with col2:

        st.metric(
            "Forecast Horizon",
            "3 Days"
        )

        st.caption(
            "Automatic daily predictions"
        )

    with col3:

        st.metric(
            "Location",
            "Karachi"
        )

        st.caption(
            "Pakistan"
        )

    with col4:

        st.metric(
            "Forecast Through",
            latest_forecast_date.strftime(
                "%d %b"
            )
        )

        st.caption(
            "Latest predicted date"
        )

    # ========================================================
    # LIVE STATUS
    # ========================================================

    st.info(
        f"📅 Forecast generated for "
        f"{future_forecast.iloc[0]['Date'].strftime('%d %b %Y')} "
        f"to "
        f"{future_forecast.iloc[-1]['Date'].strftime('%d %b %Y')}."
    )

    # ========================================================
    # HEALTH ALERT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🚨 Air Quality Health Alert'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"The highest predicted AQI during the forecast "
        f"period is **{worst_aqi:.0f}**, expected on "
        f"**{worst_row['Date'].strftime('%d %b %Y')}**."
    )

    show_aqi_alert(
        worst_aqi
    )

    # ========================================================
    # 3-DAY FORECAST
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📅 3-Day AQI Forecast'
        '</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    for i, col in enumerate(cols):

        row = future_forecast.iloc[i]

        date = row["Date"]
        aqi = row["Predicted AQI"]

        if i == 0:
            day_label = "Tomorrow"
        else:
            day_label = f"Day +{i + 1}"

        with col:

            st.markdown(
                f"""
                <div class="forecast-card">

                <div class="forecast-day">
                    {day_label}
                </div>

                <div class="forecast-date">
                    {date.strftime("%d %B %Y")}
                </div>

                <div class="forecast-aqi">
                    {aqi:.0f}
                </div>

                <div>
                    {aqi_emoji(aqi)}
                    <b>{aqi_category(aqi)}</b>
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    # ========================================================
    # FORECAST TREND
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📈 AQI Forecast Trend'
        '</div>',
        unsafe_allow_html=True
    )

    trend_data = future_forecast[
        ["Date", "Predicted AQI"]
    ].copy()

    trend_data["Date"] = trend_data[
        "Date"
    ].dt.strftime("%d %b")

    trend_data = trend_data.set_index(
        "Date"
    )

    st.line_chart(
        trend_data,
        use_container_width=True
    )

    # ========================================================
    # FORECAST DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Forecast Details'
        '</div>',
        unsafe_allow_html=True
    )

    display_data = future_forecast.copy()

    display_data["Forecast Date"] = display_data[
        "Date"
    ].dt.strftime(
        "%d %b %Y"
    )

    display_data["Predicted AQI"] = display_data[
        "Predicted AQI"
    ].round(2)

    display_data["AQI Status"] = display_data[
        "Predicted AQI"
    ].apply(
        aqi_emoji
    )

    display_data["Category"] = display_data[
        "Predicted AQI"
    ].apply(
        aqi_category
    )

    st.dataframe(
        display_data[
            [
                "Forecast Date",
                "Predicted AQI",
                "AQI Status",
                "Category"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # HISTORICAL AQI
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Historical Air Quality Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    if os.path.exists(HISTORICAL_FILE):

        try:

            historical = pd.read_csv(
                HISTORICAL_FILE
            )

            historical["Date"] = pd.to_datetime(
                historical["Date"],
                errors="coerce"
            )

            # Find AQI column robustly
            aqi_column = None

            for possible_column in [
                "AQI",
                "aqi",
                "Air Quality Index"
            ]:

                if possible_column in historical.columns:

                    aqi_column = possible_column
                    break

            if aqi_column is not None:

                historical[aqi_column] = pd.to_numeric(
                    historical[aqi_column],
                    errors="coerce"
                )

                historical = historical.dropna(
                    subset=[
                        "Date",
                        aqi_column
                    ]
                )

                historical_chart = historical[
                    ["Date", aqi_column]
                ].copy()

                historical_chart = historical_chart.set_index(
                    "Date"
                )

                historical_chart.columns = [
                    "AQI"
                ]

                st.line_chart(
                    historical_chart,
                    use_container_width=True
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Average Historical AQI",
                        f"{historical[aqi_column].mean():.1f}"
                    )

                with c2:

                    st.metric(
                        "Maximum Historical AQI",
                        f"{historical[aqi_column].max():.1f}"
                    )

                with c3:

                    st.metric(
                        "Minimum Historical AQI",
                        f"{historical[aqi_column].min():.1f}"
                    )

            else:

                st.warning(
                    "AQI column could not be identified "
                    "in historical_aqi.csv."
                )

        except Exception as e:

            st.warning(
                f"Historical analysis could not be loaded: {e}"
            )

    else:

        st.warning(
            "Historical AQI dataset was not found."
        )

    # ========================================================
    # HOW FORECAST WORKS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔄 How This Forecast Works'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        **1. Live environmental data** is collected from weather
        and air-quality APIs.

        **2. Historical pollution patterns** are combined with
        weather and temporal variables.

        **3. Feature engineering** creates lag variables such as
        previous-day AQI and PM2.5.

        **4. Random Forest** uses these features to estimate
        future AQI.

        **5. The model generates predictions for the next three
        available future dates.**

        **6. SHAP analysis explains which variables have the
        greatest influence on the model's predictions.**
        """
    )


# ============================================================
# PAGE 2 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">'
        '🤖 Model Performance & Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Two machine-learning models were evaluated using "
        "the same chronological train/test methodology."
    )

    # ========================================================
    # PERFORMANCE DATA
    # ========================================================

    performance = pd.DataFrame({

        "Model": [
            "Random Forest",
            "Ridge Regression"
        ],

        "MAE": [
            7.57,
            7.92
        ],

        "RMSE": [
            10.47,
            10.61
        ],

        "R²": [
            0.729,
            0.722
        ]

    })

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # KEY METRICS
    # ========================================================

    st.markdown(
        "### 🏆 Final Random Forest Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            "7.57 AQI points"
        )

    with col2:

        st.metric(
            "RMSE",
            "10.47 AQI points"
        )

    with col3:

        st.metric(
            "R²",
            "72.9%"
        )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.markdown(
        "### 📊 Model Comparison"
    )

    st.bar_chart(
        performance.set_index(
            "Model"
        )[[
            "MAE",
            "RMSE"
        ]]
    )

    # ========================================================
    # SELECTED MODEL
    # ========================================================

    st.markdown(
        "### 🥇 Selected Model"
    )

    st.success(
        """
        **Random Forest Regressor**

        Random Forest was selected as the final forecasting
        model because it achieved:

        • Lower MAE than Ridge Regression  
        • Lower RMSE than Ridge Regression  
        • Higher R² than Ridge Regression  

        Therefore, Random Forest provides the stronger
        predictive performance for this dataset.
        """
    )

    # ========================================================
    # METRIC EXPLANATION
    # ========================================================

    st.markdown(
        "### 📖 Understanding the Metrics"
    )

    st.write(
        """
        **MAE — Mean Absolute Error**

        Measures the average absolute difference between the
        predicted AQI and actual AQI.

        A value of **7.57** means the model's predictions are
        off by approximately 7.57 AQI points on average.
        """
    )

    st.write(
        """
        **RMSE — Root Mean Squared Error**

        Similar to MAE but penalizes larger errors more heavily.

        The model achieved an RMSE of **10.47**.
        """
    )

    st.write(
        """
        **R² — R-squared**

        Indicates how much variation in AQI is explained by
        the model.

        An R² of **0.729** means approximately **72.9% of the
        variation in the test data is explained by the model**.
        """
    )

    # ========================================================
    # TRAINING INFORMATION
    # ========================================================

    st.markdown(
        "### 📚 Training Information"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Training Samples",
            "992"
        )

    with c2:

        st.metric(
            "Testing Samples",
            "249"
        )

    with c3:

        st.metric(
            "Features",
            "20"
        )


# ============================================================
# PAGE 3 — SHAP
# ============================================================

elif page == "Model Explainability":

    st.markdown(
        '<div class="section-title">'
        '🧠 Model Explainability — SHAP Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        SHAP (SHapley Additive exPlanations) is used to
        understand which input variables have the greatest
        influence on the Random Forest model's AQI predictions.
        """
    )

    # ========================================================
    # SHAP IMAGE
    # ========================================================

    st.markdown(
        "### 📊 SHAP Feature Importance"
    )

    if os.path.exists(SHAP_FILE):

        st.image(
            SHAP_FILE,
            use_container_width=True
        )

    else:

        st.warning(
            "SHAP feature importance chart was not found."
        )

        st.info(
            "Run shap_analysis.py first."
        )

    # ========================================================
    # SHAP DATA
    # ========================================================

    shap_data = pd.DataFrame({

        "Feature": [

            "CO",
            "PM10",
            "PM2.5_Lag_1",
            "SO2",
            "O3",
            "NO2",
            "Day of Year",
            "Pressure",
            "Humidity",
            "Temperature_Lag_1",
            "Humidity_Lag_1",
            "AQI_Lag_1",
            "AQI_Lag_3",
            "Wind Speed",
            "Min Temperature",
            "Max Temperature",
            "AQI_Lag_2",
            "PM2.5_Lag_2",
            "Month",
            "Rain"

        ],

        "SHAP Importance": [

            9.981722,
            9.109949,
            4.742373,
            4.725060,
            2.401078,
            0.635882,
            0.627777,
            0.588087,
            0.578203,
            0.417039,
            0.284867,
            0.255873,
            0.253338,
            0.226564,
            0.183873,
            0.181801,
            0.170939,
            0.167632,
            0.122939,
            0.060487

        ]

    })

    # ========================================================
    # TOP SHAP FEATURES
    # ========================================================

    st.markdown(
        "### 🔍 Top Model Drivers"
    )

    top_shap = shap_data.head(10)

    st.dataframe(
        top_shap,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # SHAP BAR CHART
    # ========================================================

    st.markdown(
        "### 📈 SHAP Importance Ranking"
    )

    shap_chart = shap_data.head(10).copy()

    shap_chart = shap_chart.set_index(
        "Feature"
    )

    st.bar_chart(
        shap_chart[
            "SHAP Importance"
        ]
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown(
        "### 🧠 Interpretation"
    )

    st.success(
        """
        The SHAP analysis shows that **CO, PM10,
        previous-day PM2.5 and SO2** are among the most
        influential variables used by the forecasting model.

        This provides transparency into the machine-learning
        model instead of treating it as a complete black box.
        """
    )

    st.warning(
        """
        **Important:** SHAP importance indicates model influence,
        not causation. A high SHAP importance does not by itself
        prove that changing that variable causes AQI to change.
        """
    )

    # ========================================================
    # FEATURE GROUPS
    # ========================================================

    st.markdown(
        "### 🌫️ Environmental Variables"
    )

    st.write(
        """
        The strongest contributors include:

        • **CO** — Carbon monoxide  
        • **PM10** — Particulate matter  
        • **PM2.5_Lag_1** — Previous-day PM2.5  
        • **SO2** — Sulphur dioxide  
        • **O3** — Ozone  
        • **NO2** — Nitrogen dioxide
        """
    )


# ============================================================
# PAGE 4 — PROJECT INFORMATION
# ============================================================

elif page == "Project Information":

    st.markdown(
        '<div class="section-title">'
        '📘 AirSense Karachi — Project Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        AirSense Karachi is a machine-learning based air-quality
        forecasting system designed to predict Karachi's Air
        Quality Index for the next three days.

        The system combines historical air-quality patterns,
        pollution variables, weather conditions and temporal
        information to generate automated AQI forecasts.
        """
    )

    # ========================================================
    # ARCHITECTURE
    # ========================================================

    st.markdown(
        "### 🔄 System Architecture"
    )

    st.code(
        """
Weather API
     +
Air Quality API
     ↓
Raw Environmental Data
     ↓
Feature Engineering
     ↓
Historical Training Dataset
     ↓
Train/Test Split
     ↓
Machine Learning Models
     ├── Random Forest
     └── Ridge Regression
     ↓
Model Evaluation
     ↓
Random Forest Selected
     ↓
SHAP Explainability
     ↓
Live 3-Day Forecast
     ↓
Streamlit Dashboard
        """,
        language="text"
    )

    # ========================================================
    # FEATURES
    # ========================================================

    st.markdown(
        "### ⚙️ Model Features"
    )

    feature_list = [

        "AQI Lag 1",
        "AQI Lag 2",
        "AQI Lag 3",
        "PM2.5 Lag 1",
        "PM2.5 Lag 2",
        "PM10",
        "Temperature Lag 1",
        "Humidity Lag 1",
        "Maximum Temperature",
        "Minimum Temperature",
        "Rain",
        "Wind Speed",
        "Humidity",
        "Atmospheric Pressure",
        "Carbon Monoxide (CO)",
        "Nitrogen Dioxide (NO2)",
        "Sulphur Dioxide (SO2)",
        "Ozone (O3)",
        "Month",
        "Day of Year"

    ]

    for feature in feature_list:

        st.write(
            f"• {feature}"
        )

    # ========================================================
    # PROJECT COMPONENTS
    # ========================================================

    st.markdown(
        "### ✅ Project Components"
    )

    components = pd.DataFrame({

        "Component": [

            "Historical Data Collection",
            "Feature Engineering",
            "Forecast Training Dataset",
            "Random Forest Model",
            "Ridge Regression Model",
            "Model Evaluation",
            "Live Weather API",
            "Live Air Quality API",
            "Automatic 3-Day Forecast",
            "SHAP Explainability",
            "AQI Health Alerts",
            "Streamlit Dashboard"

        ],

        "Status": [

            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed"

        ]

    })

    st.dataframe(
        components,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # LIMITATIONS
    # ========================================================

    st.markdown(
        "### ⚠️ Current Limitations"
    )

    limitations = [

        "Forecast accuracy depends on the quality and availability of external API data.",

        "Machine-learning predictions are estimates and should not be treated as official air-quality measurements.",

        "Recursive forecasting can accumulate prediction errors across multiple days.",

        "The current model uses Random Forest and Ridge Regression rather than deep-learning forecasting models.",

        "API availability can affect live forecast generation.",

        "SHAP explains model behaviour but does not establish causal relationships."

    ]

    for item in limitations:

        st.write(
            f"• {item}"
        )

    # ========================================================
    # FUTURE IMPROVEMENTS
    # ========================================================

    st.markdown(
        "### 🚀 Future Improvements"
    )

    improvements = [

        "Automated hourly forecasting",

        "Automated model retraining",

        "LSTM / GRU time-series models",

        "Additional weather variables",

        "More historical observations",

        "Cloud deployment",

        "Model monitoring",

        "Automated model registry",

        "Real-time notifications",

        "Mobile-friendly deployment"

    ]

    for item in improvements:

        st.write(
            f"• {item}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <b>AirSense Karachi</b><br>

    AI-powered Air Quality Index Forecasting System<br><br>

    Python • Pandas • Scikit-learn • Random Forest • SHAP • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)