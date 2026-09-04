import os
import sys
import subprocess
import html
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def render_html(content: str):
    """Render an HTML string, robustly.

    st.markdown(..., unsafe_allow_html=True) runs the text through
    Streamlit's CommonMark-based markdown parser first, which treats
    any line indented 4+ spaces as a literal code block instead of
    HTML -- that's what was showing raw <div> tags on screen. st.html()
    (available in modern Streamlit) skips the markdown parser and
    injects the HTML directly, so indentation can't break it. We still
    dedent the string as a safety net, and fall back to st.markdown for
    very old Streamlit versions that lack st.html().
    """
    dedented = textwrap.dedent(content)
    try:
        st.html(dedented)
    except AttributeError:
        st.markdown(dedented, unsafe_allow_html=True)


def render_svg_chart(chart_html: str, height: int = 360):
    """
    Renders chart HTML (produced by svg_line_chart) through an
    unsanitized iframe instead of st.html().

    st.html() sanitizes everything with DOMPurify before injecting it
    into the page. Streamlit's DOMPurify configuration strips raw SVG
    elements (<svg>, <line>, <circle>, <polyline>, <text>) even though
    it keeps plain <div>/<span> markup -- that's why a chart-card's
    title/subtitle would show up but the graph itself never appeared.
    components.html() renders in an iframe with no sanitization, so
    the SVG survives. `height` should be a bit larger than the SVG's
    own `height` argument to leave room for the title/subtitle and
    card padding (roughly svg height + 60px).
    """
    components.html(textwrap.dedent(chart_html), height=height, scrolling=False)


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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AirSense Karachi",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL DARK THEME
# ============================================================

render_html(
    """
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background:
        radial-gradient(circle at 12% 0%, rgba(34,211,238,0.07), transparent 40%),
        radial-gradient(circle at 90% 15%, rgba(59,130,246,0.06), transparent 45%),
        #080b12;
    color: #f5f7fb;
}

* {
    font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.0rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2.0rem !important;
    padding-right: 2.0rem !important;
}

/* Remove ugly Streamlit top header */

header[data-testid="stHeader"] {
    background: #080b12 !important;
    height: 0rem !important;
}
/* ---------- FORCE SIDEBAR EXPAND/COLLAPSE CONTROL ---------- */

button[data-testid="stBaseButton-headerNoPadding"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

button[data-testid="stBaseButton-headerNoPadding"] svg {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

footer {
    visibility: hidden;
}

/* ==========================================================
   SIDEBAR
   ========================================================== */

/* ---------- SIDEBAR — FORCE VISIBLE ---------- */

section[data-testid="stSidebar"] {
    background: #0a0d14 !important;
    border-right: 1px solid #1d2330 !important;

    width: 255px !important;
    min-width: 255px !important;
    max-width: 255px !important;

    visibility: visible !important;
    opacity: 1 !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
}

/* Force the sidebar content itself to remain visible */
section[data-testid="stSidebar"] > div {
    width: 255px !important;
    min-width: 255px !important;
    max-width: 255px !important;

    visibility: visible !important;
    opacity: 1 !important;
}

/* Sidebar inner content */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    width: 255px !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Keep sidebar text visible */
section[data-testid="stSidebar"] * {
    color: #dbe3ef !important;
}

/* Brand */
.sidebar-brand {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff !important;
    margin-bottom: 2px;
}

.sidebar-subtitle {
    color: #69768a !important;
    font-size: 11px;
    margin-bottom: 18px;
}

section[data-testid="stSidebar"] > div {
    padding: 1.25rem 1rem !important;
}

section[data-testid="stSidebar"] * {
    color: #dce3ee;
}

.sidebar-brand {
    font-size: 20px;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff 0%, #7fd8ea 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 3px;
}

.sidebar-subtitle {
    font-size: 11px;
    color: #68758a !important;
    margin-bottom: 22px;
}

.sidebar-section {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #657187 !important;
    margin-top: 20px;
    margin-bottom: 8px;
}

/* Navigation */

div[data-testid="stRadio"] > label {
    display: none;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 5px;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label {
    border-radius: 8px;
    padding: 8px 10px;
    margin: 0;
    background: transparent;
    border: 1px solid transparent;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: #121823;
    border-color: #202b3b;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
    background: #172131;
    border-color: #29476c;
}

/* Hide radio circles */

div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none;
}

/* Sidebar button */

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: #123f73;
    color: white;
    border: 1px solid #285c91;
    border-radius: 8px;
    font-weight: 700;
    min-height: 38px;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #18548f;
}

/* ==========================================================
   TITLES
   ========================================================== */

.main-title {
    color: #ffffff;
    font-size: 34px;
    line-height: 1.1;
    font-weight: 850;
    margin: 0;
}

.main-subtitle {
    color: #78869b;
    font-size: 14px;
    margin-top: 6px;
    margin-bottom: 22px;
}

.section-title {
    color: #f5f7fb;
    font-size: 20px;
    font-weight: 800;
    margin-top: 24px;
    margin-bottom: 11px;
    padding-left: 11px;
    border-left: 3px solid #22d3ee;
}

/* ==========================================================
   GENERAL CARDS
   ========================================================== */

.card {
    background: linear-gradient(165deg, #171c27 0%, #12151d 100%);
    border: 1px solid #252c39;
    border-radius: 13px;
    padding: 18px;
    box-sizing: border-box;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
}

.card:hover {
    border-color: #355d8c;
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.35);
}

.card-label {
    color: #7e8a9e;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.card-title {
    color: #ffffff;
    font-size: 15px;
    font-weight: 750;
}

.big-value {
    background: linear-gradient(90deg, #ffffff 0%, #b7c3d6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 34px;
    font-weight: 850;
    line-height: 1;
    margin-top: 8px;
}

.card-note {
    color: #748197;
    font-size: 11px;
    margin-top: 8px;
}

/* ==========================================================
   AQI HERO
   ========================================================== */

.aqi-hero {
    background: linear-gradient(165deg, #171c27 0%, #10131b 100%);
    border: 1px solid #252c39;
    border-radius: 14px;
    padding: 22px 20px 20px 20px;
    text-align: center;
    min-height: 310px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

.aqi-hero-title {
    color: #ffffff;
    font-size: 19px;
    font-weight: 800;
}

.aqi-circle {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    margin: 18px auto 13px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 7px solid rgba(255,255,255,0.05);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.04), 0 0 40px 4px currentColor;
    opacity: 0.97;
}

.aqi-number {
    color: #ffffff;
    font-size: 54px;
    line-height: 1;
    font-weight: 900;
}

.aqi-status {
    color: #ffffff;
    font-size: 18px;
    font-weight: 800;
}

.aqi-date {
    color: #748197;
    font-size: 11px;
    margin-top: 5px;
}

/* ==========================================================
   FORECAST
   ========================================================== */

.forecast-card {
    background: linear-gradient(165deg, #171c27 0%, #12151d 100%);
    border: 1px solid #252c39;
    border-radius: 13px;
    padding: 18px;
    min-height: 155px;
    transition: 0.18s ease;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

.forecast-card:hover {
    border-color: #355d8c;
    transform: translateY(-3px);
    box-shadow: 0 12px 26px rgba(0,0,0,0.4);
}

.forecast-day {
    color: #ffffff;
    font-size: 16px;
    font-weight: 800;
}

.forecast-date {
    color: #718097;
    font-size: 11px;
    margin-top: 3px;
}

.forecast-aqi {
    color: #ffffff;
    font-size: 35px;
    line-height: 1;
    font-weight: 900;
    margin-top: 17px;
}

.forecast-category {
    color: #aeb8c7;
    font-size: 12px;
    margin-top: 7px;
}

/* ==========================================================
   ALERT
   ========================================================== */

.alert-card {
    background: linear-gradient(165deg, #1a1520 0%, #12151d 100%);
    border: 1px solid #2c2739;
    border-radius: 13px;
    padding: 19px;
    min-height: 155px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

.alert-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 800;
}

.alert-text {
    color: #aeb8c7;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 10px;
}

/* ==========================================================
   CHART
   ========================================================== */

.chart-card {
    background: linear-gradient(165deg, #171c27 0%, #12151d 100%);
    border: 1px solid #252c39;
    border-radius: 13px;
    padding: 15px 15px 8px 15px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

.chart-title {
    color: #ffffff;
    font-size: 15px;
    font-weight: 800;
}

.chart-subtitle {
    color: #718097;
    font-size: 11px;
    margin-top: 3px;
}

/* ==========================================================
   TABLES
   ========================================================== */

div[data-testid="stDataFrame"] {
    border: 1px solid #252c39;
    border-radius: 10px;
    overflow: hidden;
}

/* ==========================================================
   EXPANDERS
   ========================================================== */

div[data-testid="stExpander"] {
    background: #11151e;
    border: 1px solid #252c39;
    border-radius: 10px;
}

/* ==========================================================
   INFO / SUCCESS / ERROR
   ========================================================== */

div[data-testid="stAlert"] {
    border-radius: 9px;
}

/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 900px) {

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    section[data-testid="stSidebar"] {
        min-width: 220px !important;
        max-width: 220px !important;
    }

    .main-title {
        font-size: 28px;
    }

}

</style>
"""
)


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
    return "Hazardous"


def aqi_color(aqi):
    if aqi <= 50:
        return "#10b981"
    elif aqi <= 100:
        return "#eab308"
    elif aqi <= 150:
        return "#f97316"
    elif aqi <= 200:
        return "#ef4444"
    elif aqi <= 300:
        return "#8b5cf6"
    return "#111827"


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
    return "⚫"


def health_message(aqi):
    if aqi <= 50:
        return "Air quality is good and poses little or no health risk."
    elif aqi <= 100:
        return "Air quality is acceptable. Sensitive individuals may experience minor effects."
    elif aqi <= 150:
        return "Sensitive groups should consider reducing prolonged or heavy outdoor activity."
    elif aqi <= 200:
        return "Everyone may begin to experience health effects."
    elif aqi <= 300:
        return "Health alert: the risk of health effects is increased for everyone."
    return "Health emergency: hazardous air-quality conditions are expected."


def find_numeric_column(df, names):

    for name in names:

        if name in df.columns:

            values = pd.to_numeric(
                df[name],
                errors="coerce"
            )

            if values.notna().any():
                return name

    return None


def fmt_value(value, suffix=""):

    if value is None:
        return "—"

    try:

        if pd.isna(value):
            return "—"

        return f"{float(value):.1f}{suffix}"

    except Exception:
        return "—"


def safe_text(value):
    return html.escape(str(value))


# ============================================================
# PROFESSIONAL LINE CHART
# ============================================================

def svg_line_chart(
    values,
    labels,
    title="AQI Trend",
    subtitle="Forecast trajectory",
    width=900,
    height=300
):

    clean_values = []

    clean_labels = []

    for value, label in zip(values, labels):

        try:

            value = float(value)

            if np.isfinite(value):

                clean_values.append(value)
                clean_labels.append(str(label))

        except Exception:
            pass

    if len(clean_values) < 2:

        return textwrap.dedent("""
        <html><head><style>html,body{margin:0;padding:0;background:transparent;}</style></head>
        <body>
        <div style="
            background: linear-gradient(165deg, #171c27 0%, #12151d 100%);
            border: 1px solid #252c39;
            border-radius: 13px;
            padding: 15px 15px 8px 15px;
            box-sizing: border-box;
            font-family: Inter, 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        ">
            <div style="color:#ffffff;font-size:15px;font-weight:800;">AQI Trend</div>
            <div style="color:#718097;font-size:11px;margin-top:3px;">
                Not enough observations available.
            </div>
        </div>
        </body></html>
        """)

    values = clean_values
    labels = clean_labels

    n = len(values)

    left = 55
    right = 25
    top = 25
    bottom = 48

    chart_width = width - left - right
    chart_height = height - top - bottom

    minimum = min(values)
    maximum = max(values)

    padding = max((maximum - minimum) * 0.20, 5)

    ymin = max(0, minimum - padding)
    ymax = maximum + padding

    def x_pos(i):

        return left + (
            chart_width * i / max(n - 1, 1)
        )

    def y_pos(value):

        return top + chart_height - (
            (value - ymin)
            / max(ymax - ymin, 1)
        ) * chart_height

    points = " ".join(
        f"{x_pos(i):.1f},{y_pos(value):.1f}"
        for i, value in enumerate(values)
    )

    grid = []
    grid_labels = []

    for i in range(5):

        grid_value = ymin + (
            (ymax - ymin) * i / 4
        )

        y = y_pos(grid_value)

        grid.append(
            f"""
            <line
                x1="{left}"
                y1="{y:.1f}"
                x2="{width-right}"
                y2="{y:.1f}"
                stroke="#252c39"
                stroke-width="1"
            />
            """
        )

        grid_labels.append(
            f"""
            <text
                x="{left-8}"
                y="{y+4:.1f}"
                text-anchor="end"
                fill="#657286"
                font-size="10"
            >
                {grid_value:.0f}
            </text>
            """
        )

    xlabels = []

    for i, label in enumerate(labels):

        if i == 0 or i == n - 1 or n <= 4:

            xlabels.append(
                f"""
                <text
                    x="{x_pos(i):.1f}"
                    y="{height-14}"
                    text-anchor="middle"
                    fill="#657286"
                    font-size="10"
                >
                    {safe_text(label)}
                </text>
                """
            )

    circles = []

    for i, value in enumerate(values):

        circles.append(
            f"""
            <circle
                cx="{x_pos(i):.1f}"
                cy="{y_pos(value):.1f}"
                r="4"
                fill="#151922"
                stroke="#22d3ee"
                stroke-width="2"
            />
            """
        )

    return textwrap.dedent(f"""
    <html><head><style>html,body{{margin:0;padding:0;background:transparent;}}</style></head>
    <body>
    <div style="
        background: linear-gradient(165deg, #171c27 0%, #12151d 100%);
        border: 1px solid #252c39;
        border-radius: 13px;
        padding: 15px 15px 8px 15px;
        box-sizing: border-box;
        font-family: Inter, 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    ">

        <div style="color:#ffffff;font-size:15px;font-weight:800;">
            {safe_text(title)}
        </div>

        <div style="color:#718097;font-size:11px;margin-top:3px;">
            {safe_text(subtitle)}
        </div>

        <svg
            viewBox="0 0 {width} {height}"
            width="{width}"
            height="{height}"
            preserveAspectRatio="xMidYMid meet"
            style="display:block; width:100%; height:{height}px; max-width:100%;"
            role="img"
        >

            {''.join(grid)}

            {''.join(grid_labels)}

            <polyline
                points="{points}"
                fill="none"
                stroke="#22d3ee"
                stroke-width="3"
                stroke-linecap="round"
                stroke-linejoin="round"
            />

            {''.join(circles)}

            {''.join(xlabels)}

        </svg>

    </div>
    </body></html>
    """)


# ============================================================
# LOAD FORECAST
# ============================================================

if not os.path.exists(FORECAST_FILE):

    st.error("Forecast results were not found.")

    st.info(
        "Run src/live_forecast.py first, or use "
        "Refresh Forecast in the sidebar."
    )

    st.stop()


try:

    forecast = pd.read_csv(
        FORECAST_FILE
    )

except Exception as exc:

    st.error(
        f"Could not read forecast_results.csv: {exc}"
    )

    st.stop()


required_columns = [
    "Date",
    "Predicted AQI"
]


missing_columns = [
    column
    for column in required_columns
    if column not in forecast.columns
]


if missing_columns:

    st.error(
        "forecast_results.csv is missing: "
        + ", ".join(missing_columns)
    )

    st.stop()


forecast["Date"] = pd.to_datetime(
    forecast["Date"],
    errors="coerce"
)

forecast["Predicted AQI"] = pd.to_numeric(
    forecast["Predicted AQI"],
    errors="coerce"
)


forecast = (
    forecast
    .dropna(
        subset=[
            "Date",
            "Predicted AQI"
        ]
    )
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# KARACHI DATE LOGIC
# ============================================================

karachi_now = pd.Timestamp.now(
    tz="Asia/Karachi"
)

today = (
    karachi_now
    .normalize()
    .tz_localize(None)
)

tomorrow = (
    today
    + pd.Timedelta(days=1)
)


future_forecast = (
    forecast[
        forecast["Date"] >= tomorrow
    ]
    .sort_values("Date")
    .head(3)
    .copy()
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            🛡️ AirSense Karachi
        </div>
        """
)

    render_html(
        """
        <div class="sidebar-subtitle">
            AI-powered air-quality intelligence
        </div>
        """
)

    render_html(
        '<div class="sidebar-section">Navigation</div>'
)

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Model Performance",
            "Model Explainability",
            "Project Information"
        ],
        label_visibility="collapsed"
    )

    render_html(
        '<div class="sidebar-section">System</div>'
)

    st.caption("📍 Karachi, Pakistan")
    st.caption("🔮 3-day AQI forecasting")
    st.caption("🤖 Random Forest")
    st.caption("🧠 SHAP explainability")

    render_html(
        '<div class="sidebar-section">Live Forecast</div>'
)

    if st.button(
        "↻  Refresh Forecast",
        use_container_width=True
    ):

        with st.spinner(
            "Updating Karachi forecast..."
        ):

            try:

                result = subprocess.run(
                    [
                        sys.executable,
                        LIVE_FORECAST_FILE
                    ],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=180
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

                    if result.stderr:
                        st.code(
                            result.stderr
                        )

            except subprocess.TimeoutExpired:

                st.error(
                    "Forecast update timed out."
                )

            except Exception as exc:

                st.error(
                    f"Could not run forecast: {exc}"
                )

    st.markdown("---")

    st.caption("AirSense Karachi")
    st.caption("Automated forecasting • Streamlit")


# ============================================================
# IF NO FUTURE FORECAST
# ============================================================

if len(future_forecast) < 3:

    render_html(
        '<div class="main-title">Dashboard</div>'
)

    render_html(
        '<div class="main-subtitle">'
        'Live Karachi air-quality monitoring and 3-day AQI forecasting'
        '</div>'
)

    st.error(
        f"Only {len(future_forecast)} future forecast day(s) "
        "are currently available."
    )

    st.info(
        f"Today is {today.strftime('%d %B %Y')}. "
        "Click **Refresh Forecast** in the sidebar to generate "
        "the next three future dates."
    )

    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    render_html(
        '<div class="main-title">Dashboard</div>'
)

    render_html(
        '<div class="main-subtitle">'
        'Live Karachi air-quality monitoring and 3-day AQI forecasting'
        '</div>'
)

    # --------------------------------------------------------
    # KEY VALUES
    # --------------------------------------------------------

    tomorrow_row = future_forecast.iloc[0]

    tomorrow_aqi = float(
        tomorrow_row["Predicted AQI"]
    )

    worst_idx = (
        future_forecast["Predicted AQI"]
        .idxmax()
    )

    worst_row = future_forecast.loc[
        worst_idx
    ]

    worst_aqi = float(
        worst_row["Predicted AQI"]
    )

    # --------------------------------------------------------
    # SECONDARY METRICS
    # --------------------------------------------------------
    # forecast_results.csv only has [Date, Predicted AQI] and
    # historical_aqi.csv only has [Date, Average AQI] -- there is
    # no PM2.5 / PM10 / temperature / humidity anywhere in the
    # pipeline yet. Rather than show permanent placeholder dashes,
    # these three cards use real numbers derived from the AQI data
    # that actually exists. To get real pollutant/weather cards
    # back, the upstream data-collection step (src/live_forecast.py
    # or whatever builds these CSVs) needs to save those columns.

    latest_hist_aqi = None
    hist_avg_7d = None

    if os.path.exists(HISTORICAL_FILE):

        try:

            hist_env = pd.read_csv(HISTORICAL_FILE)

            hist_col = find_numeric_column(
                hist_env,
                ["Average AQI", "AQI", "aqi", "Air Quality Index"]
            )

            if hist_col:

                hist_series = pd.to_numeric(
                    hist_env[hist_col],
                    errors="coerce"
                ).dropna()

                if not hist_series.empty:

                    latest_hist_aqi = hist_series.iloc[-1]
                    hist_avg_7d = hist_series.tail(7).mean()

        except Exception:

            pass

    forecast_trend_delta = (
        worst_row["Predicted AQI"] - tomorrow_row["Predicted AQI"]
        if len(future_forecast) >= 2
        else 0.0
    )

    if forecast_trend_delta > 3:
        trend_label = "Worsening"
        trend_arrow = "▲"
    elif forecast_trend_delta < -3:
        trend_label = "Improving"
        trend_arrow = "▼"
    else:
        trend_label = "Stable"
        trend_arrow = "●"

    vs_latest_delta = (
        tomorrow_aqi - latest_hist_aqi
        if latest_hist_aqi is not None
        else None
    )

    # ========================================================
    # TOP SECTION
    # ========================================================

    hero_col, metrics_col = st.columns(
        [1.05, 2.2],
        gap="medium"
    )

    with hero_col:

        color = aqi_color(
            tomorrow_aqi
        )

        # IMPORTANT:
        # st.html prevents Streamlit from displaying
        # the HTML source as plain text.

        hero_html = f"""
        <div class="aqi-hero">

            <div class="aqi-hero-title">
                Air Quality Index
            </div>

            <div
                class="aqi-circle"
                style="background:{color}; color:{color};"
            >
                <div class="aqi-number">
                    {tomorrow_aqi:.0f}
                </div>
            </div>

            <div class="aqi-status">
                {safe_text(
                    aqi_category(tomorrow_aqi)
                )}
            </div>

            <div class="aqi-date">
                Forecast for
                {tomorrow_row["Date"].strftime("%d %B %Y")}
            </div>

        </div>
        """

        hero_html = textwrap.dedent(hero_html)

        try:

            st.html(hero_html)

        except AttributeError:

            # Compatibility fallback for older Streamlit

            st.markdown(hero_html, unsafe_allow_html=True)

    with metrics_col:

        m1, m2, m3 = st.columns(
            3,
            gap="small"
        )

        with m1:

            render_html(
                f"""
                <div class="card">

                    <div class="card-label">
                        Latest Observed AQI
                    </div>

                    <div class="big-value">
                        {fmt_value(latest_hist_aqi)}
                    </div>

                    <div class="card-note">
                        most recent historical reading
                    </div>

                </div>
                """
)

        with m2:

            render_html(
                f"""
                <div class="card">

                    <div class="card-label">
                        7-Day Avg AQI
                    </div>

                    <div class="big-value">
                        {fmt_value(hist_avg_7d)}
                    </div>

                    <div class="card-note">
                        trailing week, historical data
                    </div>

                </div>
                """
)

        with m3:

            render_html(
                f"""
                <div class="card">

                    <div class="card-label">
                        3-Day Trend
                    </div>

                    <div class="big-value" style="font-size:26px;">
                        {trend_arrow} {safe_text(trend_label)}
                    </div>

                    <div class="card-note">
                        tomorrow vs. worst forecast day
                    </div>

                </div>
                """
)

        render_html(
            "<div style='height:9px'></div>"
)

        render_html(
            f"""
            <div class="card">

                <div class="card-title">
                    Forecast vs. Latest Observed
                </div>

                <div class="big-value">
                    {
                        (
                            f"{vs_latest_delta:+.1f}"
                            if vs_latest_delta is not None
                            else "—"
                        )
                    }
                </div>

                <div class="card-note">
                    {
                        (
                            "AQI points higher than the latest observed reading"
                            if vs_latest_delta is not None and vs_latest_delta > 0
                            else "AQI points lower than the latest observed reading"
                            if vs_latest_delta is not None
                            else "No historical AQI available to compare against"
                        )
                    }
                </div>

            </div>
            """
)

    # ========================================================
    # LATEST ALERT
    # ========================================================

    render_html(
        '<div class="section-title">Latest Alert</div>'
)

    alert_col, summary_col = st.columns(
        [1.7, 1],
        gap="medium"
    )

    with alert_col:

        render_html(
            f"""
            <div class="alert-card">

                <div class="alert-title">

                    <span
                        style="
                        color:{aqi_color(worst_aqi)};
                        font-size:18px;
                        "
                    >
                        ●
                    </span>

                    {aqi_emoji(worst_aqi)}
                    {safe_text(
                        aqi_category(worst_aqi)
                    )}

                </div>

                <div class="alert-text">

                    The highest predicted AQI over
                    the next three days is

                    <b style="color:#ffffff;">
                        {worst_aqi:.0f}
                    </b>

                    expected on

                    <b style="color:#ffffff;">
                        {worst_row["Date"].strftime(
                            "%d %B %Y"
                        )}
                    </b>.

                    <br><br>

                    {safe_text(
                        health_message(worst_aqi)
                    )}

                </div>

            </div>
            """
)

    with summary_col:

        render_html(
            f"""
            <div class="card">

                <div class="card-label">
                    Forecast Horizon
                </div>

                <div class="big-value">
                    3
                    <span
                        style="
                        font-size:15px;
                        color:#7d899c;
                        "
                    >
                        days
                    </span>
                </div>

                <div class="card-note">
                    Through
                    {future_forecast["Date"].max().strftime(
                        "%d %b %Y"
                    )}
                </div>

            </div>
            """
)

    # ========================================================
    # 3 DAY FORECAST
    # ========================================================

    render_html(
        '<div class="section-title">3-Day AQI Forecast</div>'
)

    forecast_cols = st.columns(
        3,
        gap="medium"
    )

    for i, col in enumerate(
        forecast_cols
    ):

        row = future_forecast.iloc[i]

        aqi = float(
            row["Predicted AQI"]
        )

        if i == 0:
            day_label = "Tomorrow"
        elif i == 1:
            day_label = "Day +2"
        else:
            day_label = "Day +3"

        with col:

            render_html(
                f"""
                <div class="forecast-card">

                    <div class="forecast-day">
                        {day_label}
                    </div>

                    <div class="forecast-date">
                        {row["Date"].strftime(
                            "%d %B %Y"
                        )}
                    </div>

                    <div
                        class="forecast-aqi"
                        style="color:#ffffff;"
                    >
                        {aqi:.0f}
                    </div>

                    <div class="forecast-category">

                        <span
                            style="
                            color:{aqi_color(aqi)};
                            "
                        >
                            ●
                        </span>

                        <b>
                            {safe_text(
                                aqi_category(aqi)
                            )}
                        </b>

                    </div>

                </div>
                """
)

    # ========================================================
    # TREND
    # ========================================================

    render_html(
        '<div class="section-title">AQI Trend & Alerts</div>'
)

    trend_col, alerts_col = st.columns(
        [2.1, 1],
        gap="medium"
    )

    with trend_col:

        labels = [
            date.strftime("%d %b")
            for date in future_forecast["Date"]
        ]

        values = (
            future_forecast[
                "Predicted AQI"
            ]
            .tolist()
        )

        render_svg_chart(
            svg_line_chart(
                values,
                labels,
                title="Air Quality Index",
                subtitle="Predicted AQI trajectory",
                height=300
            ),
            height=360
        )

    with alerts_col:

        render_html(
            """
            <div class="card">

                <div class="card-title">
                    Forecast Alerts
                </div>
            """
)

        for _, row in (
            future_forecast.iterrows()
        ):

            aqi = float(
                row["Predicted AQI"]
            )

            render_html(
                f"""
                <div
                    style="
                    padding:12px 0;
                    border-bottom:1px solid #252c38;
                    "
                >

                    <div
                        style="
                        font-size:11px;
                        color:#7d899c;
                        "
                    >
                        {row["Date"].strftime(
                            "%d %b %Y"
                        )}
                    </div>

                    <div
                        style="
                        margin-top:4px;
                        color:#ffffff;
                        font-weight:750;
                        "
                    >

                        <span
                            style="
                            color:{aqi_color(aqi)};
                            "
                        >
                            ●
                        </span>

                        AQI {aqi:.0f}

                    </div>

                    <div
                        style="
                        font-size:11px;
                        color:#718097;
                        margin-top:3px;
                        "
                    >
                        {safe_text(
                            aqi_category(aqi)
                        )}
                    </div>

                </div>
                """
)

        render_html(
            "</div>"
)

    # ========================================================
    # FORECAST DETAILS
    # ========================================================

    render_html(
        '<div class="section-title">Forecast Details</div>'
)

    details = future_forecast.copy()

    details["Forecast Date"] = (
        details["Date"]
        .dt.strftime("%d %b %Y")
    )

    details["Predicted AQI"] = (
        details["Predicted AQI"]
        .round(2)
    )

    details["Status"] = (
        details["Predicted AQI"]
        .apply(aqi_category)
    )

    st.dataframe(
        details[
            [
                "Forecast Date",
                "Predicted AQI",
                "Status"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # HISTORICAL AIR QUALITY
    # ========================================================

    render_html(
        '<div class="section-title">Historical Air Quality</div>'
)

    if os.path.exists(
        HISTORICAL_FILE
    ):

        try:

            historical = pd.read_csv(
                HISTORICAL_FILE
            )

            if "Date" in historical.columns:

                historical["Date"] = (
                    pd.to_datetime(
                        historical["Date"],
                        errors="coerce"
                    )
                )

                hist_col = find_numeric_column(
                    historical,
                    [
                        "AQI",
                        "aqi",
                        "Air Quality Index"
                    ]
                )

                if hist_col:

                    hist = (
                        historical
                        .dropna(
                            subset=[
                                "Date",
                                hist_col
                            ]
                        )
                        .sort_values("Date")
                        .tail(90)
                    )

                    if len(hist) >= 2:

                        hist_values = (
                            pd.to_numeric(
                                hist[hist_col],
                                errors="coerce"
                            )
                            .tolist()
                        )

                        hist_labels = [
                            date.strftime(
                                "%d %b"
                            )
                            for date in hist["Date"]
                        ]

                        render_svg_chart(
                            svg_line_chart(
                                hist_values,
                                hist_labels,
                                title="Historical AQI",
                                subtitle="Recent historical AQI observations",
                                height=290
                            ),
                            height=350
                        )

                    else:

                        st.info(
                            "Not enough historical AQI observations."
                        )

                else:

                    st.info(
                        "Historical file does not contain "
                        "a recognized AQI column."
                    )

            else:

                st.info(
                    "Historical file does not contain a Date column."
                )

        except Exception as exc:

            st.warning(
                f"Historical analysis could not be loaded: {exc}"
            )

    else:

        st.info(
            "Historical AQI file is not available."
        )

    # ========================================================
    # HOW IT WORKS
    # ========================================================

    render_html(
        '<div class="section-title">How the Forecast Works</div>'
)

    steps = [
        (
            "01",
            "Data collection",
            "Weather and air-quality observations are collected from external data sources."
        ),
        (
            "02",
            "Feature engineering",
            "Historical pollution, lag variables, weather variables and time-based features are prepared."
        ),
        (
            "03",
            "Machine learning",
            "The Random Forest model learns relationships between environmental features and AQI."
        ),
        (
            "04",
            "Forecast generation",
            "The trained model generates AQI estimates for the next three future dates."
        ),
        (
            "05",
            "Explainability",
            "SHAP analysis identifies the variables that contribute most strongly to model predictions."
        ),
    ]

    for number, title, description in steps:

        render_html(
            f"""
            <div
                class="card"
                style="
                margin-bottom:9px;
                padding:15px 18px;
                "
            >

                <div
                    style="
                    color:#22d3ee;
                    font-size:11px;
                    font-weight:800;
                    letter-spacing:1px;
                    "
                >
                    {number}
                </div>

                <div
                    style="
                    color:#ffffff;
                    font-size:14px;
                    font-weight:800;
                    margin-top:4px;
                    "
                >
                    {title}
                </div>

                <div
                    style="
                    color:#8895a8;
                    font-size:12px;
                    line-height:1.55;
                    margin-top:4px;
                    "
                >
                    {description}
                </div>

            </div>
            """
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    render_html(
        '<div class="main-title">Model Performance</div>'
)

    render_html(
        '<div class="main-subtitle">'
        'Evaluation and comparison of the forecasting models'
        '</div>'
)

    performance = pd.DataFrame(
        {
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
        }
    )

    c1, c2, c3 = st.columns(
        3,
        gap="medium"
    )

    with c1:

        render_html(
            """
            <div class="card">

                <div class="card-label">
                    MAE
                </div>

                <div class="big-value">
                    7.57
                </div>

                <div class="card-note">
                    Average absolute AQI error
                </div>

            </div>
            """
)

    with c2:

        render_html(
            """
            <div class="card">

                <div class="card-label">
                    RMSE
                </div>

                <div class="big-value">
                    10.47
                </div>

                <div class="card-note">
                    Penalizes larger errors
                </div>

            </div>
            """
)

    with c3:

        render_html(
            """
            <div class="card">

                <div class="card-label">
                    R²
                </div>

                <div class="big-value">
                    72.9%
                </div>

                <div class="card-note">
                    Variation explained
                </div>

            </div>
            """
)

    render_html(
        '<div class="section-title">Model Comparison</div>'
)

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    render_html(
        '<div class="section-title">Selected Model</div>'
)

    render_html(
        """
        <div class="card">

            <div class="card-title">
                🥇 Random Forest Regressor
            </div>

            <div class="alert-text">

                Random Forest was selected because it achieved
                lower MAE, lower RMSE and higher R² than Ridge
                Regression under the same chronological
                train/test methodology.

            </div>

        </div>
        """
)

    render_html(
        '<div class="section-title">Metric Interpretation</div>'
)

    with st.expander(
        "MAE — Mean Absolute Error"
    ):

        st.write(
            "MAE is the average absolute difference between "
            "predicted and actual AQI. A value of 7.57 means "
            "the model's predictions differ from actual AQI by "
            "approximately 7.57 AQI points on average on the "
            "evaluated data."
        )

    with st.expander(
        "RMSE — Root Mean Squared Error"
    ):

        st.write(
            "RMSE is the square root of the average squared "
            "prediction error. Larger errors receive greater "
            "weight because the errors are squared."
        )

    with st.expander(
        "R² — R-squared"
    ):

        st.write(
            "R² measures the proportion of variation in the "
            "test target explained by the regression model. "
            "The value 0.729 corresponds to approximately "
            "72.9%."
        )

    render_html(
        '<div class="section-title">Training Information</div>'
)

    t1, t2, t3 = st.columns(
        3,
        gap="medium"
    )

    training_info = [
        (
            t1,
            "Training Samples",
            "992"
        ),
        (
            t2,
            "Testing Samples",
            "249"
        ),
        (
            t3,
            "Features",
            "20"
        )
    ]

    for col, label, value in training_info:

        with col:

            render_html(
                f"""
                <div class="card">

                    <div class="card-label">
                        {label}
                    </div>

                    <div class="big-value">
                        {value}
                    </div>

                </div>
                """
)


# ============================================================
# MODEL EXPLAINABILITY
# ============================================================

elif page == "Model Explainability":

    render_html(
        '<div class="main-title">Model Explainability</div>'
)

    render_html(
        '<div class="main-subtitle">'
        'SHAP analysis of Random Forest AQI predictions'
        '</div>'
)

    render_html(
        """
        <div class="card">

            <div class="card-title">
                What is SHAP?
            </div>

            <div class="alert-text">

                SHAP (SHapley Additive exPlanations) is used
                to understand how individual input variables
                influence the machine-learning model.

                It improves transparency by showing which
                features are most influential.

            </div>

        </div>
        """
)

    shap_data = pd.DataFrame(
        {
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
        }
    )

    left, right = st.columns(
        [1.35, 1],
        gap="medium"
    )

    with left:

        render_html(
            '<div class="section-title">Feature Importance</div>'
)

        if os.path.exists(
            SHAP_FILE
        ):

            st.image(
                SHAP_FILE,
                use_container_width=True
            )

        else:

            st.info(
                "SHAP image was not found."
            )

    with right:

        render_html(
            '<div class="section-title">Top Model Drivers</div>'
)

        st.dataframe(
            shap_data.head(10),
            use_container_width=True,
            hide_index=True
        )

    render_html(
        '<div class="section-title">Interpretation</div>'
)

    render_html(
        """
        <div class="card">

            <div class="alert-text">

                The strongest contributors in the current
                SHAP analysis are

                <b style="color:#ffffff;">CO</b>,

                <b style="color:#ffffff;">PM10</b>,

                <b style="color:#ffffff;">
                    previous-day PM2.5
                </b>,

                and

                <b style="color:#ffffff;">SO2</b>.

                <br><br>

                SHAP importance indicates model influence
                rather than causation. A high importance score
                does not by itself prove that changing a
                variable causes AQI to change.

            </div>

        </div>
        """
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

elif page == "Project Information":

    render_html(
        '<div class="main-title">Project Information</div>'
)

    render_html(
        '<div class="main-subtitle">'
        'AirSense Karachi — AI-powered AQI forecasting system'
        '</div>'
)

    render_html(
        """
        <div class="card">

            <div class="card-title">
                Project Objective
            </div>

            <div class="alert-text">

                AirSense Karachi combines historical
                air-quality patterns, pollution variables,
                weather conditions and temporal features
                to generate three-day AQI forecasts for Karachi.

            </div>

        </div>
        """
)

    render_html(
        '<div class="section-title">System Architecture</div>'
)

    st.code(
        """
External APIs
      ↓
Raw Environmental Data
      ↓
Feature Engineering
      ↓
Historical Training Dataset
      ↓
Chronological Train / Test Split
      ↓
Random Forest + Ridge Regression
      ↓
Model Evaluation
      ↓
Random Forest Selected
      ↓
SHAP Explainability
      ↓
Live 3-Day Forecast
      ↓
Flask API / Streamlit Dashboard
      ↓
GitHub Actions Automation
        """,
        language="text"
    )

    render_html(
        '<div class="section-title">Model Features</div>'
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

    feature_df = pd.DataFrame(
        {
            "Model Feature": feature_list
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    render_html(
        '<div class="section-title">Implemented Components</div>'
)

    components_table = pd.DataFrame(
        {
            "Component": [
                "Historical Data Collection",
                "Feature Engineering",
                "Forecast Training Dataset",
                "Random Forest Model",
                "Ridge Regression Model",
                "Model Evaluation",
                "Live Weather Data",
                "Live Air-Quality Data",
                "Automatic 3-Day Forecast",
                "SHAP Explainability",
                "AQI Health Alerts",
                "Streamlit Dashboard",
                "GitHub Actions Automation"
            ],
            "Status": [
                "Completed"
            ] * 13
        }
    )

    st.dataframe(
        components_table,
        use_container_width=True,
        hide_index=True
    )

    render_html(
        '<div class="section-title">Current Limitations</div>'
)

    limitations = [
        "Forecast quality depends on the availability and quality of external data.",
        "Predictions are estimates and should not be treated as official air-quality measurements.",
        "Recursive multi-day forecasting can accumulate prediction error.",
        "The current production model is Random Forest rather than a deep-learning forecasting model.",
        "API outages can affect live forecast generation.",
        "SHAP explains model behaviour but does not establish causal relationships."
    ]

    for item in limitations:

        render_html(
            f"""
            <div
                style="
                color:#aeb9c8;
                font-size:13px;
                margin:8px 0;
                "
            >
                • {safe_text(item)}
            </div>
            """
)

    render_html(
        '<div class="section-title">Future Improvements</div>'
)

    improvements = [
        "LSTM / GRU time-series benchmark",
        "Cloud feature store integration",
        "Automated model retraining",
        "Model monitoring and drift detection",
        "Automated model registry",
        "Real-time notifications",
        "Additional environmental variables",
        "Longer historical training period"
    ]

    for item in improvements:

        render_html(
            f"""
            <div
                style="
                color:#aeb9c8;
                font-size:13px;
                margin:8px 0;
                "
            >
                • {safe_text(item)}
            </div>
            """
)


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div
        style="
        text-align:center;
        color:#59667a;
        font-size:11px;
        margin-top:40px;
        padding-top:18px;
        border-top:1px solid #202633;
        "
    >

        <b>AirSense Karachi</b>
        • AI-powered AQI Forecasting System
        • Python
        • Pandas
        • Scikit-learn
        • Random Forest
        • SHAP
        • Streamlit

    </div>
    """
)
