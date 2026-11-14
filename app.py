
import streamlit as st
import plotly.express as px
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from analysis import load_data, calculate_trend, load_rainfall, calculate_rainfall_trend

st.set_page_config(page_title="TerraTrace", page_icon="🌍", layout="wide")

@st.cache_data
def get_temperature_data():
    return load_data("data/temperature.csv")

@st.cache_data
def get_rainfall_data():
    return load_rainfall("data/rainfall.csv")

@st.cache_resource
def get_ml_model():
    return joblib.load("models/temperature_rf.joblib")

st.title("TerraTrace")
st.write("Explore environmental changes using NASA Earth observation data.")

st.sidebar.header("Analysis Settings")

variable = st.sidebar.selectbox(
    "Environmental Variable",
    ["Temperature", "Rainfall"]
)

start_year, end_year = st.sidebar.slider(
    "Select Year Range",
    min_value=2000,
    max_value=2025,
    value=(2000, 2025)
)

if variable == "Temperature":
    df = get_temperature_data()
    value_column = "temperature_c"
    unit = "°C"
    source = "NASA MERRA-2"
else:
    df = get_rainfall_data()
    value_column = "rainfall_mm"
    unit = "mm"
    source = "NASA GPCP v3.3"

filtered_df = df[df["time"].dt.year.between(start_year, end_year)]

st.subheader(f"{variable} Trends — Sarawak Area")

if end_year - start_year < 4:
    st.warning("Please select at least 5 years for trend analysis.")
    st.stop()

if variable == "Temperature":
    yearly, trend = calculate_trend(filtered_df)
else:
    yearly, trend = calculate_rainfall_trend(filtered_df)

fig = px.line(
    yearly,
    x="year",
    y=value_column,
    markers=True,
    labels={
        "year": "Year",
        value_column: f"{variable} ({unit})"
    },
    title=f"Annual {variable} ({start_year}–{end_year})"
)

fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Trend Analysis")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Change per Decade",
    f"{trend['slope_per_decade']:+.3f} {unit}"
)

col2.metric(
    "P-value",
    f"{trend['p_value']:.6f}"
)

col3.metric(
    "R²",
    f"{trend['r_squared']:.3f}"
)

st.subheader("Trend Interpretation")

slope = trend["slope_per_decade"]

if slope > 0:
    direction = "an increasing"
elif slope < 0:
    direction = "a decreasing"
else:
    direction = "a stable"

st.info(
    f"Between {start_year} and {end_year}, {variable.lower()} "
    f"shows {direction} trend of approximately "
    f"{abs(slope):.3f} {unit} per decade."
)

if trend["p_value"] < 0.05:
    st.success(
        "The linear trend is statistically significant at the 5% level "
        "under the regression model assumptions."
    )
else:
    st.info(
        "The linear trend is not statistically significant at the 5% level."
    )

with st.expander("Understanding the Results"):
    st.write("**Change per Decade:** Estimated change over ten years.")
    st.write("**P-value:** Used to test whether the linear trend differs from zero.")
    st.write("**R²:** Proportion of variation explained by the linear trend.")
    st.caption(
        "Results are based on ordinary linear regression. Temporal correlation "
        "has not yet been assessed."
    )

st.caption(
    f"Source: {source} | Study area: Sarawak and surrounding regions "
    "(rectangular bounding box)"
)

if variable == "Temperature":
    st.divider()
    st.header("Machine Learning — Temperature Prediction")
    st.write(
        "Compare Random Forest predictions with actual NASA temperature "
        "measurements from 2021 to 2025."
    )

    model_path = Path("models/temperature_rf.joblib")

    if not model_path.exists():
        st.error("Trained ML model not found in the models folder.")
        st.stop()

    saved = get_ml_model()
    model = saved["model"]
    features = saved["features"]

    ml_df = get_temperature_data().copy()
    ml_df = ml_df.sort_values("time").reset_index(drop=True)

    ml_df["temp_lag1"] = ml_df["temperature_c"].shift(1)
    ml_df["temp_lag2"] = ml_df["temperature_c"].shift(2)
    ml_df["temp_lag7"] = ml_df["temperature_c"].shift(7)
    ml_df["temp_rolling7"] = ml_df["temperature_c"].shift(1).rolling(7).mean()
    ml_df["month"] = ml_df["time"].dt.month

    ml_df = ml_df.dropna(subset=features + ["temperature_c"])

    ml_test = ml_df[
        (ml_df["time"].dt.year >= 2021) &
        (ml_df["time"].dt.year <= 2025)
    ].copy()

    ml_test["predicted"] = model.predict(ml_test[features])
    ml_test["baseline"] = ml_test["temp_lag1"]

    rf_mae = mean_absolute_error(
        ml_test["temperature_c"],
        ml_test["predicted"]
    )

    rf_rmse = mean_squared_error(
        ml_test["temperature_c"],
        ml_test["predicted"]
    ) ** 0.5

    rf_r2 = r2_score(
        ml_test["temperature_c"],
        ml_test["predicted"]
    )

    baseline_mae = mean_absolute_error(
        ml_test["temperature_c"],
        ml_test["baseline"]
    )

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Random Forest MAE", f"{rf_mae:.3f} °C")
    col2.metric("Random Forest RMSE", f"{rf_rmse:.3f} °C")
    col3.metric("Random Forest R²", f"{rf_r2:.3f}")

    improvement = (baseline_mae - rf_mae) / baseline_mae * 100

    if improvement > 0:
        st.success(
            f"Random Forest reduced the average prediction error by "
            f"{improvement:.2f}% compared with the persistence baseline."
        )
    else:
        st.warning(
            "Random Forest did not outperform the persistence baseline "
            "on average absolute error."
        )

    st.subheader("Actual vs Predicted Temperature")

    available_start = max(start_year, 2021)
    available_end = min(end_year, 2025)

    if available_start > available_end:
        st.info(
            "ML evaluation is available only for 2021–2025. "
            "Adjust the year range to view predictions."
        )
    else:
        display_df = ml_test[
            ml_test["time"].dt.year.between(
                available_start,
                available_end
            )
        ].copy()

        
        display_df = display_df.rename(columns={
            "temperature_c": "Actual Temperature",
            "predicted": "Predicted Temperature"
        })

        fig_ml = px.line(
            display_df,
            x="time",
            y=["Actual Temperature", "Predicted Temperature"],
            labels={
                "time": "Date",
                "value": "Temperature (°C)",
                "variable": "Series"
            },
            title=f"Actual vs Predicted ({available_start}–{available_end})"
        )

        fig_ml.update_layout(hovermode="x unified")
        st.plotly_chart(fig_ml, use_container_width=True)

    with st.expander("About the Machine Learning Model"):
        st.write("**Algorithm:** Random Forest Regressor")
        st.write("**Training period:** 2000–2020")
        st.write("**Testing period:** 2021–2025")
        st.write("**Prediction target:** Daily average temperature")
        st.write(
            "**Input features:** Previous temperatures, seven-day "
            "average temperature, and month."
        )
        st.write(
            "**Baseline:** Predicts the current day's temperature "
            "using the previous day's observed temperature."
        )
        st.caption(
            "These results are historical one-day-ahead predictions. "
            "The model uses actual previous-day observations, not its own "
            "earlier predictions. It is not a long-term climate forecast."
        )

st.divider()
st.caption(
    "TerraTrace | NASA Space Apps Challenge 2026 | Team Super Larpathon"
)
