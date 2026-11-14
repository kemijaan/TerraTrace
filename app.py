
import streamlit as st
import plotly.express as px
import pandas as pd
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
    description = "an increasing"
elif slope < 0:
    description = "a decreasing"
else:
    description = "a stable"

st.info(
    f"Between {start_year} and {end_year}, {variable.lower()} "
    f"shows {description} trend of approximately "
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
        "Results use ordinary linear regression. Temporal correlation "
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
        "Explore historical one-day-ahead temperature predictions "
        "from our Random Forest model."
    )

    model_path = Path("models/temperature_rf.joblib")

    if not model_path.exists():
        st.error("Trained machine learning model not found.")
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
        ml_df["time"].dt.year.between(2021, 2025)
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
            f"Random Forest reduced average prediction error by "
            f"{improvement:.2f}% compared with the persistence baseline."
        )
    else:
        st.warning(
            "Random Forest did not outperform the persistence baseline."
        )

    st.subheader("Actual vs Predicted Temperature")

    st.write("Select a year and month to examine daily predictions.")

    col_year, col_month = st.columns(2)

    with col_year:
        selected_year = st.selectbox(
            "Select Year",
            options=list(range(2021, 2026)),
            index=4
        )

    with col_month:
        selected_month = st.selectbox(
            "Select Month",
            options=list(range(1, 13)),
            index=0,
            format_func=lambda x: pd.Timestamp(2025, x, 1).strftime("%B")
        )

    display_df = ml_test[
        (ml_test["time"].dt.year == selected_year) &
        (ml_test["time"].dt.month == selected_month)
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
        title=f"Actual vs Predicted — {pd.Timestamp(selected_year, selected_month, 1).strftime('%B %Y')}"
    )

    fig_ml.update_layout(hovermode="x unified")
    st.plotly_chart(fig_ml, use_container_width=True)

    st.subheader("Selected Month Performance")

    month_mae = mean_absolute_error(
        display_df["Actual Temperature"],
        display_df["Predicted Temperature"]
    )

    st.metric("Monthly MAE", f"{month_mae:.3f} °C")

    with st.expander("About the Machine Learning Model"):
        st.write("**Algorithm:** Random Forest Regressor")
        st.write("**Training period:** 2000–2020")
        st.write("**Testing period:** 2021–2025")
        st.write("**Target:** Daily average temperature")
        st.write(
            "**Features:** Previous-day temperatures, seven-day "
            "rolling average, and month."
        )
        st.write(
            "**Baseline:** Uses yesterday's temperature "
            "as today's prediction."
        )
        st.caption(
            "This is a historical one-day-ahead prediction evaluation. "
            "Actual previous-day measurements are used as model inputs. "
            "It is not a long-term temperature forecast."
        )

st.divider()
st.caption(
    "TerraTrace | NASA Space Apps Challenge 2026 | Team Super Larpathon"
)
