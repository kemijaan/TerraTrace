
import streamlit as st
import plotly.express as px
from analysis import load_data, calculate_trend, load_rainfall, calculate_rainfall_trend

st.set_page_config(page_title="TerraTrace", page_icon="🌍", layout="wide")

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
    df = load_data("data/temperature.csv")
    value_column = "temperature_c"
    unit = "°C"
    source = "NASA MERRA-2"
else:
    df = load_rainfall("data/rainfall.csv")
    value_column = "rainfall_mm"
    unit = "mm"
    source = "NASA GPCP v3.3"

filtered_df = df[df["time"].dt.year.between(start_year, end_year)]

st.subheader(f"{variable} Trends — Sarawak Region")

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
    direction = "increasing"
elif slope < 0:
    direction = "decreasing"
else:
    direction = "stable"

st.info(
    f"Between {start_year} and {end_year}, {variable.lower()} "
    f"shows an {direction} trend of approximately "
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

st.divider()
st.caption(
    f"Source: {source} | Study area: Sarawak and surrounding regions "
    "(rectangular bounding box)"
)
