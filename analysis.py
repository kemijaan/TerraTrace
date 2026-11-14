
import pandas as pd
from scipy.stats import linregress

def load_data(path):
    df = pd.read_csv(path, skiprows=8, skipinitialspace=True)
    df["time"] = pd.to_datetime(df["time"])
    df["temperature_c"] = df["mean_M2SDNXSLV_5_12_4_T2MMEAN"] - 273.15
    return df

def calculate_trend(df):
    yearly = df.groupby(df["time"].dt.year)["temperature_c"].mean().reset_index()
    yearly.columns = ["year", "temperature_c"]
    result = linregress(yearly["year"], yearly["temperature_c"])
    return yearly, {
        "slope_per_decade": result.slope * 10,
        "p_value": result.pvalue,
        "r_squared": result.rvalue ** 2
    }


def load_rainfall(path):
    df = pd.read_csv(path, skiprows=8, skipinitialspace=True)
    df["time"] = pd.to_datetime(df["time"])
    df["rainfall_mm"] = df["mean_GPCPMON_3_3_satellite_precip"] * df["time"].dt.days_in_month
    return df

def calculate_rainfall_trend(df):
    yearly = df.groupby(df["time"].dt.year)["rainfall_mm"].sum().reset_index()
    yearly.columns = ["year", "rainfall_mm"]
    result = linregress(yearly["year"], yearly["rainfall_mm"])
    return yearly, {
        "slope_per_decade": result.slope * 10,
        "p_value": result.pvalue,
        "r_squared": result.rvalue ** 2
    }
