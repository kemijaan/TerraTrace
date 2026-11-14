from analysis import load_data, calculate_trend

df = load_data("data/temperature.csv")
yearly, trend = calculate_trend(df)

print(yearly.tail())
print(trend)

from analysis import load_rainfall, calculate_rainfall_trend

rainfall_df = load_rainfall("data/rainfall.csv")
rainfall_yearly, rainfall_trend = calculate_rainfall_trend(rainfall_df)

print(rainfall_yearly.tail())
print(rainfall_trend)
