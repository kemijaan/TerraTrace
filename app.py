from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / 'data' / 'processed' / 'annual_summary.csv'
REGION_NAMES = {
    'brunei': 'Brunei', 'cambodia': 'Cambodia', 'indonesia': 'Indonesia',
    'laos': 'Laos', 'malaysia': 'Malaysia', 'myanmar': 'Myanmar',
    'philippines': 'Philippines', 'singapore': 'Singapore',
    'thailand': 'Thailand', 'timor_leste': 'Timor-Leste', 'vietnam': 'Vietnam'
}

st.set_page_config(page_title='TerraTrace', page_icon='🌍', layout='wide')

@st.cache_data
def load_processed_data():
    if not SUMMARY.exists():
        return pd.DataFrame()
    df = pd.read_csv(SUMMARY)
    required = {'region_id', 'variable', 'year', 'annual_value', 'complete_year'}
    if not required.issubset(df.columns):
        raise ValueError(f'Pipeline output missing columns: {sorted(required - set(df.columns))}')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['annual_value'] = pd.to_numeric(df['annual_value'], errors='coerce')
    df['complete_year'] = df['complete_year'].astype(str).str.lower().eq('true')
    return df.dropna(subset=['year', 'annual_value'])

@st.cache_resource
def load_model(path):
    return joblib.load(path)


def evaluate_legacy_model(variable):
    from analysis import load_data, load_rainfall
    if variable == 'Temperature':
        csv_path = ROOT / 'data' / 'temperature.csv'
        model_path = ROOT / 'models' / 'temperature_rf.joblib'
        target = 'temperature_c'
        if not csv_path.exists() or not model_path.exists():
            st.info('Original Sarawak temperature CSV or trained model is not present.')
            return
        df = load_data(csv_path).sort_values('time').copy()
        df['temp_lag1'] = df['temperature_c'].shift(1)
        df['temp_lag2'] = df['temperature_c'].shift(2)
        df['temp_lag7'] = df['temperature_c'].shift(7)
        df['temp_rolling7'] = df['temperature_c'].shift(1).rolling(7).mean()
        df['month'] = df['time'].dt.month
        baselines = {'Previous day': 'temp_lag1'}
        unit = '°C'
    else:
        csv_path = ROOT / 'data' / 'rainfall.csv'
        model_path = ROOT / 'models' / 'rainfall_rf.joblib'
        target = 'rainfall_mm'
        if not csv_path.exists() or not model_path.exists():
            st.info('Original Sarawak rainfall CSV or trained model is not present.')
            return
        df = load_rainfall(csv_path).sort_values('time').copy()
        df['rain_lag1'] = df['rainfall_mm'].shift(1)
        df['rain_lag2'] = df['rainfall_mm'].shift(2)
        df['rain_lag3'] = df['rainfall_mm'].shift(3)
        df['rain_lag12'] = df['rainfall_mm'].shift(12)
        df['rain_rolling3'] = df['rainfall_mm'].shift(1).rolling(3).mean()
        df['month'] = df['time'].dt.month
        baselines = {'Previous month': 'rain_lag1', 'Same month last year': 'rain_lag12'}
        unit = 'mm'

    saved = load_model(str(model_path))
    features = saved['features']
    df = df.dropna(subset=features + [target])
    df = df[df['time'].dt.year.between(2021, 2025)].copy()
    if df.empty:
        st.warning('No 2021–2025 test observations available.')
        return
    df['prediction'] = saved['model'].predict(df[features])
    actual = df[target]
    rows = [{
        'Model': 'Random Forest',
        'MAE': mean_absolute_error(actual, df['prediction']),
        'RMSE': mean_squared_error(actual, df['prediction']) ** 0.5,
        'R²': r2_score(actual, df['prediction'])
    }]
    for name, col in baselines.items():
        rows.append({
            'Model': name,
            'MAE': mean_absolute_error(actual, df[col]),
            'RMSE': mean_squared_error(actual, df[col]) ** 0.5,
            'R²': r2_score(actual, df[col])
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f'Units: {unit}. Results are historical one-step-ahead tests (2021–2025), not future forecasts.')
    year = st.selectbox('Evaluation year', list(range(2021, 2026)), index=4, key='legacy_year')
    period = df[df['time'].dt.year == year].copy()
    if variable == 'Temperature':
        month = st.selectbox('Evaluation month', list(range(1, 13)),
                             format_func=lambda m: pd.Timestamp(2025, m, 1).strftime('%B'))
        period = period[period['time'].dt.month == month]
    period = period.rename(columns={target: 'Actual', 'prediction': 'Predicted'})
    fig = px.line(period, x='time', y=['Actual', 'Predicted'],
                  labels={'time': 'Date', 'value': unit, 'variable': 'Series'})
    st.plotly_chart(fig, use_container_width=True)

st.title('🌍 TerraTrace')
st.write('Explore environmental trends across Southeast Asia using NASA Earth observation data.')
explorer_tab, ml_tab = st.tabs(['Southeast Asia Explorer', 'Original ML Model Evaluation'])

with explorer_tab:
    st.subheader('Choose a country')
    try:
        from map_view import render_country_map
        selected_country = render_country_map()
    except Exception as exc:
        st.error(f'Interactive map failed: {type(exc).__name__}: {exc}')
        st.info('Check that map_view.py and data/geo/countries.geojson are in the project, and that folium and streamlit-folium are installed.')
        selected_country = st.selectbox('Select a country instead', list(REGION_NAMES), format_func=lambda k: REGION_NAMES[k])
    country_label = REGION_NAMES.get(selected_country, selected_country.replace('_', ' ').title())
    st.write(f'**Selected country:** {country_label}')
    try:
        annual = load_processed_data()
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f'The processed dataset could not be loaded: {exc}')
        annual = pd.DataFrame()
    if annual.empty:
        st.info('No processed data found. Run `python -m pipeline.run` from the TerraTrace project root.')
    else:
        available = annual[
            (annual['region_id'] == selected_country) & annual['complete_year']
        ].copy()
        available_variables = sorted(available['variable'].dropna().unique().tolist())
        if not available_variables:
            st.info(f'No verified country-wide NASA datasets are available for {country_label} yet. Download the country-shape CSVs and rerun the pipeline.')
        else:
            labels = {'temperature': 'Temperature', 'rainfall': 'Rainfall'}
            variable = st.selectbox('Environmental variable', available_variables,
                                    format_func=lambda v: labels.get(v, v.title()))
            series = available[available['variable'] == variable].sort_values('year')
            years = series['year'].astype(int).tolist()
            start_year, end_year = st.slider(
                'Historical year range', min_value=min(years), max_value=max(years),
                value=(min(years), max(years))
            )
            series = series[series['year'].between(start_year, end_year)].copy()
            unit = '°C' if variable == 'temperature' else 'mm/year'
            name = 'Average temperature' if variable == 'temperature' else 'Annual rainfall total'
            fig = px.line(series, x='year', y='annual_value', markers=True,
                          title=f'{country_label} — {name} ({start_year}–{end_year})',
                          labels={'year': 'Year', 'annual_value': f'{name} ({unit})'})
            st.plotly_chart(fig, use_container_width=True)
            if len(series) >= 5:
                fit = linregress(series['year'], series['annual_value'])
                a, b, c = st.columns(3)
                a.metric('Trend per decade', f'{fit.slope * 10:+.3f} {unit}')
                b.metric('Regression p-value', f'{fit.pvalue:.6f}')
                c.metric('R²', f'{fit.rvalue ** 2:.3f}')
                st.info('Estimated historical trend is increasing.' if fit.slope > 0 else
                        'Estimated historical trend is decreasing.' if fit.slope < 0 else
                        'Estimated historical trend is approximately flat.')
                st.caption('Ordinary linear regression is a preliminary significance test; time-series correlation has not been accounted for.')
            else:
                st.warning('Select at least five complete years for trend analysis.')
            st.caption(f'Source: NASA Giovanni processed data. Region: {country_label}. Country-shape coverage must be verified for each dataset.')

with ml_tab:
    st.subheader('Historical machine learning evaluations')
    st.warning('These existing Random Forest models were trained on the original Sarawak-area bounding box. They are NOT country-wide forecasts and must not be applied to selected countries.')
    legacy_variable = st.selectbox('Original model', ['Temperature', 'Rainfall'])
    if st.checkbox('Load and evaluate saved model', value=False):
        try:
            evaluate_legacy_model(legacy_variable)
        except Exception as exc:
            st.error(f'Model evaluation failed: {type(exc).__name__}: {exc}')
            st.info('Country exploration can still be used independently from the existing ML models.')

st.divider()
st.caption('TerraTrace · NASA Space Apps Challenge 2026 · Super Larpathon')
