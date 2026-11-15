from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / 'data' / 'processed' / 'annual_summary.csv'
REGION_NAMES = {
    'brunei': 'Brunei', 'cambodia': 'Cambodia', 'indonesia': 'Indonesia',
    'laos': 'Laos', 'malaysia': 'Malaysia', 'myanmar': 'Myanmar',
    'philippines': 'Philippines', 'singapore': 'Singapore',
    'thailand': 'Thailand', 'timor_leste': 'Timor-Leste', 'vietnam': 'Vietnam'
}
VARIABLE_NAMES = {'temperature': 'Temperature', 'rainfall': 'Rainfall'}

st.set_page_config(page_title='TerraTrace | Earth Trend Detective', page_icon='🌍', layout='wide')
st.markdown('''
<style>
.stApp { background: #07131f; color: #e8f1f7; }
.block-container { max-width: 1240px; padding-top: 1.7rem; padding-bottom: 2.2rem; }
[data-testid="stMetric"] { background: #112638; border: 1px solid #254255; border-radius: 12px; padding: 14px 16px; }
[data-testid="stMetricLabel"] { color: #a8c3d4; }
[data-testid="stMetricValue"] { color: #f0f8ff; }
.stTabs [data-baseweb="tab-list"] { gap: 12px; }
.stTabs [data-baseweb="tab"] { background: #112638; border-radius: 9px; padding: 10px 18px; }
.stTabs [aria-selected="true"] { background: #175c65; }
.hero-kicker { color: #5edecb; font-weight: 700; letter-spacing: 2px; font-size: 12px; }
.hero-title { font-size: clamp(2rem,4vw,3.2rem); font-weight: 800; letter-spacing: -1px; margin: 0.3rem 0; }
.hero-subtitle { color: #b5cad7; max-width: 760px; line-height: 1.65; }
</style>
''', unsafe_allow_html=True)

@st.cache_data
def load_processed_data():
    if not SUMMARY.exists():
        return pd.DataFrame()
    df = pd.read_csv(SUMMARY)
    required = {'region_id', 'variable', 'year', 'annual_value', 'complete_year'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing pipeline output columns: {sorted(required - set(df.columns))}')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['annual_value'] = pd.to_numeric(df['annual_value'], errors='coerce')
    df['complete_year'] = df['complete_year'].astype(str).str.strip().str.lower().isin(['true', '1'])
    df = df[df['complete_year'] & df['region_id'].isin(REGION_NAMES) & df['variable'].isin(VARIABLE_NAMES)]
    df = df.dropna(subset=['year', 'annual_value']).copy()
    df['year'] = df['year'].astype(int)
    duplicates = df.duplicated(['region_id', 'variable', 'year'])
    if duplicates.any():
        raise ValueError('The processed data contains duplicate region/variable/year rows.')
    return df

def unit_for(variable):
    return '°C' if variable == 'temperature' else 'mm/year'

def trend_metrics(series):
    if len(series) < 5 or series['year'].nunique() < 5:
        return None
    return linregress(series['year'], series['annual_value'])

def explain_trend(fit, variable, label, years):
    slope = fit.slope * 10
    unit = unit_for(variable)
    if slope > 0:
        direction = 'increased'
    elif slope < 0:
        direction = 'decreased'
    else:
        direction = 'showed no linear change'
    signal = ('The fitted linear trend is statistically distinguishable from zero under ordinary regression assumptions.'
              if fit.pvalue < 0.05 else
              'The fitted linear trend is not statistically distinguishable from zero at the 5% threshold under ordinary regression assumptions.')
    reliability = ('R² describes how much of the variation across these annual values is explained by a straight-line fit; '
                   'it does not measure data quality or prove a cause.')
    st.markdown('#### What does this mean?')
    st.write(f'From **{years[0]} to {years[-1]}**, **{label}** {direction} at an estimated rate of **{abs(slope):.3f} {unit} per decade**. '
             f'This is a fitted historical rate, not a predicted future change.')
    st.write(signal)
    st.caption(reliability + ' The p-value is preliminary: serial correlation and other time-series effects are not accounted for.')

def chart_style(fig):
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#d9eaf2'),
        hovermode='x unified', margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text='', xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='rgba(180,215,230,0.12)')
    )
    return fig

st.markdown('<div class="hero-kicker">NASA SPACE APPS CHALLENGE · 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🌍 TerraTrace</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Explore NASA temperature and rainfall observations across 11 Southeast Asian countries. Discover historical trends, compare countries, and understand how the environment is changing.</div>', unsafe_allow_html=True)
st.write('')

try:
    annual = load_processed_data()
except Exception as exc:
    st.error(f'Unable to read processed NASA data: {type(exc).__name__}: {exc}')
    annual = pd.DataFrame()

explorer_tab, compare_tab, guide_tab = st.tabs(['🌏 Country Explorer', '📊 Comparison', '📖 Reading the Results'])

with explorer_tab:
    explorer_mode = st.radio('Map view', ['Explore country', 'Trend heatmap'], horizontal=True, key='explorer_map_mode')
    if explorer_mode == 'Trend heatmap':
        st.subheader('Geographic trend heatmap')
        st.write('Explore how historical temperature or rainfall trends vary across Southeast Asia. Change the variable and years to update the map.')
        heat_variable = st.radio('Environmental variable', ['temperature', 'rainfall'], format_func=lambda v: VARIABLE_NAMES[v], horizontal=True, key='heat_variable')
        if annual.empty:
            heat_years = (2000, 2025)
        else:
            pool = annual[annual['variable'] == heat_variable]
            heat_years = (int(pool['year'].min()), int(pool['year'].max())) if not pool.empty else (2000, 2025)
        if heat_years[0] < heat_years[1]:
            heat_start, heat_end = st.slider('Heatmap year range', heat_years[0], heat_years[1], heat_years, key='heat_years')
        else:
            heat_start = heat_end = heat_years[0]
        try:
            from trend_map import calculate_country_trends, render_trend_heatmap
            trends = calculate_country_trends(annual, heat_variable, heat_start, heat_end)
            render_trend_heatmap(trends, heat_variable)
            if not trends.empty:
                unit = '°C/decade' if heat_variable == 'temperature' else 'mm/decade'
                display = trends.sort_values('trend', ascending=False).rename(columns={
                    'country': 'Country', 'trend': f'Trend ({unit})', 'years': 'Years of data',
                    'start': 'First year', 'end': 'Last year'
                })
                st.dataframe(display[['Country', f'Trend ({unit})', 'Years of data', 'First year', 'Last year']].style.format({f'Trend ({unit})': '{:+.3f}'}), use_container_width=True, hide_index=True)
            else:
                st.info('No countries have five complete observations in this year range yet. Their map areas will be gray until processed NASA country datasets are available.')
            st.caption('The heatmap displays fitted historical slopes, not absolute temperatures or rainfall amounts. Comparisons across countries may use different available years. Ordinary regression assumptions apply; color intensity does not indicate statistical significance.')
        except Exception as exc:
            st.error(f'Trend heatmap unavailable: {type(exc).__name__}: {exc}')
    else:
        st.subheader('Explore a country')
        try:
            from map_view import render_country_map
            selected_country = render_country_map()
        except Exception as exc:
            st.warning(f'Interactive map unavailable: {type(exc).__name__}: {exc}')
            selected_country = st.selectbox('Select a country', list(REGION_NAMES), format_func=lambda k: REGION_NAMES[k])
        if selected_country not in REGION_NAMES:
            st.error('Select one of the 11 supported countries.')
        else:
            country_label = REGION_NAMES[selected_country]
            st.write(f'**Selected country:** {country_label}')
            if annual.empty:
                st.info('No country-level processed data yet. Download country-shape CSVs and run `python -m pipeline.run`.')
            else:
                available = annual[annual['region_id'] == selected_country].copy()
                choices = sorted(available['variable'].unique())
                if not choices:
                    st.info(f'No complete country-level NASA data has been processed for {country_label} yet.')
                else:
                    variable = st.selectbox('Environmental variable', choices, format_func=lambda v: VARIABLE_NAMES[v], key='explorer_variable')
                    series = available[available['variable'] == variable].sort_values('year')
                    years = sorted(series['year'].unique().tolist())
                    if len(years) > 1:
                        start, end = st.slider('Historical year range', min(years), max(years), (min(years), max(years)), key='explorer_years')
                        series = series[series['year'].between(start, end)]
                    unit = unit_for(variable)
                    name = 'Annual average temperature' if variable == 'temperature' else 'Annual rainfall total'
                    fig = px.line(series, x='year', y='annual_value', markers=True, title=f'{country_label} — {name}', labels={'year': 'Year', 'annual_value': f'{name} ({unit})'})
                    fig.update_traces(line=dict(color='#59d4c5', width=3), marker=dict(size=7))
                    st.plotly_chart(chart_style(fig), use_container_width=True)
                    y = series['year'].astype(int).tolist()
                    st.caption(f'Available in this selection: {len(series)} complete-year observations, {min(y)}–{max(y)}.')
                    fit = trend_metrics(series)
                    if fit is None:
                        st.warning('At least five complete years are needed for trend analysis.')
                    else:
                        a, b, c = st.columns(3)
                        a.metric('Trend per decade', f'{fit.slope * 10:+.3f} {unit}/decade')
                        b.metric('Regression p-value', f'{fit.pvalue:.4g}')
                        c.metric('R²', f'{fit.rvalue ** 2:.3f}')
                        explain_trend(fit, variable, country_label, y)
                    st.caption('Source: NASA Giovanni, as processed by TerraTrace. Country-shape coverage must be verified. Linear regression p-values are preliminary and do not account for time-series autocorrelation.')

with compare_tab:
    comparison_mode = st.radio('Comparison type', ['Country vs country', 'Period vs period'], horizontal=True, key='comparison_mode')
    if comparison_mode == 'Country vs country':
        st.subheader('Compare countries')
        st.write('Select two countries and one variable. TerraTrace compares only years with complete observations for **both** countries.')
        col1, col2 = st.columns(2)
        with col1:
            country_a = st.selectbox('Country A', list(REGION_NAMES), index=4, format_func=lambda k: REGION_NAMES[k], key='compare_a')
        with col2:
            country_b = st.selectbox('Country B', list(REGION_NAMES), index=2, format_func=lambda k: REGION_NAMES[k], key='compare_b')
        variable = st.radio('Environmental variable', ['temperature', 'rainfall'], format_func=lambda v: VARIABLE_NAMES[v], horizontal=True, key='compare_variable')
        if country_a == country_b:
            st.warning('Choose two different countries to compare.')
        elif annual.empty:
            st.info('Country comparison is ready, but there are no processed country datasets yet. Add genuine country-shape NASA CSVs and run `python -m pipeline.run`.')
        else:
            a = annual[(annual['region_id'] == country_a) & (annual['variable'] == variable)][['year', 'annual_value']].rename(columns={'annual_value': 'value_a'})
            b = annual[(annual['region_id'] == country_b) & (annual['variable'] == variable)][['year', 'annual_value']].rename(columns={'annual_value': 'value_b'})
            missing = [REGION_NAMES[c] for c, series in [(country_a, a), (country_b, b)] if series.empty]
            if missing:
                st.info(f'No complete {VARIABLE_NAMES[variable].lower()} dataset available for: {", ".join(missing)}. Select another pair or download the missing data.')
            else:
                merged = a.merge(b, on='year', how='inner').sort_values('year')
                if merged.empty:
                    st.warning('The datasets have no overlapping complete years. A valid comparison is not possible.')
                else:
                    first, last = int(merged['year'].min()), int(merged['year'].max())
                    if first == last:
                        start, end = first, last
                        st.caption(f'Common year: {first}')
                    else:
                        start, end = st.slider('Comparison year range', first, last, (first, last), key='comparison_years')
                    merged = merged[merged['year'].between(start, end)].copy()
                    label_a, label_b = REGION_NAMES[country_a], REGION_NAMES[country_b]
                    units = unit_for(variable)
                    plot_data = merged.melt(id_vars='year', value_vars=['value_a', 'value_b'], var_name='country', value_name='value')
                    plot_data['country'] = plot_data['country'].map({'value_a': label_a, 'value_b': label_b})
                    fig = px.line(plot_data, x='year', y='value', color='country', markers=True, labels={'year': 'Year', 'value': f'{VARIABLE_NAMES[variable]} ({units})', 'country': 'Country'}, title=f'{label_a} vs {label_b}: {VARIABLE_NAMES[variable]}', color_discrete_sequence=['#59d4c5', '#f6b86a'])
                    fig.update_traces(line=dict(width=3), marker=dict(size=6))
                    st.plotly_chart(chart_style(fig), use_container_width=True)
                    st.caption(f'Compared {len(merged)} matching complete years ({int(merged.year.min())}–{int(merged.year.max())}).')
                    st.markdown('#### Historical comparison')
                    if len(merged) < 5:
                        st.warning('At least five common complete years are required to calculate comparable regression trends.')
                    else:
                        fit_a = linregress(merged['year'], merged['value_a'])
                        fit_b = linregress(merged['year'], merged['value_b'])
                        results = pd.DataFrame([
                            {'Country': label_a, f'Mean ({units})': merged['value_a'].mean(), f'Trend ({units}/decade)': fit_a.slope * 10, 'p-value (preliminary)': fit_a.pvalue, 'R²': fit_a.rvalue ** 2},
                            {'Country': label_b, f'Mean ({units})': merged['value_b'].mean(), f'Trend ({units}/decade)': fit_b.slope * 10, 'p-value (preliminary)': fit_b.pvalue, 'R²': fit_b.rvalue ** 2}
                        ])
                        st.dataframe(results.style.format({f'Mean ({units})': '{:.2f}', f'Trend ({units}/decade)': '{:+.3f}', 'p-value (preliminary)': '{:.4g}', 'R²': '{:.3f}'}), hide_index=True, use_container_width=True)
                        difference = (fit_a.slope - fit_b.slope) * 10
                        st.metric(f'Trend difference ({label_a} − {label_b})', f'{difference:+.3f} {units}/decade')
                        st.markdown('#### How to interpret this comparison')
                        st.write(f'Across the **same {len(merged)} years**, the historical linear trend was **{fit_a.slope * 10:+.3f} {units}/decade** in {label_a} and **{fit_b.slope * 10:+.3f} {units}/decade** in {label_b}.')
                        st.write(f'The estimated trend difference is **{difference:+.3f} {units}/decade** ({label_a} minus {label_b}). This alone does **not** establish that the countries have statistically different trends.')
                        if variable == 'rainfall':
                            st.caption('Rainfall totals can vary sharply year to year; a trend in annual totals does not describe flood risk or rainfall extremes.')
                        else:
                            st.caption('Annual average temperature is not the same as daily maximum temperature or the frequency of heatwaves.')
                        st.caption('The trend difference is descriptive, not a statistical test of whether the countries’ trends differ. Ordinary regression p-values do not account for autocorrelation.')
                    st.caption('Source: NASA Giovanni data processed by TerraTrace. Only complete common years are used. Verify each country was selected using its actual country shape.')

    else:
        st.subheader('Compare periods within a country')
        st.write('Compare two different time periods for the same country using annual NASA measurements.')
        period_country = st.selectbox('Country', list(REGION_NAMES), index=4, format_func=lambda k: REGION_NAMES[k], key='period_country')
        period_variable = st.radio('Environmental variable', ['temperature', 'rainfall'], format_func=lambda v: VARIABLE_NAMES[v], horizontal=True, key='period_variable')
        selected = annual[(annual['region_id'] == period_country) & (annual['variable'] == period_variable)].sort_values('year') if not annual.empty else pd.DataFrame()
        if selected.empty:
            st.info(f'No complete {VARIABLE_NAMES[period_variable].lower()} data is available for {REGION_NAMES[period_country]} yet.')
        else:
            years = sorted(selected['year'].unique().tolist())
            first_year, last_year = int(min(years)), int(max(years))
            if last_year - first_year < 3:
                st.warning('At least four calendar years are needed to select two separate periods.')
            else:
                midpoint = (first_year + last_year) // 2
                c1, c2 = st.columns(2)
                with c1:
                    period_a = st.slider('Period A', first_year, last_year, (first_year, midpoint), key='period_a')
                with c2:
                    period_b = st.slider('Period B', first_year, last_year, (midpoint + 1, last_year), key='period_b')
                if period_a[1] >= period_b[0] and period_b[1] >= period_a[0]:
                    st.warning('Period A and Period B must not overlap. Adjust their year ranges.')
                else:
                    a = selected[selected['year'].between(*period_a)].copy()
                    b = selected[selected['year'].between(*period_b)].copy()
                    if a.empty or b.empty:
                        st.warning('One of the chosen periods has no complete-year observations. Adjust the year ranges.')
                    else:
                        unit = unit_for(period_variable)
                        mean_a = a['annual_value'].mean()
                        mean_b = b['annual_value'].mean()
                        delta = mean_b - mean_a
                        title_a = f'{period_a[0]}–{period_a[1]}'
                        title_b = f'{period_b[0]}–{period_b[1]}'
                        x, y, z = st.columns(3)
                        x.metric(f'Period A mean ({title_a})', f'{mean_a:.2f} {unit}')
                        y.metric(f'Period B mean ({title_b})', f'{mean_b:.2f} {unit}')
                        z.metric('B minus A', f'{delta:+.2f} {unit}')
                        plot_data = pd.concat([a.assign(period='Period A'), b.assign(period='Period B')])
                        fig = px.line(plot_data, x='year', y='annual_value', color='period', markers=True,
                                      color_discrete_map={'Period A': '#59d4c5', 'Period B': '#f6b86a'},
                                      title=f'{REGION_NAMES[period_country]} — {VARIABLE_NAMES[period_variable]}: period comparison',
                                      labels={'year': 'Year', 'annual_value': f'Annual value ({unit})', 'period': 'Period'})
                        fig.update_traces(line=dict(width=3), marker=dict(size=6))
                        st.plotly_chart(chart_style(fig), use_container_width=True)
                        st.caption(f'Period A: {len(a)} complete-year observations. Period B: {len(b)} complete-year observations. Missing years are excluded.')
                        st.markdown('#### What does this mean?')
                        direction = 'higher' if delta > 0 else 'lower' if delta < 0 else 'the same as'
                        st.write(f'For **{REGION_NAMES[period_country]}**, the average annual {VARIABLE_NAMES[period_variable].lower()} in **{title_b}** was **{abs(delta):.2f} {unit} {direction}** the average in **{title_a}**.')
                        if period_variable == 'rainfall' and mean_a != 0:
                            st.write(f'This represents a **{delta / mean_a * 100:+.1f}%** difference relative to Period A.')
                        st.caption('This compares period averages only. It does not establish statistical significance, identify causes, or predict future changes. Unequal period lengths and missing years can affect the comparison.')

with guide_tab:
    st.subheader('How to read TerraTrace')
    st.markdown('**Trend per decade** shows the estimated change in the annual measurement over ten years. A positive number indicates an upward fitted trend; a negative number indicates a downward fitted trend.')
    st.markdown('**p-value** measures how unusual the estimated slope would be under a statistical model with zero true linear slope. A small p-value is not proof of causation, and these preliminary values do not adjust for serial correlation.')
    st.markdown('**R²** indicates the share of observed annual variation accounted for by the fitted straight line, not how trustworthy the underlying NASA dataset is.')
    st.markdown('**Temperature** is annual mean air temperature in °C. **Rainfall** is estimated annual total precipitation in mm/year, calculated from the monthly average daily precipitation rate using the number of days in each month.')
    st.markdown('**Country coverage** must be confirmed in NASA Giovanni when downloading each country-shape dataset. A Sarawak-only rectangle is not a Malaysia-wide measurement.')
    st.info('TerraTrace explores historical changes. It does not assign causes, predict future climate, or measure local conditions at a particular city.')

st.divider()
st.caption('TerraTrace · NASA Space Apps Challenge 2026 · Super Larpathon')
