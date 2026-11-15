import json
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from scipy.stats import linregress
from streamlit_folium import st_folium

CODES = {
    'BRN': 'brunei', 'KHM': 'cambodia', 'IDN': 'indonesia',
    'LAO': 'laos', 'MYS': 'malaysia', 'MMR': 'myanmar',
    'PHL': 'philippines', 'SGP': 'singapore', 'THA': 'thailand',
    'TLS': 'timor_leste', 'VNM': 'vietnam'
}
NAMES = {
    'brunei': 'Brunei', 'cambodia': 'Cambodia', 'indonesia': 'Indonesia',
    'laos': 'Laos', 'malaysia': 'Malaysia', 'myanmar': 'Myanmar',
    'philippines': 'Philippines', 'singapore': 'Singapore',
    'thailand': 'Thailand', 'timor_leste': 'Timor-Leste', 'vietnam': 'Vietnam'
}

@st.cache_data

def read_shapes():
    path = Path(__file__).resolve().parent / 'data' / 'geo' / 'countries.geojson'
    with path.open(encoding='utf-8') as f:
        geojson = json.load(f)
    features = []
    for item in geojson['features']:
        props = item.get('properties', {})
        code = props.get('ISO3166-1-Alpha-3') or props.get('ADM0_A3') or props.get('ISO_A3')
        if code in CODES:
            feature = dict(item)
            feature['properties'] = dict(props)
            feature['properties']['country_id'] = CODES[code]
            feature['properties']['country_name'] = NAMES[CODES[code]]
            features.append(feature)
    if len({f['properties']['country_id'] for f in features}) != len(CODES):
        found = {f['properties']['country_id'] for f in features}
        missing = [NAMES[c] for c in CODES.values() if c not in found]
        raise ValueError('Missing country boundaries: ' + ', '.join(missing))
    return features


def calculate_country_trends(annual, variable, first_year, last_year):
    rows = []
    if annual.empty:
        return pd.DataFrame(columns=['region_id', 'country', 'trend', 'years', 'start', 'end'])
    filtered = annual[
        (annual['variable'] == variable) &
        (annual['year'].between(first_year, last_year)) &
        (annual['region_id'].isin(NAMES))
    ]
    for country_id, group in filtered.groupby('region_id'):
        group = group.dropna(subset=['year', 'annual_value']).sort_values('year')
        if group['year'].nunique() < 5:
            continue
        fit = linregress(group['year'], group['annual_value'])
        rows.append({
            'region_id': country_id,
            'country': NAMES[country_id],
            'trend': fit.slope * 10,
            'years': len(group),
            'start': int(group['year'].min()),
            'end': int(group['year'].max())
        })
    return pd.DataFrame(rows, columns=['region_id', 'country', 'trend', 'years', 'start', 'end'])


def render_trend_heatmap(trends, variable):
    features = read_shapes()
    lookup = trends.set_index('region_id').to_dict('index') if not trends.empty else {}
    bound = max([abs(x['trend']) for x in lookup.values()] + [0.01])
    scale = LinearColormap(['#2786c7', '#e7edf0', '#e76b42'], vmin=-bound, vmax=bound)
    unit = '°C/decade' if variable == 'temperature' else 'mm/decade'
    heatmap = folium.Map(
        location=[5, 112], zoom_start=4, tiles=None,
        zoom_control=True, control_scale=False, prefer_canvas=True
    )
    for feature in features:
        slug = feature['properties']['country_id']
        info = lookup.get(slug)
        color = scale(info['trend']) if info else '#526175'
        feature['properties']['trend_label'] = f"{info['trend']:+.2f} {unit}" if info else 'Insufficient data'
        feature['properties']['period_label'] = f"{info['start']}–{info['end']} ({info['years']} years)" if info else '—'
        folium.GeoJson(
            feature,
            name=NAMES[slug],
            style_function=lambda feat, fill=color: {
                'fillColor': fill, 'fillOpacity': 0.85,
                'color': '#172635', 'weight': 1.5
            },
            highlight_function=lambda feat: {
                'fillOpacity': 1, 'color': '#ffffff', 'weight': 3
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['country_name', 'trend_label', 'period_label'],
                aliases=['Country:', 'Trend:', 'Observed years:'],
                sticky=True
            )
        ).add_to(heatmap)
    response = st_folium(
        heatmap, height=530, width=None,
        key=f'heatmap_{variable}',
        returned_objects=[]
    )
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;max-width:540px;margin:auto">'
        '<span style="font-size:0.85rem">Decreasing</span>'
        '<div style="flex:1;height:13px;border-radius:6px;background:linear-gradient(90deg,#2786c7,#e7edf0,#e76b42)"></div>'
        '<span style="font-size:0.85rem">Increasing</span></div>',
        unsafe_allow_html=True
    )
    st.caption(f'Color scale: −{bound:.2f} to +{bound:.2f} {unit}, centered on zero. Gray = fewer than five complete years. Hover over a country for its value.')
    return response
