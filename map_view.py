
import json
from pathlib import Path
import folium
import streamlit as st
from streamlit_folium import st_folium

COUNTRIES = {
    "BRN": ("brunei", "Brunei"),
    "KHM": ("cambodia", "Cambodia"),
    "IDN": ("indonesia", "Indonesia"),
    "LAO": ("laos", "Laos"),
    "MYS": ("malaysia", "Malaysia"),
    "MMR": ("myanmar", "Myanmar"),
    "PHL": ("philippines", "Philippines"),
    "SGP": ("singapore", "Singapore"),
    "THA": ("thailand", "Thailand"),
    "TLS": ("timor_leste", "Timor-Leste"),
    "VNM": ("vietnam", "Vietnam")
}

@st.cache_data
def load_country_shapes():
    path = Path("data/geo/countries.geojson")
    with path.open("r", encoding="utf-8") as file:
        geojson = json.load(file)

    features = []

    for feature in geojson["features"]:
        code = feature["properties"].get("ISO3166-1-Alpha-3")

        if code in COUNTRIES:
            slug, name = COUNTRIES[code]
            feature["properties"]["name"] = name
            feature["properties"]["slug"] = slug
            features.append(feature)

    if len(features) != len(COUNTRIES):
        raise ValueError(
            f"Expected {len(COUNTRIES)} countries, found {len(features)}."
        )

    return {
        "type": "FeatureCollection",
        "features": features
    }

def render_country_map():
    if "selected_country" not in st.session_state:
        st.session_state.selected_country = "malaysia"

    shapes = load_country_shapes()

    country_map = folium.Map(
        location=[5, 112],
        zoom_start=4,
        tiles=None,
        zoom_control=True,
        prefer_canvas=True
)

    selected = st.session_state.selected_country

    for feature in shapes["features"]:
        slug = feature["properties"]["slug"]
        active = slug == selected

        folium.GeoJson(
            feature,
            name=slug,
            style_function=lambda x, active=active: {
                "fillColor": "#16a085" if active else "#3498db",
                "color": "#ffffff",
                "weight": 2 if active else 1,
                "fillOpacity": 0.65 if active else 0.30
            },
            highlight_function=lambda x: {
                "fillColor": "#f1c40f",
                "color": "#ffffff",
                "weight": 2,
                "fillOpacity": 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name"],
                labels=False,
                sticky=True
            )
        ).add_to(country_map)

    result = st_folium(
        country_map,
        height=550,
        width=None,
        key="southeast_asia_map",
        returned_objects=["last_object_clicked_tooltip"]
    )

    clicked = result.get("last_object_clicked_tooltip")

    if clicked:
        clicked = str(clicked).strip()

        for slug, name in COUNTRIES.values():
            if clicked == name and slug != selected:
                st.session_state.selected_country = slug
                st.rerun()

    return st.session_state.selected_country
