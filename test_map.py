import streamlit as st
from map_view import render_country_map

st.set_page_config(page_title="TerraTrace Map", layout="wide")

st.title("TerraTrace — Southeast Asia Explorer")

selected = render_country_map()

st.subheader("Selected Country")
st.write(selected.replace("_", " ").title())
