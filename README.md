# TerraTrace 🌍

https://terratracespace.streamlit.app/

**Exploring 26 years of temperature and rainfall changes across Southeast Asia.**

TerraTrace was built by **Super Larpathon**, a two-person team, for the **NASA Space Apps Challenge 2026** under the challenge **“Be An Earth System Trend Detective!”**

This is a **hackathon project**, developed within a limited timeframe. Our goal was to turn large NASA Earth observation datasets into something people could actually explore without needing to know how to process climate data or run statistical analysis themselves.

## What is TerraTrace?

TerraTrace is an interactive Streamlit web application that uses NASA Earth observation data from **2000 to 2025** to investigate temperature and rainfall patterns across **11 Southeast Asian countries**:

Brunei, Cambodia, Indonesia, Laos, Malaysia, Myanmar, the Philippines, Singapore, Thailand, Timor-Leste, and Vietnam.

We built it around four questions from the challenge:

- **What** is changing?
- **Where** is it changing?
- **How much** has it changed?
- **Are the observed trends statistically significant?**

## Features

- **Country explorer:** Select a country and explore historical temperature and rainfall measurements through interactive charts and a map.
- **Historical trend analysis:** Estimate changes per decade using linear regression, with p-values and R² values to help interpret the results.
- **Country comparison:** Compare the same environmental variable between two countries over a shared period.
- **Period comparison:** Compare two different time periods within one country.
- **Dynamic trend map:** Visualize geographical differences in estimated trends, with colors updating based on the selected variable and time range.
- **Annual anomalies:** See how individual years differ from a selected reference-period average.
- **Plain-language explanations:** Make the results easier to follow for users without a background in statistics.

## Data sources

Our datasets were accessed through **NASA Giovanni**:

| Variable | Dataset | Original resolution | Processing |
| --- | --- | --- | --- |
| 2-meter air temperature | MERRA-2 — M2SDNXSLV v5.12.4 | Daily | Convert Kelvin to °C and calculate annual averages |
| Multisatellite precipitation | GPCP — GPCPMON v3.3 | Monthly | Convert monthly mean precipitation rates (mm/day) into monthly totals, then sum into annual totals |

All analyses in the app focus on **2000–2025**. Country-level results depend on the geographic shapes selected when exporting each dataset from Giovanni.

- [NASA Giovanni](https://giovanni.gsfc.nasa.gov/giovanni/)
- [MERRA-2 temperature dataset](https://disc.gsfc.nasa.gov/datasets/M2SDNXSLV_5.12.4/summary)
- [GPCP precipitation dataset](https://disc.gsfc.nasa.gov/datasets/GPCPMON_3.3/summary)

## Built with

**Python · Streamlit · Pandas · NumPy · SciPy · Plotly · Folium · Git · GitHub**

We used Python for the analysis pipeline and Streamlit for the application. Pandas and NumPy handle the datasets, SciPy supports trend calculations, Plotly provides interactive graphs, and Folium handles geographical exploration.

## Running locally

1. Clone the repository and open the project directory.
2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

The application needs the processed datasets (including `data/processed/annual_summary.csv`) and geographical boundaries (`data/geo/countries.geojson`) to be available at their expected paths.

If you have the original Giovanni CSV exports and the pipeline configuration, you can regenerate the processed data with:

```bash
python -m pipeline.run
```

## What we couldn't finish during the hackathon

We had more ideas than we could realistically complete in the hackathon timeframe. Some of the features we considered were:

- **Machine-learning predictions:** We explored using Random Forest models to forecast temperature and rainfall. We chose not to include an unvalidated forecasting feature in the final application. Historical trend analysis remained our priority.
- **More environmental variables:** We originally wanted to investigate more Earth-system measurements, but narrowed the scope to temperature and rainfall so we could complete the data collection, processing, and visualizations.
- **Deeper analysis:** We considered additional analyses such as seasonal patterns and relationships between temperature and rainfall, but prioritized a working, understandable historical exploration tool.

These are **ideas for future development**, not features we claim are already available.

## Limitations

- The project uses historical observations and reanalysis products; it **does not predict future climate conditions**.
- A linear trend describes the selected historical period, not necessarily what will happen next.
- Regression p-values are preliminary indicators of significance; ordinary linear regression does not fully address autocorrelation and other complications in environmental time series.
- Country-level values are based on the geographic regions and spatial averaging used in NASA Giovanni. They should not be treated as local weather measurements.
- Different periods, datasets, and geographic definitions may produce different results.

## Why we built it

We wanted to make environmental data less intimidating. There is a huge amount of scientific information available, but not everyone has the time or experience to download datasets, clean them, and interpret the numbers.

TerraTrace is our attempt to make that process simpler—and to encourage people to explore how their environment has changed.

---

**Team:** Super Larpathon  
**Event:** NASA Space Apps Challenge 2026  
**Challenge:** Be An Earth System Trend Detective!  
**Status:** Hackathon prototype

*This is an independent hackathon project using NASA data. It is not an official NASA product or an endorsement by NASA.*
