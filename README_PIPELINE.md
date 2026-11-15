# TerraTrace — NASA Giovanni batch pipeline

This package processes the same two Giovanni data products used in the existing TerraTrace prototype:

- **Temperature:** MERRA-2 `M2SDNXSLV_5_12_4_T2MMEAN`, daily, Kelvin → Celsius.
- **Rainfall:** GPCP `GPCPMON_3_3_satellite_precip`, monthly, mm/day → estimated monthly total (mm) using calendar-month length.

The outputs are **area-averaged measurements**. The included Sarawak source files use a **bounding box**, NOT Sarawak's administrative boundary.

## Install into the existing TerraTrace project

Copy these folders and files into your existing TerraTrace root: `pipeline/`, `config/`, `data/raw/`, `tests/`, `requirements-pipeline.txt`, and `README_PIPELINE.md`.

**Do not replace** your existing `app.py`, `analysis.py`, `models/` or `notebooks/`. Keep your original `data/temperature.csv` and `data/rainfall.csv` until the dashboard is updated. The supplied `data/raw/malaysia_sarawak/` copies are the same raw NASA CSVs supplied in the conversation.

From the TerraTrace project root, run:

```powershell
python -m pip install -r requirements-pipeline.txt
python -m pipeline.run
python -m unittest discover -s tests -v
```

You will get:

- `data/processed/observations.csv`: one row per raw daily/monthly observation, standardized columns (`region_id`, `variable`, `time`, `value`, `unit`, ...).
- `data/processed/annual_summary.csv`: one row per region, variable and year; temperature uses **annual average °C**, rainfall uses **annual total mm/year**.
- `data/processed/catalog.json`: dataset provenance, source, coverage, selection geometry metadata, observation count and completeness checks.

## Add a country or Malaysian subregion

1. Use NASA Giovanni `Time Series, Area-Averaged`, **2000-01-01 to 2025-12-31**, with the same two dataset IDs above.
2. Select a verified country **shape** (not the capital city) in Giovanni, or an explicitly described bounding box.
3. Save the downloads as:

```text
data/raw/indonesia/temperature.csv
data/raw/indonesia/rainfall.csv
```

4. Register `indonesia` in `config/regions.json`.

For a bounding box:

```json
"example_region": {
  "label": "Example bounding-box region",
  "boundary_type": "bounding_box",
  "bbox": [100.0, 0.0, 110.0, 10.0],
  "description": "Example only; replace with the exact Giovanni selection."
}
```

For an actual country shape:

```json
"indonesia": {
  "label": "Indonesia",
  "boundary_type": "country_shape",
  "description": "Indonesia country shape selected in Giovanni; manually confirmed against downloaded source.",
  "shape_verified": true
}
```

**Set `shape_verified: true` only after checking that you selected Indonesia's country shape in Giovanni.** Do not mark an approximate bounding box as a country shape. The CSV time series may not contain enough information to independently prove the shape was selected; this step requires your confirmation. The pipeline preserves the source metadata so reviewers can audit it.

5. Run `python -m pipeline.run` again. No processing code changes are needed for more regions using the **same two dataset identifiers and units**.

Other variables or alternate NASA products **must** be registered and given an explicit unit conversion before they can be processed. This is intentional to avoid invalid results.

## Important quality rules

- The pipeline searches for the `time, mean_...` header instead of hardcoding `skiprows=8`.
- Metadata fill values are converted to missing data, not physical measurements.
- Default mode **rejects** missing periods and fill values rather than silently creating false annual trends.
- To process a partially complete dataset intentionally, run `python -m pipeline.run --allow-gaps`; incomplete annual summaries will be labeled `complete_year=False` and should be excluded from trend analysis.
- Configured bounding boxes are checked against Giovanni's `User Bounding Box` in the CSV header.
- Historical **trend analysis** can be run on a new region's annual summaries; existing saved ML models remain specific to their original Sarawak-area training data and **must not be treated as validated models for new regions**.

## Simple Streamlit integration example

The pipeline does **not** replace the existing dashboard automatically. To read standardized annual values:

```python
from pipeline.api import get_available_regions, load_annual_series

regions = get_available_regions()
annual_temp = load_annual_series(regions[0], "temperature")
annual_rain = load_annual_series(regions[0], "rainfall")
```

The fields returned are `year`, `annual_value`, `annual_unit`, and additional provenance/completeness fields. Connect the actual region selected in your future map to `load_annual_series(region_id, variable)`.
