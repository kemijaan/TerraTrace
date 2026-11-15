"""Read and standardize NASA Giovanni area-averaged time-series CSV files."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class PipelineError(ValueError):
    """Raised when a source file fails a provenance or data-quality check."""


DATASETS = {
    "temperature": {
        "column": "mean_M2SDNXSLV_5_12_4_T2MMEAN",
        "source": "NASA MERRA-2 M2SDNXSLV v5.12.4",
        "frequency": "daily",
        "raw_unit": "K",
        "unit": "°C",
    },
    "rainfall": {
        "column": "mean_GPCPMON_3_3_satellite_precip",
        "source": "NASA GPCP GPCPMON v3.3",
        "frequency": "monthly",
        "raw_unit": "mm/day",
        "unit": "mm/month",
    },
}


@dataclass
class ProcessedDataset:
    observations: pd.DataFrame
    annual: pd.DataFrame
    metadata: dict


def load_region_config(path: Path) -> dict:
    try:
        regions = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot load region config {path}: {exc}") from exc
    if not isinstance(regions, dict) or not regions:
        raise PipelineError("Region configuration must contain at least one region")
    for region_id, region in regions.items():
        if not isinstance(region, dict) or not region.get("label"):
            raise PipelineError(f"Region {region_id!r} needs a label")
        kind = region.get("boundary_type")
        if kind == "bounding_box":
            bbox = region.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise PipelineError(f"Region {region_id!r} needs bbox [west,south,east,north]")
            if not all(isinstance(n, (float, int)) for n in bbox):
                raise PipelineError(f"Region {region_id!r} has non-numeric bbox coordinates")
            west, south, east, north = bbox
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise PipelineError(f"Region {region_id!r} has an invalid bounding box")
        elif kind == "country_shape":
            if not region.get("shape_verified") or not region.get("description"):
                raise PipelineError(
                    f"Region {region_id!r} requires a description and shape_verified=true; "
                    "verify the Giovanni country shape manually before processing"
                )
        else:
            raise PipelineError(f"Region {region_id!r}: unsupported boundary_type {kind!r}")
    return regions


def _read_giovanni_csv(path: Path) -> tuple[dict, str, pd.DataFrame]:
    try:
        contents = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise PipelineError(f"Cannot read {path}: {exc}") from exc
    lines = list(csv.reader(io.StringIO(contents)))
    header_at = None
    for i, fields in enumerate(lines):
        if fields and fields[0].strip().lower() == "time":
            header_at = i
            break
    if header_at is None:
        raise PipelineError(f"{path}: no time header found")
    header = [field.strip() for field in lines[header_at]]
    if len(header) != 2 or not header[1].startswith("mean_"):
        raise PipelineError(f"{path}: expected exactly two columns: time and mean_<dataset>")
    metadata = {}
    for fields in lines[:header_at]:
        if fields and fields[0].strip():
            key = fields[0].strip().removesuffix(":")
            metadata[key] = ",".join(fields[1:]).strip()
    try:
        table = pd.read_csv(io.StringIO(contents), skiprows=header_at, skipinitialspace=True)
    except Exception as exc:
        raise PipelineError(f"{path}: malformed CSV body: {exc}") from exc
    table.columns = table.columns.str.strip()
    if list(table.columns) != header:
        raise PipelineError(f"{path}: unexpected columns {list(table.columns)}")
    if table.empty:
        raise PipelineError(f"{path}: empty time series")
    return metadata, header[1], table


def _check_spatial_selection(metadata: dict, region: dict, path: Path) -> None:
    if region["boundary_type"] != "bounding_box":
        return
    raw_bbox = metadata.get("User Bounding Box")
    if not raw_bbox:
        raise PipelineError(f"{path}: no User Bounding Box metadata to verify area")
    try:
        actual = [float(v.strip()) for v in raw_bbox.split(",")]
    except ValueError as exc:
        raise PipelineError(f"{path}: invalid User Bounding Box {raw_bbox!r}") from exc
    expected = region["bbox"]
    if len(actual) != 4 or any(abs(a - b) > 0.00001 for a, b in zip(actual, expected)):
        raise PipelineError(
            f"{path}: region mismatch, file bbox={actual}, configured bbox={expected}"
        )


def _dates_from_table(table: pd.DataFrame, path: Path) -> pd.Series:
    try:
        dates = pd.to_datetime(table["time"], errors="raise", utc=True)
    except (ValueError, TypeError) as exc:
        raise PipelineError(f"{path}: invalid timestamp: {exc}") from exc
    if dates.isna().any():
        raise PipelineError(f"{path}: missing timestamps")
    return dates.dt.tz_localize(None).dt.normalize()


def process_file(path: Path, region_id: str, region: dict, variable: str,
                 allow_gaps: bool = False) -> ProcessedDataset:
    if variable not in DATASETS:
        raise PipelineError(f"Unsupported variable: {variable!r}")
    settings = DATASETS[variable]
    meta, value_col, table = _read_giovanni_csv(path)
    if value_col != settings["column"]:
        raise PipelineError(
            f"{path}: expected {settings['column']!r}, found {value_col!r}. "
            "Check the Giovanni variable and dataset version before processing."
        )
    _check_spatial_selection(meta, region, path)
    dates = _dates_from_table(table, path)
    if dates.duplicated().any():
        raise PipelineError(f"{path}: duplicate timestamps detected")
    if variable == "rainfall" and (dates.dt.day != 1).any():
        raise PipelineError(f"{path}: monthly rainfall observations must be dated on day 1")
    if "User Start Date" in meta and "User End Date" in meta:
        try:
            start_requested = pd.Timestamp(meta["User Start Date"][:10])
            end_requested = pd.Timestamp(meta["User End Date"][:10])
        except (ValueError, TypeError) as exc:
            raise PipelineError(f"{path}: invalid requested date-range metadata") from exc
        if settings["frequency"] == "monthly":
            start_requested = start_requested.to_period("M").to_timestamp()
            end_requested = end_requested.to_period("M").to_timestamp()
        if dates.min() < start_requested or dates.max() > end_requested:
            raise PipelineError(f"{path}: observations fall outside requested date range")
    else:
        start_requested, end_requested = dates.min(), dates.max()

    try:
        values = pd.to_numeric(table[value_col], errors="raise")
    except (ValueError, TypeError) as exc:
        raise PipelineError(f"{path}: non-numeric measurements: {exc}") from exc
    fill_key = f"Fill Value ({value_col})"
    if fill_key in meta:
        try:
            fill = float(meta[fill_key])
        except ValueError as exc:
            raise PipelineError(f"{path}: invalid fill value {meta[fill_key]!r}") from exc
        values = values.mask(values == fill)
    missing_values = int(values.isna().sum())
    if variable == "temperature":
        bad = (values.notna()) & (~values.between(180, 350))
        if bad.any():
            raise PipelineError(f"{path}: temperature values outside plausible Kelvin range")
        converted = values - 273.15
    else:
        bad = (values.notna()) & ((values < 0) | (values > 1000))
        if bad.any():
            raise PipelineError(f"{path}: invalid precipitation rate (mm/day)")
        converted = values * dates.dt.days_in_month

    period_start = start_requested
    period_end = end_requested
    if settings["frequency"] == "monthly":
        expected_dates = pd.date_range(period_start, period_end, freq="MS")
    else:
        expected_dates = pd.date_range(period_start, period_end, freq="D")
    observed_dates = pd.DatetimeIndex(dates)
    missing_dates = expected_dates.difference(observed_dates)
    if len(missing_dates) and not allow_gaps:
        show = ", ".join(str(d.date()) for d in missing_dates[:3])
        raise PipelineError(f"{path}: {len(missing_dates)} missing periods (first: {show}). Use --allow-gaps to permit incomplete coverage.")
    if not allow_gaps and missing_values:
        raise PipelineError(f"{path}: {missing_values} missing/fill values. Use --allow-gaps to allow incomplete coverage.")

    observations = pd.DataFrame({
        "region_id": region_id,
        "region_label": region["label"],
        "boundary_type": region["boundary_type"],
        "variable": variable,
        "time": dates,
        "value": converted,
        "unit": settings["unit"],
        "frequency": settings["frequency"],
        "source": settings["source"],
    }).sort_values("time").reset_index(drop=True)

    annual_rows = []
    for year, frame in observations.groupby(observations["time"].dt.year, sort=True):
        year = int(year)
        if settings["frequency"] == "daily":
            expected_count = 366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365
        else:
            expected_count = 12
        valid = frame["value"].notna()
        valid_count = int(valid.sum())
        annual_value = frame.loc[valid, "value"].mean() if variable == "temperature" else frame.loc[valid, "value"].sum(min_count=1)
        annual_rows.append({
            "region_id": region_id,
            "region_label": region["label"],
            "variable": variable,
            "year": year,
            "annual_value": annual_value,
            "annual_unit": "°C" if variable == "temperature" else "mm/year",
            "valid_observations": valid_count,
            "expected_observations": expected_count,
            "complete_year": valid_count == expected_count,
        })
    annual = pd.DataFrame(annual_rows)
    details = {
        "region_id": region_id,
        "region_label": region["label"],
        "boundary_type": region["boundary_type"],
        "boundary_description": region.get("description", ""),
        "variable": variable,
        "source_file": str(path.as_posix()),
        "source_dataset": settings["source"],
        "raw_column": value_col,
        "raw_unit": settings["raw_unit"],
        "standard_unit": settings["unit"],
        "frequency": settings["frequency"],
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "observations": int(len(table)),
        "missing_timestamps": int(len(missing_dates)),
        "missing_values": missing_values,
        "complete_years": int(annual["complete_year"].sum()),
        "incomplete_years": annual.loc[~annual["complete_year"], "year"].astype(int).tolist(),
        "user_bounding_box": meta.get("User Bounding Box"),
        "data_bounding_box": meta.get("Data Bounding Box"),
        "source_reproduction_url": meta.get("URL to Reproduce Results"),
        "original_title": meta.get("Title"),
    }
    return ProcessedDataset(observations=observations, annual=annual, metadata=details)
