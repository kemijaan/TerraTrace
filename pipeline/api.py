"""Simple loading helpers for the future Streamlit region selector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_available_regions(project_root: str | Path = ".") -> list[str]:
    path = Path(project_root) / "data" / "processed" / "annual_summary.csv"
    return sorted(pd.read_csv(path)["region_id"].unique().tolist())


def load_annual_series(region_id: str, variable: str, project_root: str | Path = ".",
                       complete_only: bool = True) -> pd.DataFrame:
    if variable not in {"temperature", "rainfall"}:
        raise ValueError("variable must be 'temperature' or 'rainfall'")
    path = Path(project_root) / "data" / "processed" / "annual_summary.csv"
    df = pd.read_csv(path)
    selection = df[(df["region_id"] == region_id) & (df["variable"] == variable)].copy()
    if complete_only:
        selection = selection[selection["complete_year"] == True].copy()
    if selection.empty:
        raise ValueError(f"No usable {variable} data for region {region_id!r}")
    return selection.sort_values("year").reset_index(drop=True)
