"""Checks applied before standardized files are written."""

from __future__ import annotations

import pandas as pd

from .process import PipelineError


def validate_outputs(observations: pd.DataFrame, annual: pd.DataFrame) -> None:
    if observations.empty or annual.empty:
        raise PipelineError("No processed data available")
    if observations.duplicated(["region_id", "variable", "time"]).any():
        raise PipelineError("Duplicate (region, variable, time) key in processed data")
    if annual.duplicated(["region_id", "variable", "year"]).any():
        raise PipelineError("Duplicate (region, variable, year) key in annual summary")
    if annual["annual_value"].isna().any():
        raise PipelineError("One or more years contain no usable observations")
    if ((annual["valid_observations"] > annual["expected_observations"]) |
            (annual["valid_observations"] < 0)).any():
        raise PipelineError("Annual sample counts are inconsistent")
    for region_id, grp in annual.groupby("region_id"):
        if set(grp["variable"]) == {"temperature", "rainfall"}:
            spans = grp.groupby("variable")["year"].agg(["min", "max"])
            if len(set(zip(spans["min"], spans["max"]))) != 1:
                print(f"WARNING: {region_id} temperature and rainfall have different year coverage")
