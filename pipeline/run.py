"""Run all registered local NASA Giovanni datasets through one pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .process import DATASETS, PipelineError, load_region_config, process_file
from .validate import validate_outputs


def run_pipeline(root: Path, allow_gaps: bool = False) -> dict:
    root = root.resolve()
    raw_folder = root / "data" / "raw"
    output_folder = root / "data" / "processed"
    regions = load_region_config(root / "config" / "regions.json")
    if not raw_folder.is_dir():
        raise PipelineError(f"Raw data directory does not exist: {raw_folder}")

    discovered = sorted(raw_folder.glob("*/*.csv"))
    if not discovered:
        raise PipelineError("No source files found at data/raw/<region_id>/{temperature,rainfall}.csv")
    for csv_file in discovered:
        if csv_file.parent.name not in regions:
            raise PipelineError(f"{csv_file}: region not declared in config/regions.json")
        if csv_file.stem not in DATASETS:
            raise PipelineError(f"{csv_file}: filename must be temperature.csv or rainfall.csv")

    observations_list = []
    annual_list = []
    catalogue = []
    for csv_file in discovered:
        region_id = csv_file.parent.name
        variable = csv_file.stem
        item = process_file(csv_file, region_id, regions[region_id], variable, allow_gaps)
        item.metadata["source_file"] = str(csv_file.relative_to(root).as_posix())
        observations_list.append(item.observations)
        annual_list.append(item.annual)
        catalogue.append(item.metadata)
        print(
            f"OK {region_id:24s} {variable:12s} "
            f"{item.metadata['observations']:5d} rows | "
            f"{item.metadata['complete_years']} complete years"
        )

    observations = pd.concat(observations_list, ignore_index=True)
    annual = pd.concat(annual_list, ignore_index=True)
    observations = observations.sort_values(["region_id", "variable", "time"]).reset_index(drop=True)
    annual = annual.sort_values(["region_id", "variable", "year"]).reset_index(drop=True)
    validate_outputs(observations, annual)
    output_folder.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output_folder / "observations.csv", index=False, date_format="%Y-%m-%d", float_format="%.9f")
    annual.to_csv(output_folder / "annual_summary.csv", index=False, float_format="%.9f")
    report = {
        "schema_version": 1,
        "region_count": len({row["region_id"] for row in catalogue}),
        "dataset_count": len(catalogue),
        "observation_count": len(observations),
        "catalogue": catalogue,
    }
    (output_folder / "catalog.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"DONE: {len(catalogue)} datasets, {len(observations)} observations")
    print(f"Output: {output_folder}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="TerraTrace NASA Giovanni batch-processing pipeline")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allow-gaps", action="store_true",
                        help="Allow missing periods/measurements (incomplete years are flagged)")
    args = parser.parse_args()
    try:
        run_pipeline(args.root, allow_gaps=args.allow_gaps)
    except PipelineError as exc:
        parser.exit(1, f"PIPELINE FAILED: {exc}\n")


if __name__ == "__main__":
    main()
