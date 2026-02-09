#!/usr/bin/env python3
"""Generate country_list.csv from all_region_to_country mappings.

This script extracts unique countries from the all_region_to_country CSVs,
which is much faster than scanning the full layer files.

Output: data/processed/country_list.csv
  Columns: country_ISO2, country_ISO3, country_name, n_regions_gadm1, n_regions_nuts3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from task44_utils import (
    extract_countries_and_regions_from_mapping,
    iso2_to_iso3,
    iso2_to_name,
)


def _extract_countries_and_regions(csv_path: Path) -> dict[str, set[str]]:
    """Wrapper for compatibility - delegates to task44_utils."""
    return extract_countries_and_regions_from_mapping(csv_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate country_list.csv from all_region_to_country CSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Task 44 data directory (default: projects/task_44/data).",
    )
    
    args = parser.parse_args(argv)
    
    data_dir: Path = args.data_dir
    mapping_dir = data_dir / "all_region_to_country"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if not mapping_dir.exists():
        raise FileNotFoundError(f"Missing all_region_to_country directory: {mapping_dir}")
    
    print("Extracting countries from all_region_to_country CSVs...")
    
    # Extract regions per country for each layer
    gadm1_csv = mapping_dir / "gadm1_to_country.csv"
    nuts3_csv = mapping_dir / "nuts3_2024_to_country.csv"
    
    print("  Processing gadm1_to_country.csv...")
    iso2_gadm1 = _extract_countries_and_regions(gadm1_csv)
    
    print("  Processing nuts3_2024_to_country.csv...")
    iso2_nuts3 = _extract_countries_and_regions(nuts3_csv)
    
    # Merge all countries (exclude USA by default)
    all_iso2 = (set(iso2_gadm1.keys()) | set(iso2_nuts3.keys())) - {"US"}
    
    rows = []
    for iso2 in sorted(all_iso2):
        iso2 = iso2.strip().upper()
        if not iso2 or iso2 in {"", "NAN"}:
            continue

        iso3 = iso2_to_iso3(iso2)
        name = iso2_to_name(iso2)

        if iso3 is None:
            # Fallback for codes not in pycountry (e.g., XK for Kosovo)
            iso3 = iso2 + iso2[0]
        if name is None:
            name = iso2
        
        n_gadm1 = len(iso2_gadm1.get(iso2, set()))
        n_nuts3 = len(iso2_nuts3.get(iso2, set()))
        
        rows.append({
            "country_ISO2": iso2,
            "country_ISO3": iso3,
            "country_name": name,
            "n_regions_gadm1": n_gadm1,
            "n_regions_nuts3": n_nuts3,
        })
    
    df = pd.DataFrame(rows).sort_values("n_regions_gadm1", ascending=False).reset_index(drop=True)

    out_path = processed_dir / "country_list.csv"
    df.to_csv(out_path, index=False)

    print(f"\nWrote: {out_path}")
    print(f"  Total countries: {len(df)}")
    print("\nTop 15 by GADM1 regions:")
    print(
        df[df["n_regions_gadm1"] > 0]
        .head(15)[["country_ISO3", "country_name", "n_regions_gadm1", "n_regions_nuts3"]]
        .to_string(index=False)
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
