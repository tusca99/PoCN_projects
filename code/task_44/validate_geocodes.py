#!/usr/bin/env python3
"""Validate compatibility between SCI layer region codes and centroid sources.

Why this exists
---------------
The SCI layer CSVs only contain region *codes* (e.g. GADM GID_1, NUTS_ID).
Coordinates are added later by joining a centroid table built from external
boundary datasets (GADM / GISCO NUTS).

If versions are mismatched (e.g. NUTS year, GADM release), many region codes
will not be found in the centroid table. This script quantifies that mismatch.

Outputs
-------
By default it prints a small summary per layer and (optionally) writes CSVs:
- <outdir>/summary.csv
- <outdir>/missing_<layer>.csv (first N missing codes)

Layers checked (paths are under projects/task_44/data/):
- gadm1/gadm1.csv
- nuts_2024/nuts3_2024.csv

Notes
-----
- We only consider *within-country* rows (user_country == friend_country).
- We optionally restrict to a selected country set (same flags as builders).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from task44_io import default_country_file, load_centroids_table, load_country_table


@dataclass(frozen=True)
class LayerSpec:
    name: str
    path: Path
    code_regex: str | None


def _iter_within_region_codes(
    layer_csv: Path,
    *,
    iso2_set: set[str],
    chunksize: int,
    max_rows: int | None,
) -> set[str]:
    usecols = ["user_country", "friend_country", "user_region", "friend_region"]
    seen = 0
    codes: set[str] = set()

    for chunk in pd.read_csv(layer_csv, chunksize=chunksize, dtype=str, usecols=usecols):
        if max_rows is not None and seen >= max_rows:
            break

        uc = chunk["user_country"].astype(str).str.upper()
        fc = chunk["friend_country"].astype(str).str.upper()
        within = (uc == fc) & uc.isin(iso2_set)
        if within.any():
            sub = chunk.loc[within, ["user_region", "friend_region"]]
            u = sub["user_region"].astype(str)
            v = sub["friend_region"].astype(str)
            codes.update(set(u.tolist()))
            codes.update(set(v.tolist()))

        seen += len(chunk)

    # Drop obvious empties
    codes = {c for c in codes if c and str(c).strip() and str(c).strip().lower() != "nan"}
    return codes


def _regex_match_fraction(values: list[str], pattern: str) -> float:
    if not values:
        return float("nan")
    rx = re.compile(pattern)
    ok = sum(1 for v in values if rx.fullmatch(v) is not None)
    return ok / len(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SCI region codes against centroid sources")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Task 44 data directory (default: projects/task_44/data).",
    )
    parser.add_argument(
        "--centroids",
        type=Path,
        default=None,
        help="Centroids CSV (default: <data-dir>/processed/centroids.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for summary/missing codes (default: <data-dir>/processed/geocodes_check)",
    )

    # Country selection (same semantics as builders)
    parser.add_argument(
        "--country-file",
        type=Path,
        default=None,
        help="CSV with country_ISO2,country_ISO3,country_name (or coverage table).",
    )
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--all-countries", action="store_true")
    parser.add_argument(
        "--exclude-iso3",
        type=str,
        default=None,
        help="Exclude this ISO3 (default: USA). Use 'NONE' to include all.",
    )
    parser.add_argument("--iso3", action="append", default=None)

    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--max-rows", type=int, default=None, help="For quick tests only")
    parser.add_argument(
        "--write-missing",
        type=int,
        default=500,
        help="Write up to this many missing codes per layer to CSV (0 disables).",
    )

    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    centroids_path = args.centroids or (data_dir / "processed" / "centroids.csv")
    if not centroids_path.exists():
        raise FileNotFoundError(
            f"Centroids file not found: {centroids_path}. Build it first (task44_cli.py centroids)."
        )

    outdir = args.outdir or (data_dir / "processed" / "geocodes_check")
    outdir.mkdir(parents=True, exist_ok=True)

    processed_dir = data_dir / "processed"
    country_file = args.country_file or default_country_file(processed_dir)
    selected = load_country_table(
        country_file,
        exclude_iso3=args.exclude_iso3,
        top_k=args.top_k,
        only_iso3=args.iso3,
        all_countries=args.all_countries,
        min_coverage=args.min_coverage,
    )
    if selected.empty:
        raise ValueError("No countries selected. Check --exclude-iso3/--iso3/--top-k.")

    iso2_set = set(selected["country_ISO2"].astype(str).str.upper().tolist())

    layers = [
        LayerSpec(
            name="gadm1",
            path=data_dir / "gadm1" / "gadm1.csv",
            code_regex=r"^[A-Z]{3}\..+_\d+$",
        ),
        LayerSpec(
            name="nuts3_2024",
            path=data_dir / "nuts_2024" / "nuts3_2024.csv",
            code_regex=r"^[A-Z]{2}[A-Z0-9]{3}$",
        ),
    ]

    for spec in layers:
        if not spec.path.exists():
            raise FileNotFoundError(f"Missing layer CSV: {spec.path}")

    centroids = load_centroids_table(centroids_path)
    centroid_codes = set(centroids.index.astype(str).tolist())

    rows: list[dict[str, object]] = []

    for spec in layers:
        codes = _iter_within_region_codes(
            spec.path,
            iso2_set=iso2_set,
            chunksize=int(args.chunksize),
            max_rows=args.max_rows,
        )
        codes_list = sorted(codes)

        matched = [c for c in codes_list if c in centroid_codes]
        missing = [c for c in codes_list if c not in centroid_codes]

        rx_frac = float("nan")
        if spec.code_regex is not None:
            rx_frac = _regex_match_fraction(codes_list[:50_000], spec.code_regex)

        rows.append(
            {
                "layer": spec.name,
                "layer_csv": str(spec.path),
                "countries_selected": len(iso2_set),
                "region_codes_used": len(codes_list),
                "centroids_rows": len(centroid_codes),
                "matched_codes": len(matched),
                "missing_codes": len(missing),
                "missing_frac": (len(missing) / len(codes_list)) if codes_list else float("nan"),
                "regex_match_frac_sample": rx_frac,
            }
        )

        print(
            f"{spec.name}: used={len(codes_list)} matched={len(matched)} missing={len(missing)} "
            f"(missing_frac={(len(missing)/len(codes_list)) if codes_list else float('nan'):.3f})"
        )
        if missing:
            print(f"  sample missing: {missing[:8]}")

        if int(args.write_missing) > 0 and missing:
            out_missing = outdir / f"missing_{spec.name}.csv"
            pd.DataFrame({"region_code": missing[: int(args.write_missing)]}).to_csv(out_missing, index=False)

    summary = pd.DataFrame(rows)
    summary_path = outdir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path}")

    # Helpful interpretation hint
    worst = summary.sort_values("missing_frac", ascending=False).head(1)
    if not worst.empty:
        layer = worst.iloc[0]["layer"]
        frac = worst.iloc[0]["missing_frac"]
        if pd.notna(frac) and float(frac) > 0.05:
            print(
                f"WARNING: High missing_frac for {layer} (>{0.05:.2f}). "
                "This often indicates a version mismatch between the layer codes and the boundary dataset used for centroids."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
