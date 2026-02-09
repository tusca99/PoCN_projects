"""Build a centroids table (region_code -> lat/lon + label) for Task 44.

This script downloads (or reuses cached) administrative boundary datasets and
extracts representative points for region codes used in the SCI layer CSVs.

Outputs a CSV usable by build_global_network.py via --centroids with columns:
  region_code, latitude, longitude, nodeLabel

Supported code types (matching this workspace's SCI layers):
- GADM1: region_code == GADM GID_1 (e.g. "KOR.15_1")
- NUTS3: region_code == NUTS_ID (e.g. "ITH36")

Notes
-----
- Downloads are cached under projects/task_44/data/raw/geo/.
- Data files are large; do not commit them.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from task44_io import default_country_file, load_country_table, normalize_exclude_iso3


def _download(url: str, outpath: Path) -> None:
    import requests  # noqa: WPS433

    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.exists() and outpath.stat().st_size > 0:
        return

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = outpath.with_suffix(outpath.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(outpath)


def _extract_zip(zip_path: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(outdir)


def _representative_points(gdf):
    # Use representative points instead of centroids for multipolygons.
    # Ensure EPSG:4326
    try:
        if gdf.crs is not None and str(gdf.crs).lower() != "epsg:4326":
            gdf = gdf.to_crs(4326)
    except Exception:  # noqa: BLE001
        pass
    pts = gdf.geometry.representative_point()
    return pts.y.astype(float), pts.x.astype(float)


def _download_and_extract_gadm1(iso3: str, cache_dir: Path) -> Path | None:
    iso3 = iso3.strip().upper()
    if not iso3 or not re.fullmatch(r"[A-Z]{3}", iso3):
        return None

    zip_url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso3}_shp.zip"
    zip_path = cache_dir / "gadm41_shp" / f"gadm41_{iso3}_shp.zip"
    try:
        _download(zip_url, zip_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to download GADM for {iso3}: {exc}")
        return None

    extract_dir = cache_dir / "gadm41_shp" / f"gadm41_{iso3}"
    if not extract_dir.exists() or not any(extract_dir.glob("*.shp")):
        _extract_zip(zip_path, extract_dir)

    shp = extract_dir / f"gadm41_{iso3}_1.shp"
    return shp if shp.exists() else None


def _gadm1_centroids(iso3_list: list[str], cache_dir: Path, *, workers: int) -> pd.DataFrame:
    import geopandas as gpd  # noqa: WPS433

    rows: list[dict[str, object]] = []
    workers = max(1, int(workers))
    shp_paths: list[Path] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_and_extract_gadm1, iso3, cache_dir): iso3
            for iso3 in iso3_list
        }
        for future in as_completed(futures):
            try:
                shp = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: GADM worker failed: {exc}")
                continue
            if shp is not None:
                shp_paths.append(shp)

    for shp in sorted(shp_paths):
        gdf = gpd.read_file(shp)
        if "GID_1" not in gdf.columns:
            continue

        lat, lon = _representative_points(gdf)
        code = gdf["GID_1"].astype(str)
        label_col = "NAME_1" if "NAME_1" in gdf.columns else None
        label = gdf[label_col].astype(str) if label_col else code

        rows.extend(
            {
                "region_code": c,
                "latitude": float(la),
                "longitude": float(lo),
                "nodeLabel": str(lb),
            }
            for c, la, lo, lb in zip(code, lat, lon, label, strict=False)
        )

    return pd.DataFrame(rows).drop_duplicates("region_code")


def _nuts3_centroids(cache_dir: Path) -> pd.DataFrame:
    import geopandas as gpd  # noqa: WPS433

    # GISCO distribution endpoint (GeoJSON, EPSG:4326, level 3)
    url = (
        "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
        "NUTS_RG_01M_2024_4326_LEVL_3.geojson"
    )
    geojson_path = cache_dir / "nuts" / "NUTS_RG_01M_2024_4326_LEVL_3.geojson"
    _download(url, geojson_path)

    gdf = gpd.read_file(geojson_path)
    if "NUTS_ID" not in gdf.columns:
        raise ValueError("NUTS GeoJSON missing NUTS_ID")

    lat, lon = _representative_points(gdf)
    code = gdf["NUTS_ID"].astype(str)
    label_col = "NAME_LATN" if "NAME_LATN" in gdf.columns else None
    label = gdf[label_col].astype(str) if label_col else code

    return (
        pd.DataFrame(
            {
                "region_code": code,
                "latitude": lat.astype(float),
                "longitude": lon.astype(float),
                "nodeLabel": label.astype(str),
            }
        )
        .drop_duplicates("region_code")
        .reset_index(drop=True)
    )



def _infer_iso3_from_processed(processed_dir: Path, *, exclude_iso3: str, min_coverage: float | None) -> list[str]:
    """Infer ISO3 list from processed country tables."""
    country_file = default_country_file(processed_dir)
    df = load_country_table(
        country_file,
        exclude_iso3=exclude_iso3,
        top_k=10_000,
        only_iso3=None,
        all_countries=True,
        min_coverage=min_coverage,
    )
    return sorted(set(df["country_ISO3"].astype(str).str.upper().tolist()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a centroids table for Task 44.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Task 44 data directory (default: projects/task_44/data).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for downloaded geo data (default: <data-dir>/raw/geo).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output centroids CSV path (default: <data-dir>/processed/centroids.csv).",
    )
    parser.add_argument(
        "--no-gadm1",
        action="store_true",
        help="Do not include GADM1 centroids.",
    )
    parser.add_argument(
        "--no-nuts3",
        action="store_true",
        help="Do not include NUTS3 centroids.",
    )
    parser.add_argument(
        "--country-file",
        type=Path,
        default=None,
        help="Optional country table to infer ISO3 list for GADM1 downloads.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=None,
        help="If using a coverage table, keep only countries with coverage >= this value.",
    )
    parser.add_argument(
        "--exclude-iso3",
        type=str,
        default=None,
        help="Exclude this ISO3 when inferring ISO3 list (default: USA). Use 'NONE' to disable.",
    )
    parser.add_argument(
        "--iso3",
        action="append",
        default=None,
        help="ISO3 to download for GADM (repeatable). Default: infer from data/nodes/*.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for GADM downloads (default: min(8, cpu_count)).",
    )

    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    cache_dir = args.cache_dir or (data_dir / "raw" / "geo")
    out_csv = args.out or (data_dir / "processed" / "centroids.csv")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []

    if not args.no_gadm1:
        iso3_list = [c.strip().upper() for c in (args.iso3 or []) if c.strip()]
        if not iso3_list:
            exclude_norm = normalize_exclude_iso3(args.exclude_iso3)
            processed_dir = data_dir / "processed"
            try:
                if args.country_file is not None and args.country_file.exists():
                    df = load_country_table(
                        args.country_file,
                        exclude_iso3=exclude_norm,
                        top_k=10_000,
                        only_iso3=None,
                        all_countries=True,
                        min_coverage=args.min_coverage,
                    )
                    iso3_list = sorted(set(df["country_ISO3"].astype(str).str.upper().tolist()))
                else:
                    iso3_list = _infer_iso3_from_processed(
                        processed_dir,
                        exclude_iso3=exclude_norm,
                        min_coverage=args.min_coverage,
                    )
            except Exception:  # noqa: BLE001
                iso3_list = []

        if iso3_list:
            default_workers = min(8, os.cpu_count() or 4)
            workers = args.workers or default_workers
            parts.append(_gadm1_centroids(iso3_list, cache_dir, workers=workers))

    if not args.no_nuts3:
        parts.append(_nuts3_centroids(cache_dir))

    if not parts:
        raise ValueError("No centroid sources enabled.")

    centroids = pd.concat(parts, ignore_index=True)
    centroids = centroids.dropna(subset=["region_code", "latitude", "longitude"]).copy()
    centroids["region_code"] = centroids["region_code"].astype(str)
    centroids = centroids.drop_duplicates("region_code").reset_index(drop=True)

    centroids.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} (rows={len(centroids)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
