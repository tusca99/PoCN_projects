"""Shared I/O helpers for Task 44 scripts.

This folder is intentionally script-first (not a Python package). The helpers in
this module keep the CSV schemas consistent across build/validate/plot steps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_exclude_iso3(exclude_iso3: str | None, *, default: str = "USA") -> str:
    """Normalize an exclusion ISO3 argument.

    Returns:
        - ""  -> no exclusion
        - "XXX" -> exclude that ISO3
    """

    if exclude_iso3 is None:
        return default
    val = exclude_iso3.strip().upper()
    if val in {"", "NONE", "NO", "NULL"}:
        return ""
    return val


def require_columns(df: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def default_country_file(processed_dir: Path) -> Path:
    for candidate in [
        processed_dir / "country_list.csv",
        processed_dir / "top_countries.csv",
        processed_dir / "country_coverage.csv",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing country list. Expected data/processed/country_list.csv or country_coverage.csv."
    )


def load_country_table(
    path: Path,
    *,
    exclude_iso3: str | None,
    top_k: int,
    only_iso3: list[str] | None,
    all_countries: bool,
    min_coverage: float | None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, {"country_ISO2", "country_ISO3", "country_name"}, name=f"Country table {path}")

    df = df.copy()
    df["country_ISO2"] = df["country_ISO2"].astype(str).str.upper()
    df["country_ISO3"] = df["country_ISO3"].astype(str).str.upper()

    exclude_norm = normalize_exclude_iso3(exclude_iso3)
    if exclude_norm:
        df = df[df["country_ISO3"] != exclude_norm]

    if min_coverage is not None:
        if "coverage" not in df.columns:
            raise ValueError(f"--min-coverage requires a 'coverage' column in: {path}")
        df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
        df = df[df["coverage"].fillna(0.0) >= float(min_coverage)]

    if only_iso3:
        wanted = {c.strip().upper() for c in only_iso3 if c.strip()}
        df = df[df["country_ISO3"].isin(wanted)]
        return df.reset_index(drop=True)

    if all_countries:
        return df.reset_index(drop=True)

    return df.head(int(top_k)).reset_index(drop=True)


def load_centroids_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, {"region_code", "latitude", "longitude"}, name=f"Centroids table {path}")
    df = df.copy()
    df["region_code"] = df["region_code"].astype(str)
    return df.set_index("region_code")


def read_nodes_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, {"nodeID", "nodeLabel", "latitude", "longitude"}, name=f"nodes.csv {path}")
    df = df.copy()
    df["nodeID"] = pd.to_numeric(df["nodeID"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df


def read_edges_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(
        df,
        {"nodeID_from", "nodeID_to", "country_name", "country_ISO3"},
        name=f"edges.csv {path}",
    )
    df = df.copy()
    df["nodeID_from"] = pd.to_numeric(df["nodeID_from"], errors="coerce").astype("Int64")
    df["nodeID_to"] = pd.to_numeric(df["nodeID_to"], errors="coerce").astype("Int64")
    df["country_ISO3"] = df["country_ISO3"].astype(str).str.upper()
    df["country_name"] = df["country_name"].astype(str)
    return df
