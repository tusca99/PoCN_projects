"""Shared utilities for Task 44 scripts.

This module provides common helper functions used across the Task 44 pipeline,
including ISO code conversions, data extraction helpers, and validation utilities.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


def iso2_to_iso3(iso2: str) -> str | None:
    """Convert ISO2 country code to ISO3 using pycountry.
    
    Args:
        iso2: Two-letter country code (e.g., 'IT', 'US')
    
    Returns:
        Three-letter country code (e.g., 'ITA', 'USA') or None if not found.
    """
    try:
        import pycountry
    except ImportError:
        raise ImportError(
            "pycountry is required for ISO code conversion. "
            "Install with: pip install pycountry"
        )
    
    iso2 = iso2.strip().upper()
    if not iso2 or len(iso2) != 2:
        return None
    
    try:
        country = pycountry.countries.get(alpha_2=iso2)
        return country.alpha_3 if country else None
    except (KeyError, LookupError):
        return None


def iso2_to_name(iso2: str) -> str | None:
    """Convert ISO2 country code to country name using pycountry.
    
    Args:
        iso2: Two-letter country code (e.g., 'IT', 'US')
    
    Returns:
        Country name (e.g., 'Italy', 'United States') or None if not found.
    """
    try:
        import pycountry
    except ImportError:
        raise ImportError(
            "pycountry is required for ISO code conversion. "
            "Install with: pip install pycountry"
        )
    
    iso2 = iso2.strip().upper()
    if not iso2 or len(iso2) != 2:
        return None
    
    try:
        country = pycountry.countries.get(alpha_2=iso2)
        return country.name if country else None
    except (KeyError, LookupError):
        return None


def iso3_to_iso2(iso3: str) -> str | None:
    """Convert ISO3 country code to ISO2 using pycountry.
    
    Args:
        iso3: Three-letter country code (e.g., 'ITA', 'USA')
    
    Returns:
        Two-letter country code (e.g., 'IT', 'US') or None if not found.
    """
    try:
        import pycountry
    except ImportError:
        raise ImportError(
            "pycountry is required for ISO code conversion. "
            "Install with: pip install pycountry"
        )
    
    iso3 = iso3.strip().upper()
    if not iso3 or len(iso3) != 3:
        return None
    
    try:
        country = pycountry.countries.get(alpha_3=iso3)
        return country.alpha_2 if country else None
    except (KeyError, LookupError):
        return None


def extract_countries_and_regions_from_mapping(
    csv_path: Path,
) -> dict[str, set[str]]:
    """Extract ISO2 -> {region_codes} from a *_to_country.csv file.
    
    Args:
        csv_path: Path to a file like gadm1_to_country.csv
    
    Returns:
        Dict mapping ISO2 code -> set of region codes for that country.
    """
    if not csv_path.exists():
        return {}
    
    iso2_to_regions = defaultdict(set)
    needed_cols = ["user_country", "friend_country", "user_region", "friend_region"]
    
    df = pd.read_csv(csv_path, dtype=str, usecols=needed_cols)
    
    # Extract within-country regions only
    uc = df["user_country"].str.upper()
    fc = df["friend_country"].str.upper()
    within = uc == fc
    
    for iso2, region in zip(uc[within], df.loc[within, "user_region"], strict=False):
        iso2_to_regions[iso2].add(str(region))
    for iso2, region in zip(fc[within], df.loc[within, "friend_region"], strict=False):
        iso2_to_regions[iso2].add(str(region))
    
    return dict(iso2_to_regions)


def extract_iso2_from_layer_csv(
    csv_path: Path,
    *,
    chunksize: int = 1_000_000,
) -> set[str]:
    """Extract unique ISO2 country codes from a layer CSV (within-country only).
    
    Args:
        csv_path: Path to a layer CSV (e.g., gadm1.csv, nuts3_2024.csv)
        chunksize: Chunk size for reading large CSVs
    
    Returns:
        Set of ISO2 country codes present in the layer.
    """
    if not csv_path.exists():
        return set()
    
    iso2_set: set[str] = set()
    needed_cols = ["user_country", "friend_country"]
    
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str, usecols=needed_cols):
        uc = chunk["user_country"].str.upper()
        fc = chunk["friend_country"].str.upper()
        within = uc == fc
        if within.any():
            iso2_set.update(set(uc[within].unique().tolist()))
    
    return iso2_set


def validate_iso2(iso2: str) -> bool:
    """Check if an ISO2 code is valid.
    
    Args:
        iso2: Two-letter country code
    
    Returns:
        True if valid, False otherwise.
    """
    iso2 = iso2.strip().upper()
    if not iso2 or len(iso2) != 2:
        return False
    return iso2_to_iso3(iso2) is not None


def validate_iso3(iso3: str) -> bool:
    """Check if an ISO3 code is valid.
    
    Args:
        iso3: Three-letter country code
    
    Returns:
        True if valid, False otherwise.
    """
    iso3 = iso3.strip().upper()
    if not iso3 or len(iso3) != 3:
        return False
    return iso3_to_iso2(iso3) is not None
