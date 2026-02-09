"""Build one global nodes.csv and edges.csv for Task 44.

This is the same construction as the per-country builders, but concatenated into
single outputs with globally unique nodeIDs.

Resolution policy (best available per country):
- EU countries present in NUTS3 -> nuts3_2024 layer
- Everyone else -> gadm1 layer

Inputs (layer CSVs as downloaded from HDX/Meta):
  user_country, friend_country, user_region, friend_region, scaled_sci

Outputs (default):
- projects/task_44/data/global/nodes.csv  (nodeID,nodeLabel,latitude,longitude)
- projects/task_44/data/global/edges.csv  (nodeID_from,nodeID_to,country_name,country_ISO3)
- projects/task_44/data/global/edges_weighted.csv (optional, includes scaled_sci)

Notes
-----
- Graphs are built *within-country* (user_country == friend_country).
- Self-loops (same region) are dropped by default.
- If you pass a centroids table, labels/coordinates will be filled where available.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from task44_io import default_country_file, load_centroids_table, load_country_table


_load_country_table = load_country_table
_default_country_file = default_country_file


def _available_within_countries(layer_csv: Path, *, chunksize: int) -> set[str]:
    needed = ["user_country", "friend_country"]
    available: set[str] = set()
    for chunk in pd.read_csv(layer_csv, chunksize=chunksize, dtype=str, usecols=needed):
        uc = chunk["user_country"].astype(str).str.upper()
        fc = chunk["friend_country"].astype(str).str.upper()
        within = uc == fc
        if within.any():
            available.update(set(uc[within].unique().tolist()))
    return available


def _iter_within_rows(layer_csv: Path, *, iso2_set: set[str], chunksize: int, max_rows: int | None):
    needed_cols = ["user_country", "friend_country", "user_region", "friend_region", "scaled_sci"]
    seen_rows = 0
    for chunk in pd.read_csv(layer_csv, chunksize=chunksize, dtype=str, usecols=needed_cols):
        if max_rows is not None and seen_rows >= max_rows:
            break

        uc = chunk["user_country"].astype(str).str.upper()
        fc = chunk["friend_country"].astype(str).str.upper()
        within = (uc == fc) & uc.isin(iso2_set)
        if not within.any():
            seen_rows += len(chunk)
            continue

        sub = chunk.loc[within, ["user_country", "user_region", "friend_region", "scaled_sci"]].copy()
        sub["user_country"] = sub["user_country"].astype(str).str.upper()
        sub["scaled_sci"] = pd.to_numeric(sub["scaled_sci"], errors="coerce").fillna(1.0)
        for iso2, region_u, region_v, w in zip(
            sub["user_country"],
            sub["user_region"],
            sub["friend_region"],
            sub["scaled_sci"],
            strict=False,
        ):
            yield str(iso2), str(region_u), str(region_v), float(w)

        seen_rows += len(chunk)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build one global nodes.csv and edges.csv from SCI layer CSVs (best resolution)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Task 44 data directory (default: projects/task_44/data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-dir>/global).",
    )
    parser.add_argument(
        "--country-file",
        type=Path,
        default=None,
        help="CSV with country_ISO2,country_ISO3,country_name.",
    )
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--all-countries", action="store_true")
    parser.add_argument("--iso3", action="append", default=None)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--keep-self-loops", action="store_true")
    parser.add_argument("--write-weighted", action="store_true")
    parser.add_argument(
        "--centroids",
        type=Path,
        default=None,
        help="CSV: region_code,latitude,longitude,(nodeLabel)",
    )
    parser.add_argument(
        "--label-prefix",
        choices=["none", "iso3"],
        default="iso3",
        help="Prefix nodeLabel with ISO3 to avoid collisions in the single nodes.csv.",
    )

    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    processed_dir = data_dir / "processed"
    out_dir = args.out_dir or (data_dir / "global")
    out_dir.mkdir(parents=True, exist_ok=True)

    nuts3_csv = data_dir / "nuts_2024" / "nuts3_2024.csv"
    gadm1_csv = data_dir / "gadm1" / "gadm1.csv"

    for p in [nuts3_csv, gadm1_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing layer CSV: {p}")

    country_file = args.country_file or _default_country_file(processed_dir)
    selected = _load_country_table(
        country_file,
        exclude_iso3=None,
        top_k=args.top_k,
        only_iso3=args.iso3,
        all_countries=args.all_countries,
        min_coverage=args.min_coverage,
    )
    if not selected.empty:
        selected = selected.copy()
        selected["country_ISO2"] = selected["country_ISO2"].astype(str).str.upper()
        selected["country_ISO3"] = selected["country_ISO3"].astype(str).str.upper()
        selected = selected[(selected["country_ISO2"] != "US") & (selected["country_ISO3"] != "USA")]

    if selected.empty:
        raise ValueError("No countries selected. Check country file or selection filters.")

    iso2_to_iso3 = dict(zip(selected["country_ISO2"], selected["country_ISO3"]))
    iso2_to_name = dict(zip(selected["country_ISO2"], selected["country_name"]))
    selected_iso2 = set(selected["country_ISO2"].tolist())

    centroids_df = None
    if args.centroids is not None:
        if not args.centroids.exists():
            raise FileNotFoundError(f"Centroids file not found: {args.centroids}")
        centroids_df = load_centroids_table(args.centroids)

    nuts_available = _available_within_countries(nuts3_csv, chunksize=int(args.chunksize))

    # Decide which layer to use per country
    nuts_set = selected_iso2 & nuts_available
    gadm_set = selected_iso2 - nuts_set

    print(
        "Selected countries:",
        f"total={len(selected_iso2)} | NUTS3={len(nuts_set)} | GADM1={len(gadm_set)}",
    )

    # Global mapping: (ISO3, region_code) -> global nodeID
    node_id_by_key: dict[tuple[str, str], int] = {}
    node_rows: list[dict[str, object]] = []

    def get_global_node(iso2: str, region_code: str) -> int:
        iso3 = iso2_to_iso3.get(iso2, iso2)
        key = (iso3, region_code)
        if key in node_id_by_key:
            return node_id_by_key[key]

        node_id = len(node_id_by_key) + 1
        node_id_by_key[key] = node_id

        lat = ""
        lon = ""
        label = region_code
        if centroids_df is not None and region_code in centroids_df.index:
            row = centroids_df.loc[region_code]
            lat = row.get("latitude", "")
            lon = row.get("longitude", "")
            if "nodeLabel" in centroids_df.columns:
                label = row.get("nodeLabel", region_code)

        if args.label_prefix == "iso3":
            label = f"{iso3}:{label}"

        node_rows.append(
            {
                "nodeID": node_id,
                "nodeLabel": label,
                "latitude": lat,
                "longitude": lon,
            }
        )
        return node_id

    # Aggregate weights per within-country pair using global IDs
    weights_by_country: dict[str, dict[tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))

    def consume_layer(layer_csv: Path, iso2_set: set[str]) -> None:
        if not iso2_set:
            return
        for iso2, region_u, region_v, w in _iter_within_rows(
            layer_csv,
            iso2_set=iso2_set,
            chunksize=int(args.chunksize),
            max_rows=args.max_rows,
        ):
            u = get_global_node(iso2, region_u)
            v = get_global_node(iso2, region_v)

            if (not args.keep_self_loops) and u == v:
                continue

            if not args.directed and u > v:
                u, v = v, u

            iso3 = iso2_to_iso3.get(iso2, iso2)
            weights_by_country[iso3][(u, v)] += float(w)

    consume_layer(nuts3_csv, nuts_set)
    consume_layer(gadm1_csv, gadm_set)

    nodes_df = pd.DataFrame(node_rows)
    nodes_out = out_dir / "nodes.csv"
    nodes_df[["nodeID", "nodeLabel", "latitude", "longitude"]].to_csv(nodes_out, index=False)

    edges_rows: list[dict[str, object]] = []
    edges_w_rows: list[dict[str, object]] = []

    for iso3, edge_dict in weights_by_country.items():
        # Only within-country edges exist
        country_name = None
        # Find ISO2 from ISO3 by scanning mapping once (small)
        for iso2, iso3c in iso2_to_iso3.items():
            if iso3c == iso3:
                country_name = iso2_to_name.get(iso2, iso3)
                break
        if country_name is None:
            country_name = iso3

        for (u, v), w in edge_dict.items():
            edges_rows.append(
                {
                    "nodeID_from": int(u),
                    "nodeID_to": int(v),
                    "country_name": country_name,
                    "country_ISO3": iso3,
                }
            )
            if args.write_weighted:
                edges_w_rows.append(
                    {
                        "nodeID_from": int(u),
                        "nodeID_to": int(v),
                        "country_name": country_name,
                        "country_ISO3": iso3,
                        "scaled_sci": float(w),
                    }
                )

    edges_df = pd.DataFrame(edges_rows)
    edges_out = out_dir / "edges.csv"
    edges_df[["nodeID_from", "nodeID_to", "country_name", "country_ISO3"]].to_csv(edges_out, index=False)

    if args.write_weighted:
        edges_w_df = pd.DataFrame(edges_w_rows)
        edges_w_out = out_dir / "edges_weighted.csv"
        edges_w_df[
            ["nodeID_from", "nodeID_to", "country_name", "country_ISO3", "scaled_sci"]
        ].to_csv(edges_w_out, index=False)

    print(f"Wrote: {nodes_out} (N={len(nodes_df)})")
    print(f"Wrote: {edges_out} (E={len(edges_df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
