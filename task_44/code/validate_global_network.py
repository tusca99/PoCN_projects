"""Validate a global nodes.csv + edges.csv pair.

Checks:
- nodeIDs are unique and consecutive (1..N)
- edges refer to existing nodeIDs
- (optional) self-loops fraction
- missing coordinates fraction
- per-country counts of nodes/edges in the edges.csv (since edges include country columns)

This is meant as a quick sanity-check before submitting/exporting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from task44_io import read_edges_csv, read_nodes_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate global nodes.csv + edges.csv")
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("projects/task_44/data/global/nodes.csv"),
        help="Path to nodes.csv",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("projects/task_44/data/global/edges.csv"),
        help="Path to edges.csv",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=15,
        help="How many top countries to print by edges.",
    )

    args = parser.parse_args(argv)

    if not args.nodes.exists():
        raise FileNotFoundError(f"nodes not found: {args.nodes}")
    if not args.edges.exists():
        raise FileNotFoundError(f"edges not found: {args.edges}")

    nodes = read_nodes_csv(args.nodes)
    edges = read_edges_csv(args.edges)

    N = int(nodes["nodeID"].nunique())
    E = int(len(edges))

    node_ids = nodes["nodeID"].dropna().astype(int)
    unique_ids = set(node_ids.tolist())

    consecutive_ok = unique_ids == set(range(1, N + 1))
    dup_nodes = int(nodes["nodeID"].duplicated().sum())

    bad_from = edges.loc[~edges["nodeID_from"].isin(unique_ids)]
    bad_to = edges.loc[~edges["nodeID_to"].isin(unique_ids)]

    self_loops = edges.loc[edges["nodeID_from"] == edges["nodeID_to"]]

    # Coordinate coverage
    has_coord = nodes["latitude"].notna() & nodes["longitude"].notna()

    print(f"N={N}, E={E}")
    print(f"nodeID consecutive: {consecutive_ok} | duplicated nodeIDs: {dup_nodes}")
    print(f"edges with missing nodeID_from: {len(bad_from)} | missing nodeID_to: {len(bad_to)}")
    print(f"self-loops in edges: {len(self_loops)}")
    print(f"nodes with coordinates: {int(has_coord.sum())}/{N} ({has_coord.mean()*100:.1f}%)")

    if len(bad_from) > 0:
        print("Sample bad_from rows:")
        print(bad_from.head(5).to_string(index=False))

    if len(bad_to) > 0:
        print("Sample bad_to rows:")
        print(bad_to.head(5).to_string(index=False))

    if "country_ISO3" in edges.columns:
        by_country = edges.groupby("country_ISO3").size().sort_values(ascending=False)
        print("Top countries by E:")
        print(by_country.head(int(args.show_top)).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
