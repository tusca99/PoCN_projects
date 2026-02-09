#!/usr/bin/env python3
"""Sanity-check plots for the global SCI deliverable.

Reads the Moodle-style global CSVs:
- data/global/nodes.csv
- data/global/edges.csv
Optional:
- data/global/edges_weighted.csv (adds scaled_sci)

Outputs summary + a couple of quick plots under:
- data/global/summary_by_country.csv
- data/global/plots/

Plots are intentionally simple and robust even when networks are dense.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from task44_io import read_edges_csv, read_nodes_csv


def _read_nodes(path: Path) -> pd.DataFrame:
    df = read_nodes_csv(path)
    df = df.copy()
    df["nodeID"] = df["nodeID"].astype(int)
    df["has_coords"] = df["latitude"].notna() & df["longitude"].notna()
    return df[["nodeID", "has_coords"]]


def _read_edges(path: Path) -> pd.DataFrame:
    df = read_edges_csv(path)
    df = df.copy()
    df["nodeID_from"] = df["nodeID_from"].astype(int)
    df["nodeID_to"] = df["nodeID_to"].astype(int)
    return df


def _read_edges_weighted(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"nodeID_from", "nodeID_to", "country_ISO3", "scaled_sci"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"weighted edges file missing columns: {sorted(missing)}")

    df = df.copy()
    df["nodeID_from"] = pd.to_numeric(df["nodeID_from"], errors="raise").astype(int)
    df["nodeID_to"] = pd.to_numeric(df["nodeID_to"], errors="raise").astype(int)
    df["country_ISO3"] = df["country_ISO3"].astype(str)
    df["scaled_sci"] = pd.to_numeric(df["scaled_sci"], errors="coerce")
    return df


def build_summary(nodes: pd.DataFrame, edges: pd.DataFrame, edges_w: pd.DataFrame | None) -> pd.DataFrame:
    # Nodes per country are inferred from edges (since nodes.csv has no ISO3 column).
    iso3 = edges["country_ISO3"].to_numpy()
    from_ids = edges["nodeID_from"].to_numpy()
    to_ids = edges["nodeID_to"].to_numpy()

    data = {}
    for i, country in enumerate(iso3):
        if country not in data:
            data[country] = {"edges": 0, "nodes": set()}
        data[country]["edges"] += 1
        data[country]["nodes"].add(int(from_ids[i]))
        data[country]["nodes"].add(int(to_ids[i]))

    nodes_has = dict(zip(nodes["nodeID"].to_numpy(), nodes["has_coords"].to_numpy()))

    rows = []
    for country, obj in data.items():
        node_ids = np.fromiter(obj["nodes"], dtype=int)
        n = int(node_ids.size)
        e = int(obj["edges"])
        max_e = n * (n - 1) // 2
        density = float(e / max_e) if max_e > 0 else np.nan

        coords = np.array([bool(nodes_has.get(int(x), False)) for x in node_ids], dtype=bool)
        coords_frac = float(coords.mean()) if coords.size else np.nan

        rows.append(
            {
                "country_ISO3": country,
                "N_nodes": n,
                "E_edges": e,
                "density_undirected": density,
                "coords_frac": coords_frac,
            }
        )

    out = pd.DataFrame(rows).sort_values(["E_edges", "N_nodes"], ascending=False).reset_index(drop=True)

    if edges_w is not None and not edges_w.empty:
        wsum = edges_w.groupby("country_ISO3", as_index=False)["scaled_sci"].sum().rename(
            columns={"scaled_sci": "total_scaled_sci"}
        )
        out = out.merge(wsum, on="country_ISO3", how="left")
    return out


def plot_scatter_ne(edges_summary: pd.DataFrame, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    n = edges_summary["N_nodes"].to_numpy()
    e = edges_summary["E_edges"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(n, e, s=14, alpha=0.75)

    n_line = np.linspace(max(2, n.min(initial=2)), max(2, n.max(initial=2)), 200)
    e_complete = n_line * (n_line - 1) / 2
    ax.plot(n_line, e_complete, lw=1.5, color="black", alpha=0.7, label="Complete graph")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N nodes (per country)")
    ax.set_ylabel("E edges (per country)")
    ax.set_title("Global SCI: per-country size")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(outdir / "scatter_N_vs_E.pdf")
    fig.savefig(outdir / "scatter_N_vs_E.png", dpi=200)
    plt.close(fig)


def plot_coords_frac_vs_n(edges_summary: pd.DataFrame, outdir: Path) -> None:
    """Square plot: coordinate coverage versus network size."""

    import matplotlib.pyplot as plt

    df = edges_summary.dropna(subset=["coords_frac", "N_nodes"]).copy()
    if df.empty:
        return

    n = df["N_nodes"].to_numpy()
    frac = df["coords_frac"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(n, frac, s=18, alpha=0.75)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("N nodes (per country)")
    ax.set_ylabel("Fraction of nodes with coordinates")
    ax.set_title("Coordinate coverage vs network size")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(outdir / "coords_frac_vs_N.pdf")
    fig.savefig(outdir / "coords_frac_vs_N.png", dpi=200)
    plt.close(fig)


def plot_coords_coverage(edges_summary: pd.DataFrame, outdir: Path, top_k: int) -> None:
    import matplotlib.pyplot as plt

    df = edges_summary.copy()
    df = df.sort_values("N_nodes", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(7.0, 0.28 * len(df) + 1.6))
    y = np.arange(len(df))

    ax.barh(y, df["coords_frac"].to_numpy(), color="#4C78A8", alpha=0.9)
    ax.set_yticks(y, labels=df["country_ISO3"].to_list())
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of nodes with coordinates")
    ax.set_title(f"Coordinate coverage (top {top_k} countries by N)")
    ax.grid(True, axis="x", ls=":", lw=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(outdir / "coords_coverage_topK.pdf")
    fig.savefig(outdir / "coords_coverage_topK.png", dpi=200)
    plt.close(fig)


def plot_total_weight(edges_summary: pd.DataFrame, outdir: Path, top_k: int) -> None:
    if "total_scaled_sci" not in edges_summary.columns:
        return

    import matplotlib.pyplot as plt

    df = edges_summary.dropna(subset=["total_scaled_sci"]).copy()
    if df.empty:
        return

    df = df.sort_values("total_scaled_sci", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(df) + 1.6))
    y = np.arange(len(df))

    ax.barh(y, df["total_scaled_sci"].to_numpy(), color="#F58518", alpha=0.9)
    ax.set_yticks(y, labels=df["country_ISO3"].to_list())
    ax.invert_yaxis()
    ax.set_xlabel("Total scaled_sci (sum over edges)")
    ax.set_title(f"Total within-country connectedness (top {top_k})")
    ax.grid(True, axis="x", ls=":", lw=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(outdir / "total_scaled_sci_topK.pdf")
    fig.savefig(outdir / "total_scaled_sci_topK.png", dpi=200)
    plt.close(fig)


def plot_total_weight_rank(edges_summary: pd.DataFrame, outdir: Path, top_k: int) -> None:
    """Square plot: total scaled_sci as a function of rank.

    This avoids long country label lists in the report.
    """

    if "total_scaled_sci" not in edges_summary.columns:
        return

    import matplotlib.pyplot as plt

    df = edges_summary.dropna(subset=["total_scaled_sci"]).copy()
    if df.empty:
        return

    df = df.sort_values("total_scaled_sci", ascending=False).head(top_k)
    y = df["total_scaled_sci"].to_numpy()
    x = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(x, y, marker="o", lw=1.8, ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("Country rank")
    ax.set_ylabel("Total scaled_sci (sum over edges)")
    ax.set_title(f"Total within-country connectedness (top {top_k})")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(outdir / "total_scaled_sci_rank_topK.pdf")
    fig.savefig(outdir / "total_scaled_sci_rank_topK.png", dpi=200)
    plt.close(fig)


def plot_mean_weight_rank(edges_summary: pd.DataFrame, outdir: Path, top_k: int) -> None:
    """Square plot: mean scaled_sci per edge as a function of rank.

    This normalizes for the number of edges and highlights whether differences
    in total weight are mainly driven by graph size/density.
    """

    if "total_scaled_sci" not in edges_summary.columns:
        return

    import matplotlib.pyplot as plt

    df = edges_summary.dropna(subset=["total_scaled_sci", "E_edges"]).copy()
    df = df[df["E_edges"].to_numpy() > 0]
    if df.empty:
        return

    df["mean_scaled_sci"] = df["total_scaled_sci"] / df["E_edges"]
    df = df.sort_values("mean_scaled_sci", ascending=False).head(top_k)

    y = df["mean_scaled_sci"].to_numpy()
    x = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(x, y, marker="o", lw=1.8, ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("Country rank")
    ax.set_ylabel("Mean scaled_sci per edge")
    ax.set_title(f"Mean within-country connectedness (top {top_k})")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(outdir / "mean_scaled_sci_rank_topK.pdf")
    fig.savefig(outdir / "mean_scaled_sci_rank_topK.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sanity-check plots for global SCI CSVs")
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("projects/task_44/data/global/nodes.csv"),
        help="Path to global nodes.csv",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("projects/task_44/data/global/edges.csv"),
        help="Path to global edges.csv",
    )
    parser.add_argument(
        "--edges-weighted",
        type=Path,
        default=Path("projects/task_44/data/global/edges_weighted.csv"),
        help="Optional path to global edges_weighted.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("projects/task_44/data/global"),
        help="Output directory (summary + plots/) ",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many countries to show in bar plots",
    )

    args = parser.parse_args(argv)

    nodes = _read_nodes(args.nodes)
    edges = _read_edges(args.edges)

    edges_w: pd.DataFrame | None
    if args.edges_weighted.exists():
        edges_w = _read_edges_weighted(args.edges_weighted)
    else:
        edges_w = None

    summary = build_summary(nodes=nodes, edges=edges, edges_w=edges_w)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / "summary_by_country.csv"
    summary.to_csv(summary_path, index=False)

    plots_dir = args.outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_scatter_ne(summary, plots_dir)
    plot_coords_frac_vs_n(summary, plots_dir)
    plot_coords_coverage(summary, plots_dir, top_k=args.top_k)
    plot_total_weight(summary, plots_dir, top_k=args.top_k)
    plot_total_weight_rank(summary, plots_dir, top_k=args.top_k)
    plot_mean_weight_rank(summary, plots_dir, top_k=args.top_k)

    print(f"Wrote: {summary_path}")
    print(f"Wrote plots under: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
