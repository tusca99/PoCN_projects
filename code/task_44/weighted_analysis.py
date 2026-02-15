#!/usr/bin/env python3
"""Weighted network analysis for the global SCI deliverable.

Reads:
    data/global/edges_weighted.csv   (nodeID_from, nodeID_to, country_ISO3, scaled_sci)
    data/global/nodes.csv            (nodeID, nodeLabel, latitude, longitude)
    data/global/summary_by_country.csv

Produces report-ready figures under latex/figures/task44/:
    weight_and_strength.pdf   – left: global weight CCDF; right: per-country strength Gini
    clustering_comparison.pdf – left: weighted vs unweighted clustering; right: example weight matrix

Usage (standalone):
    python weighted_analysis.py --data-dir ../../data/task_44 --fig-dir ../../latex/figures/task44

Integrated via task44.py:
    python task44.py analyze
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _gini(x: np.ndarray) -> float:
    """Gini coefficient of array *x* (0 = perfectly equal, 1 = maximally unequal)."""
    x = np.sort(x)
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def _weighted_clustering_node(adj: np.ndarray, i: int) -> float:
    """Onnela et al. (2005) weighted clustering for node *i*.

    C_i^w = 1/(s_i (k_i - 1)) * sum_{j,h} (w_ij w_ih w_jh)^{1/3}

    where the sum runs over triangles incident on i.
    """
    n = adj.shape[0]
    k_i = int(np.sum(adj[i] > 0))
    if k_i < 2:
        return 0.0
    s_i = float(np.sum(adj[i]))
    # cube-root of product of three weights for each triangle
    total = 0.0
    nbrs = np.where(adj[i] > 0)[0]
    for j_idx in range(len(nbrs)):
        j = nbrs[j_idx]
        for h_idx in range(j_idx + 1, len(nbrs)):
            h = nbrs[h_idx]
            if adj[j, h] > 0:
                total += (adj[i, j] * adj[i, h] * adj[j, h]) ** (1.0 / 3.0)
    return float(2.0 * total / (s_i * (k_i - 1)))


def _weighted_clustering_country(weights: pd.DataFrame, n_nodes: int,
                                 node_ids: np.ndarray) -> tuple[float, float]:
    """Return (mean_C_unweighted, mean_C_weighted) for one country.

    For a complete graph C_unw = 1.0, so this is mainly about C_w.
    We normalise weights to [0, 1] within the country (max-normalised)
    so the Onnela formula gives interpretable values.
    """
    # build adjacency
    id_map = {int(v): i for i, v in enumerate(node_ids)}
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for _, row in weights.iterrows():
        u = id_map.get(int(row["nodeID_from"]))
        v = id_map.get(int(row["nodeID_to"]))
        if u is not None and v is not None and u != v:
            adj[u, v] = row["scaled_sci"]
            adj[v, u] = row["scaled_sci"]

    # normalise by max weight
    wmax = adj.max()
    if wmax > 0:
        adj /= wmax

    c_w = np.mean([_weighted_clustering_node(adj, i) for i in range(n_nodes)])

    # unweighted clustering (for a complete graph it's 1)
    bin_adj = (adj > 0).astype(np.float64)
    k = bin_adj.sum(axis=1)
    # triangles via matrix cube trace
    t = np.diag(bin_adj @ bin_adj @ bin_adj)
    denom = k * (k - 1)
    c_u_arr = np.divide(t, denom, out=np.zeros_like(t, dtype=float), where=denom > 0)
    c_u = float(np.mean(c_u_arr))

    return c_u, c_w


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------

def analyse(data_dir: Path, fig_dir: Path, top_k: int = 15,
            max_nodes_clustering: int = 120) -> None:
    """Run all weighted analyses and produce report figures."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global_dir = data_dir / "global"
    ew_path = global_dir / "edges_weighted.csv"
    nodes_path = global_dir / "nodes.csv"
    summary_path = global_dir / "summary_by_country.csv"

    if not ew_path.exists():
        raise FileNotFoundError(f"Missing {ew_path}; run  task44.py build  first.")

    edges_w = pd.read_csv(ew_path)
    nodes = pd.read_csv(nodes_path)
    summary = pd.read_csv(summary_path)

    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Weight CCDF  (global, pooled across all countries after
    #     within-country normalisation w / w_max)
    # ------------------------------------------------------------------
    grp = edges_w.groupby("country_ISO3")

    normed_weights = []
    for iso3, sub in grp:
        w = sub["scaled_sci"].to_numpy(dtype=float)
        wmax = w.max()
        if wmax > 0:
            normed_weights.append(w / wmax)
    all_w = np.concatenate(normed_weights)
    all_w_sorted = np.sort(all_w)[::-1]
    ccdf = np.arange(1, len(all_w_sorted) + 1) / len(all_w_sorted)

    # ------------------------------------------------------------------
    # 2.  Per-country strength Gini
    # ------------------------------------------------------------------
    gini_rows = []
    for iso3, sub in grp:
        w = sub["scaled_sci"].to_numpy(dtype=float)
        # build strength per node
        f = sub["nodeID_from"].to_numpy(dtype=int)
        t = sub["nodeID_to"].to_numpy(dtype=int)
        strength: dict[int, float] = {}
        for i in range(len(w)):
            strength[f[i]] = strength.get(f[i], 0.0) + w[i]
            strength[t[i]] = strength.get(t[i], 0.0) + w[i]
        s = np.array(list(strength.values()))
        gini_rows.append({
            "country_ISO3": iso3,
            "gini": _gini(s),
            "N": len(s),
            "mean_strength": float(s.mean()),
            "cv_strength": float(s.std() / s.mean()) if s.mean() > 0 else np.nan,
        })
    gini_df = pd.DataFrame(gini_rows).sort_values("N", ascending=False)

    # ------------------------------------------------------------------
    # 3.  Weighted clustering for countries with N <= max_nodes_clustering
    # ------------------------------------------------------------------
    clust_rows = []
    for iso3, sub in grp:
        node_ids = np.union1d(
            sub["nodeID_from"].to_numpy(dtype=int),
            sub["nodeID_to"].to_numpy(dtype=int),
        )
        n = len(node_ids)
        if n > max_nodes_clustering or n < 3:
            continue
        c_u, c_w = _weighted_clustering_country(sub, n, node_ids)
        clust_rows.append({
            "country_ISO3": iso3,
            "N": n,
            "C_unweighted": c_u,
            "C_weighted": c_w,
        })
    clust_df = pd.DataFrame(clust_rows).sort_values("N", ascending=False)

    # ------------------------------------------------------------------
    # save analytics CSV
    # ------------------------------------------------------------------
    analytics_path = global_dir / "weighted_analytics.csv"
    merged = gini_df.merge(clust_df[["country_ISO3", "C_weighted"]], on="country_ISO3", how="left")
    merged.to_csv(analytics_path, index=False)
    print(f"Wrote: {analytics_path}")

    # ==================================================================
    # FIGURE 1 – weight_and_strength.pdf  (2 panels)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    # — Panel A: normalised-weight CCDF —
    ax = axes[0]
    # thin out for plotting (every 50th point + last)
    step = max(1, len(all_w_sorted) // 2000)
    idx = np.arange(0, len(all_w_sorted), step)
    if idx[-1] != len(all_w_sorted) - 1:
        idx = np.append(idx, len(all_w_sorted) - 1)
    ax.loglog(all_w_sorted[idx], ccdf[idx], lw=0.8, color="#4C78A8")
    ax.set_xlabel(r"Normalised weight $w / w_{\max}$")
    ax.set_ylabel(r"CCDF $P(W \geq w)$")
    ax.set_title("(a) Edge-weight CCDF (all countries pooled)")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)

    # — Panel B: strength Gini vs N —
    ax = axes[1]
    gd = gini_df.dropna(subset=["gini"])
    sc = ax.scatter(gd["N"], gd["gini"], s=20, c=np.log10(gd["mean_strength"]),
                    cmap="viridis", edgecolors="k", linewidths=0.3, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("N (regions per country)")
    ax.set_ylabel("Strength Gini coefficient")
    ax.set_title("(b) Strength heterogeneity")
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}\langle s \rangle$")

    # annotate a few outliers
    for _, row in gd.nlargest(3, "gini").iterrows():
        ax.annotate(row["country_ISO3"], (row["N"], row["gini"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(4, 2), textcoords="offset points")

    fig.tight_layout()
    out1 = fig_dir / "weight_and_strength.pdf"
    fig.savefig(out1)
    fig.savefig(fig_dir / "weight_and_strength.png", dpi=200)
    plt.close(fig)
    print(f"Wrote: {out1}")

    # ==================================================================
    # FIGURE 2 – clustering_comparison.pdf  (2 panels)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    # — Panel A: C_w vs N  +  C_unw reference line —
    ax = axes[0]
    if not clust_df.empty:
        ax.scatter(clust_df["N"], clust_df["C_weighted"], s=22,
                   label=r"$C^w$ (Onnela)", zorder=3, edgecolors="k", linewidths=0.3)
        ax.axhline(1.0, ls="--", color="grey", lw=1, label=r"$C^{\rm unw}=1$ (complete)")
        ax.set_xscale("log")
        ax.set_xlabel("N (regions per country)")
        ax.set_ylabel("Clustering coefficient")
        ax.set_title(r"(a) Weighted clustering $C^w$")
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)

        # annotate extremes
        for _, row in clust_df.nsmallest(3, "C_weighted").iterrows():
            ax.annotate(row["country_ISO3"], (row["N"], row["C_weighted"]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(4, 2), textcoords="offset points")

    # — Panel B: example weight heatmap (pick ITA as a well-known EU country at NUTS3) —
    ax = axes[1]
    example_iso3 = "ITA"
    sub = grp.get_group(example_iso3)
    node_ids = np.sort(np.union1d(
        sub["nodeID_from"].to_numpy(dtype=int),
        sub["nodeID_to"].to_numpy(dtype=int),
    ))
    id_map = {int(v): i for i, v in enumerate(node_ids)}
    n = len(node_ids)
    adj = np.zeros((n, n), dtype=np.float64)
    for _, row in sub.iterrows():
        u, v = id_map[int(row["nodeID_from"])], id_map[int(row["nodeID_to"])]
        adj[u, v] = row["scaled_sci"]
        adj[v, u] = row["scaled_sci"]
    # log-scale for visibility
    adj_log = np.log10(adj + 1)
    im = ax.imshow(adj_log, cmap="YlOrRd", aspect="equal", interpolation="none")
    ax.set_xlabel("Region index")
    ax.set_ylabel("Region index")
    ax.set_title(f"(b) Weight matrix ({example_iso3}, $N={n}$, log-scale)")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}(w_{ij}+1)$")

    fig.tight_layout()
    out2 = fig_dir / "clustering_comparison.pdf"
    fig.savefig(out2)
    fig.savefig(fig_dir / "clustering_comparison.png", dpi=200)
    plt.close(fig)
    print(f"Wrote: {out2}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Weighted network analysis for the global SCI deliverable."
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Task 44 data directory",
    )
    parser.add_argument(
        "--fig-dir", type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "latex" / "figures" / "task44",
        help="Output directory for report figures",
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--max-nodes-clustering", type=int, default=120,
        help="Skip weighted clustering for countries with N > this (O(N^2) cost)",
    )
    args = parser.parse_args(argv)
    analyse(args.data_dir, args.fig_dir, args.top_k, args.max_nodes_clustering)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
