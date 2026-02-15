"""BTW sandpile on diverse network topologies – multi-topology CCDF comparison.

Runs the BTW sandpile (degree threshold z_c(i) = k_i) on several network types
and produces a single overlay figure of avalanche-area CCDFs, demonstrating how
topology affects avalanche statistics.

Networks:
  - 2D lattice (open boundary)
  - Erdős–Rényi (ER)
  - Watts–Strogatz (WS, two rewiring probs)
  - Barabási–Albert (BA)
  - Scale-free static model (SF, two γ values)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

import networkx as nx

from generate_networks import (
    build_lattice_2d,
    build_er,
    build_ba,
    build_ws,
    build_sf_static_model,
)
from numba_sandpile import graph_to_csr_undirected

try:
    from numba_sandpile import simulate_btw
except Exception:
    simulate_btw = None


# ── helpers ─────────────────────────────────────────────────────────────────

def log_binned_ccdf(data: np.ndarray, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Complementary CDF with log-spaced bin edges."""
    data = data[np.isfinite(data) & (data > 0)]
    if len(data) == 0:
        return np.array([]), np.array([])
    xmin, xmax = float(np.min(data)), float(np.max(data))
    if xmin <= 0 or xmax <= xmin:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(xmin), np.log10(xmax), int(bins) + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ccdf = np.array([np.mean(data >= e) for e in edges[:-1]])
    mask = ccdf > 0
    return centers[mask], ccdf[mask]


def _numba_warmup() -> None:
    if simulate_btw is None:
        return
    indptr = np.array([0, 2, 4, 6], dtype=np.int64)
    indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
    z_c = np.array([2, 2, 2], dtype=np.int32)
    active = np.array([0, 1, 2], dtype=np.int32)
    simulate_btw(indptr, indices, z_c, 0.0, 0, 1, 0, 1, active, 0)


# ── single-topology run ───────────────────────────────────────────────────

def run_single(
    label: str,
    G: nx.Graph,
    steps: int,
    transient: int,
    loss_prob: float,
    seed: int,
    bins: int,
) -> dict:
    """Run BTW on graph G with degree threshold, return area CCDF."""
    N = G.number_of_nodes()
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    degrees = np.array([G.degree(i) for i in range(N)], dtype=np.int32)

    # Threshold = degree (Goh convention)
    z_c = degrees.copy()
    z_c[z_c < 2] = 2  # safety

    # Identify bulk vs boundary (nodes with degree < 2 are effectively boundary)
    active_nodes = np.arange(N, dtype=np.int32)

    if simulate_btw is not None:
        indptr, indices, _deg = graph_to_csr_undirected(G)
        # loss_mode=0 → per-grain
        # signature: simulate_btw(indptr, indices, z_c, loss_prob, loss_mode,
        #                         n_steps, transient, seed, active_nodes, log_every)
        out = simulate_btw(
            indptr, indices, z_c,
            loss_prob, 0,          # loss_prob, loss_mode=per-grain
            steps, transient,
            seed, active_nodes,
            0,                     # log_every
        )
        areas = np.asarray(out[0], dtype=np.int64)
    else:
        raise RuntimeError("Numba kernel required for efficiency")

    # Filter trivial avalanches (area == 0 means no toppling)
    areas = areas[areas > 0]

    cx, cy = log_binned_ccdf(areas.astype(float), bins=bins)
    k_mean = float(np.mean(degrees))
    k_max_val = int(np.max(degrees))

    return {
        "label": label,
        "N": N,
        "k_mean": round(k_mean, 2),
        "k_max": k_max_val,
        "n_events": int(len(areas)),
        "area_max": int(np.max(areas)) if len(areas) else 0,
        "ccdf_x": cx,
        "ccdf_y": cy,
    }


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BTW on diverse topologies")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--N", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--transient", type=int, default=20_000)
    parser.add_argument("--loss-prob", type=float, default=1e-4)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    N = args.N
    seed = args.seed

    print("Numba warmup...")
    _numba_warmup()

    # Build networks ─────────────────────────────────────────────────
    L = int(np.sqrt(N))
    N_lattice = L * L  # adjust to exact square

    networks = {
        f"2D lattice ({L}×{L})": build_lattice_2d(L),
        r"ER ($\langle k \rangle \approx 4$)": build_er(N, 4.0 / N, seed),
        r"WS ($k{=}4,\, \beta{=}0.1$)": build_ws(N, 4, 0.1, seed),
        r"BA ($m{=}2$)": build_ba(N, 2, seed),
        r"SF static ($\gamma{=}2.5$)": build_sf_static_model(N, 2, 2.5, seed),
        r"SF static ($\gamma{=}3.0$)": build_sf_static_model(N, 2, 3.0, seed),
    }

    # Ensure all connected
    for label, G in list(networks.items()):
        if not nx.is_connected(G):
            cc = max(nx.connected_components(G), key=len)
            networks[label] = G.subgraph(cc).copy()
        networks[label] = nx.convert_node_labels_to_integers(networks[label])

    # Run simulations (parallel) ─────────────────────────────────────
    results = []
    tasks = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, (label, G) in enumerate(networks.items()):
            fut = pool.submit(
                run_single, label, G,
                args.steps, args.transient, args.loss_prob,
                seed + i * 1000, args.bins,
            )
            tasks.append((label, fut))

        for label, fut in tasks:
            print(f"  waiting: {label} ...", end=" ", flush=True)
            res = fut.result()
            print(f"done (N={res['N']}, <k>={res['k_mean']}, events={res['n_events']}, max_area={res['area_max']})")
            results.append(res)

    # Plot: overlay CCDFs ─────────────────────────────────────────────
    # Color palette
    colors = ["#2166ac", "#d6604d", "#4daf4a", "#ff7f00", "#984ea3", "#a65628"]
    markers = ["o", "s", "^", "D", "v", "p"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, res in enumerate(results):
        cx, cy = res["ccdf_x"], res["ccdf_y"]
        if len(cx) < 3:
            continue
        ax.loglog(
            cx, cy,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=4,
            markevery=3,
            linewidth=1.4,
            alpha=0.85,
            label=res["label"],
        )

    # Guide lines for reference slopes
    xg = np.logspace(0.5, 3.5, 50)
    ax.loglog(xg, 2.0 * xg ** (-0.5), "k--", linewidth=1.0, alpha=0.4,
              label=r"$A^{-0.5}$ guide (MF)")
    ax.loglog(xg, 2.0 * xg ** (-1.0), "k:", linewidth=1.0, alpha=0.4,
              label=r"$A^{-1}$ guide")

    ax.set_xlabel("Avalanche area $A$ (distinct toppled nodes)", fontsize=12)
    ax.set_ylabel(r"$P(A^\prime \geq A)$", fontsize=12)
    ax.set_title(f"BTW avalanche-area CCDF – topology comparison ($N \\approx$ {N})", fontsize=12)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.grid(True, which="major", alpha=0.2)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    fig.tight_layout()
    fig.savefig(outdir / "topology_comparison_ccdf.pdf", bbox_inches="tight")
    fig.savefig(outdir / "topology_comparison_ccdf.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Summary to CSV
    import csv
    with open(outdir / "topology_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "N", "k_mean", "k_max", "n_events", "area_max"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})

    print(f"\nWrote: {outdir}")


if __name__ == "__main__":
    main()
