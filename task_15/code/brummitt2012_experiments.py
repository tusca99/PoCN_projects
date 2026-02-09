"""Experiments inspired by Brummitt, D'Souza, Leicht (PNAS 2012).

Paper focus:
- BTW sandpile on two interdependent networks.
- Coupling is beneficial up to an optimum p*, then detrimental.

We implement the synthetic model R(za)-B(p)-R(zb):
- Two random regular graphs with internal degrees za and zb.
- Add 'bridges' by selecting m ~ Bin(N, p) nodes in EACH network (same m),
  then pairing them uniformly at random to create m inter-network edges.
  This matches Bernoulli coupling closely and guarantees equal stubs.

We measure (for network A):
- Local cascades: S_AA (origin A, topplings in A)
- Inflicted cascades: S_BA (origin B, topplings in A)
- Combined: S_A (topplings in A, regardless of origin)
- Global size: S = topplings in A + B

Outputs:
- summary_by_p.csv with probabilities of large cascades vs p
- a Fig.4-like plot Pr(S_A > C) and its components vs p
- optional rank-size plot of global avalanche sizes

All outputs should be placed under projects/task_15/data/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import NullLocator

import networkx as nx

from generate_networks import build_sf_static_model
from numba_sandpile import graph_to_csr_undirected, simulate_btw_two_modules

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _normalize_numpy_seed(seed: int) -> int:
    return int(seed) % (2**32)


def _parse_p_list(s: str) -> list[float]:
    out: list[float] = []
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def _parse_gamma_list(s: str) -> list[float]:
    out: list[float] = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t in {"inf", "infty", "infinite", "infinity", "∞"}:
            out.append(float("inf"))
        else:
            out.append(float(t))
    return out


def _gamma_label(gamma: float) -> str:
    if np.isinf(gamma):
        return "inf"
    return f"{gamma:g}"


def _setup_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", alpha=0.25)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def build_coupled_regular(
    *,
    N: int,
    za: int,
    zb: int,
    p: float,
    seed: int,
) -> tuple[nx.Graph, int]:
    """Build a 2-module graph with 2N nodes.

    Returns (G, split_index) where nodes [0, split_index) are module A.
    """

    rng = np.random.default_rng(_normalize_numpy_seed(seed))

    Ga = nx.random_regular_graph(int(za), int(N), seed=int(rng.integers(0, 2**32 - 1)))
    Gb = nx.random_regular_graph(int(zb), int(N), seed=int(rng.integers(0, 2**32 - 1)))

    # Relabel B nodes to N..2N-1.
    mapping_b = {i: i + N for i in range(N)}
    Gb = nx.relabel_nodes(Gb, mapping_b, copy=True)

    G = nx.Graph()
    G.add_nodes_from(range(2 * N))
    G.add_edges_from(Ga.edges())
    G.add_edges_from(Gb.edges())

    # Bernoulli coupling approximation: choose same m on both sides.
    p = float(p)
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be in [0,1]")

    m = int(rng.binomial(N, p))
    if m > 0:
        a_nodes = rng.choice(np.arange(0, N, dtype=np.int32), size=m, replace=False)
        b_nodes = rng.choice(np.arange(N, 2 * N, dtype=np.int32), size=m, replace=False)
        rng.shuffle(b_nodes)
        bridges = list(zip(a_nodes.tolist(), b_nodes.tolist(), strict=False))
        G.add_edges_from(bridges)

    return G, N


def build_coupled_sf_static(
    *,
    N: int,
    m: int,
    gamma: float,
    p: float,
    seed: int,
) -> tuple[nx.Graph, int]:
    """New/connecting idea: do Brummitt-style coupling but with SF modules.

    Each module is a PRL2003 static-model SF network with the same mean degree (2m).
    """

    rng = np.random.default_rng(_normalize_numpy_seed(seed))

    Ga = build_sf_static_model(N=int(N), m=int(m), gamma=float(gamma), seed=int(rng.integers(0, 2**32 - 1)))
    Gb = build_sf_static_model(N=int(N), m=int(m), gamma=float(gamma), seed=int(rng.integers(0, 2**32 - 1)))

    mapping_b = {i: i + N for i in range(N)}
    Gb = nx.relabel_nodes(Gb, mapping_b, copy=True)

    G = nx.Graph()
    G.add_nodes_from(range(2 * N))
    G.add_edges_from(Ga.edges())
    G.add_edges_from(Gb.edges())

    p = float(p)
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be in [0,1]")

    mbridges = int(rng.binomial(N, p))
    if mbridges > 0:
        a_nodes = rng.choice(np.arange(0, N, dtype=np.int32), size=mbridges, replace=False)
        b_nodes = rng.choice(np.arange(N, 2 * N, dtype=np.int32), size=mbridges, replace=False)
        rng.shuffle(b_nodes)
        G.add_edges_from(list(zip(a_nodes.tolist(), b_nodes.tolist(), strict=False)))

    return G, N


@dataclass
class MetricsByP:
    p: float
    n_events: int
    cutoff: int
    pr_local_large_a: float
    pr_inflicted_large_a: float
    pr_any_large_a: float
    pr_global_large: float
    global_cutoff: int


def run_one_p(
    *,
    outdir: Path,
    tag: str,
    N: int,
    network_model: str,
    za: int,
    zb: int,
    sf_m: int,
    sf_gamma: float,
    p: float,
    steps: int,
    transient: int,
    loss_prob: float,
    loss_mode: str,
    bulk_only: bool,
    seed: int,
    cutoff: int,
    global_cutoff: int,
    log_every: int,
    save_rank_size: bool,
) -> dict[str, Any]:
    rng_seed = _normalize_numpy_seed(seed)

    if network_model == "regular":
        G, split = build_coupled_regular(N=N, za=za, zb=zb, p=p, seed=rng_seed)
    elif network_model == "sf_static":
        G, split = build_coupled_sf_static(N=N, m=sf_m, gamma=sf_gamma, p=p, seed=rng_seed)
    else:
        raise ValueError(f"Unknown network_model: {network_model}")

    z_c = np.array([G.degree(i) for i in range(G.number_of_nodes())], dtype=np.int32)
    active_nodes = np.where(z_c > 0)[0].astype(np.int32)

    indptr, indices, _deg = graph_to_csr_undirected(G)

    lm = 0 if str(loss_mode) == "per-grain" else 1
    ta, tb, dur, losses, origin = simulate_btw_two_modules(
        indptr,
        indices,
        z_c,
        float(loss_prob),
        int(lm),
        int(steps),
        int(transient),
        int(rng_seed),
        active_nodes,
        int(log_every),
        int(split),
    )

    ta = ta.astype(np.int64)
    tb = tb.astype(np.int64)
    losses = losses.astype(np.int64)
    origin = origin.astype(np.int8)

    # Selection
    total = ta + tb
    msel = total > 0
    if bulk_only:
        msel = msel & (losses == 0)

    ta = ta[msel]
    tb = tb[msel]
    origin = origin[msel]
    total = total[msel]

    # Local/inflicted into A.
    # origin==0 => began in A
    # origin==1 => began in B
    local_a = ta[origin == 0]
    inflicted_a = ta[origin == 1]

    pr_local_large_a = float(np.mean(local_a > cutoff)) if local_a.size else float("nan")
    pr_inflicted_large_a = float(np.mean(inflicted_a > cutoff)) if inflicted_a.size else float("nan")
    pr_any_large_a = float(np.mean(ta > cutoff)) if ta.size else float("nan")

    pr_global_large = float(np.mean(total > global_cutoff)) if total.size else float("nan")

    if save_rank_size:
        # Save the top K global avalanche sizes (rank-size plot data)
        K = min(10000, int(total.size))
        if K > 0:
            top = np.sort(total)[::-1][:K]
            pd.DataFrame({"rank": np.arange(1, K + 1), "S": top}).to_csv(outdir / f"rank_size_{tag}.csv", index=False)

    metrics = MetricsByP(
        p=float(p),
        n_events=int(total.size),
        cutoff=int(cutoff),
        pr_local_large_a=pr_local_large_a,
        pr_inflicted_large_a=pr_inflicted_large_a,
        pr_any_large_a=pr_any_large_a,
        pr_global_large=pr_global_large,
        global_cutoff=int(global_cutoff),
    )

    return {
        "tag": tag,
        "metrics": metrics.__dict__,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent / "../data/brummitt2012"))

    ap.add_argument("--N", type=int, default=2000, help="Nodes per module")
    ap.add_argument("--za", type=int, default=3)
    ap.add_argument("--zb", type=int, default=3)

    ap.add_argument(
        "--network-model",
        type=str,
        default="regular",
        choices=["regular", "sf_static"],
        help="regular reproduces the paper's R(z)-B(p)-R(z) toy model; sf_static is a new bridging experiment.",
    )
    ap.add_argument("--sf-m", type=int, default=2, help="(sf_static) static model parameter, mean degree=2m")
    ap.add_argument("--sf-gamma", type=float, default=2.5, help="(sf_static) degree exponent")
    ap.add_argument(
        "--sf-gamma-list",
        type=str,
        default="",
        help="(sf_static) Comma-separated gammas sweep, e.g. '2.2,2.5,3.0,3.5,inf'. If set, overrides --sf-gamma.",
    )

    ap.add_argument(
        "--p-list",
        type=str,
        default="0.001,0.003,0.01,0.03,0.05,0.075,0.1,0.2",
        help="Comma-separated coupling probabilities",
    )

    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--transient", type=int, default=50000)

    ap.add_argument("--loss-prob", type=float, default=0.01)
    ap.add_argument(
        "--loss-mode",
        type=str,
        default="per-toppling",
        choices=["per-grain", "per-toppling"],
    )
    ap.add_argument("--bulk-only", action="store_true")

    ap.add_argument("--cutoff", type=int, default=1000, help="Large-cascade cutoff for S_A (paper uses N/2 when N=2000)")
    ap.add_argument("--global-cutoff", type=int, default=2000, help="Large-cascade cutoff for global S")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replicates", type=int, default=1)

    ap.add_argument("--numba-log-every", type=int, default=0)
    ap.add_argument("--save-rank-size", action="store_true")

    ap.add_argument("--no-progress", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    p_values = _parse_p_list(args.p_list)
    reps = max(1, int(args.replicates))

    gamma_values = [float(args.sf_gamma)]
    if str(args.sf_gamma_list).strip():
        gamma_values = _parse_gamma_list(str(args.sf_gamma_list))

    use_progress = (not args.no_progress) and (tqdm is not None)

    rows: list[dict[str, Any]] = []

    it: list[tuple[float, float, int, str]] = []
    for gamma in gamma_values:
        for p in p_values:
            for rep in range(reps):
                tag = f"{args.network_model}_N{args.N}_za{args.za}_zb{args.zb}_p{p:g}"
                if args.network_model == "sf_static":
                    tag += f"_m{args.sf_m}_g{_gamma_label(gamma)}"
                if reps > 1:
                    tag += f"_rep{rep}"
                it.append((float(gamma), float(p), int(rep), tag))

    if use_progress:
        it_iter = tqdm(it, total=len(it), desc="p-sweep", unit="run")
    else:
        it_iter = it

    for gamma, p, rep, tag in it_iter:
        seed = int((args.seed + 1) * 1000003 + rep * 17 + int(round(p * 1e6)) * 31 + int(round((0 if np.isinf(gamma) else gamma) * 1000.0)) * 7)
        res = run_one_p(
            outdir=outdir,
            tag=tag,
            N=int(args.N),
            network_model=str(args.network_model),
            za=int(args.za),
            zb=int(args.zb),
            sf_m=int(args.sf_m),
            sf_gamma=float(gamma),
            p=float(p),
            steps=int(args.steps),
            transient=int(args.transient),
            loss_prob=float(args.loss_prob),
            loss_mode=str(args.loss_mode),
            bulk_only=bool(args.bulk_only),
            seed=seed,
            cutoff=int(args.cutoff),
            global_cutoff=int(args.global_cutoff),
            log_every=int(args.numba_log_every),
            save_rank_size=bool(args.save_rank_size),
        )
        row = dict(res["metrics"])
        row.update(
            {
                "tag": tag,
                "N": int(args.N),
                "za": int(args.za),
                "zb": int(args.zb),
                "network_model": str(args.network_model),
                "sf_m": int(args.sf_m),
                "sf_gamma": float(gamma),
                "loss_prob": float(args.loss_prob),
                "loss_mode": str(args.loss_mode),
                "bulk_only": bool(args.bulk_only),
                "steps": int(args.steps),
                "transient": int(args.transient),
                "rep": int(rep),
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "summary_by_p.csv", index=False)

    # Aggregate over replicates
    gcols = ["sf_gamma", "p"] if args.network_model == "sf_static" else ["p"]
    agg = (
        df.groupby(gcols, as_index=False)
        .agg(
            {
                "pr_local_large_a": "mean",
                "pr_inflicted_large_a": "mean",
                "pr_any_large_a": "mean",
                "pr_global_large": "mean",
                "n_events": "sum",
            }
        )
        .sort_values(gcols)
    )
    agg.to_csv(outdir / "summary_by_p_agg.csv", index=False)

    def _plot_one_curve(xp: np.ndarray, y: np.ndarray, *, color, label: str) -> None:
        plt.semilogx(xp, y, marker="o", markersize=4, linewidth=1.6, alpha=0.9, color=color, label=label)

    # Fig.4-like plot(s)
    plt.figure(figsize=(7.2, 4.8))
    if args.network_model == "sf_static":
        gammas_sorted = sorted(gamma_values, key=lambda g: (np.isinf(g), g))
        colors = cm.plasma(np.linspace(0.12, 0.9, len(gammas_sorted)))
        for g, c in zip(gammas_sorted, colors, strict=False):
            sub = agg[np.isclose(agg["sf_gamma"].values, g) | (np.isinf(g) & np.isinf(agg["sf_gamma"].values))]
            if sub.empty:
                continue
            _plot_one_curve(sub["p"].values, sub["pr_any_large_a"].values, color=c, label=f"gamma={_gamma_label(g)}")
        plt.title("New: optimal coupling p* depends on SF exponent")
    else:
        _plot_one_curve(agg["p"].values, agg["pr_local_large_a"].values, color="#1f77b4", label="local in A (origin A)")
        _plot_one_curve(agg["p"].values, agg["pr_inflicted_large_a"].values, color="#d62728", label="inflicted on A (origin B)")
        _plot_one_curve(agg["p"].values, agg["pr_any_large_a"].values, color="#bcbd22", label="A overall")
        plt.title("Interdependence can suppress large cascades (Brummitt 2012)")

    ax = plt.gca()
    _setup_axes(ax)
    plt.xlabel("interconnectivity p")
    plt.ylabel(f"Pr(S_A > {int(args.cutoff)})")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / "pr_large_in_A_vs_p.pdf")
    plt.savefig(outdir / "pr_large_in_A_vs_p.png", dpi=220)
    plt.close()

    # Global large probability
    plt.figure(figsize=(7.2, 4.8))
    if args.network_model == "sf_static":
        gammas_sorted = sorted(gamma_values, key=lambda g: (np.isinf(g), g))
        colors = cm.viridis(np.linspace(0.12, 0.9, len(gammas_sorted)))
        for g, c in zip(gammas_sorted, colors, strict=False):
            sub = agg[np.isclose(agg["sf_gamma"].values, g) | (np.isinf(g) & np.isinf(agg["sf_gamma"].values))]
            if sub.empty:
                continue
            _plot_one_curve(sub["p"].values, sub["pr_global_large"].values, color=c, label=f"gamma={_gamma_label(g)}")
        plt.legend(frameon=False)
        plt.title("Global cascades vs coupling (SF modules)")
    else:
        _plot_one_curve(agg["p"].values, agg["pr_global_large"].values, color="#2ca02c", label="global")
        plt.title("Global cascades tend to grow with coupling")

    ax = plt.gca()
    _setup_axes(ax)
    plt.xlabel("interconnectivity p")
    plt.ylabel(f"Pr(S > {int(args.global_cutoff)})")
    plt.tight_layout()
    plt.savefig(outdir / "pr_large_global_vs_p.pdf")
    plt.savefig(outdir / "pr_large_global_vs_p.png", dpi=220)
    plt.close()

    # p* estimate table/plot (only meaningful for sf_static sweep)
    if args.network_model == "sf_static":
        rows_star = []
        for g in gamma_values:
            sub = agg[np.isclose(agg["sf_gamma"].values, g) | (np.isinf(g) & np.isinf(agg["sf_gamma"].values))]
            if sub.empty:
                continue
            j = int(np.nanargmin(sub["pr_any_large_a"].values))
            pstar = float(sub["p"].values[j])
            rows_star.append({"sf_gamma": float(g), "p_star": pstar, "pr_min": float(sub["pr_any_large_a"].values[j])})
        df_star = pd.DataFrame(rows_star).sort_values("sf_gamma")
        df_star.to_csv(outdir / "p_star_vs_gamma.csv", index=False)

        plt.figure(figsize=(7.2, 4.4))
        xs = [g if not np.isinf(g) else np.nan for g in df_star["sf_gamma"].values]
        plt.plot(xs, df_star["p_star"].values, marker="o", linewidth=1.6)
        ax = plt.gca()
        _setup_axes(ax)
        plt.xlabel("gamma (SF exponent)")
        plt.ylabel("p* (argmin Pr(S_A > C))")
        plt.title("New: optimal interconnectivity p* vs gamma")
        plt.tight_layout()
        plt.savefig(outdir / "p_star_vs_gamma.pdf")
        plt.savefig(outdir / "p_star_vs_gamma.png", dpi=220)
        plt.close()

    print(f"Wrote: {outdir}")


if __name__ == "__main__":
    main()
