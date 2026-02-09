import argparse
from pathlib import Path
import pickle

import numpy as np
import networkx as nx


def add_sink_connected_to_all(G: nx.Graph, sink_label: str = "sink") -> nx.Graph:
    if sink_label in G:
        raise ValueError("sink label already in graph")
    G = G.copy()
    G.add_node(sink_label)
    G.add_edges_from((sink_label, n) for n in G.nodes() if n != sink_label)
    return G


def build_lattice_2d(L: int) -> nx.Graph:
    # 2D grid with open boundaries
    G = nx.grid_2d_graph(L, L)
    # Relabel nodes to integers for easier handling
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return G


def build_er(N: int, p: float, seed: int) -> nx.Graph:
    return nx.erdos_renyi_graph(N, p, seed=seed)


def build_ba(N: int, m: int, seed: int) -> nx.Graph:
    return nx.barabasi_albert_graph(N, m, seed=seed)


def build_ws(N: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(N, k, p, seed=seed)


def build_sf_configuration(
    N: int,
    gamma: float,
    k_min: int,
    seed: int,
    k_max: int | None = None,
    ensure_connected: bool = True,
) -> nx.Graph:
    """Scale-free random graph from a power-law degree sequence via configuration model.

    Notes:
    - The simple-graph projection (removing multi-edges/self-loops) slightly perturbs the degree sequence.
    - If ensure_connected=True, we keep the largest connected component and relabel nodes to 0..n-1.
    """
    if N < 10:
        raise ValueError("N too small")
    if gamma <= 2.0:
        raise ValueError("gamma must be > 2 for a well-behaved mean degree")
    if k_min < 1:
        raise ValueError("k_min must be >= 1")

    rng = np.random.default_rng(seed)
    if k_max is None:
        # Natural cutoff ~ N^{1/(gamma-1)}; cap to N-1.
        k_max = int(min(N - 1, max(k_min + 1, round(N ** (1.0 / (gamma - 1.0))))))
    k_max = int(min(N - 1, max(k_min, k_max)))

    # Inverse-CDF sampling for continuous power law, then discretize.
    a = 1.0 - gamma
    lo = float(k_min) ** a
    hi = float(k_max) ** a
    u = rng.random(N)
    k = (lo + u * (hi - lo)) ** (1.0 / a)
    deg = np.clip(np.rint(k).astype(int), k_min, k_max)

    # Ensure graphical parity (even sum of degrees).
    s = int(deg.sum())
    if s % 2 == 1:
        idx = int(rng.integers(0, N))
        if deg[idx] < k_max:
            deg[idx] += 1
        elif deg[idx] > k_min:
            deg[idx] -= 1
        else:
            # Fallback: flip a different index.
            idx2 = int((idx + 1) % N)
            if deg[idx2] < k_max:
                deg[idx2] += 1
            elif deg[idx2] > k_min:
                deg[idx2] -= 1

    MG = nx.configuration_model(deg.tolist(), seed=seed)
    G = nx.Graph(MG)
    G.remove_edges_from(nx.selfloop_edges(G))

    if ensure_connected and G.number_of_nodes() > 0:
        if not nx.is_connected(G):
            cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(cc).copy()
        G = nx.convert_node_labels_to_integers(G, first_label=0)

    return G


def build_sf_static_model(N: int, m: int, gamma: float, seed: int) -> nx.Graph:
    """Scale-free network via the static model (Goh et al. 2001/2003).

    Procedure (as used in PRL 2003 sandpile paper):
    - Assign weights w_i = i^{-alpha} with alpha = 1/(gamma-1).
    - Repeatedly pick (i,j) independently with probabilities proportional to w.
    - Add edge if not already present and i!=j.
    - Stop when number of edges reaches m*N (mean degree = 2m).

    Notes:
    - Produces a simple undirected graph (no multi-edges/self-loops).
    - For alpha=0 (gamma=inf) this approaches ER.
    """
    if N < 10:
        raise ValueError("N too small")
    if m < 1:
        raise ValueError("m must be >= 1")
    if not np.isinf(gamma) and gamma <= 2.0:
        raise ValueError("gamma must be > 2 (or inf)")

    rng = np.random.default_rng(seed)
    alpha = 0.0 if np.isinf(gamma) else 1.0 / (gamma - 1.0)

    # Nodes are indexed 1..N in the paper; we keep 0..N-1 and use i+1.
    idx = np.arange(1, N + 1, dtype=float)
    w = idx ** (-alpha)
    p = w / w.sum()
    cdf = np.cumsum(p)
    cdf[-1] = 1.0

    target_edges = int(m) * int(N)
    edges: set[tuple[int, int]] = set()

    # Oversample in batches to reduce Python overhead.
    while len(edges) < target_edges:
        remaining = target_edges - len(edges)
        batch = int(min(max(10000, remaining * 2), 500000))
        u1 = rng.random(batch)
        u2 = rng.random(batch)
        a = np.searchsorted(cdf, u1, side="right").astype(int)
        b = np.searchsorted(cdf, u2, side="right").astype(int)
        for i, j in zip(a.tolist(), b.tolist(), strict=False):
            if i == j:
                continue
            if i > j:
                i, j = j, i
            edges.add((i, j))
            if len(edges) >= target_edges:
                break

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    return G


def save_graph(G: nx.Graph, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(G, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent / "../data/networks"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--add-sink", action="store_true", help="Add a sink node connected to all nodes")

    ap.add_argument("--lattice-L", type=int, default=32)
    ap.add_argument("--er-N", type=int, default=1000)
    ap.add_argument("--er-p", type=float, default=0.01)
    ap.add_argument("--ba-N", type=int, default=1000)
    ap.add_argument("--ba-m", type=int, default=5)
    ap.add_argument("--ws-N", type=int, default=1000)
    ap.add_argument("--ws-k", type=int, default=10)
    ap.add_argument("--ws-p", type=float, default=0.1)

    ap.add_argument("--sf-N", type=int, default=0, help="If >0, also generate a scale-free config-model graph")
    ap.add_argument("--sf-gamma", type=float, default=3.0)
    ap.add_argument("--sf-kmin", type=int, default=2)
    ap.add_argument("--sf-kmax", type=int, default=0, help="Optional max degree (0 => automatic)")

    ap.add_argument("--static-N", type=int, default=0, help="If >0, also generate a scale-free static-model graph")
    ap.add_argument("--static-m", type=int, default=2, help="Static model parameter (mean degree = 2m)")
    ap.add_argument("--static-gamma", type=float, default=3.0)

    args = ap.parse_args()
    outdir = Path(args.outdir).resolve()

    graphs = {
        f"lattice_L{args.lattice_L}": build_lattice_2d(args.lattice_L),
        f"ER_N{args.er_N}_p{args.er_p}": build_er(args.er_N, args.er_p, args.seed),
        f"BA_N{args.ba_N}_m{args.ba_m}": build_ba(args.ba_N, args.ba_m, args.seed),
        f"WS_N{args.ws_N}_k{args.ws_k}_p{args.ws_p}": build_ws(args.ws_N, args.ws_k, args.ws_p, args.seed),
    }

    if int(args.sf_N) > 0:
        kmax = None if int(args.sf_kmax) <= 0 else int(args.sf_kmax)
        graphs[
            f"SFcfg_N{args.sf_N}_g{args.sf_gamma:g}_kmin{args.sf_kmin}_kmax{(kmax if kmax is not None else 'auto')}"
        ] = build_sf_configuration(
            N=int(args.sf_N),
            gamma=float(args.sf_gamma),
            k_min=int(args.sf_kmin),
            k_max=kmax,
            seed=int(args.seed),
            ensure_connected=True,
        )

    if int(args.static_N) > 0:
        graphs[f"SFstatic_N{args.static_N}_m{args.static_m}_g{args.static_gamma:g}"] = build_sf_static_model(
            N=int(args.static_N),
            m=int(args.static_m),
            gamma=float(args.static_gamma),
            seed=int(args.seed),
        )

    for name, G in graphs.items():
        if args.add_sink:
            G = add_sink_connected_to_all(G, sink_label="sink")
        save_graph(G, outdir / f"{name}.pkl")
        print(f"Saved {name}: N={G.number_of_nodes()} E={G.number_of_edges()} -> {outdir / (name + '.pkl')}")


if __name__ == "__main__":
    main()
