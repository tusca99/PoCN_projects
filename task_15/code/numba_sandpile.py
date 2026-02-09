"""Numba-accelerated sandpile dynamics on a graph.

This module provides:
- conversion of a NetworkX graph to CSR (compressed sparse row) adjacency arrays
- Numba-compiled kernels for BTW-style add+relax dynamics

The main purpose in this project is to run the coupled 2-module experiments
inspired by Brummitt et al. (PNAS 2012) efficiently.

Observables produced by the kernels:
- avalanche size S (#topplings)
- duration T (#waves)
- (optional) area A (#distinct toppled nodes)
- losses (grains dissipated)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore


def graph_to_csr_undirected(G) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an undirected NetworkX graph to CSR arrays.

    Returns (indptr, indices, degree) for nodes assumed labeled 0..N-1.
    """
    N = G.number_of_nodes()
    deg = np.zeros(N, dtype=np.int32)

    # First pass: degrees
    for u in G.nodes():
        deg[int(u)] = int(G.degree(u))

    indptr = np.zeros(N + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(deg, dtype=np.int64)

    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    cursor = indptr[:-1].copy()

    # Second pass: fill adjacency
    for u, v in G.edges():
        u = int(u)
        v = int(v)
        pu = int(cursor[u])
        indices[pu] = v
        cursor[u] = pu + 1

        pv = int(cursor[v])
        indices[pv] = u
        cursor[v] = pv + 1

    return indptr, indices, deg


def _missing_numba(*_args, **_kwargs):  # pragma: no cover
    raise ImportError(
        "Numba is required for Task 15 simulations but is not available. "
        "Install it (e.g. 'pip install numba') and rerun."
    )


if njit is None:
    simulate_btw = _missing_numba  # type: ignore
    simulate_btw_two_modules = _missing_numba  # type: ignore
else:

    @njit(cache=True)
    def _xorshift64star(state):
        x = state[0]
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        state[0] = x
        return x * np.uint64(2685821657736338717)


    @njit(cache=True)
    def _rand_u01(state) -> float:
        return float(_xorshift64star(state) >> np.uint64(11)) * (1.0 / (1 << 53))


    @njit(cache=True)
    def simulate_btw(
        indptr: np.ndarray,
        indices: np.ndarray,
        z_c: np.ndarray,
        loss_prob: float,
        loss_mode: int,
        n_steps: int,
        transient: int,
        seed: int,
        active_nodes: np.ndarray,
        log_every: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run BTW-like sandpile with per-transfer loss.

        Assumes the system is stable before each grain addition.
        Records arrays of length n_steps (post-transient additions):
        - area[t], size[t], duration[t], losses[t]
        Zeros correspond to trivial avalanches.
        """
        N = z_c.shape[0]
        n_active = active_nodes.shape[0]
        if n_active <= 0:
            raise ValueError("active_nodes must be non-empty")
        z = np.zeros(N, dtype=np.int32)

        out_area = np.zeros(n_steps, dtype=np.int32)
        out_size = np.zeros(n_steps, dtype=np.int32)
        out_dur = np.zeros(n_steps, dtype=np.int32)
        out_loss = np.zeros(n_steps, dtype=np.int32)

        # Frontier buffers (worst-case size N, but practical is much smaller).
        frontier = np.empty(N, dtype=np.int32)
        next_frontier = np.empty(N, dtype=np.int32)

        # Dedup marks
        in_next = np.zeros(N, dtype=np.uint32)
        in_next_stamp = np.uint32(1)

        toppled = np.zeros(N, dtype=np.uint32)
        toppled_stamp = np.uint32(1)

        rng = np.empty(1, dtype=np.uint64)
        rng[0] = np.uint64(seed if seed != 0 else 1)

        total = n_steps + transient
        rec_i = 0

        for step in range(total):
            if log_every > 0 and (step + 1) % log_every == 0:
                # Numba prints are slow if too frequent; keep log_every large (e.g. 1e5+).
                print("[numba] step", step + 1, "/", total)
            # Add grain at a random node.
            aidx = int(_rand_u01(rng) * n_active)
            if aidx >= n_active:
                aidx = n_active - 1
            node = int(active_nodes[aidx])
            z[node] += 1

            # Avalanche from this node only.
            f_len = 1
            frontier[0] = node
            size = 0
            dur = 0
            area = 0
            losses = 0

            while f_len > 0:
                # Build wave list = unstable nodes in frontier.
                w_len = 0
                for ii in range(f_len):
                    i = frontier[ii]
                    if z[i] >= z_c[i]:
                        frontier[w_len] = i
                        w_len += 1

                if w_len == 0:
                    break

                dur += 1
                nf_len = 0

                # Reset stamp for dedup of next_frontier.
                in_next_stamp = np.uint32(in_next_stamp + 1)
                if in_next_stamp == np.uint32(0):
                    in_next[:] = 0
                    in_next_stamp = np.uint32(1)

                for wi in range(w_len):
                    i = frontier[wi]
                    size += 1

                    if toppled[i] != toppled_stamp:
                        toppled[i] = toppled_stamp
                        area += 1

                    start = indptr[i]
                    end = indptr[i + 1]
                    k_i = int(end - start)

                    z[i] -= k_i
                    # i might remain unstable
                    if z[i] >= z_c[i] and in_next[i] != in_next_stamp:
                        in_next[i] = in_next_stamp
                        next_frontier[nf_len] = i
                        nf_len += 1

                    if loss_prob > 0.0 and loss_mode == 1 and k_i > 0:
                        # per-toppling: with prob=f lose exactly 1 grain by skipping one neighbor transfer.
                        lost_one = _rand_u01(rng) < loss_prob
                        skip = -1
                        if lost_one:
                            losses += 1
                            skip = int(_rand_u01(rng) * k_i)
                            if skip >= k_i:
                                skip = k_i - 1
                        for p in range(start, end):
                            if lost_one and (p - start) == skip:
                                continue
                            nb = indices[p]
                            z[nb] += 1
                            if z[nb] >= z_c[nb] and in_next[nb] != in_next_stamp:
                                in_next[nb] = in_next_stamp
                                next_frontier[nf_len] = nb
                                nf_len += 1
                    else:
                        # per-grain: each outgoing grain is lost independently with prob=f.
                        for p in range(start, end):
                            nb = indices[p]
                            if loss_prob > 0.0 and loss_mode == 0 and _rand_u01(rng) < loss_prob:
                                losses += 1
                                continue
                            z[nb] += 1
                            if z[nb] >= z_c[nb] and in_next[nb] != in_next_stamp:
                                in_next[nb] = in_next_stamp
                                next_frontier[nf_len] = nb
                                nf_len += 1

                # Swap frontiers
                f_len = nf_len
                for ii in range(nf_len):
                    frontier[ii] = next_frontier[ii]

            if step >= transient:
                out_area[rec_i] = area
                out_size[rec_i] = size
                out_dur[rec_i] = dur
                out_loss[rec_i] = losses
                rec_i += 1

            # Advance stamp once per avalanche
            toppled_stamp = np.uint32(toppled_stamp + 1)
            if toppled_stamp == np.uint32(0):
                toppled[:] = 0
                toppled_stamp = np.uint32(1)

        return out_area, out_size, out_dur, out_loss


    @njit(cache=True)
    def simulate_btw_two_modules(
        indptr: np.ndarray,
        indices: np.ndarray,
        z_c: np.ndarray,
        loss_prob: float,
        loss_mode: int,
        n_steps: int,
        transient: int,
        seed: int,
        active_nodes: np.ndarray,
        log_every: int,
        split_index: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """BTW on a graph with two node types (modules).

        Nodes [0, split_index) are module A, nodes [split_index, N) are module B.
        Records arrays of length n_steps (post-transient additions):
        - topplings_a[t], topplings_b[t], duration[t], losses[t], origin[t] (0=A, 1=B)
        """

        N = z_c.shape[0]
        if split_index <= 0 or split_index >= N:
            raise ValueError("split_index must satisfy 0 < split_index < N")

        n_active = active_nodes.shape[0]
        if n_active <= 0:
            raise ValueError("active_nodes must be non-empty")

        z = np.zeros(N, dtype=np.int32)

        out_ta = np.zeros(n_steps, dtype=np.int32)
        out_tb = np.zeros(n_steps, dtype=np.int32)
        out_dur = np.zeros(n_steps, dtype=np.int32)
        out_loss = np.zeros(n_steps, dtype=np.int32)
        out_origin = np.zeros(n_steps, dtype=np.int8)

        frontier = np.empty(N, dtype=np.int32)
        next_frontier = np.empty(N, dtype=np.int32)

        in_next = np.zeros(N, dtype=np.uint32)
        in_next_stamp = np.uint32(1)

        rng = np.empty(1, dtype=np.uint64)
        rng[0] = np.uint64(seed if seed != 0 else 1)

        total = n_steps + transient
        rec_i = 0

        for step in range(total):
            if log_every > 0 and (step + 1) % log_every == 0:
                print("[numba] step", step + 1, "/", total)

            aidx = int(_rand_u01(rng) * n_active)
            if aidx >= n_active:
                aidx = n_active - 1
            node = int(active_nodes[aidx])
            z[node] += 1
            origin = np.int8(0 if node < split_index else 1)

            f_len = 1
            frontier[0] = node

            top_a = 0
            top_b = 0
            dur = 0
            losses = 0

            while f_len > 0:
                w_len = 0
                for ii in range(f_len):
                    i = frontier[ii]
                    if z[i] >= z_c[i]:
                        frontier[w_len] = i
                        w_len += 1
                if w_len == 0:
                    break

                dur += 1
                nf_len = 0

                in_next_stamp = np.uint32(in_next_stamp + 1)
                if in_next_stamp == np.uint32(0):
                    in_next[:] = 0
                    in_next_stamp = np.uint32(1)

                for wi in range(w_len):
                    i = frontier[wi]
                    if i < split_index:
                        top_a += 1
                    else:
                        top_b += 1

                    start = indptr[i]
                    end = indptr[i + 1]
                    k_i = int(end - start)
                    z[i] -= k_i

                    if z[i] >= z_c[i] and in_next[i] != in_next_stamp:
                        in_next[i] = in_next_stamp
                        next_frontier[nf_len] = i
                        nf_len += 1

                    if loss_prob > 0.0 and loss_mode == 1 and k_i > 0:
                        lost_one = _rand_u01(rng) < loss_prob
                        skip = -1
                        if lost_one:
                            losses += 1
                            skip = int(_rand_u01(rng) * k_i)
                            if skip >= k_i:
                                skip = k_i - 1
                        for p in range(start, end):
                            if lost_one and (p - start) == skip:
                                continue
                            nb = indices[p]
                            z[nb] += 1
                            if z[nb] >= z_c[nb] and in_next[nb] != in_next_stamp:
                                in_next[nb] = in_next_stamp
                                next_frontier[nf_len] = nb
                                nf_len += 1
                    else:
                        for p in range(start, end):
                            nb = indices[p]
                            if loss_prob > 0.0 and loss_mode == 0 and _rand_u01(rng) < loss_prob:
                                losses += 1
                                continue
                            z[nb] += 1
                            if z[nb] >= z_c[nb] and in_next[nb] != in_next_stamp:
                                in_next[nb] = in_next_stamp
                                next_frontier[nf_len] = nb
                                nf_len += 1

                f_len = nf_len
                for ii in range(nf_len):
                    frontier[ii] = next_frontier[ii]

            if step >= transient:
                out_ta[rec_i] = top_a
                out_tb[rec_i] = top_b
                out_dur[rec_i] = dur
                out_loss[rec_i] = losses
                out_origin[rec_i] = origin
                rec_i += 1

        return out_ta, out_tb, out_dur, out_loss, out_origin
