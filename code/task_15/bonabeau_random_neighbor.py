"""Random-neighbor sandpile model (Bonabeau-style, annealed topology).

This is not the classic BTW on a fixed lattice: each toppling redistributes grains
to randomly chosen neighbors (annealed connectivity).

We support:
- k fixed (same for all nodes)
- k_i i.i.d. drawn once per node from a distribution (quenched thresholds)

Dissipation:
- each grain sent during a toppling is dissipated with probability epsilon.

Measurements:
- avalanche size s: number of toppling events
- avalanche duration T: number of parallel waves (generations) until stable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import numpy as np


KMode = Literal["fixed", "gaussian", "poisson"]


@dataclass
class RandomNeighborParams:
    N: int = 10_000
    k_mode: KMode = "fixed"
    k_fixed: int = 4
    k_mean: float = 4.0
    k_sigma: float = 1.0
    epsilon: float = 1e-3  # per-grain dissipation probability
    allow_replacement: bool = False


class RandomNeighborSandpile:
    def __init__(self, params: RandomNeighborParams, seed: Optional[int] = None):
        if params.N <= 1:
            raise ValueError("N must be > 1")
        if params.k_fixed < 1:
            raise ValueError("k_fixed must be >= 1")
        if not (0.0 <= params.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0,1]")

        self.p = params
        self.rng = np.random.default_rng(seed)

        # heights
        self.z = np.zeros(self.p.N, dtype=np.int64)

        # Quenched thresholds k_i (or constant k).
        self.k = self._sample_k_sequence()

    def _sample_k_sequence(self) -> np.ndarray:
        if self.p.k_mode == "fixed":
            return np.full(self.p.N, int(self.p.k_fixed), dtype=np.int64)
        if self.p.k_mode == "poisson":
            k = self.rng.poisson(self.p.k_mean, size=self.p.N)
            k = np.maximum(1, k)
            return k.astype(np.int64)
        if self.p.k_mode == "gaussian":
            k = np.rint(self.rng.normal(self.p.k_mean, self.p.k_sigma, size=self.p.N)).astype(np.int64)
            k = np.maximum(1, k)
            return k
        raise ValueError(f"Unknown k_mode: {self.p.k_mode}")

    def add_grain(self, node: Optional[int] = None) -> Tuple[int, int]:
        if node is None:
            node = int(self.rng.integers(0, self.p.N))
        self.z[node] += 1
        # The configuration is stable between grain additions, so the only possible
        # instability after adding a grain is at the perturbed node.
        if self.z[node] < self.k[node]:
            return 0, 0
        return self.avalanche([node])

    def avalanche(self, start_nodes: list[int]) -> Tuple[int, int]:
        size = 0
        duration = 0

        # Generation-based queue to compute parallel-wave duration.
        current = start_nodes

        in_next = np.zeros(self.p.N, dtype=bool)

        while current:
            duration += 1
            next_gen = []

            for i in current:
                if self.z[i] < self.k[i]:
                    continue

                size += 1
                k_i = int(self.k[i])
                self.z[i] -= k_i

                # Dissipate per grain: sample how many grains survive, then only sample those destinations.
                if self.p.epsilon <= 0:
                    m = k_i
                elif self.p.epsilon >= 1:
                    m = 0
                else:
                    m = int(self.rng.binomial(k_i, 1.0 - self.p.epsilon))

                if m <= 0:
                    continue

                # Choose m random destinations (annealed)
                if self.p.allow_replacement:
                    nbrs = self.rng.integers(0, self.p.N, size=m)
                else:
                    if m >= self.p.N:
                        nbrs = self.rng.integers(0, self.p.N, size=m)
                    else:
                        candidates = self.rng.choice(self.p.N - 1, size=m, replace=False)
                        nbrs = candidates + (candidates >= i)

                np.add.at(self.z, nbrs, 1)

                # Any node that becomes unstable enters next generation once.
                # We check condition after increment; use mask to avoid duplicates.
                # Note: iterating over nbrs is O(k_i) and avoids np.unique overhead.
                for j in nbrs:
                    jj = int(j)
                    if not in_next[jj] and self.z[jj] >= self.k[jj]:
                        in_next[jj] = True
                        next_gen.append(jj)

            # reset mask for nodes we added
            for j in next_gen:
                in_next[j] = False

            current = next_gen

        return size, duration

    def run(self, steps: int, transient: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        sizes = []
        durations = []
        for t in range(steps + transient):
            s, T = self.add_grain()
            if t >= transient and s > 0:
                sizes.append(s)
                durations.append(T)
        return np.asarray(sizes, dtype=int), np.asarray(durations, dtype=int)


if __name__ == "__main__":
    p = RandomNeighborParams(N=2000, k_mode="fixed", k_fixed=4, epsilon=1e-3)
    m = RandomNeighborSandpile(p, seed=0)
    s, T = m.run(steps=2000, transient=200)
    print("events", len(s), "mean size", float(s.mean()) if len(s) else None)
