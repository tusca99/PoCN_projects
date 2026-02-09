"""Plotting utilities for Brummitt-style experiments.

Reads existing CSV outputs produced by brummitt2012_experiments.py and creates
cleaner, report-ready plots without rerunning simulations.

Usage:
  ./.venv/bin/python projects/task_15/code/plot_brummitt2012.py \
    --indir projects/task_15/data/brummitt2012_SF_final_2026-02-02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def _wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for binomial proportion.

    Returns (lo, hi). Works with vector inputs.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    n_safe = np.maximum(n, 1.0)
    phat = np.clip(k / n_safe, 0.0, 1.0)

    denom = 1.0 + (z**2) / n_safe
    center = (phat + (z**2) / (2.0 * n_safe)) / denom
    half = (
        z
        * np.sqrt((phat * (1.0 - phat) / n_safe) + (z**2) / (4.0 * (n_safe**2)))
        / denom
    )
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    return lo, hi


def _gamma_label(gamma: float) -> str:
    if np.isinf(gamma):
        return "inf"
    return f"{gamma:g}"


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png")
    plt.close(fig)


def plot_sf(indir: Path, *, outdir: Path | None = None) -> None:
    outdir = indir if outdir is None else outdir
    _ensure_dir(outdir)

    df = pd.read_csv(indir / "summary_by_p.csv")
    if "sf_gamma" not in df.columns:
        raise ValueError("summary_by_p.csv does not look like sf_static output (missing sf_gamma).")

    # Aggregate over replicates by reconstructing binomial counts.
    # NOTE: summary_by_p.csv stores probabilities and n_events, not raw counts.
    # We approximate counts as round(p*n). This is sufficient for clean CIs.
    df = df.copy()
    df["k_any_large_a"] = np.rint(df["pr_any_large_a"].to_numpy(dtype=float) * df["n_events"].to_numpy(dtype=float)).astype(int)
    df["k_global_large"] = np.rint(df["pr_global_large"].to_numpy(dtype=float) * df["n_events"].to_numpy(dtype=float)).astype(int)

    g = (
        df.groupby(["sf_gamma", "p"], as_index=False)
        .agg(
            k_any_large_a=("k_any_large_a", "sum"),
            k_global_large=("k_global_large", "sum"),
            n_events=("n_events", "sum"),
            nrep=("rep", "nunique"),
        )
        .sort_values(["sf_gamma", "p"])
    )

    g["pr_any_large_a"] = g["k_any_large_a"] / np.maximum(1, g["n_events"])
    g["pr_global_large"] = g["k_global_large"] / np.maximum(1, g["n_events"])

    lo_any, hi_any = _wilson_interval(g["k_any_large_a"].to_numpy(), g["n_events"].to_numpy())
    lo_glob, hi_glob = _wilson_interval(g["k_global_large"].to_numpy(), g["n_events"].to_numpy())
    g["pr_any_large_a_lo"] = lo_any
    g["pr_any_large_a_hi"] = hi_any
    g["pr_global_large_lo"] = lo_glob
    g["pr_global_large_hi"] = hi_glob

    gammas = sorted(g["sf_gamma"].unique(), key=lambda x: (np.isinf(x), float(x)))
    cmap = plt.get_cmap("plasma")
    colors = [cmap(v) for v in np.linspace(0.12, 0.9, len(gammas))]

    # Plot: Pr(S_A > C) vs p
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for gamma, color in zip(gammas, colors, strict=False):
        sub = g[(g["sf_gamma"] == gamma)].sort_values("p")
        x = sub["p"].to_numpy()
        y = sub["pr_any_large_a"].to_numpy()

        ax.semilogx(x, y, marker="o", color=color, label=rf"$\gamma={_gamma_label(float(gamma))}$")
        ax.fill_between(x, sub["pr_any_large_a_lo"].to_numpy(), sub["pr_any_large_a_hi"].to_numpy(), color=color, alpha=0.18, linewidth=0)

        # Mark p* (argmin on sampled grid)
        j = int(np.nanargmin(y))
        ax.scatter([x[j]], [y[j]], color=color, s=28, zorder=5)

    ax.set_title(r"SF modules: optimal coupling $p^*(\gamma)$")
    ax.set_xlabel("interconnectivity $p$")
    ax.set_ylabel(r"$\mathrm{Pr}(S_A > C)$")
    ax.legend(frameon=False, ncol=2)
    ax.set_xlim(min(g["p"].min(), 1e-3), max(g["p"].max(), 1e-1))

    _save(fig, outdir, "pr_large_in_A_vs_p_pretty")

    # Plot: global Pr(S > Cg) vs p
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    cmap2 = plt.get_cmap("viridis")
    colors2 = [cmap2(v) for v in np.linspace(0.12, 0.9, len(gammas))]
    for gamma, color in zip(gammas, colors2, strict=False):
        sub = g[(g["sf_gamma"] == gamma)].sort_values("p")
        x = sub["p"].to_numpy()
        y = sub["pr_global_large"].to_numpy()

        ax.semilogx(x, y, marker="o", color=color, label=rf"$\gamma={_gamma_label(float(gamma))}$")
        ax.fill_between(x, sub["pr_global_large_lo"].to_numpy(), sub["pr_global_large_hi"].to_numpy(), color=color, alpha=0.18, linewidth=0)

    ax.set_title("SF modules: global large cascades")
    ax.set_xlabel("interconnectivity $p$")
    ax.set_ylabel(r"$\mathrm{Pr}(S > C_g)$")
    ax.legend(frameon=False, ncol=2)
    ax.set_xlim(min(g["p"].min(), 1e-3), max(g["p"].max(), 1e-1))

    _save(fig, outdir, "pr_large_global_vs_p_pretty")

    # p* vs gamma table/plot (use a numeric x with a special tick for infinity).
    rows = []
    for gamma in gammas:
        sub = g[g["sf_gamma"] == gamma].sort_values("p")
        x = sub["p"].to_numpy()
        y = sub["pr_any_large_a"].to_numpy()
        j = int(np.nanargmin(y))
        rows.append(
            {
                "sf_gamma": float(gamma),
                "p_star": float(x[j]),
                "pr_min": float(y[j]),
                "n_events": int(sub["n_events"].sum()),
                "nrep": int(sub["nrep"].max()),
            }
        )

    df_star = pd.DataFrame(rows)
    df_star.to_csv(outdir / "p_star_vs_gamma_pretty.csv", index=False)

    # Map inf to a finite coordinate for plotting.
    finite = df_star[np.isfinite(df_star["sf_gamma"])]["sf_gamma"].to_numpy()
    x_inf = (finite.max() + 0.6) if finite.size else 4.0
    xs = np.array([x if np.isfinite(x) else x_inf for x in df_star["sf_gamma"].to_numpy()], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(xs, df_star["p_star"].to_numpy(), marker="o")
    ax.set_title(r"$p^*$ vs $\gamma$ (grid argmin)")
    ax.set_xlabel(r"SF exponent $\gamma$")
    ax.set_ylabel(r"$p^*$")

    # Custom ticks including infinity.
    xticks = xs
    xlabels = [_gamma_label(float(g)) for g in df_star["sf_gamma"].to_numpy()]
    ax.set_xticks(xticks, xlabels)

    _save(fig, outdir, "p_star_vs_gamma_pretty")


def plot_regular(indir: Path, *, outdir: Path | None = None) -> None:
    outdir = indir if outdir is None else outdir
    _ensure_dir(outdir)

    df = pd.read_csv(indir / "summary_by_p.csv")

    g = (
        df.groupby(["p"], as_index=False)
        .agg(
            pr_local_large_a_mean=("pr_local_large_a", "mean"),
            pr_inflicted_large_a_mean=("pr_inflicted_large_a", "mean"),
            pr_any_large_a_mean=("pr_any_large_a", "mean"),
            pr_global_large_mean=("pr_global_large", "mean"),
        )
        .sort_values(["p"])
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.semilogx(g["p"], g["pr_local_large_a_mean"], marker="o", label="local in A")
    ax.semilogx(g["p"], g["pr_inflicted_large_a_mean"], marker="o", label="inflicted on A")
    ax.semilogx(g["p"], g["pr_any_large_a_mean"], marker="o", label="overall in A")
    ax.set_title("Regular modules: optimum coupling")
    ax.set_xlabel("interconnectivity $p$")
    ax.set_ylabel(r"$\mathrm{Pr}(S_A > C)$")
    ax.legend(frameon=False)
    _save(fig, outdir, "pr_large_in_A_vs_p_pretty")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.semilogx(g["p"], g["pr_global_large_mean"], marker="o", color="#2ca02c")
    ax.set_title("Regular modules: global large cascades")
    ax.set_xlabel("interconnectivity $p$")
    ax.set_ylabel(r"$\mathrm{Pr}(S > C_g)$")
    _save(fig, outdir, "pr_large_global_vs_p_pretty")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True, help="Directory containing summary_by_p.csv")
    ap.add_argument("--outdir", type=str, default="", help="Optional output directory (default: indir)")
    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else None

    _style()

    df = pd.read_csv(indir / "summary_by_p.csv")
    if "sf_gamma" in df.columns and df["network_model"].iloc[0] == "sf_static":
        plot_sf(indir, outdir=outdir)
    else:
        plot_regular(indir, outdir=outdir)

    print(f"Wrote pretty plots under: {outdir or indir}")


if __name__ == "__main__":
    main()
