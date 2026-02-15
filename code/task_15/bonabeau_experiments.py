import argparse
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib import cm

from bonabeau_random_neighbor import RandomNeighborParams, RandomNeighborSandpile


def log_binned_ccdf(data: np.ndarray, bins: int = 60):
    data = data[data > 0]
    if len(data) == 0:
        return np.array([]), np.array([])
    xmin = data.min()
    xmax = data.max()
    if xmin <= 0 or xmax <= 0 or xmin == xmax:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(xmin), np.log10(xmax), bins + 1)
    # Guard against duplicate edges (can happen with extreme ranges)
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.array([]), np.array([])
    centers = np.sqrt(edges[:-1] * edges[1:])

    # CCDF: P(X >= x) computed via searchsorted on sorted data
    sorted_data = np.sort(data)
    n = len(sorted_data)
    idx = np.searchsorted(sorted_data, centers, side="left")
    ccdf = (n - idx) / n
    mask = ccdf > 0
    return centers[mask], ccdf[mask]


def log_binned_pdf(data: np.ndarray, bins: int = 60):
    data = data[data > 0]
    if len(data) == 0:
        return np.array([]), np.array([])
    xmin = data.min()
    xmax = data.max()
    if xmin <= 0 or xmax <= 0 or xmin == xmax:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(xmin), np.log10(xmax), bins + 1)
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.array([]), np.array([])

    # Safer than density=True (avoids divide-by-zero warnings)
    counts, edges = np.histogram(data, bins=edges, density=False)
    total = counts.sum()
    if total <= 0:
        return np.array([]), np.array([])
    widths = edges[1:] - edges[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        hist = counts / (total * widths)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = np.isfinite(hist) & (hist > 0)
    return centers[mask], hist[mask]


def discrete_ccdf(data: np.ndarray):
    """Exact CCDF for discrete positive integer data."""
    data = data[data > 0]
    if len(data) == 0:
        return np.array([]), np.array([])
    xs, counts = np.unique(data.astype(int), return_counts=True)
    # P(X >= x) = 1 - CDF(x-1)
    cdf = np.cumsum(counts) / counts.sum()
    ccdf = 1.0 - np.concatenate(([0.0], cdf[:-1]))
    return xs.astype(float), ccdf.astype(float)


def discrete_pmf(data: np.ndarray):
    """PMF for discrete positive integer data."""
    data = data[data > 0]
    if len(data) == 0:
        return np.array([]), np.array([])
    xs, counts = np.unique(data.astype(int), return_counts=True)
    pmf = counts / counts.sum()
    return xs.astype(float), pmf.astype(float)


def compute_curves(sizes: np.ndarray, durations: np.ndarray, bins: int):
    # Duration is discrete and often noisy in PMF; prefer log-binned PDF when range allows.
    dur_pdf = log_binned_pdf(durations.astype(float), bins=max(10, min(bins, 60)))
    if len(dur_pdf[0]) == 0:
        dur_pdf = discrete_pmf(durations)
    curves = {
        "size": {
            "pdf": log_binned_pdf(sizes, bins=bins),
            "ccdf": log_binned_ccdf(sizes, bins=bins),
        },
        "duration": {
            "pdf": dur_pdf,
            "ccdf": discrete_ccdf(durations),
        },
    }
    return curves


def cutoff_quantile(data: np.ndarray, q: float = 0.99) -> float:
    data = data[data > 0]
    if len(data) == 0:
        return float("nan")
    return float(np.quantile(data, q))


def _log_edges(values: list[float]) -> np.ndarray:
    """Compute log-spaced bin edges for monotone positive values."""
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        # arbitrary +- one octave
        return np.array([v[0] / np.sqrt(10.0), v[0] * np.sqrt(10.0)], dtype=float)
    # geometric means between neighbors
    mids = np.sqrt(v[:-1] * v[1:])
    first = v[0] ** 2 / mids[0]
    last = v[-1] ** 2 / mids[-1]
    return np.concatenate([[first], mids, [last]]).astype(float)


def add_reference_powerlaw(ax, x: np.ndarray, y: np.ndarray, slope: float, label: str):
    """Draw y ~ x^slope reference line anchored at the median x."""
    if len(x) == 0 or len(y) == 0:
        return
    x0 = float(np.median(x))
    # anchor at closest point
    idx = int(np.argmin(np.abs(x - x0)))
    y0 = float(y[idx])
    xs = np.array([x.min(), x.max()], dtype=float)
    ys = y0 * (xs / x0) ** slope
    ax.loglog(xs, ys, linestyle="--", linewidth=1.2, color="0.35", label=label)


def setup_axes(ax):
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", alpha=0.25)


def run_one(
    outdir: Path,
    tag: str,
    params: RandomNeighborParams,
    steps: int,
    transient: int,
    seed: int,
    bins: int,
    cutoff_q: float = 0.99,
    save_per_run: bool = True,
    save_events_csv: bool = True,
):
    print(f"[run] start {tag} (N={params.N}, eps={params.epsilon:g})")
    model = RandomNeighborSandpile(params, seed=seed)
    sizes, durations = model.run(steps=steps, transient=transient)
    print(f"[run] done  {tag}: events={len(sizes)}")

    curves = compute_curves(sizes, durations, bins=bins)

    if save_events_csv:
        df = pd.DataFrame({"size": sizes, "duration": durations})
        df.to_csv(outdir / f"events_{tag}.csv", index=False)

    # PDF + CCDF plots (per-run)
    if save_per_run:
        for name, arr in [("size", sizes), ("duration", durations)]:
            # CCDF
            x_ccdf, y_ccdf = log_binned_ccdf(arr, bins=bins)
            # PDF
            x_pdf, y_pdf = log_binned_pdf(arr, bins=bins)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            if len(x_pdf):
                axes[0].loglog(x_pdf, y_pdf, marker="o", linestyle="none", markersize=3, alpha=0.85)
            setup_axes(axes[0])
            axes[0].set_xlabel(name)
            axes[0].set_ylabel(f"D({name})")
            axes[0].set_title("PDF (log-binned)")

            if len(x_ccdf):
                axes[1].loglog(x_ccdf, y_ccdf, marker="o", linestyle="none", markersize=3, alpha=0.85)
            setup_axes(axes[1])
            axes[1].set_xlabel(name)
            axes[1].set_ylabel(f"P({name} ≥ x)")
            axes[1].set_title("CCDF")

            # Reference slopes: if PDF ~ s^{-3/2}, then CCDF ~ s^{-1/2}; if PDF(T)~T^{-2}, CCDF~T^{-1}
            if name == "size":
                add_reference_powerlaw(axes[1], x_ccdf, y_ccdf, slope=-0.5, label="ref slope -1/2")
            if name == "duration":
                add_reference_powerlaw(axes[1], x_ccdf, y_ccdf, slope=-1.0, label="ref slope -1")

            axes[1].legend(fontsize=8, frameon=False, loc="best")

            fig.suptitle(f"Random-neighbor ({tag})")
            fig.savefig(outdir / f"pdf_ccdf_{name}_{tag}.png", dpi=220, bbox_inches="tight")
            fig.savefig(outdir / f"pdf_ccdf_{name}_{tag}.pdf", bbox_inches="tight")
            plt.close(fig)

    summary = {
        "tag": tag,
        "N": params.N,
        "k_mode": params.k_mode,
        "k_fixed": params.k_fixed,
        "k_mean": params.k_mean,
        "k_sigma": params.k_sigma,
        "epsilon": params.epsilon,
        "steps": steps,
        "transient": transient,
        "n_events": int(len(sizes)),
        "size_max": int(sizes.max()) if len(sizes) else None,
        "dur_max": int(durations.max()) if len(durations) else None,
        "cutoff_q": float(cutoff_q),
        "size_q": cutoff_quantile(sizes, q=float(cutoff_q)),
        "dur_q": cutoff_quantile(durations, q=float(cutoff_q)),
    }
    return summary, sizes, durations, curves


def _job_seed(base_seed: int, N: int, eps: float, rep: int) -> int:
    # Deterministic seed per job across processes.
    e = int(round(eps * 1e12))
    x = (base_seed + 1) * 1000003
    x ^= (N + 0x9E3779B9) * 2654435761
    x ^= (e + 0x85EBCA6B) * 1597334677
    x ^= (rep + 0xC2B2AE35) * 2246822519
    return int(x % (2**32 - 1))


def _run_job(
    outdir_str: str,
    tag: str,
    params_dict: dict,
    steps: int,
    transient: int,
    seed: int,
    bins: int,
    cutoff_q: float,
    save_per_run: bool,
    save_events_csv: bool,
):
    outdir = Path(outdir_str)
    params = RandomNeighborParams(**params_dict)
    summary, _, _, curves = run_one(
        outdir=outdir,
        tag=tag,
        params=params,
        steps=steps,
        transient=transient,
        seed=seed,
        bins=bins,
        cutoff_q=cutoff_q,
        save_per_run=save_per_run,
        save_events_csv=save_events_csv,
    )
    return {"summary": summary, "curves": curves}


def _parse_eps_rep(tag: str):
    # Expected: ..._eps{val}[_repX]
    eps = None
    rep = None
    if "_eps" in tag:
        tail = tag.split("_eps", 1)[1]
        eps_str = tail.split("_", 1)[0]
        try:
            eps = float(eps_str)
        except ValueError:
            eps = None
        if "_rep" in tail:
            rep_str = tail.split("_rep", 1)[1]
            try:
                rep = int(rep_str)
            except ValueError:
                rep = None
    return eps, rep


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent / "../data/bonabeau"))
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--transient", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--N", type=int, default=10_000)
    ap.add_argument(
        "--N-list",
        type=str,
        default="",
        help="Comma-separated N values for a sweep (overrides --N). Example: 5000,20000,50000",
    )
    ap.add_argument("--k-mode", type=str, default="fixed", choices=["fixed", "gaussian", "poisson"])
    ap.add_argument("--k-fixed", type=int, default=4)
    ap.add_argument("--k-mean", type=float, default=4.0)
    ap.add_argument("--k-sigma", type=float, default=1.0)

    ap.add_argument(
        "--replacement",
        action="store_true",
        help="Sample destinations with replacement (faster, classic random-neighbor).",
    )

    ap.add_argument("--eps-list", type=str, default="1e-2,3e-3,1e-3")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes for the sweep (recommended: 4).",
    )
    ap.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Independent replicates per (N, epsilon). Outputs are tagged with _repX.",
    )
    ap.add_argument("--no-duration", action="store_true", help="Skip duration plots (faster)")
    ap.add_argument(
        "--no-per-run",
        action="store_true",
        help="Skip per-epsilon PDF+CCDF panels (faster, only comparison plots + summary).",
    )
    ap.add_argument(
        "--heatmap",
        action="store_true",
        help="If sweeping N and epsilon, plot heatmap of cutoff proxy (q99).",
    )
    ap.add_argument(
        "--cutoff-q",
        type=float,
        default=0.99,
        help="Quantile used as cutoff proxy (default: 0.99).",
    )
    ap.add_argument(
        "--heatmap-min-events",
        type=int,
        default=500,
        help="Mask heatmap cells with fewer than this many recorded avalanches.",
    )
    ap.add_argument(
        "--no-events-csv",
        action="store_true",
        help="Do not write per-run events CSVs (much faster for very large runs).",
    )
    args = ap.parse_args()

    cutoff_q = float(args.cutoff_q)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    eps_values = [float(x.strip()) for x in args.eps_list.split(",") if x.strip()]

    if args.N_list.strip():
        N_values = [int(x.strip()) for x in args.N_list.split(",") if x.strip()]
    else:
        N_values = [int(args.N)]

    summaries = []
    # For overlays we only compare epsilon at a fixed N (the first N in the sweep)
    overlay_curves = {"size": [], "duration": []}  # list[(tag, curves_dict)]

    jobs = max(1, int(args.jobs))
    reps = max(1, int(args.replicates))
    if jobs > 1:
        jobs = min(jobs, (os.cpu_count() or jobs))

    task_specs: list[tuple[str, RandomNeighborParams, int]] = []
    for N in N_values:
        for eps in eps_values:
            for rep in range(reps):
                params = RandomNeighborParams(
                    N=N,
                    k_mode=args.k_mode,
                    k_fixed=args.k_fixed,
                    k_mean=args.k_mean,
                    k_sigma=args.k_sigma,
                    epsilon=eps,
                    allow_replacement=bool(args.replacement),
                )
                tag = f"{args.k_mode}_N{N}_eps{eps:g}"
                if reps > 1:
                    tag += f"_rep{rep}"
                seed = _job_seed(int(args.seed), N=N, eps=eps, rep=rep)
                task_specs.append((tag, params, seed))

    print(
        f"Planned runs: {len(task_specs)} (jobs={jobs}, per-run-plots={not args.no_per_run}, events_csv={not args.no_events_csv})"
    )

    if jobs == 1:
        for tag, params, seed in task_specs:
            summary, _, _, curves = run_one(
                outdir,
                tag,
                params,
                args.steps,
                args.transient,
                seed=seed,
                bins=args.bins,
                cutoff_q=cutoff_q,
                save_per_run=not args.no_per_run,
                save_events_csv=not args.no_events_csv,
            )
            summaries.append(summary)
            if params.N == int(N_values[0]):
                overlay_curves["size"].append((tag, curves))
                if not args.no_duration:
                    overlay_curves["duration"].append((tag, curves))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = []
            for tag, params, seed in task_specs:
                futs.append(
                    ex.submit(
                        _run_job,
                        str(outdir),
                        tag,
                        asdict(params),
                        int(args.steps),
                        int(args.transient),
                        int(seed),
                        int(args.bins),
                        float(cutoff_q),
                        bool(not args.no_per_run),
                        bool(not args.no_events_csv),
                    )
                )
            for fut in as_completed(futs):
                res = fut.result()
                summary = res["summary"]
                summaries.append(summary)
                if int(summary.get("N", -1)) == int(N_values[0]):
                    overlay_curves["size"].append((summary["tag"], res["curves"]))
                    if not args.no_duration:
                        overlay_curves["duration"].append((summary["tag"], res["curves"]))

    pd.DataFrame(summaries).to_csv(outdir / "summary.csv", index=False)

    # Overlay comparisons across epsilon (for the first N)
    # Color by epsilon (stable even with replicates)
    eps_present = []
    for tag, _c in overlay_curves["size"]:
        e, _r = _parse_eps_rep(tag)
        if e is not None:
            eps_present.append(e)
    eps_unique = [e for e in eps_values if e in set(eps_present)]
    if not eps_unique:
        eps_unique = sorted(set(eps_present))
    color_map = {
        e: c
        for e, c in zip(
            eps_unique,
            cm.viridis(np.linspace(0.15, 0.85, max(1, len(eps_unique)))),
            strict=False,
        )
    }

    for name in ["size", "duration"]:
        if name == "duration" and args.no_duration:
            continue
        if len(overlay_curves[name]) == 0:
            continue

        # CCDF overlay
        plt.figure(figsize=(6.2, 4.8))
        last_xy = None
        for tag, curves in overlay_curves[name]:
            e, r = _parse_eps_rep(tag)
            c = color_map.get(e, "0.2")
            label = f"eps={e:g}" if e is not None else tag
            if r is not None:
                label += f" (rep {r})"

            x, y = curves[name]["ccdf"]
            if len(x) == 0:
                continue
            if name == "duration":
                plt.semilogy(
                    x,
                    y,
                    drawstyle="steps-post",
                    linewidth=1.8,
                    alpha=0.9,
                    color=c,
                    label=label,
                )
            else:
                plt.loglog(
                    x,
                    y,
                    marker="o",
                                linestyle="none",
                                markersize=3,
                                alpha=0.85,
                                color=c,
                                label=label,
                            )
            last_xy = (x, y)

        ax = plt.gca()
        if name == "duration":
            ax.grid(True, which="major", alpha=0.25)
        else:
            setup_axes(ax)
            if last_xy is not None:
                x_last, y_last = last_xy
                add_reference_powerlaw(ax, x_last, y_last, slope=-0.5, label="ref slope -1/2")
        plt.xlabel(name)
        plt.ylabel(f"P({name} >= x)")
        plt.title(f"Random-neighbor CCDF ({args.k_mode}, N={N_values[0]})")
        plt.legend(title="epsilon", fontsize=8, frameon=False)
        plt.savefig(outdir / f"compare_ccdf_{name}.png", dpi=240, bbox_inches="tight")
        plt.savefig(outdir / f"compare_ccdf_{name}.pdf", bbox_inches="tight")
        plt.close()

        # PDF overlay
        plt.figure(figsize=(6.2, 4.8))
        for tag, curves in overlay_curves[name]:
            e, r = _parse_eps_rep(tag)
            c = color_map.get(e, "0.2")
            label = f"eps={e:g}" if e is not None else tag
            if r is not None:
                label += f" (rep {r})"

            x, y = curves[name]["pdf"]
            if len(x) == 0:
                continue
            if name == "duration":
                # If log-binned, log-log is readable; if PMF fallback (small integer support), this still works.
                plt.loglog(
                    x,
                    y,
                    marker="o",
                    linestyle="-",
                    linewidth=1.4,
                    markersize=3,
                    alpha=0.9,
                    color=c,
                    label=label,
                )
            else:
                plt.loglog(
                    x,
                    y,
                    marker="o",
                    linestyle="none",
                    markersize=3,
                    alpha=0.85,
                    color=c,
                    label=label,
                )
        ax = plt.gca()
        if name == "duration":
            ax.grid(True, which="major", alpha=0.25)
        else:
            setup_axes(ax)
        plt.xlabel(name)
        plt.ylabel(f"D({name})")
        plt.title(f"Random-neighbor PDF (log-binned) ({args.k_mode}, N={N_values[0]})")
        plt.legend(title="epsilon", fontsize=8, frameon=False)
        plt.savefig(outdir / f"compare_pdf_{name}.png", dpi=240, bbox_inches="tight")
        plt.savefig(outdir / f"compare_pdf_{name}.pdf", bbox_inches="tight")
        plt.close()

    # Optional heatmap of cutoff proxy across (N, epsilon)
    if args.heatmap and len(N_values) > 1 and len(eps_values) > 1:
        sdf = pd.DataFrame(summaries)
        sdf["event_rate"] = sdf["n_events"] / sdf["steps"].replace(0, np.nan)
        # If replicates are used, aggregate to one value per (N, epsilon)
        if sdf.duplicated(subset=["N", "epsilon"]).any():
            sdf = (
                sdf.groupby(["N", "epsilon"], as_index=False)
                .median(numeric_only=True)
                .sort_values(["N", "epsilon"])
            )
        N_sorted = sorted(sdf["N"].unique().tolist())
        eps_sorted = sorted(sdf["epsilon"].unique().tolist())

        def _plot_heatmap(metric: str, title: str, fname: str):
            pivot = sdf.pivot(index="N", columns="epsilon", values=metric).reindex(index=N_sorted, columns=eps_sorted)
            pivot_ne = sdf.pivot(index="N", columns="epsilon", values="n_events").reindex(index=N_sorted, columns=eps_sorted)

            data = pivot.to_numpy(dtype=float)
            ne = pivot_ne.to_numpy(dtype=float)
            # Mask unreliable cells
            min_ev = int(args.heatmap_min_events)
            data = np.where((np.isfinite(ne)) & (ne >= min_ev), data, np.nan)
            data = np.where(np.isfinite(data) & (data > 0), data, np.nan)

            X = _log_edges([float(e) for e in eps_sorted])
            Y = _log_edges([float(n) for n in N_sorted])

            plt.figure(figsize=(7.4, 4.9))
            Z = np.log10(data)
            m = np.ma.masked_invalid(Z)
            im = plt.pcolormesh(X, Y, m, shading="auto", cmap="magma")
            plt.xscale("log")
            plt.yscale("log")
            plt.colorbar(im, label=f"log10({title})")
            plt.xticks(eps_sorted, [f"{e:g}" for e in eps_sorted], rotation=45)
            plt.yticks(N_sorted, [str(n) for n in N_sorted])
            plt.xlabel("epsilon")
            plt.ylabel("N")
            plt.title(f"Cutoff proxy heatmap ({args.k_mode}, q={cutoff_q:g})")
            plt.savefig(outdir / f"{fname}.png", dpi=240, bbox_inches="tight")
            plt.savefig(outdir / f"{fname}.pdf", bbox_inches="tight")
            plt.close()

        # Cutoff heatmaps
        _plot_heatmap("size_q", f"size q{cutoff_q:g}", "heatmap_size_q")
        if not args.no_duration:
            _plot_heatmap("dur_q", f"duration q{cutoff_q:g}", "heatmap_dur_q")

        # Event-rate heatmap (diagnostic: shows where stats are unreliable)
        _plot_heatmap("event_rate", "event_rate", "heatmap_event_rate")

    print(f"Wrote: {outdir}")


if __name__ == "__main__":
    main()
