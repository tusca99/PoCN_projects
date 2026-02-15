#!/usr/bin/env python3
"""Task 44 - Social Connectedness Index: Simple CLI

This is the main entrypoint for the Task 44 pipeline. It provides three
simple commands:

  build    - Build the entire network (country_list + centroids + global network)
  plot     - Generate sanity plots and summary statistics  
  validate - Validate the generated network and geocodes

Usage Examples
--------------
Build everything:
    python task44.py build 
    or 
    python task44.py build --rebuild-country-list --rebuild-centroids to force regeneration of intermediate files

Generate plots:
    python task44.py plot

Validate outputs:
    python task44.py validate
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import os

def get_data_dir() -> Path:
    """Get the default data directory for Task 44."""
    return Path(__file__).resolve().parent.parent / "data"


def cmd_build(args: argparse.Namespace) -> int:
    """Build the complete network pipeline."""
    data_dir = args.data_dir
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    global_dir = data_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Task 44 - Build Pipeline")
    print("=" * 70)
    
    # Step 1: Generate country_list.csv (if needed)
    country_list_file = processed_dir / "country_list.csv"
    if not country_list_file.exists() or args.rebuild_country_list:
        print("\n[1/3] Generating country_list.csv...")
        import build_country_list
        try:
            build_country_list.main(["--data-dir", str(data_dir)])
        except Exception as e:
            print(f"Error generating country_list: {e}")
            return 1
    else:
        print(f"\n[1/3] Using existing country_list.csv: {country_list_file}")
    
    # Step 2: Generate centroids (optional but recommended)
    centroids_file = processed_dir / "centroids.csv"
    if args.skip_centroids:
        print("\n[2/3] Skipping centroids generation (--skip-centroids)")
        centroids_arg = None
    elif centroids_file.exists() and not args.rebuild_centroids:
        print(f"\n[2/3] Using existing centroids: {centroids_file}")
        centroids_arg = centroids_file
    else:
        print("\n[2/3] Generating centroids (lat/lon + labels)...")
        import build_centroids
        try:
            cent_argv = [
                "--data-dir", str(data_dir),
                "--out", str(centroids_file),
            ]
            if args.centroid_workers is not None:
                cent_argv.extend(["--workers", str(args.centroid_workers)])
            build_centroids.main(cent_argv)
            centroids_arg = centroids_file
        except Exception as e:
            print(f"Warning: centroids generation failed: {e}")
            print("Continuing without centroids (nodes will have empty lat/lon)")
            centroids_arg = None
    
    # Step 3: Build global network
    print("\n[3/3] Building global network (nodes.csv + edges.csv)...")
    import build_global_network
    
    build_argv = [
        "--data-dir", str(data_dir),
        "--out-dir", str(global_dir),
    ]
    
    if args.all_countries:
        build_argv.append("--all-countries")
    
    if centroids_arg:
        build_argv.extend(["--centroids", str(centroids_arg)])
    
    if not args.no_weighted:
        build_argv.append("--write-weighted")
    
    if args.max_rows:
        build_argv.extend(["--max-rows", str(args.max_rows)])
    
    try:
        build_global_network.main(build_argv)
    except Exception as e:
        print(f"Error building global network: {e}")
        return 1
    
    print("\n" + "=" * 70)
    print("Build complete!")
    print(f"  nodes.csv: {global_dir / 'nodes.csv'}")
    print(f"  edges.csv: {global_dir / 'edges.csv'}")
    if not args.no_weighted:
        print(f"  edges_weighted.csv: {global_dir / 'edges_weighted.csv'}")
    print("=" * 70)
    
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Generate sanity plots and summary statistics."""
    data_dir = args.data_dir
    global_dir = data_dir / "global"
    
    print("=" * 70)
    print("Task 44 - Generate Plots")
    print("=" * 70)
    
    import sanity_plots_global
    
    plot_argv = [
        "--nodes", str(global_dir / "nodes.csv"),
        "--edges", str(global_dir / "edges.csv"),
        "--outdir", str(global_dir),
    ]
    
    edges_weighted = global_dir / "edges_weighted.csv"
    if edges_weighted.exists():
        plot_argv.extend(["--edges-weighted", str(edges_weighted)])
    
    if args.top_k:
        plot_argv.extend(["--top-k", str(args.top_k)])
    
    try:
        sanity_plots_global.main(plot_argv)
    except Exception as e:
        print(f"Error generating plots: {e}")
        return 1
    
    print("\n" + "=" * 70)
    print("Plots complete!")
    print(f"  Output: {global_dir / 'plots'}")
    print(f"  Summary: {global_dir / 'summary_by_country.csv'}")
    print("=" * 70)
    
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run weighted network analysis and produce report figures."""
    data_dir = args.data_dir

    print("=" * 70)
    print("Task 44 - Weighted Network Analysis")
    print("=" * 70)

    import weighted_analysis

    try:
        weighted_analysis.analyse(
            data_dir=data_dir,
            fig_dir=args.fig_dir,
            top_k=args.top_k,
            max_nodes_clustering=args.max_nodes_clustering,
        )
    except Exception as e:
        print(f"Error in weighted analysis: {e}")
        return 1

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate the generated network and geocodes."""
    data_dir = args.data_dir
    processed_dir = data_dir / "processed"
    global_dir = data_dir / "global"
    
    print("=" * 70)
    print("Task 44 - Validation")
    print("=" * 70)
    
    # Validate network structure
    print("\n[1/2] Validating network structure...")
    import validate_global_network
    
    val_argv = [
        "--nodes", str(global_dir / "nodes.csv"),
        "--edges", str(global_dir / "edges.csv"),
    ]
    
    if args.show_top:
        val_argv.extend(["--show-top", str(args.show_top)])
    
    try:
        validate_global_network.main(val_argv)
    except Exception as e:
        print(f"Error in network validation: {e}")
        return 1
    
    # Validate geocodes (if validation script exists)
    print("\n[2/2] Validating geocodes...")
    try:
        import validate_geocodes
        geocode_argv = [
            "--data-dir", str(data_dir),
            "--centroids", str(processed_dir / "centroids.csv"),
        ]
        validate_geocodes.main(geocode_argv)
    except ImportError:
        print("  (validate_geocodes.py not found, skipping)")
    except Exception as e:
        print(f"Warning in geocode validation: {e}")
    
    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)
    
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task 44 - Social Connectedness Index: Simple Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_data_dir(),
        help="Task 44 data directory (default: projects/task_44/data)",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")
    
    # BUILD command
    build_parser = subparsers.add_parser(
        "build",
        help="Build the complete network (country_list + centroids + network)",
    )
    build_parser.add_argument(
        "--all-countries",
        action="store_true",
        help="Include all available countries",
    )
    build_parser.add_argument(
        "--skip-centroids",
        action="store_true",
        help="Skip centroids generation (nodes will have empty lat/lon)",
    )
    build_parser.add_argument(
        "--rebuild-country-list",
        action="store_true",
        help="Force rebuild of country_list.csv even if it exists",
    )
    build_parser.add_argument(
        "--rebuild-centroids",
        action="store_true",
        help="Force rebuild of centroids.csv even if it exists",
    )
    build_parser.add_argument(
        "--centroid-workers",
        type=int,
        default=min(4, (os.cpu_count() or 1) - 1),
        help="Parallel workers for centroid downloads (default: min(4, cpu_count - 1))",
    )
    build_parser.add_argument(
        "--write-weighted",
        action="store_true",
        help="Deprecated: weighted output is on by default.",
    )
    build_parser.add_argument(
        "--no-weighted",
        action="store_true",
        help="Do not write edges_weighted.csv",
    )
    build_parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of rows read from layer CSVs (for testing)",
    )
    build_parser.set_defaults(func=cmd_build)
    
    # PLOT command
    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate sanity plots and summary statistics",
    )
    plot_parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top countries to show in plots (default: 15)",
    )
    plot_parser.set_defaults(func=cmd_plot)
    
    # ANALYZE command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run weighted network analysis and produce report figures",
    )
    analyze_parser.add_argument(
        "--fig-dir", type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "latex" / "figures" / "task44",
        help="Output directory for report figures",
    )
    analyze_parser.add_argument("--top-k", type=int, default=15)
    analyze_parser.add_argument(
        "--max-nodes-clustering", type=int, default=120,
        help="Skip weighted clustering for countries with N > this",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # VALIDATE command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate generated network and geocodes",
    )
    validate_parser.add_argument(
        "--show-top",
        type=int,
        default=15,
        help="Number of top countries to show (default: 15)",
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    args = parser.parse_args(argv)
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
