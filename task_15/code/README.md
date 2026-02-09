# Code map (Task 15)

This folder contains scripts and modules for Task 15.

The final report is Brummitt-centric; Bonabeau/Goh code is kept under `legacy/`.

## Core (used in the final report)
- `numba_sandpile.py`: Numba CSR kernels for BTW (two-modules engine for Brummitt)
- `generate_networks.py`: generators (random regular graphs + SF static model)
- `brummitt2012_experiments.py`: p-sweep runner (regular modules + SF extension)
- `plot_brummitt2012.py`: report-ready plots from CSV outputs (no re-run of simulations)

## Optional (real-network demo, not in the report)
- `soc_real_network.py`: single-network SOC on US power grid (Opsahl). Reads a local edge list, runs kernel, saves CCDF, rank-size, and degree plots.

## Legacy (not used in the final report)
- `legacy/`: Bonabeau/Goh experiments and a pure-Python BTW reference implementation.

## Practical notes
- Keep outputs under `projects/task_15/data/`.
- For long Numba runs, use `--numba-log-every` in scripts that support it.
