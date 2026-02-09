# Task 15: SOC Model (Self-Organized Criticality)

This folder contains the code and data used for Task 15. The authoritative write-up is the LaTeX report (the section for this task lives under the shared report project).

This README is intentionally focused on:
- what each script does;
- how to reproduce the figures used in the report;
- where outputs are written.

## References used (and where they appear)

The Task 15 report section is Brummitt-centric:
1. Brummitt et al. (2012): Coupled modules and the effect of interdependence (local vs inflicted cascades; optimal coupling idea).

Older Bonabeau/Goh replication code is kept under `code/legacy/` but is not used in the final report.

## Code map (what is what)

Used for the final report figures:
- `code/numba_sandpile.py`: Numba-accelerated CSR simulation kernels (including two-module engine).
- `code/generate_networks.py`: generators (random regular modules, static SF modules).
- `code/brummitt2012_experiments.py`: coupled-module experiments (regular modules + SF extension).
- `code/plot_brummitt2012.py`: regenerates report-ready (“pretty”) plots from CSV outputs.

Not used in the final report:
- `code/legacy/`: older Bonabeau/Goh experiments and a pure-Python BTW reference implementation.

## Reproduce the report figures

All commands below assume you run them from `projects/task_15`.

### Brummitt coupled modules (regular)

```bash
python code/brummitt2012_experiments.py \
  --outdir data/brummitt2012_R3_final2_2026-02-07 \
  --network-model regular --N 2000 --za 3 --zb 3 \
  --p-list 0.001,0.003,0.01,0.03,0.05,0.075,0.1,0.2 \
  --steps 500000 --transient 50000 \
  --loss-prob 0.01 --loss-mode per-toppling \
  --cutoff 1000 --global-cutoff 2000
```

### Brummitt coupled modules (scale-free extension, static model)

```bash
python code/brummitt2012_experiments.py \
  --outdir data/brummitt2012_SF_microP_final2_2026-02-07 \
  --network-model sf_static --N 2000 --sf-m 2 \
  --sf-gamma-list 2.2,2.5,3.0,3.5,inf \
  --p-list 0.0001,0.0002,0.0003,0.0005,0.0007,0.001,0.0015,0.002,0.003,0.004,0.005,0.0075,0.01,0.02,0.03,0.05,0.075,0.1 \
  --steps 300000 --transient 30000 \
  --loss-prob 0.01 --loss-mode per-toppling \
  --cutoff 1000 --global-cutoff 2000 \
  --replicates 2
```

### Report-ready plots for Brummitt outputs

```bash
python code/plot_brummitt2012.py --indir data/brummitt2012_R3_final2_2026-02-07
python code/plot_brummitt2012.py --indir data/brummitt2012_SF_microP_final2_2026-02-07
```

Note on the shaded band (“glow”) in the scale-free Brummitt plots: it is a 95% Wilson score confidence interval for the estimated probability of a “large event”, computed from (approximate) aggregated binomial counts in `plot_brummitt2012.py`.

## What is not used in the report

Bonabeau/Goh replication scripts are not used in the final Task 15 write-up and are kept under `code/legacy/`.

## Optional demo (not cited in the report)

A single-network SOC run on the US power grid (Opsahl dataset) is available as a demo:
- Input edge list: `data/opsahl-powergrid/out.opsahl-powergrid`
- Output folder: `data/powergrid_soc_2026-02-07/`
- Script: `code/soc_real_network.py`

Command used:

```bash
python code/soc_real_network.py \
  --outdir data/powergrid_soc_2026-02-07 \
  --edge-file data/opsahl-powergrid/out.opsahl-powergrid \
  --dataset opsahl-powergrid
```

---

Last updated: 2026-02-07
