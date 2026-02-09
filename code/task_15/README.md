# Task 15: SOC Model (Self-Organized Criticality)

**Score**: 0.6
**Type**: Theoretical

This folder contains the code used for Task 15. Outputs live under `data/task_15/`.

This README is intentionally focused on:
- what each script does;
- how to reproduce the figures used in the report;
- where outputs are written.

## Code map

Used for the final report figures:
- `code/numba_sandpile.py`: Numba-accelerated CSR simulation kernels (including two-module engine).
- `code/generate_networks.py`: generators (random regular modules, static SF modules).
- `code/brummitt2012_experiments.py`: coupled-module experiments (regular modules + SF extension).
- `code/plot_brummitt2012.py`: regenerates report-ready (“pretty”) plots from CSV outputs.

## Reproduce the report figures

All commands below assume you run them from the repo root.

### Brummitt coupled modules (regular)

```bash
python code/task_15/brummitt2012_experiments.py \
  --outdir data/task_15/brummitt2012_R3_final2_2026-02-07 \
  --network-model regular --N 2000 --za 3 --zb 3 \
  --p-list 0.001,0.003,0.01,0.03,0.05,0.075,0.1,0.2 \
  --steps 500000 --transient 50000 \
  --loss-prob 0.01 --loss-mode per-toppling \
  --cutoff 1000 --global-cutoff 2000
```

### Brummitt coupled modules (scale-free extension, static model)

```bash
python code/task_15/brummitt2012_experiments.py \
  --outdir data/task_15/brummitt2012_SF_microP_final2_2026-02-07 \
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
python code/task_15/plot_brummitt2012.py --indir data/task_15/brummitt2012_R3_final2_2026-02-07
python code/task_15/plot_brummitt2012.py --indir data/task_15/brummitt2012_SF_microP_final2_2026-02-07
```

---

Last updated: 2026-02-09
