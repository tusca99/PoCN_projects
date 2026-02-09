# Task 44: Social Connectedness Index II (Facebook)

**Score**: 1.0
**Type**: Data Analytics

## Objective

Build the global deliverable: two CSV files (`nodes.csv` + `edges.csv`) representing within-country SCI networks (USA excluded by default).

## Data sources

- **Primary**: [Social Connectedness Index (HumData)](https://data.humdata.org/dataset/social-connectedness-index)
- **Alternative**: [Meta AI SCI Official](https://ai.meta.com/ai-for-good/datasets/social-connectedness-index/#accessdata)
- **GADM**: [gadm.org](https://gadm.org/download_world.html)
- **NUTS reference**: [Wikipedia](https://en.wikipedia.org/wiki/Nomenclature_of_Territorial_Units_for_Statistics)

## Inputs

Layer CSVs (as downloaded from HDX/Meta):

```
user_country,friend_country,user_region,friend_region,scaled_sci
PH,PH,PHL.10_1,PHL.11_1,7258
PH,KR,PHL.10_1,KOR.15_1,885
```

In this workspace:
- `data/task_44/gadm1/gadm1.csv` (GADM level 1 regions)
- `data/task_44/nuts_2024/nuts3_2024.csv` (EU NUTS3)
- `data/task_44/all_region_to_country/*.csv` (fast country list source)

## Outputs

Canonical outputs live under `data/task_44/global/`:

- `nodes.csv` with `nodeID,nodeLabel,latitude,longitude`
- `edges.csv` with `nodeID_from,nodeID_to,country_name,country_ISO3`
- `edges_weighted.csv` (optional) adds `scaled_sci`

Plots and summary (optional) are written to:
- `data/task_44/global/plots/`
- `data/task_44/global/summary_by_country.csv`

## Quickstart

### Step 0: Manual downloads (required)

Download the SCI layer CSVs from the HumData/Meta page and place them exactly here:

- `data/task_44/all_region_to_country/`
	- `gadm1_to_country.csv`
	- `nuts3_2024_to_country.csv`
- `data/task_44/gadm1/`
	- `gadm1.csv`
- `data/task_44/nuts_2024/`
	- `nuts3_2024.csv`
- `data/task_44/country/`
	- `country.csv`

Only these layer files are used by the pipeline.

### Step 1: Run the pipeline

Run from the repo root:

```bash
python code/task_44/task44.py build
python code/task_44/task44.py plot
python code/task_44/task44.py validate
```

Useful flags:
- `--rebuild-country-list` and `--rebuild-centroids`
- `--skip-centroids` (nodes will have empty lat/lon)
- `--all-countries` (ignore top-k selection)
- `--no-weighted` (skip `edges_weighted.csv`)
- `--centroid-workers N` (parallel GADM downloads)

Intermediate files:
- `data/task_44/processed/country_list.csv`
- `data/task_44/processed/centroids.csv`

## Code map

All scripts live in `code/task_44/`.

- `task44.py`: single CLI (build, plot, validate)
- `build_country_list.py`: builds `data/processed/country_list.csv`
- `build_centroids.py`: builds `data/processed/centroids.csv`
- `build_global_network.py`: builds `data/global/nodes.csv` + `data/global/edges.csv`
- `sanity_plots_global.py`: plots + summary for the global output
- `validate_global_network.py`: integrity checks for nodes/edges
- `validate_geocodes.py`: checks region codes vs centroids
- `task44_io.py`, `task44_utils.py`: shared helpers

## Dependencies

See `code/requirements.txt`.

---

Last updated: 2026-02-09
