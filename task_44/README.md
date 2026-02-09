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
- `projects/task_44/data/gadm1/gadm1.csv` (GADM level 1 regions)
- `projects/task_44/data/nuts_2024/nuts3_2024.csv` (EU NUTS3)
- `projects/task_44/data/all_region_to_country/*.csv` (fast country list source)

## Outputs

Canonical outputs live under `projects/task_44/data/global/`:

- `nodes.csv` with `nodeID,nodeLabel,latitude,longitude`
- `edges.csv` with `nodeID_from,nodeID_to,country_name,country_ISO3`
- `edges_weighted.csv` (optional) adds `scaled_sci`

Plots and summary (optional) are written to:
- `projects/task_44/data/global/plots/`
- `projects/task_44/data/global/summary_by_country.csv`

## Quickstart

Run from `projects/task_44`:

```bash
python code/task44.py build
python code/task44.py plot
python code/task44.py validate
```

Useful flags:
- `--rebuild-country-list` and `--rebuild-centroids`
- `--skip-centroids` (nodes will have empty lat/lon)
- `--all-countries` (ignore top-k selection)
- `--no-weighted` (skip `edges_weighted.csv`)
- `--centroid-workers N` (parallel GADM downloads)

Intermediate files:
- `projects/task_44/data/processed/country_list.csv`
- `projects/task_44/data/processed/centroids.csv`

## Code map

All scripts live in `projects/task_44/code/`.

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
