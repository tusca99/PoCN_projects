# Task 44: Social Connectedness Index II (Facebook)

**Score**: 1.0  
**Type**: Data Analytics  

---

## Objective

Build the **global deliverable** : exactly **two CSV files** (`nodes.csv` + `edges.csv`) representing within-country SCI networks (USA excluded by default).

## Data Sources

- **Primary**: [Social Connectedness Index (HumData)](https://data.humdata.org/dataset/social-connectedness-index)
- **Alternative**: [Meta AI SCI Official](https://ai.meta.com/ai-for-good/datasets/social-connectedness-index/#accessdata)
- **GADM Shapefiles**: [gadm.org](https://gadm.org/download_world.html)
- **NUTS Reference**: [Wikipedia](https://en.wikipedia.org/wiki/Nomenclature_of_Territorial_Units_for_Statistics)

## Dataset Format

**Input (SCI CSVs as downloaded from HDX/Meta)**:
```
user_country,friend_country,user_region,friend_region,scaled_sci
PH,PH,PHL.10_1,PHL.11_1,7258
PH,KR,PHL.10_1,KOR.15_1,885
```

In this workspace the main global regional file you downloaded is:
- `projects/task_44/data/gadm1/gadm1.csv` (GADM level 1 regions)

EU-only higher resolution is available via:
- `projects/task_44/data/nuts_2024/nuts3_2024.csv`

**GADM Code Format**: `COUNTRY.NUTS1.NUTS2_NUTS3`
- Example: `ITA.1.2_1` = Italy, NUTS1=1, NUTS2=2, NUTS3=1

## Output Format

The canonical output is under `projects/task_44/data/global/`.

### Node File: `nodes.csv`
```
nodeID,nodeLabel,latitude,longitude
1,Milano,45.4642,9.1900
2,Roma,41.9028,12.4964
...
```

### Edge File: `edges.csv`
```
nodeID_from,nodeID_to,country_name,country_ISO3
1,2,Italy,ITA
1,15,Italy,ITA
...
```

Optional (for analysis/debug): `edges_weighted.csv` with an extra `scaled_sci` column.

## Workflow

## Code map (what does what)

All scripts live in `projects/task_44/code/`.

Recommended entrypoint (for reviewers):

- `task44_cli.py`: **single CLI** that runs the whole pipeline (data check / centroids / build global or per-country / export / plots / validate).

- `download_data.py`: prints instructions / checks expected input files under `projects/task_44/data/`.
- `build_centroids.py`: builds `data/processed/centroids.csv` (region_code -> lat/lon + optional label) from polygons.
- `build_global_network.py`: builds the **Moodle deliverable** (single `data/global/nodes.csv` + `data/global/edges.csv`).
- `validate_global_network.py`: quick integrity checks for the global CSVs.
- `sanity_plots_global.py`: produces summary tables + PDF/PNG sanity plots under `data/global/plots/`.
- `validate_geocodes.py`: checks that SCI region codes (GADM/NUTS/US counties) match the centroid sources (useful to detect version mismatches).

Per-country outputs (legacy layout kept on purpose):

- `build_country_networks.py`: builds per-country networks from the layer CSVs using the same *Option C* resolution policy
   (US counties / EU NUTS3 / else GADM1) and writes them under a dated archive folder:
   `data/_archive_YYYY-MM-DD/{nodes,edges}`.
- `export_country_networks.py`: splits an **already built** global `nodes.csv/edges.csv` into per-country
   `data/_archive_YYYY-MM-DD/{nodes,edges}` (useful when you only want the country folders).

Maintenance:

- `cleanup_data.py`: moves non-essential artifacts under `data/_archive_YYYY-MM-DD/` (conservative, no deletes).
- `task44_io.py`: small shared I/O helpers to keep CSV schemas consistent.

For the submission handout you can ignore any non-mentioned files/folders: the pipeline above is self-contained.

### Step 1: Data (already downloaded)

In this workspace the SCI release is already present under `projects/task_44/data/` as *layer CSVs*.

Examples:
- `projects/task_44/data/gadm1/gadm1.csv` (global, subnational level 1)
- `projects/task_44/data/nuts_2024/nuts3_2024.csv` (EU NUTS3)
- `projects/task_44/data/country/country.csv` (country-country)

### Step 2: Country selection + coordinates (optional)

You already have:
- `projects/task_44/data/processed/top_countries.csv` (top 100 excluding USA)
- `projects/task_44/data/processed/country_coverage.csv`

The layer CSVs do not include region coordinates. To fill `latitude/longitude` you need a centroid source (polygons -> representative points) and pass it to the builder via `--centroids`.

You can generate a reusable centroids table (GADM1 + NUTS3 + US counties) with:
```bash
./.venv/bin/python projects/task_44/code/task44_cli.py centroids \
   --country-file projects/task_44/data/processed/country_coverage.csv \
   --min-coverage 1.0
```
This writes:
- `projects/task_44/data/processed/centroids.csv`

Quick compatibility check (recommended if you're unsure about boundary dataset versions):
```bash
./.venv/bin/python projects/task_44/code/task44_cli.py geo \
   --country-file projects/task_44/data/processed/country_coverage.csv \
   --all-countries \
   --min-coverage 1.0
```
This writes a short report under:
- `projects/task_44/data/processed/geocodes_check/summary.csv`

### Step 3: Build Global CSVs

This produces the **single pair** required for submission:
```bash
./.venv/bin/python projects/task_44/code/task44_cli.py build --global yes \
   --country-file projects/task_44/data/processed/country_coverage.csv \
   --all-countries \
   --min-coverage 1.0 \
   --write-weighted \
   --centroids projects/task_44/data/processed/centroids.csv
```

Outputs:
- `projects/task_44/data/global/nodes.csv`
- `projects/task_44/data/global/edges.csv`
- `projects/task_44/data/global/edges_weighted.csv` (optional)

Quick validation:
```bash
./.venv/bin/python projects/task_44/code/task44_cli.py validate
```

### Step 3b (optional): Build per-country folders (best resolution)

This regenerates the two per-country folders (one `nodes_<ISO3>.csv` and one `edges_<ISO3>.csv` per country) under:
`projects/task_44/data/_archive_YYYY-MM-DD/`.

```bash
./.venv/bin/python projects/task_44/code/task44_cli.py build --global no \
   --country-file projects/task_44/data/processed/country_coverage.csv \
   --all-countries \
   --min-coverage 1.0 \
   --centroids projects/task_44/data/processed/centroids.csv
```

If you want weighted edges too:

```bash
./.venv/bin/python projects/task_44/code/task44_cli.py build --global no \
   --country-file projects/task_44/data/processed/country_coverage.csv \
   --all-countries \
   --min-coverage 1.0 \
   --centroids projects/task_44/data/processed/centroids.csv \
   --write-weighted
```

### Step 3c (optional): Export per-country folders from the global deliverable

If you already have `data/global/nodes.csv` + `data/global/edges.csv` and only want the per-country folders:

```bash
./.venv/bin/python projects/task_44/code/task44_cli.py export
```

Example: export only Italy into a custom folder:

```bash
./.venv/bin/python projects/task_44/code/task44_cli.py export \
   --include-iso3 ITA \
   --out-root projects/task_44/data/_archive_2026-02-04
```

### Step 4: Sanity-check plots (fast)
```bash
./.venv/bin/python projects/task_44/code/task44_cli.py plots
```

This writes:
- `projects/task_44/data/global/summary_by_country.csv`
- `projects/task_44/data/global/plots/` (PDF/PNG)

By default self-loops (region -> same region) are dropped. They tend to make densities/clustering misleading (and can even yield density > 1 with the usual simple-graph formula). To keep them, add `--keep-self-loops`.

**Tasks**:
- For each of top 100 countries:
  - Filter SCI data for that country
  - Create node list (GADM NUTS3 areas)
  - Create edge list (SCI connections)
  - Remap to sequential nodeID (1-indexed)

**Outputs**:
- `projects/task_44/data/nodes/nodes_<ISO3>.csv` (one per country)
- `projects/task_44/data/edges/edges_<ISO3>.csv` (one per country)

Optional:
- `projects/task_44/data/edges/weighted/edges_<ISO3>_weighted.csv` (includes aggregated `scaled_sci`)

Note: the SCI CSVs do not include region coordinates. To fill `latitude/longitude` you need a centroid source (e.g. GADM/geoBoundaries polygons -> representative points) and pass it via `--centroids`.

## Notes on code organization

The directory `projects/task_44/code/` is kept minimal for the Moodle deliverable. Older/experimental scripts (per-country builders, heavier analysis) were moved to:
- `projects/task_44/code/deprecated/`

If you want a quick check of whether your manually downloaded layer CSVs are in the right place (no auto-download), run:
```bash
./.venv/bin/python projects/task_44/code/download_data.py
```

## Key Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| N | Number of nodes | - |
| E | Number of edges | - |
| ⟨k⟩ | Average degree | $\frac{2E}{N}$ |
| C | Clustering coefficient | Average local clustering |
| Q | Modularity | Newman's Q |
| ρ | Density | $\frac{E}{N(N-1)/2}$ |

## Expected Results

**Comparative Table (top 10-20 countries)**:

| Country | ISO3 | N | E | ⟨k⟩ | C | Q |
|---------|------|---|---|-----|---|---|
| Italy | ITA | ... | ... | ... | ... | ... |
| France | FRA | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

## Challenges & Solutions

1. **GADM parsing complexity**
   - Solution: Use regex, robust error handling
   
2. **Large dataset (GB)**
   - Solution: Process in chunks, filter early
   
3. **Missing coordinates**
   - Solution: Use GADM shapefiles or geocoding API
   
4. **Sparse NUTS3 in some countries**
   - Solution: Document coverage, include disclaimer

## Dependencies

See `code/requirements.txt`:
```
pandas>=1.3.0
numpy>=1.21.0
geopandas>=0.10.0
networkx>=2.6.0
matplotlib>=3.4.0
scipy>=1.7.0
```

## Timeline

- **Days 1-2**: Download, exploration
- **Days 3-4**: Parsing, network construction
- **Day 5**: Analysis
- **Day 6**: Visualizations, report

**Total**: ~6 days (~30-35 hours)

---

**Status**: In progress  
**Last Updated**: 2026-02-04
