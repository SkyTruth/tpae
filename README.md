# Terrestrial Protected Area Relative Assessment

This repository contains preliminary code related to the development of a Terrestrial Protected Area Relative Assessent (TPARA) model. TPARA is a tool for evaluating the impact of a protected area.

## Repository Contents

- **notebooks/**: Notebooks for running analysis on a single site and visualizing results.
  - `run_absolute_effectiveness.ipynb`: Absolute habitat condition and loss for one site, calculated by the pipeline's own code, with maps.
  - `MDM_match_cells.ipynb`: Mahalanobis Distance Matching (MDM) and covariate-balance diagnostics for one site.
  - `run_relative_effectiveness.ipynb`: Per-cell scores and site-level relative effectiveness for one site.
  - `archive/`: Retired PSM notebooks.
- **src/sites/**: Select and clean analysis sites.
  - `chunker.py`: Select every eligible site from WDPA (Terrestrial and Coastal PAs that are not Proposed, with a designation year up to `ANALYSIS_END_YR - 1`, and whose `GIS_AREA` is at least `MIN_GIS_AREA_KM2`), sort them by country, split them into chunks of up to `CHUNK_SIZE` sites, and save each site's chunk to GCS.
  - `clean_sites.py`: Select the active site group (a chunk, or a group in `SITE_GROUPS`) from the eligible sites, clean geometries, and save to GCS.
  - `unchunker.py`: Check every chunk's outputs and merge them into global tables of the kept sites, with a list of every dropped site and why.
  - `site_selector.py`: Load the cleaned sites, reduce Earth Engine images over a site, split large sites into pieces for intactness, and build rectangles for selecting input images.
  - `dropped_sites.py`: Record and read the sites each step drops, with the step and why (a rule such as too few treatment cells, or an error).
- **src/cells/**: Build treatment and control cells.
  - `get_treatment_cells.py`: Sample 1 km land cells fully inside each site, on the covariate stack's grid, and drop sites with fewer than `MIN_TREATMENT_CELLS`.
  - `get_control_cells.py`: Sample 1 km land cells 10–200 km from each site, away from other PAs and OECMs.
  - `load_all_cells.py`: Load treatment and control cells and assign `cell_ID`s.
- **src/covariates/**: Extract matching covariates.
  - `export_covariate_stack.py`: Build the global 1 km covariate stack and save it as an Earth Engine asset, via GeoTIFFs in GCS (run once).
  - `cell_features.py`: Read each cell's covariates as one pixel of the covariate stack, with country / ecoregion / biome painted onto the same grid.
  - `extract_cell_covariates.py`: Run extraction for every site in the active group, in tiles of the stack's grid, and save one parquet file per site.
- **src/matching/**: Match treatment and control cells with MDM.
  - `match_cells.py`: Neighbor search, greedy/optimal assignment, and saving match outputs.
  - `diagnostics.py`: Covariate-balance diagnostics and the per-site report card.
  - `MDM_match_cells.py`: Run MDM on every site in the active group.
- **src/absolute_effectiveness/**: Site-level habitat condition and loss.
  - `absolute_effectiveness.py`: Run the analysis for every site in the active group and save one results row per site.
  - `data_processor.py`: Load and process Earth Engine datasets (GLC, GPW, ForTy, HGFC).
  - `habitat_condition.py`: Habitat extent, intactness, and overall condition.
  - `habitat_loss.py`: Habitat loss, plus drivers and types of loss.
  - `visualization.py`: Cloud-masked Sentinel-2 composites for maps.
- **src/relative_effectiveness/**: Cell-level scores and site-level relative effectiveness.
  - `relative_effectiveness.py`: Run the analysis for every site in the active group and save one results row per site.
  - `metrics_per_cell.py`: Reuses the absolute-effectiveness analyzers, scoring a FeatureCollection of cells.
  - `relative_scores.py`: Pairwise treatment–control differences and site-level aggregates.
- **src/utils/**: Shared settings and helpers.
  - `variables.py`: Shared constants, the active `SITE_GROUP`, and GCS paths.
  - `retries.py`: Retry transient Earth Engine errors and report sites that still failed.
  - `timing.py`: Time a pipeline script and log its run time and Earth Engine compute to GCS.
- **src/archive/**: Retired scripts.
- **src/disturbance/**: DIST-ALERT helpers (separate from the main pipeline).

Site lists, cell files, covariates, and match outputs are stored in GCS under prefixes tagged with `SITE_GROUP` (set in `src/utils/variables.py`): `chunk_<n>` for chunk n of the eligible sites, or a group in `SITE_GROUPS`. Change that one value to switch groups without overwriting another group's files.

## Script Order

0. `src/covariates/export_covariate_stack.py`: Write the global covariate stack to the `covariates_1km` Earth Engine asset (once for all site groups; re-run only after changing a covariate).
1. `src/sites/chunker.py`: Write `chunks/site_chunks.parquet` (each eligible site's chunk) and `chunks/chunks.json` (`last_chunk_index`; chunks are numbered from 0). Run once, before any group is cleaned: every group is chosen from this table, and re-running after a WDPA update moves chunk boundaries.
2. `src/sites/clean_sites.py`: Write `site_geoms/site_geoms_{SITE_GROUP}.parquet`, with a summary of the run in `site_geoms/site_geoms_{SITE_GROUP}.json`. Sites no longer in WDPA, or with no polygon parts, are dropped.
3. `src/cells/get_treatment_cells.py`: Write `cells/treatment_cells_{SITE_GROUP}.parquet`. Sites with too few land cells are dropped.
4. `src/cells/get_control_cells.py`: Write `cells/control_cells_{SITE_GROUP}.parquet`. Sites with no control cells are dropped.
5. `src/covariates/extract_cell_covariates.py`: Write one covariate file per site under `cell_covariates/{SITE_GROUP}/`. Sites that are already up to date are skipped; use `--overwrite` to redo every site.
6. `src/matching/MDM_match_cells.py`: Match every site and, with `save_match_outputs = True`, write match tables and matched cells under `matches/{SITE_GROUP}/`. Sites with too few matched treatment cells are dropped. Use `notebooks/MDM_match_cells.ipynb` to inspect one site.
7. `src/relative_effectiveness/relative_effectiveness.py`: Score matched cells and compute relative effectiveness, written to `relative_results/relative_results_{SITE_GROUP}.csv`. Each site's scored matched pairs (treatment and control scores, and their raw and regression-adjusted differences) are written to `matches/{SITE_GROUP}/scored_pairs_mdm_{site_id}.parquet`, for calculating uncertainty or re-aggregating later.
8. `src/absolute_effectiveness/absolute_effectiveness.py`: Calculate absolute effectiveness for every site in the active group, written to `absolute_results/absolute_results_{SITE_GROUP}.csv`.
9. `src/sites/unchunker.py`: Once steps 2–8 have run for every chunk, merge each chunk's relative and absolute results, report card, and match diagnostics into `global_results/`, with one row per kept site (cleaned, not dropped, and analyzed without errors). Also writes `dropped_sites.csv` (every dropped site, with the step and reason), `site_geoms.parquet` (every cleaned site), and `summary.json`. It first checks that every chunk is complete (every step has run, and none was interrupted), holds only its own sites, ran its steps in order, and was cleaned from the chunk table's WDPA release, and writes nothing if a check fails. Use `--allow-partial` to merge incomplete chunks while the run is in progress.

Steps 2–8 record the sites they drop in `site_geoms/dropped_sites_{SITE_GROUP}.csv`, with the step and reason. That includes sites that fail with an error after retries (reason `failed: <error>`), so one failed site never stops the others from being run and saved. Dropped sites are skipped by every later step, except that steps 7 and 8 are independent: neither skips or clears the other's dropped sites, so a site can be dropped at both, and they can run in either order (but not at once, since they share the file). Re-running a step replaces its records and clears those of later steps, which then need to re-run. To retry failed sites, re-run their step: step 5 picks up only the sites without up-to-date files, and steps 7 and 8 take `--retry-failed` to re-run only the sites that failed or are missing in the saved results. To re-run only some sites, pass `--sites <site_id> ...` to steps 4–8 in order: the other sites' saved outputs and dropped records are kept. To time a step and log its Earth Engine compute, run it through `src/utils/timing.py` (e.g. `poetry run python src/utils/timing.py src/cells/get_control_cells.py`).

## Working in this Repository

- This repository uses [ruff](https://docs.astral.sh/ruff/) [pre-commit hooks](https://pre-commit.com/).
- This repository uses [Poetry](https://python-poetry.org/) for package and dependency management (see below for installation and set-up).



# Poetry Installation and Set-up

Note: Python needs to be installed before poetry.

## Mac / Linux

1. [Install Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
2. After installing poetry, add the Poetry bin to PATH by adding the following line to .zshrc:

```shell
export PATH="$HOME/.local/bin:$PATH"
```



## Windows

1. Open PowerShell in Administrator mode (windows menu > Powershell > right click > Administrator)
2. Install Poetry:
  ```shell
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
  ```

- It is probably possible (and better) to add the PATH to the Poetry app to your environment somehow, but we could not figure out how to make that work so instead we set up an Alias so that we could run poetry commands

1. Set execution policy (this makes your computer slightly less secure, but if you aren't downloading potential malware via Powershell, you're alright)
  ```shell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```
2. create user profile: open $PROFILE in notepad:
  ```shell
  notepad $PROFILE
  ```
3. Add the following to that file (replace YourUsername with the correct identifier):
  ```
  Set-Alias poetry "C:\Users\<YourUsername>\AppData\Roaming\Python\Scripts\poetry.exe"
  ```



## Using Poetry



### Working in the poetry environment

This repo already has a `pyproject.toml` and a `poetry.lock` file. These define the virtual environment (sort of like how requirements.txt defines a pip environment, but these files resolve dependencies and ensure everyone has the same dependencies, since those are defined in the lockfile).

To work in the poetry virtual environment, you just need to prepend all command line statements with poetry run (i.e. instead of `python hello_world.py` you would simply run `poetry run python hello_world.py`)

Alternatively, you can work work entirely in the environment (similar to `conda activate`) with the command `poetry shell`, and then you can just run `python hello_world.py`

### Install environment

Do this each time you pull the repo in case there have been changes to the dependencies:

```shell
poetry install
```



### Updating the environment

Adding a new library:

Deleting a library

# License

This work is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). See LICENSE.txt