# Terrestrial Protected Area Relative Assessment

This repository contains preliminary code related to the development of a Terrestrial Protected Area Relative Assessent (TPARA) model. TPARA is a tool for evaluating the impact of a protected area.

## Repository Contents

- **notebooks/**: Notebooks for running analysis on a single site and visualizing results.
  - `run_absolute_effectiveness.ipynb`: Absolute habitat condition and loss for one site, calculated by the pipeline's own code, with maps.
  - `MDM_match_cells.ipynb`: Mahalanobis Distance Matching (MDM) and covariate-balance diagnostics for one site.
  - `run_relative_effectiveness.ipynb`: Per-cell scores and site-level relative effectiveness for one site.
  - `archive/`: Retired PSM notebooks.
- **src/sites/**: Select and clean analysis sites.
  - `clean_sites.py`: Select the active site group from WDPA (Terrestrial and Coastal PAs that are not Proposed and whose `GIS_AREA` is large enough for `MIN_TREATMENT_CELLS` cells), clean geometries, and save to GCS.
  - `site_selector.py`: Load the cleaned sites, reduce Earth Engine images over a site, split large sites into pieces for intactness, and build rectangles for selecting input images.
  - `dropped_sites.py`: Record and read sites dropped for too few treatment cells.
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

Site lists, cell files, covariates, and match outputs are stored in GCS under prefixes tagged with `SITE_GROUP` (set in `src/utils/variables.py`). Change that one value to switch groups without overwriting another group's files.

## Script Order

0. `src/covariates/export_covariate_stack.py`: Write the global covariate stack to the `covariates_1km` Earth Engine asset (once for all site groups; re-run only after changing a covariate).
1. `src/sites/clean_sites.py`: Write `site_geoms/site_geoms_{SITE_GROUP}.parquet`, with a summary of the run in `site_geoms/site_geoms_{SITE_GROUP}.json`.
2. `src/cells/get_treatment_cells.py`: Write `cells/treatment_cells_{SITE_GROUP}.parquet`. Sites with too few land cells are recorded in `site_geoms/dropped_sites_{SITE_GROUP}.csv`.
3. `src/cells/get_control_cells.py`: Write `cells/control_cells_{SITE_GROUP}.parquet`.
4. `src/covariates/extract_cell_covariates.py`: Write one covariate file per site under `cell_covariates/{SITE_GROUP}/`. Sites that are already up to date are skipped; use `--overwrite` to redo every site.
5. `src/matching/MDM_match_cells.py`: Match every site and, with `save_match_outputs = True`, write match tables and matched cells under `matches/{SITE_GROUP}/`. Sites with too few matched treatment cells are added to the dropped sites file. Use `notebooks/MDM_match_cells.ipynb` to inspect one site.
6. `src/relative_effectiveness/relative_effectiveness.py`: Score matched cells and compute relative effectiveness, written to `relative_results/relative_results_{SITE_GROUP}.csv`.
7. `src/absolute_effectiveness/absolute_effectiveness.py`: Calculate absolute effectiveness for every site in the active group, written to `absolute_results/absolute_results_{SITE_GROUP}.csv`.

Dropped sites are skipped by every later step. Steps 6 and 7 take `--retry-failed` to re-run only the sites that failed or are missing in the saved results. To time a step and log its Earth Engine compute, run it through `src/utils/timing.py` (e.g. `poetry run python src/utils/timing.py src/cells/get_control_cells.py`).

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