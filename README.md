# Terrestrial Protected Area Relative Assessment

This repository contains preliminary code related to the development of a Terrestrial Protected Area Relative Assessent (TPARA) model. TPARA is a tool for evaluating the impact of a protected area.

## Repository Contents

- **notebooks/**: Notebooks for running analysis on a single site and visualizing results.
  - `run_absolute_effectiveness.ipynb`: Absolute habitat condition and loss for one site.
  - `MDM_match_cells.ipynb`: Mahalanobis Distance Matching (MDM) and covariate-balance diagnostics for one site.
  - `run_relative_effectiveness.ipynb`: Per-cell scores and site-level relative effectiveness for one site.
  - `archive/`: Retired PSM notebooks.
- **src/sites/**: Select and clean analysis sites.
  - `clean_sites.py`: Extract the active `SITE_IDS` from WDPA/WDOECM, clean geometries, and save them to GCS.
  - `site_selector.py`: Load the selected sites and derive site-specific variables (start year, geometry).
- **src/cells/**: Build treatment and control cells.
  - `get_treatment_cells.py`: Sample 1 km interior cells for each site.
  - `get_control_cells.py`: Sample 1 km unprotected cells in a buffer around each site.
  - `load_all_cells.py`: Load one site's treatment and control cells and assign `cell_ID`s.
- **src/covariates/**: Extract matching covariates.
  - `covariates.py`: Build the 1 km resampled covariate stack.
  - `cell_features.py`: Aggregate covariates within cells and join country / ecoregion / biome.
  - `extract_cell_covariates.py`: Run extraction for every site in the active group and save one parquet file per site.
- **src/matching/**: Match treatment and control cells with MDM.
  - `match_cells.py`: Neighbor search, greedy/optimal assignment, and saving match outputs.
  - `diagnostics.py`: Covariate-balance diagnostics and the per-site report card.
  - `MDM_match_cells.py`: Run MDM on every site in the active group.
- **src/absolute_effectiveness/**: Site-level habitat condition and loss.
  - `data_processor.py`: Load and process Earth Engine datasets (GLC, GPW, ForTy, HGFC).
  - `habitat_condition.py`: Habitat extent, intactness, and overall condition.
  - `habitat_loss.py`: Habitat loss score, plus drivers and types of loss.
  - `visualization.py`: Cloud-masked Sentinel-2 composites for maps.
- **src/relative_effectiveness/**: Cell-level scores and site-level relative effectiveness.
  - `metrics_per_cell.py`: Reuses the absolute-effectiveness analyzers, scoring a FeatureCollection of cells.
  - `relative_scores.py`: Pairwise treatment–control differences and site-level aggregates.
- **src/utils/variables.py**: Shared constants, the active `SITE_GROUP`, and GCS paths.
- **src/archive/**: Retired scripts.
- **src/disturbance/**: DIST-ALERT helpers (separate from the main pipeline).

Site lists, cell files, covariates, and match outputs are stored in GCS under prefixes tagged with `SITE_GROUP` (set in `src/utils/variables.py`). Change that one value to switch groups without overwriting another group's files.

## Script Order

Relative effectiveness, for the active `SITE_GROUP`:

1. `src/sites/clean_sites.py`: Write `sites/sites_{SITE_GROUP}.parquet`.
2. `src/cells/get_treatment_cells.py`: Write `cells/treatment_cells_{SITE_GROUP}.parquet`.
3. `src/cells/get_control_cells.py`: Write `cells/control_cells_{SITE_GROUP}.parquet`.
4. `src/covariates/extract_cell_covariates.py`: Write one covariate file per site under `cell_covariates/{SITE_GROUP}/`.
5. `src/matching/MDM_match_cells.py`: Match every site and, with `save_match_outputs = True`, write match tables and matched cells under `matches/{SITE_GROUP}/`. Use `notebooks/MDM_match_cells.ipynb` to inspect one site.
6. `notebooks/run_relative_effectiveness.ipynb`: Score matched cells and compute relative effectiveness for one site.

Absolute effectiveness does not depend on matching. Run `notebooks/run_absolute_effectiveness.ipynb` for any `site_id` in `SITE_IDS`.

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