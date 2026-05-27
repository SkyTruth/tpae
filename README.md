# Terrestrial Protected Area Effectiveness
This repository contains preliminary code related to the development of a Terrestrial Protected Area Effectiveness (TPAE) model. TPAE is part of SkyTruth's 30x30 progress tracking initiative.

## Repository Contents
- **data/**: Input and output datasets.
- **models/**: Saved propensity score models.
- **notebooks/**: Notebooks currently used for developing and testing methods on individual PAs; will later be converted into scripts.
- **src/absolute_effectiveness/**: Functions for evaluating absolute effectiveness.
  - **`SiteSelector`**: Retrieves selected test sites and derives site-specific variables.
  - **`DataProcessor`**: Loads and processes Earth Engine datasets (GLC, GPW, NFW, HGFC) for the selected sites and analysis period.
  - **`HabitatConditionAnalyzer`**: Calculates habitat extent, intactness, and overall habitat condition score.
  - **`HabitatLossAnalyzer`**: Calculates habitat loss score and summarizes the drivers and types of habitat loss.
  - **`VisualizationService`**: Builds cloud-masked Sentinel-2 composites to aid map visualization.
- **src/psm/**: Functions to aid in propensity score matching.
  - **`tiling.py`**: Divides the globe into tiles for sampling efficiency.
  - **`allocation.py`**: Stratified sample allocation.
  - **`sampling.py`**: Samples covariates by tile based on the stratified sample allocation.
  - **`get_treatment_cells.py`**: Generate a set of candidate treatment (interior, protected) cells for each PA.
  - **`get_control_cells.py`**: Generate a set of candidate control (nearby unprotected) cells for each PA.
  - **`predict.py`**: Predicts propensity scores using the saved propensity model.
- **src/relative_effectiveness/**: Functions for evaluating relative effectiveness.
  - **`metrics_per_cell.py`**: Reuses absolute effectiveness code, but computes scores across a series of cells rather than within a single PA.
- **src/utils/variables.py:** Constant variables.

## Script Order

Scripts should be run in this order:
1. **`notebooks/run_absolute_effectiveness.ipynb`**: Generate habitat condition and loss metrics within a given PA.
2. **`notebooks/global_psm.ipynb`**: Fit a global propensity score model that predicts the likelihood of a location of being protected given a set of covariates. Model is saved to **`models/propensity_model_{timestamp}.pkl`**
3. **`src/psm/get_treatment_cells.py`** and **`src/psm/get_control_cells.py`**: Generate a set of candidate treatment (protected) and control (nearby unprotected) 1km2 cells for each PA. Outputs are saved to **`data/treatment_cells.parquet`** and **`data/control_cells.parquet`**.
4. **`notebooks/match_cells.ipynb`**: Use the global propensity model to predict a propensity score for each of the candidate treatment and control cells, and match each treatment cell to a set of control cells with similar propensity scores.
5. **`notebooks/run_relative_effectiveness.ipynb`**: Generate habitat condition and loss metrics within each cell, and calculate relative effectiveness metrics by comparing scores for matched treatment and control cells.

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

3. Set execution policy (this makes your computer slightly less secure, but if you aren't downloading potential malware via Powershell, you're alright)
  ```shell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```
4. create user profile: open $PROFILE in notepad:
  ```shell
  notepad $PROFILE
  ```
5. Add the following to that file (replace YourUsername with the correct identifier):
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
  ```shell
  poetry add <new-package>
  ```
Deleting a library
  ```shell
  poetry remove <old-package>
  ```

# License
This work is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). See LICENSE.txt