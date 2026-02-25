# Terrestrial Protected Area Effectiveness
This repository contains preliminary code related to the development of a Terrestrial Protected Area Effectiveness (TPAE) model. TPAE is part of SkyTruth's 30x30 progress tracking initiative.

## Repository Contents
- **src/absolute_effectiveness_functions.py:** Functions for evaluating the absolute effectiveness of a PA (measuring only within the boundaries of the PA, with no comparison to an unprotected spatial control).
- **notebooks/absolute_effectiveness.ipynb:** Notebook for testing absolute effectiveness code individually on various test sites and visualizing results.
- **src/utils/variables.py:** Constants related to absolute effectiveness code.

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