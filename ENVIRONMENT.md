This repository includes a Conda environment for development on Windows.

Quick steps (PowerShell):

```powershell
# Create the conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate sdcard

# Verify Python and installed packages
python --version
python -c "import typer, pandas, psutil, yaml; print('imports ok')"
```

Notes:
- This environment targets Python 3.11 per `pyproject.toml`.
- If you don't have conda installed, install Miniconda or Anaconda first: https://docs.conda.io/en/latest/miniconda.html
- If you prefer to use Poetry for dependency management instead of the pip list above, install `poetry` in the environment and run `poetry install` from the repo root.
- In VS Code, choose the `sdcard` interpreter (or let the Python extension auto-activate the environment) to run and debug using this env.

Windows-only packages `wmi` and `pywinusb` are included because this project contains Windows-specific code paths.
