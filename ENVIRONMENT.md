# Development environment (UV)

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Python 3.11+ available (on HPC: `module load python/3.11.0`)

On systems with a tight home quota (space or inode), keep UV caches and the venv on scratch:

```bash
export UV_CACHE_DIR=/scratch3/$USER/uv/cache
export UV_PYTHON_INSTALL_DIR=/scratch3/$USER/uv/python
export UV_PROJECT_ENVIRONMENT=/scratch3/$USER/uv/venvs/sdcard
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"
```

## Setup

```bash
cd /path/to/sdcard
module load python/3.11.0   # HPC only, if needed
uv sync --group dev
# Optional: make IDEs find the env via .venv
ln -sfn "$UV_PROJECT_ENVIRONMENT" .venv
```

## Verify

```bash
uv run python --version
uv run python -c "import typer, pandas, psutil, yaml; print('imports ok')"
uv run sdcard --help
uv run pytest
```

## Notes

- Runtime deps and the `sdcard` console script are defined in `pyproject.toml`.
- Dev tools (pytest) live in the `dev` dependency group.
- Windows-only packages `wmi` and `pywinusb` install automatically on Windows via environment markers.
- A Conda `environment.yml` remains for Windows/Conda users who prefer that path.
