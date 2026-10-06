# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Industrial defect detection with YOLO (Ultralytics): training pipeline, FastAPI service and Docker deployment. The repo is currently a scaffold: the subpackages under `src/defect_detection/` are empty `__init__.py` files, and `configs/`, `scripts/`, `notebooks/` and `.github/workflows/` are empty directories. The only test is an import smoke test.

## Commands

The project is managed with `uv` (Python 3.11, pinned in `.python-version`; `requires-python` is `>=3.11,<3.13`).

```bash
uv sync                      # install runtime deps + dev group (pytest, ruff)
uv sync --extra api          # also install FastAPI / uvicorn / python-multipart

uv run pytest                                              # all tests
uv run pytest tests/test_smoke.py::test_package_is_importable   # single test

uv run ruff check .          # lint (rules: E, F, I, B, UP; line length 100)
uv run ruff check . --fix
uv run ruff format .
```

## Layout and intent

`src/` layout, built with hatchling; the importable package is `defect_detection`, split by pipeline stage:

- `data/`: dataset preparation
- `training/`: YOLO training
- `inference/`: prediction
- `api/`: FastAPI service; its dependencies live in the optional `api` extra, so code outside `api/` must not import FastAPI or uvicorn

Supporting directories: `configs/` (YAML, loaded with `pyyaml`), `scripts/` (entry points), `notebooks/`, `data/raw` and `data/processed`, `models/`.

## Things to know

- **PyTorch is CPU-only locally.** `pyproject.toml` pins `torch` and `torchvision` to the `pytorch-cpu` index through `[tool.uv.sources]`. Do not remove this or add a CUDA build without being asked.
- **Data and model weights are not in git.** `.gitignore` excludes everything under `data/` and `models/` except `.gitkeep`, plus `*.pt` and `*.onnx`; the plan is to version them with DVC. Ultralytics `runs/` and `mlruns/` are also ignored.
- Existing comments in `pyproject.toml` and `.gitignore` are written in Spanish.

## Cómo trabajar conmigo
- Responde siempre en español, directo y conciso.
- Antes de escribir o editar código: explica qué vas a hacer y POR QUÉ
  (decisión de ingeniería y alternativas descartadas), y espera mi confirmación.
- No avances de paso/fase sin que yo lo confirme.
- Rigor: no inventes métricas, resultados, versiones ni comandos. Si no lo sabes, dilo.
- Objetivo: repo de portfolio con nivel de producción (estructura limpia, tests,
  logging, README serio). Enfoque MLOps: peso en despliegue, Docker, CI/CD, versionado.
- Mi nivel: PyTorch medio, OpenCV sólido, Docker básico. Quiero entender, no copiar.
- Commits en Conventional Commits (feat:, fix:, chore:, docs:, test:).

## Contexto
- torch se fija al índice CPU porque en local solo se desarrolla/testea/sirve;
  el entrenamiento se hace en Google Colab (GPU) con `pip install -e .`, que
  ignora tool.uv.sources y usa el torch+CUDA de Colab.
- Dataset: MVTec AD (licencia CC BY-NC-SA 4.0). Nunca se commitea; vive en data/ (gitignored).
