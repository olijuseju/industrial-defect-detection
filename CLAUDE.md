# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Industrial defect detection with YOLO (Ultralytics): training pipeline, FastAPI service and Docker deployment. Only the dataset download step is implemented (`src/defect_detection/data/download.py`, CLI in `scripts/download_data.py`). The `training`, `inference` and `api` subpackages are empty `__init__.py` files, and `notebooks/` and `.github/workflows/` are empty directories. See the roadmap at the end for the current step.

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

# MVTec AD into data/raw (idempotent). Source: --archive, --url or the MVTEC_AD_URL env var
uv run python scripts/download_data.py --archive /path/to/mvtec_anomaly_detection.tar.xz
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

## Roadmap y decisiones tomadas
Pasos (no avanzar sin mi confirmación):
1. Setup ✅ (uv, src layout, WSL2, torch CPU local)
2. Descarga dataset MVTec AD ✅ (15 categorías en data/raw, SHA256 en configs/dataset_checksums.json)
3. Preparación de datos: máscaras → etiquetas YOLO + splits ← SIGUIENTE
4. Entrenamiento baseline YOLO (en Colab) + métricas (precision, recall, mAP)
5. Inferencia + API FastAPI (imagen → JSON + imagen anotada)
6. Docker + tests + logging estructurado + CI (GitHub Actions)
7. Versionado (DVC/MLflow) + README con resultados reales
8. (Fase 2 del portfolio, otro momento) anomaly detection con anomalib

Hechos del dataset que condicionan el diseño:
- MVTec AD es de anomaly detection: el train solo tiene piezas buenas; los defectos
  están solo en test, con máscaras de segmentación a nivel de píxel, sin bboxes.
- Por tanto hay que convertir máscaras → etiquetas YOLO (bboxes por componentes conexas)
  y re-particionar las imágenes defectuosas en train/val/test propios.
- Pocas imágenes defectuosas por categoría: documentarlo como limitación en el README.
- Categoría baseline: hazelnut (defectos localizados: crack, cut, hole, print).
  Evitar defectos globales sin localización (p. ej. metal_nut/flip).
- Detección (bbox) primero; segmentación (YOLO-seg) se valora después.

Diseño acordado del paso 2 (descarga):
- Lógica en src/defect_detection/data/download.py; scripts/download_data.py solo es CLI.
- Dos vías: --url (descarga en streaming) o --archive (fichero ya bajado a mano).
- La URL de MVTec no se hardcodea ni se commitea (va por argumento o variable de entorno).
- SHA256: se calcula en la primera descarga y se guarda en configs/; después se verifica.
  No inventar checksums oficiales.
- Extracción segura con tarfile filter="data" (evita path traversal).
- Idempotente: si la categoría ya está extraída, no hace nada.
- Solo librería estándar + logging.
- Tests sin red, con un .tar diminuto generado en el propio test.
