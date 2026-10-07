# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Industrial defect detection with YOLO (Ultralytics): training pipeline, FastAPI service and Docker deployment. Implemented so far: dataset download and data preparation, both under `src/defect_detection/data/` with thin CLIs in `scripts/`. The `training`, `inference` and `api` subpackages are empty `__init__.py` files, and `notebooks/` and `.github/workflows/` are empty directories. See the roadmap at the end for the current step.

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

# YOLO dataset into data/processed/<category> (rebuilt from scratch on every run)
uv run python scripts/prepare_data.py                       # uses configs/data_hazelnut.yaml
uv run python scripts/preview_labels.py --merge-gap-px 31 --min-area-px 300   # contact sheets of the boxes
```

## Layout and intent

`src/` layout, built with hatchling; the importable package is `defect_detection`, split by pipeline stage:

- `data/`: dataset download and preparation. `prepare.py` is the entry point and composes the rest: `mvtec.py` (on-disk layout of a category), `masks.py` (mask → boxes), `split.py` (test + k folds), `config.py` (`DataConfig` read from `configs/data_<category>.yaml`)
- `training/`: YOLO training
- `inference/`: prediction
- `api/`: FastAPI service; its dependencies live in the optional `api` extra, so code outside `api/` must not import FastAPI or uvicorn

Supporting directories: `configs/` (YAML, loaded with `pyyaml`), `scripts/` (entry points), `notebooks/`, `data/raw` and `data/processed`, `models/`.

## Things to know

- **PyTorch is CPU-only locally.** `pyproject.toml` pins `torch` and `torchvision` to the `pytorch-cpu` index through `[tool.uv.sources]`. Do not remove this or add a CUDA build without being asked.
- **Data and model weights are not in git.** `.gitignore` excludes everything under `data/` and `models/` except `.gitkeep`, plus `*.pt` and `*.onnx`; the plan is to version them with DVC. Ultralytics `runs/` and `mlruns/` are also ignored.
- Existing comments in `pyproject.toml` and `.gitignore` are written in Spanish. Code follows the same rule: docstrings and comments in Spanish, identifiers, log messages and exception messages in English.
- **`configs/splits/<category>.json` is the source of truth for the split** and is committed. `load_or_create_split` reuses it and fails if the config no longer matches; changing which images are in test requires `scripts/prepare_data.py --force-split`, which invalidates any metric already reported.
- **Processed dataset layout.** `data/processed/<category>/` has flat `images/` and `labels/` plus `test.txt`, `foldN_train.txt`, `foldN_val.txt` and `foldN.yaml` in its root. The lists use `./images/...` paths and the YAMLs set no `path`, so Ultralytics resolves everything relative to the files themselves and the directory can be moved as is. Do not move the lists into subfolders: Ultralytics resolves `./` against the folder of the `.txt`.

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
3. Preparación de datos: máscaras → etiquetas YOLO + splits ✅ (hazelnut: 140 imágenes, 77 cajas, test + 4 folds)
4. Entrenamiento baseline YOLO (en Colab) + métricas (precision, recall, mAP) ← SIGUIENTE
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

Diseño acordado del paso 3 (preparación de datos):
- hazelnut tiene 70 imágenes defectuosas (crack 18, cut 17, hole 18, print 17): las métricas
  serán ruidosas. Documentarlo como limitación en el README.
- Split: test fijo estratificado (20%) + 4 folds, semilla 42, guardado en
  configs/splits/hazelnut.json. El baseline valida con un fold; el k-fold completo queda
  como experimento posterior sin rehacer datos.
- Negativos: las buenas son un estrato más, muestreadas 1:1 con las defectuosas de entre
  train/good y test/good, y repartidas en test y folds igual que los defectos. Así val y
  test permiten medir falsas alarmas sobre piezas buenas.
- Máscara → bbox: merge_gap_px=31 y min_area_px=300, elegidos mirando las hojas de
  preview. 31 px une las letras de un mismo sello en `print` (65 cajas → 18); 300 px quita
  dos restos sueltos (189 y 192 px) sin tocar el defecto real más pequeño (405 px).
- Una imagen defectuosa que se quede sin cajas es un error, no un negativo.
- Clases: las etiquetas llevan siempre las 4 clases. El baseline de una sola clase `defect`
  se entrena con single_cls=True de Ultralytics; 4 clases queda como experimento comparado.
  single_cls aún no está probado: se valida al entrenar en el paso 4.
- Pendiente para el paso 4: resolución de entrada (imgsz; las imágenes son 1024×1024) y
  cómo llega el dataset a Colab.
