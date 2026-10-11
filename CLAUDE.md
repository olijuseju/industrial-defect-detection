# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Industrial defect detection with YOLO (Ultralytics): training pipeline, FastAPI service and Docker deployment. Implemented so far: dataset download and preparation (`data/`), config-driven training (`training/`), evaluation (`evaluation/`), inference (`inference/`) and the FastAPI service (`api/`), each with a thin CLI in `scripts/` except the API, which is started with uvicorn. `.github/workflows/` is an empty directory. See the roadmap at the end for the current step.

## Commands

The project is managed with `uv` (Python 3.11, pinned in `.python-version`; `requires-python` is `>=3.11,<3.14`).

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

# Processed dataset <-> reproducible zip for Colab (SHA256 kept in configs/dataset_checksums.json)
uv run python scripts/package_dataset.py pack                # writes data/processed/hazelnut.zip
uv run python scripts/package_dataset.py unpack <zip>        # verifies the checksum, then extracts

# Training. Real runs happen in Colab through notebooks/train_colab.ipynb; locally only as a smoke test
uv run python scripts/train.py --config configs/train_lr5e-4.yaml --epochs 1 --device cpu

# Evaluation on validation; writes reports/metrics_<name>.json. --test also evaluates the test split
uv run python scripts/evaluate.py --weights models/lr5e-4.pt --config configs/train_lr5e-4.yaml
# False alarms on the good images left out of the split; writes reports/false_alarms_<name>.json
uv run python scripts/false_alarms.py --weights models/lr5e-4.pt --config configs/train_lr5e-4.yaml

# One image: JSON on stdout, optional annotated image
uv run python scripts/predict.py data/raw/hazelnut/test/crack/000.png --out /tmp/crack.png

# API (needs the `api` extra). Settings via MODEL_PATH, CONFIDENCE, MAX_UPLOAD_MB; docs at /docs
uv run uvicorn defect_detection.api.app:create_app --factory
```

## Layout and intent

`src/` layout, built with hatchling; the importable package is `defect_detection`, split by pipeline stage:

- `data/`: dataset download and preparation. `prepare.py` is the entry point and composes the rest: `mvtec.py` (on-disk layout of a category), `masks.py` (mask → boxes), `split.py` (test + k folds), `config.py` (`DataConfig` read from `configs/data_<category>.yaml`)
- `training/`: YOLO training. `TrainConfig` is read from `configs/train_<name>.yaml`; its optional `hyperparameters` mapping is passed straight to `YOLO.train`
- `evaluation/`: box-level metrics (from Ultralytics) and image-level metrics (false alarms on good parts, defective parts flagged)
- `inference/`: `Predictor` loads the weights once and returns plain dataclasses (`Prediction`, `Detection`), never Ultralytics objects; `annotate` draws the boxes. Knows nothing about HTTP
- `api/`: FastAPI service (`app.py` with the `create_app` factory, `settings.py` from environment variables, `schemas.py` for responses). It only validates the upload and calls the `Predictor`. Its dependencies live in the optional `api` extra, so code outside `api/` must not import FastAPI or uvicorn

Supporting directories: `configs/` (YAML, loaded with `pyyaml`), `scripts/` (entry points), `notebooks/` (Colab notebook that only orchestrates the scripts), `reports/` (committed metrics JSON), `data/raw` and `data/processed`, `models/`.

## Things to know

- **PyTorch is CPU-only locally.** `pyproject.toml` pins `torch` and `torchvision` to the `pytorch-cpu` index through `[tool.uv.sources]`. Do not remove this or add a CUDA build without being asked.
- **Data and model weights are not in git.** `.gitignore` excludes everything under `data/` and `models/` except `.gitkeep`, plus `*.pt` and `*.onnx`; the plan is to version them with DVC. Ultralytics `runs/` and `mlruns/` are also ignored.
- Existing comments in `pyproject.toml` and `.gitignore` are written in Spanish. Code follows the same rule: docstrings and comments in Spanish, identifiers, log messages and exception messages in English.
- **`configs/splits/<category>.json` is the source of truth for the split** and is committed. `load_or_create_split` reuses it and fails if the config no longer matches; changing which images are in test requires `scripts/prepare_data.py --force-split`, which invalidates any metric already reported.
- **Colab sets the version floor.** The notebook runs `pip install -e .` on Colab's Python (3.13 when step 4 was done) and must keep Colab's CUDA build of torch, so `requires-python` and the lower bounds of `torch`/`torchvision` in `pyproject.toml` have to stay compatible with what Colab ships. Local versions are pinned by `uv.lock`.
- **`single_cls=True` renames the class to `item`.** Labels always carry the 4 defect classes; the trained checkpoints have a single class named `item`, not `defect`. Evaluation must also pass `single_cls=True`.
- **The test split is evaluated once.** `scripts/evaluate.py` only touches it with `--test`. It was used for `lr5e-4` (`reports/metrics_lr5e-4.json`); do not re-run it to compare experiments, use validation.
- **`uv add` / `uv remove` drop the `api` extra from the venv.** Run `uv sync --extra api` again afterwards, or `tests/test_api.py` is skipped (it uses `pytest.importorskip`) and uvicorn is missing.
- **The test client needs `httpx2`, not `httpx`.** The installed Starlette deprecates `httpx` for `TestClient`; `httpx2` is in the dev group.
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
4. Entrenamiento baseline YOLO (en Colab) + métricas (precision, recall, mAP) ✅ (modelo `lr5e-4`; métricas en reports/)
5. Inferencia + API FastAPI (imagen → JSON + imagen anotada) ✅ (/health, /predict, /predict/annotated)
6. Docker + tests + logging estructurado + CI (GitHub Actions) ← SIGUIENTE
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
  Validado en el paso 4: funciona, pero el checkpoint nombra la clase `item`, no `defect`.

Diseño acordado del paso 4 (entrenamiento y evaluación):
- Entrenamiento desde config (`configs/train_<nombre>.yaml`) con `scripts/train.py`; el
  notebook de Colab solo clona el repo, instala, descomprime el dataset y llama al script.
- El dataset llega a Colab como zip reproducible subido a Drive; su SHA256 se verifica antes
  de extraer. Los runs se copian a Drive como `<name>-<commit corto>`.
- Modelo YOLO11n preentrenado, imgsz 640, una clase (single_cls), valida con el fold 0,
  semilla 42 y deterministic=True. El entrenamiento es reproducible entre sesiones de Colab.
- `baseline` (optimizer=auto de Ultralytics: AdamW, lr0=0.002): las métricas caen a cero en
  las épocas 6-16 y tarda ~100 épocas en recuperarse. Con patience=30 paraba en la 35
  guardando la época 5; por eso epochs=300 y patience=100. Agota las 300 sin converger.
- `lr5e-4`: igual salvo lr0=0.0005 (optimizer, momentum y warmup_bias_lr fijados a lo que
  elige `auto`, para que solo cambie lr0). El bache dura ~12 épocas; mejor época 157.
  Es el modelo elegido. La causa del bache no está verificada (hipótesis: AMP se salta las
  primeras actualizaciones por gradientes desbordados; no afirmarlo como hecho).
- Validación en Colab (best.pt): baseline P 0.969, R 0.667, mAP50 0.807, mAP50-95 0.523;
  lr5e-4 P 0.812, R 0.864, mAP50 0.889, mAP50-95 0.682.
- Umbral de confianza para métricas por imagen: `confidence: 0.25` en la config, fijado
  mirando solo validación. El de máximo F1 se descartó: inestable entre Colab y local y
  dejaba pasar 4 de 14 piezas defectuosas en validación.
- Test (una sola evaluación, lr5e-4, 28 imágenes): P 0.846, R 0.857, mAP50 0.803,
  mAP50-95 0.571; 14 de 14 defectuosas detectadas y 1 falsa alarma en 14 buenas.
- Falsas alarmas sobre las 361 buenas de hazelnut que no entraron en el split (modelo y
  umbral ya fijados): 5 de 361, 1,4 % (IC 95 % de Wilson: 0,6 %–3,2 %). Sustituye como
  estimación a la de test (1 de 14). Para las defectuosas no hay equivalente: las 70 están
  todas en el split.
- Se dejó de ajustar aquí a propósito: con 15 defectos en validación, más experimentos
  ajustarían al ruido. Pendientes como experimentos: 4 clases, imgsz 1024, k-fold completo.

Diseño acordado del paso 5 (inferencia y API):
- Capa de inferencia separada de la API: `Predictor` devuelve dataclasses propias. La API
  solo traduce entre HTTP y esa capa.
- La clase `item` del checkpoint se expone como `defect` mediante la tabla `LABELS` de
  `inference/predictor.py`; otras clases conservan su nombre. Descartado reentrenar.
- Umbral fijado en el servidor (0.25, `CONFIDENCE`); el cliente no puede cambiarlo, para que
  las métricas del paso 4 sigan describiendo al servicio.
- Dos endpoints de predicción: `/predict` (JSON) y `/predict/annotated` (PNG, con el
  veredicto en las cabeceras `X-Defective` y `X-Detections`). Descartado base64 en el JSON.
- Configuración por variables de entorno (`MODEL_PATH`, `CONFIDENCE`, `MAX_UPLOAD_MB`), no
  por los YAML de entrenamiento.
- El modelo se carga y se calienta al arrancar; si faltan los pesos el servicio no arranca.
- Endpoints síncronos (`def`): la inferencia ocupa CPU y FastAPI los ejecuta en un pool de
  hilos. `Predictor` serializa las predicciones con un cerrojo porque un modelo de
  Ultralytics no admite llamadas simultáneas. Para más capacidad se escalan procesos.
- Validación de la subida: content-type declarado (415), tamaño (413) e imagen que no se
  decodifica (400). El límite de tamaño se comprueba tras recibir el fichero: el límite real
  del cuerpo lo debe poner el proxy.
- Medido en la CPU local (i5-14400F), servicio caliente, imágenes PNG de 1024 px: ~32 ms de
  inferencia del modelo, pero ~84 ms por petición completa (subida, decodificación, pre y
  posproceso), es decir, ~12 imágenes/s con un proceso. 40 peticiones con 8 en paralelo:
  todas 200 y con resultados idénticos por imagen.
- JPEG (calidad 90) de una imagen que en PNG da 1 caja da 2 cajas solapadas: el modelo solo
  se entrenó con PNG. El veredicto no cambia. No se ha medido el efecto de forma sistemática.
- Defecto conocido: los `logger.info` de la app y del Predictor no se ven al arrancar con
  uvicorn (solo salen sus logs de acceso), porque nadie configura el logging del paquete.
  Se resuelve en el paso 6 con el logging estructurado.
- Pendiente para el paso 6: logging estructurado, Dockerfile (cómo entran los pesos en la
  imagen), CI instalando el extra `api`, y número de workers.
