# Industrial defect detection

[![CI](https://github.com/olijuseju/industrial-defect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/olijuseju/industrial-defect-detection/actions/workflows/ci.yml)

Defect detection on industrial parts with YOLO (Ultralytics), built as an end-to-end
project rather than a notebook: reproducible data pipeline, config-driven training,
evaluation with versioned metrics, a FastAPI inference service, a Docker image and CI.

> **Status: work in progress.** Data, training, evaluation, the API, Docker and CI are done.
> Data and model versioning and cross-validated results are not (see [Roadmap](#roadmap)).

## What it does

- Converts the pixel-level masks of [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
  (an anomaly-detection dataset with no bounding boxes) into YOLO labels.
- Trains a YOLO11n detector on the `hazelnut` category (Google Colab GPU, same code as local).
- Evaluates per box (precision, recall, mAP) and per image (defective parts flagged,
  false alarms on good parts).
- Serves the model over HTTP: an image in, JSON or an annotated image out.
- Ships as a Docker image, built and smoke-tested on every push.

## Results

Model `lr5e-4` (YOLO11n, 640 px, single `defect` class), confidence threshold 0.25.
Numbers come from the JSON files in [`reports/`](reports/).

| Test split (28 images, evaluated once) | |
|---|---|
| Precision / recall | 0.846 / 0.857 |
| mAP50 / mAP50-95 | 0.803 / 0.571 |
| Defective parts flagged | 14 of 14 |

| False alarms on good parts | |
|---|---|
| 361 good images never used for training or tuning | 5 of 361 (1.4 %, 95 % CI 0.6 %–3.2 %) |

### Limitations

- **Very little defect data.** `hazelnut` has 70 defective images. The test split holds 14,
  so "14 of 14" only bounds the true detection rate above roughly 77 % (95 % confidence).
- **One fold, one seed.** The split has 4 folds ready, but cross-validation has not been run.
- **Loose boxes.** mAP50-95 of 0.57: good enough to flag a part, not to measure the defect.
- **One category, studio images.** Nothing here says how the model behaves on other parts,
  lighting or cameras.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13.

```bash
uv sync --extra api
uv run pytest
```

The dataset and the trained weights are not in the repository (see [Data](#data)).

```bash
# 1. Dataset: extract MVTec AD and build the YOLO dataset
uv run python scripts/download_data.py --archive /path/to/mvtec_anomaly_detection.tar.xz
uv run python scripts/prepare_data.py

# 2. Train (in Colab: notebooks/train_colab.ipynb) and put the weights in models/lr5e-4.pt

# 3. Evaluate on validation
uv run python scripts/evaluate.py --weights models/lr5e-4.pt --config configs/train_lr5e-4.yaml

# 4. Predict one image
uv run python scripts/predict.py data/raw/hazelnut/test/crack/000.png --out annotated.png
```

## API

```bash
uv run uvicorn defect_detection.api.app:create_app --factory
```

| Endpoint | Returns |
|---|---|
| `GET /health` | Service status, loaded model and confidence threshold |
| `POST /predict` | JSON with the verdict and the boxes |
| `POST /predict/annotated` | PNG with the boxes drawn; verdict in `X-Defective` / `X-Detections` headers |

```bash
curl -F "file=@part.png;type=image/png" http://127.0.0.1:8000/predict
```

```json
{
  "defective": true,
  "detections": [{"label": "defect", "confidence": 0.852, "box": [264, 267, 822, 535]}],
  "width": 1024,
  "height": 1024,
  "inference_ms": 31.6
}
```

Interactive documentation is served at `/docs`. The service is configured with the
`MODEL_PATH`, `CONFIDENCE` and `MAX_UPLOAD_MB` environment variables.

Logs are one JSON object per line (`LOG_FORMAT=text` for a readable format). Every request
gets an ID, taken from the `X-Request-ID` header or generated, which is returned to the
client and attached to every log line written while handling it.

## Docker

The image contains the service but not the model: the weights are mounted at run time.

```bash
docker build -t defect-detection .
docker run --rm -p 8000:8000 \
  -v "$PWD/models/lr5e-4.pt:/models/model.pt:ro" defect-detection
```

- Two-stage build with dependencies installed from `uv.lock` (CPU-only PyTorch).
- Runs as a non-root user and has a health check against `/health`.
- `scripts/docker_smoke_test.sh <image> <weights.pt>` starts a container and checks the
  endpoints; CI runs the same script with the public `yolo11n.pt` weights, because the
  trained weights are not in the repository.

The image is large: 2.5 GB on disk, about 570 MB compressed, most of it PyTorch. Serving
an ONNX export without PyTorch would shrink it; that has not been done.

## Project layout

```
src/defect_detection/
  data/        download, mask -> box conversion, split, YOLO dataset
  training/    config-driven YOLO training
  evaluation/  box-level and image-level metrics
  inference/   Predictor (no HTTP, no Ultralytics types in its output)
  api/         FastAPI service
  logging_config.py  text or JSON logging for scripts and service
scripts/       thin command-line entry points
configs/       data and training configs, dataset checksums, the committed split
reports/       metrics produced by the evaluation scripts
notebooks/     Colab notebook that only orchestrates the scripts
```

## Data

MVTec AD is licensed under CC BY-NC-SA 4.0 and is not redistributed here: `data/` is
git-ignored. Download it from MVTec and pass the archive to `scripts/download_data.py`.
Its SHA256 is recorded in `configs/dataset_checksums.json` on first use and verified
afterwards. The train/validation/test split is committed in `configs/splits/`.

## Roadmap

- [x] Project setup
- [x] Dataset download with checksum verification
- [x] Data preparation: masks to YOLO labels, stratified test split + 4 folds
- [x] Training and evaluation
- [x] Inference layer and FastAPI service
- [x] Structured logging, CI (GitHub Actions) and Docker image
- [ ] Data and model versioning (DVC/MLflow), cross-validated results
