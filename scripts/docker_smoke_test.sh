#!/usr/bin/env bash
# Prueba de humo de la imagen: arranca el contenedor con unos pesos montados y comprueba
# que el servicio responde. No mide la calidad del modelo, solo que la imagen sirve.
#
# Uso: scripts/docker_smoke_test.sh <imagen> <pesos.pt>
# En CI se usan los pesos públicos yolo11n.pt; en local valen también los entrenados.
set -euo pipefail

IMAGE="${1:?usage: docker_smoke_test.sh <image> <weights.pt>}"
WEIGHTS="$(realpath "${2:?usage: docker_smoke_test.sh <image> <weights.pt>}")"
PORT="${PORT:-8000}"
NAME="defect-detection-smoke-$$"
URL="http://127.0.0.1:${PORT}"
WORKDIR="$(mktemp -d)"

cleanup() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "--- container logs ---"
        docker logs "$NAME" 2>&1 | tail -40 || true
    fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    rm -rf "$WORKDIR"
    exit "$status"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $1" >&2
    exit 1
}

echo "Starting $IMAGE with $WEIGHTS"
docker run -d --name "$NAME" -p "${PORT}:8000" -v "${WEIGHTS}:/models/model.pt:ro" "$IMAGE" >/dev/null

# La carga del modelo tarda unos segundos; se espera a que /health responda.
for _ in $(seq 1 60); do
    if curl -sf "$URL/health" >/dev/null; then break; fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
        fail "container exited during startup"
    fi
    sleep 1
done

health="$(curl -sf "$URL/health")" || fail "/health did not respond"
echo "health: $health"
echo "$health" | grep -q '"status":"ok"' || fail "unexpected /health body"

# La imagen de prueba se genera con el OpenCV del propio contenedor: así el runner no
# necesita Python ni dependencias.
docker exec "$NAME" python -c "
import sys, cv2, numpy as np
sys.stdout.buffer.write(cv2.imencode('.png', np.zeros((320, 320, 3), np.uint8))[1].tobytes())
" >"$WORKDIR/blank.png"

body="$(curl -sf -F "file=@$WORKDIR/blank.png;type=image/png" "$URL/predict")" \
    || fail "/predict did not return 200"
echo "predict: $body"
echo "$body" | grep -q '"defective":' || fail "unexpected /predict body"

code="$(curl -s -o "$WORKDIR/annotated.png" -w '%{http_code} %{content_type}' \
    -F "file=@$WORKDIR/blank.png;type=image/png" "$URL/predict/annotated")"
[ "$code" = "200 image/png" ] || fail "/predict/annotated returned '$code'"

echo "not an image" >"$WORKDIR/notes.txt"
code="$(curl -s -o /dev/null -w '%{http_code}' \
    -F "file=@$WORKDIR/notes.txt;type=text/plain" "$URL/predict")"
[ "$code" = "415" ] || fail "a text upload returned $code instead of 415"

[ "$(docker exec "$NAME" id -un)" != "root" ] || fail "service runs as root"

echo "Smoke test passed"
