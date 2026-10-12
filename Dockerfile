# syntax=docker/dockerfile:1

# ---- Etapa 1: construir el entorno virtual ----
# Aquí viven las herramientas de instalación (uv, cachés). Nada de esto llega a la imagen
# final: solo se copia el entorno ya montado.
FROM python:3.11-slim-trixie AS builder

# uv se trae de su imagen oficial, con la misma versión que se usa en desarrollo.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv

# Compilar a bytecode acelera el arranque; copy porque la caché está en otro sistema de
# ficheros; y se prohíbe descargar otro Python: debe usarse el de la imagen base.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Primero solo las dependencias, a partir del lock. Es la capa más pesada (torch) y así
# Docker la reutiliza mientras no cambien pyproject.toml ni uv.lock, aunque cambie el código.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --extra api

# Después el código del proyecto. --no-editable lo instala dentro del entorno, de modo que
# la imagen final no necesita la carpeta src/.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra api --no-editable


# ---- Etapa 2: imagen de ejecución ----
FROM python:3.11-slim-trixie

# Librerías del sistema que opencv-python carga al importarse y que la imagen slim no trae
# (sin ellas falla con "libxcb.so.1: cannot open shared object file"). Es la variante con
# interfaz gráfica, que es la que exige Ultralytics, aunque aquí no se abra ninguna ventana.
# Los nombres son los de Debian 13 (trixie); por eso la imagen base fija esa versión.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios: si alguien compromete el servicio, no es root en el contenedor.
RUN useradd --create-home --uid 1000 app

# Misma ruta que en la etapa anterior: un entorno virtual no se puede cambiar de sitio.
# Queda propiedad de root: el usuario del servicio puede ejecutarlo pero no modificarlo.
COPY --from=builder /app/.venv /app/.venv

# Los pesos no van en la imagen: se montan en /models al arrancar el contenedor.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/model.pt \
    LOG_FORMAT=json

USER app
WORKDIR /home/app
EXPOSE 8000

# Docker marca el contenedor como unhealthy si /health deja de responder. Se usa Python
# porque la imagen slim no trae curl. start-period da margen a la carga del modelo.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# 0.0.0.0: dentro de un contenedor, 127.0.0.1 no es accesible desde fuera.
CMD ["uvicorn", "defect_detection.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
