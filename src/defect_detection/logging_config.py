"""Configuración del logging de todo el proyecto.

Dos formatos, elegidos con la variable de entorno ``LOG_FORMAT``:

- ``text``: una línea legible, para trabajar en la terminal (por defecto en los scripts).
- ``json``: un objeto JSON por línea, para que un agregador de logs pueda filtrar por campo
  (por defecto en el servicio).

Los datos de un mensaje se pasan con ``extra={...}`` y salen como campos del JSON. Cada
registro lleva además el ``request_id`` de la petición HTTP en curso, si la hay.
"""

import contextvars
import json
import logging
import os
import sys
from datetime import UTC, datetime

TEXT = "text"
JSON = "json"
_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Identificador de la petición HTTP en curso; lo fija el middleware de la API.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Atributos que trae cualquier LogRecord: lo que no esté aquí llegó por `extra`.
# `color_message` es una copia del mensaje con códigos de color que añade uvicorn.
_STANDARD_ATTRIBUTES = frozenset(
    set(vars(logging.LogRecord("", 0, "", 0, "", None, None)))
    | {"message", "asctime", "taskName", "color_message"}
)


class JsonFormatter(logging.Formatter):
    """Un objeto JSON por registro, con los campos de ``extra`` en el primer nivel."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRIBUTES and value is not None:
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        # default=str: un Path o una fecha en `extra` no deben tumbar el log.
        return json.dumps(entry, default=str, ensure_ascii=False)


class _ProjectHandler(logging.StreamHandler):
    """Marca el handler que instala este módulo, para poder sustituirlo sin tocar otros."""


def _install_request_id_factory() -> None:
    """Añade ``request_id`` a cada registro, lo emita quien lo emita."""
    current = logging.getLogRecordFactory()
    if getattr(current, "adds_request_id", False):
        return

    def factory(*args, **kwargs) -> logging.LogRecord:
        record = current(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    factory.adds_request_id = True
    logging.setLogRecordFactory(factory)


def configure_logging(default_format: str = TEXT, capture_uvicorn: bool = False) -> None:
    """Configura el logger raíz. Se puede llamar varias veces.

    ``LOG_FORMAT`` y ``LOG_LEVEL`` del entorno mandan sobre los valores por defecto.
    ``capture_uvicorn`` hace que los logs de uvicorn salgan con el mismo formato y apaga
    su log de acceso, que duplicaría la línea por petición que escribe la API.
    """
    log_format = os.environ.get("LOG_FORMAT", default_format).lower()
    if log_format not in (TEXT, JSON):
        raise ValueError(f"LOG_FORMAT must be '{TEXT}' or '{JSON}', got {log_format!r}")
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # A stderr: stdout queda libre para la salida de los scripts (p. ej. el JSON de predict).
    handler = _ProjectHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if log_format == JSON else logging.Formatter(_TEXT_FORMAT))
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, _ProjectHandler)]
    root.addHandler(handler)
    root.setLevel(level)
    _install_request_id_factory()

    if capture_uvicorn:
        for name in ("uvicorn", "uvicorn.error"):
            uvicorn_logger = logging.getLogger(name)
            uvicorn_logger.handlers.clear()
            uvicorn_logger.propagate = True
        access = logging.getLogger("uvicorn.access")
        access.handlers.clear()
        access.propagate = False
