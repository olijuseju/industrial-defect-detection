"""Servicio HTTP de detección de defectos.

Se arranca con::

    uvicorn defect_detection.api.app:create_app --factory

Solo traduce entre HTTP y la capa de inferencia: valida la subida, llama al ``Predictor`` y
da formato a la respuesta. El modelo se carga una vez al arrancar.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status

from defect_detection import __version__
from defect_detection.api.schemas import HealthOut, PredictionOut
from defect_detection.api.settings import Settings
from defect_detection.inference.predictor import Prediction, Predictor, annotate
from defect_detection.logging_config import JSON, configure_logging, request_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

ALLOWED_CONTENT_TYPES = ("image/png", "image/jpeg")

_UPLOAD_ERRORS = {
    status.HTTP_400_BAD_REQUEST: {"description": "El fichero no es una imagen válida"},
    status.HTTP_413_CONTENT_TOO_LARGE: {"description": "La imagen supera el tamaño máximo"},
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "Solo se admite PNG o JPEG"},
}


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def read_image(file: UploadFile, max_bytes: int) -> np.ndarray:
    """Valida la subida y la decodifica a una imagen BGR."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported content type {file.content_type!r}; use PNG or JPEG",
        )
    # Se lee un byte de más para detectar el exceso sin cargar el fichero entero. Esto
    # protege la decodificación; el límite del cuerpo de la petición lo pone el proxy.
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"Image larger than {max_bytes} bytes"
        )
    # El content-type lo declara el cliente; lo que decide es si los bytes se decodifican.
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is not a valid image")
    return image


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    """Crea la aplicación. ``predictor`` permite inyectar uno falso en los tests."""
    settings = settings or Settings.from_env()
    configure_logging(default_format=JSON, capture_uvicorn=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Si los pesos no existen el servicio no arranca: mejor fallar aquí que aceptar
        # peticiones que no se pueden atender.
        app.state.predictor = predictor or Predictor(settings.model_path, settings.confidence)
        app.state.settings = settings
        # La primera predicción es más lenta; se hace ahora y no con la primera petición.
        app.state.predictor.predict(np.zeros((640, 640, 3), np.uint8))
        logger.info(
            "Service ready",
            extra={
                "model": app.state.predictor.weights.name,
                "confidence": app.state.predictor.confidence,
            },
        )
        yield

    app = FastAPI(
        title="Industrial defect detection",
        description="Detecta defectos en imágenes de piezas con un modelo YOLO.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        """Una línea de log por petición, con un identificador que se devuelve al cliente.

        Se respeta el ``X-Request-ID`` que envíe el cliente o el proxy, para poder seguir
        una petición a través de varios servicios.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            if status_code >= 500:
                level = logging.ERROR
            elif request.url.path == "/health":
                # El orquestador la consulta cada pocos segundos: a INFO taparía el resto.
                level = logging.DEBUG
            else:
                level = logging.INFO
            logger.log(
                level,
                "%s %s -> %d in %.1f ms",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": duration_ms,
                },
            )
            request_id_var.reset(token)

    def predict_upload(file: UploadFile, predictor: Predictor, settings: Settings):
        image = read_image(file, settings.max_upload_bytes)
        prediction = predictor.predict(image)
        logger.info(
            "Prediction: defective=%s detections=%d inference_ms=%.1f",
            prediction.defective,
            len(prediction.detections),
            prediction.inference_ms,
            extra={
                "defective": prediction.defective,
                "detections": len(prediction.detections),
                "inference_ms": prediction.inference_ms,
                "width": prediction.width,
                "height": prediction.height,
            },
        )
        return image, prediction

    @app.get("/health", response_model=HealthOut, summary="Estado del servicio")
    def health(
        predictor: Annotated[Predictor, Depends(get_predictor)],
    ) -> dict:
        return {
            "status": "ok",
            "model": predictor.weights.name,
            "confidence": predictor.confidence,
        }

    # Los endpoints son `def` y no `async def`: la inferencia ocupa CPU, y FastAPI ejecuta
    # las funciones síncronas en un hilo aparte sin bloquear el bucle de eventos.
    @app.post(
        "/predict",
        response_model=PredictionOut,
        responses=_UPLOAD_ERRORS,
        summary="Detecta defectos y devuelve JSON",
    )
    def predict(
        file: Annotated[UploadFile, File(description="Imagen PNG o JPEG de la pieza")],
        predictor: Annotated[Predictor, Depends(get_predictor)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> dict:
        _, prediction = predict_upload(file, predictor, settings)
        return prediction.to_dict()

    @app.post(
        "/predict/annotated",
        response_class=Response,
        responses={
            status.HTTP_200_OK: {
                "content": {"image/png": {}},
                "description": "La imagen con las cajas pintadas",
            },
            **_UPLOAD_ERRORS,
        },
        summary="Detecta defectos y devuelve la imagen anotada",
    )
    def predict_annotated(
        file: Annotated[UploadFile, File(description="Imagen PNG o JPEG de la pieza")],
        predictor: Annotated[Predictor, Depends(get_predictor)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> Response:
        image, prediction = predict_upload(file, predictor, settings)
        ok, encoded = cv2.imencode(".png", annotate(image, prediction))
        if not ok:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Cannot encode image")
        return Response(
            content=encoded.tobytes(),
            media_type="image/png",
            headers=_summary_headers(prediction),
        )

    return app


def _summary_headers(prediction: Prediction) -> dict[str, str]:
    """Veredicto en cabeceras, para quien solo pide la imagen y no quiere otra llamada."""
    return {
        "X-Defective": str(prediction.defective).lower(),
        "X-Detections": str(len(prediction.detections)),
    }
