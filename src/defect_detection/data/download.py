"""Descarga, verificación y extracción del dataset MVTec AD.

El dataset tiene licencia CC BY-NC-SA 4.0 y no se versiona en git: vive en ``data/raw``.
La URL de descarga tampoco se guarda en el repo; llega por argumento desde la CLI.

Sobre el checksum: no se usa ningún hash "oficial". El SHA256 se calcula la primera vez
que se ve un archivo y se guarda; a partir de ahí se verifica. Eso detecta corrupción y
garantiza que todas las descargas son idénticas a la primera, pero no prueba que la
primera fuese el fichero auténtico de MVTec.
"""

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

MVTEC_AD_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

_CHUNK_SIZE = 1024 * 1024
_PROGRESS_EVERY = 256 * _CHUNK_SIZE


class ChecksumMismatchError(Exception):
    """El SHA256 de un archivo no coincide con el registrado."""


def sha256_file(path: Path) -> str:
    """Calcula el SHA256 de un fichero leyéndolo por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path, timeout: float = 60) -> Path:
    """Descarga ``url`` en streaming a ``dest``.

    Se escribe en ``<dest>.part`` y se renombra al terminar, de modo que una descarga
    interrumpida nunca deja un fichero que parezca completo.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    logger.info("Downloading %s", dest.name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, part.open("wb") as out:
            total = response.headers.get("Content-Length")
            done = 0
            next_report = _PROGRESS_EVERY
            while chunk := response.read(_CHUNK_SIZE):
                out.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    logger.info("Downloaded %d MB of %s bytes", done // _CHUNK_SIZE, total or "?")
                    next_report += _PROGRESS_EVERY
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    logger.info("Download finished: %s (%d bytes)", dest, done)
    return dest


def verify_or_record_checksum(archive: Path, checksums_path: Path) -> str:
    """Verifica el SHA256 de ``archive`` o lo registra si es la primera vez que se ve.

    Los hashes se guardan en un JSON ``{nombre_de_fichero: sha256}``.
    """
    checksums: dict[str, str] = {}
    if checksums_path.exists():
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))

    actual = sha256_file(archive)
    expected = checksums.get(archive.name)
    if expected is None:
        checksums[archive.name] = actual
        checksums_path.parent.mkdir(parents=True, exist_ok=True)
        checksums_path.write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        logger.info("Recorded SHA256 for %s in %s: %s", archive.name, checksums_path, actual)
    elif actual != expected:
        raise ChecksumMismatchError(
            f"SHA256 mismatch for {archive.name}: expected {expected}, got {actual}"
        )
    else:
        logger.info("SHA256 verified for %s", archive.name)
    return actual


def extract_categories(archive: Path, raw_dir: Path, categories: Iterable[str]) -> None:
    """Extrae de ``archive`` las carpetas ``<categoría>/`` pedidas a ``raw_dir``.

    Se extrae a un directorio temporal dentro de ``raw_dir`` y solo al final se mueve cada
    categoría a su sitio: una extracción interrumpida no deja categorías a medias.
    ``filter="data"`` rechaza rutas que escapan del destino (path traversal).
    """
    wanted = set(categories)
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".extract-", dir=raw_dir))
    try:
        # Una sola pasada: listar antes los miembros obligaría a descomprimir dos veces.
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                parts = [p for p in PurePosixPath(member.name).parts if p != "."]
                if parts and parts[0] in wanted:
                    tar.extract(member, tmp_dir, filter="data")

        missing = sorted(c for c in wanted if not (tmp_dir / c).is_dir())
        if missing:
            raise ValueError(f"Categories not found in {archive.name}: {', '.join(missing)}")

        for category in sorted(wanted):
            (tmp_dir / category).rename(raw_dir / category)
            logger.info("Extracted %s to %s", category, raw_dir / category)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def ensure_dataset(
    raw_dir: Path,
    checksums_path: Path,
    categories: Iterable[str] | None = None,
    url: str | None = None,
    archive: Path | None = None,
) -> None:
    """Deja en ``raw_dir`` las categorías pedidas (por defecto, las 15 de MVTec AD).

    Es idempotente: las categorías que ya existen no se tocan y, si están todas, no se
    descarga ni se abre nada. El archivo llega por ``url`` (se descarga a
    ``raw_dir/archives``) o por ``archive`` (fichero ya bajado a mano).
    """
    requested = list(categories) if categories else list(MVTEC_AD_CATEGORIES)
    missing = [c for c in requested if not (raw_dir / c).is_dir()]
    if not missing:
        logger.info("All requested categories already present in %s, nothing to do", raw_dir)
        return

    if (url is None) == (archive is None):
        raise ValueError("Provide exactly one of url or archive")

    if archive is None:
        name = PurePosixPath(urllib.parse.urlparse(url).path).name
        if not name:
            raise ValueError("Cannot derive an archive file name from the URL")
        archive = raw_dir / "archives" / name
        if archive.exists():
            logger.info("Archive already downloaded: %s", archive)
        else:
            download_file(url, archive)
    elif not archive.is_file():
        raise FileNotFoundError(archive)

    verify_or_record_checksum(archive, checksums_path)
    extract_categories(archive, raw_dir, missing)
