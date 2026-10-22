"""Extraccion de datos de Past Performances (DRF / Brisnet) con la API de Anthropic."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

import anthropic
from dotenv import load_dotenv
from PIL import Image, ImageOps
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

from services.simulator import (
    CLASS_RATINGS,
    FOREIGN_LEVELS,
    FOREIGN_ORIGINS,
    FORM_TRENDS,
    JOCKEY_SWITCHES,
    RUNNING_STYLES,
    SURFACES,
    TRACK_APTITUDES,
    normalize_horse,
)

load_dotenv()

# El modelo se puede cambiar con la variable ANTHROPIC_MODEL (ver .env.example).
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_PDF_BYTES = 32 * 1024 * 1024  # limite de la API para PDFs
MAX_PDF_PAGES = 100  # limite de la API para analisis visual de PDFs
MAX_IMAGES = 30  # imagenes por extraccion (limite practico para controlar el costo)
MAX_IMAGE_BYTES = 4_500_000  # la API acepta hasta 5 MB por imagen
MAX_IMAGE_SIDE = 3000  # lado mayor maximo en pixeles antes de reducir
MAX_REQUEST_BYTES = 24 * 1024 * 1024  # el limite de 32 MB de la API cuenta el Base64 (+33 %)
MAX_OUTPUT_TOKENS = 12_000
MAX_ATTEMPTS = 2


class ExtractionError(RuntimeError):
    """Error controlado durante la extraccion del PDF."""


SYSTEM_PROMPT = f"""Eres un analista experto en handicapping de carreras de caballos de pura sangre \
(Breeders' Cup, Keeneland, Del Mar) y lees Past Performances de Daily Racing Form y Brisnet, \
ya sea en PDF o en fotos y capturas de pantalla.
Tu unica salida es UN objeto JSON valido. Sin texto antes o despues, sin markdown, sin comentarios.
Nunca inventes datos: si algo no aparece en el PDF y no puede deducirse, usa null (o el valor por \
defecto indicado). Los nombres de claves y los valores de catalogo deben coincidir EXACTAMENTE.

ESTRUCTURA DE SALIDA:
{{
  "race": {{
    "race_name": string|null,
    "race_number": integer|null,
    "track": string|null,
    "race_date": "YYYY-MM-DD"|null,
    "surface": one of {list(SURFACES)}|null,
    "distance": string|null
  }},
  "horses": [
    {{
      "program_number": string,
      "name": string,
      "morning_line": string|null,
      "base_speed_figure": number|null,
      "foreign_origin": one of {list(FOREIGN_ORIGINS)},
      "foreign_level": one of {list(FOREIGN_LEVELS)},
      "dirt_debut": boolean,
      "form_trend": one of {list(FORM_TRENDS)},
      "class_rating": one of {list(CLASS_RATINGS)},
      "jockey": string|null,
      "is_regular_jockey": boolean,
      "jockey_switch": one of {list(JOCKEY_SWITCHES)},
      "track_aptitude": one of {list(TRACK_APTITUDES)},
      "mud_proven": boolean,
      "running_style": one of {list(RUNNING_STYLES)},
      "weight_lbs": number|null,
      "last_weight_lbs": number|null,
      "scratched": boolean
    }}
  ]
}}

DEFINICION DE CADA CAMPO:
- program_number / name / jockey: tal como aparecen en el programa. Incluye todos los caballos, \
tambien los marcados como scratched o also-eligible (marca scratched=true los retirados).
- morning_line: linea matutina en formato fraccion, p. ej. "5/2", "9-2", "even". Si no aparece, null.
- base_speed_figure: figura de velocidad en escala 0-120. Usa Beyer; si no hay, Brisnet Speed Rating; \
si solo hay Timeform/RPR, conviertela de forma conservadora a la escala Beyer. Usa el promedio de las \
2 mejores figuras de las ultimas 3 salidas. Si no hay ninguna, null.
- foreign_origin: "Europe" o "Japan" si su ultima salida o su base de campana es en Europa/Japon; \
si no, "None".
- foreign_level: nivel de su mejor resultado reciente en el extranjero: "G1_Winner" (gano un Grupo 1/G1), \
"G1_Placed" (2do o 3ro en un G1), "Other" (cualquier otro nivel). "None" si foreign_origin es "None".
- dirt_debut: true si el caballo nunca ha corrido en pista de arena (dirt).
- form_trend: analiza las ultimas 3 salidas. "Ascending" = figuras/posiciones mejorando; "Plateau" = \
estable; "Descending" = empeorando.
- class_rating: "G1_Winner" = ha ganado un Grado 1; "Graded_Placed" = ha sido colocado en stakes \
graduados o corre en su nivel; "Class_Riser" = sube claramente de categoria o de nivel.
- is_regular_jockey: true si monta el jinete que lo montó en la mayoria de sus ultimas salidas.
- jockey_switch: "Upgrade" si cambia a un jinete claramente mejor (p. ej. I. Ortiz Jr., F. Prat, \
R. Moore, W. Buick); "Downgrade" si cambia a uno claramente inferior; "Neutral" en los demas casos \
(incluido el jinete habitual).
- track_aptitude: rendimiento historico en este hipodromo/trazado: "Proven_Winner" (ha ganado aqui), \
"Positive" (buenos resultados, sin victoria), "Negative" (malos resultados aqui), "First_Time" (nunca \
ha corrido aqui).
- mud_proven: true si tiene victorias o actuaciones fuertes comprobadas en pista fangosa/pesada/suave.
- running_style: "E" = puntero (Early speed), "P" = presser/perseguidor, "S" = sustained/rematador \
(closer). Deducelo de los simbolos de estilo (E, E/P, P, S) y de los parciales de sus carreras.
- weight_lbs: peso asignado en esta carrera (libras). last_weight_lbs: peso en su ultima salida.

REGLAS: no agregues claves extra; numeros como numeros (no strings); booleanos como true/false; \
conserva el orden del programa."""


def _open_pdf(pdf_bytes: bytes) -> PdfReader:
    if not pdf_bytes:
        raise ExtractionError("El PDF esta vacio.")
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ExtractionError("El PDF esta protegido con contrasena.")
        if len(reader.pages) == 0:
            raise ExtractionError("El PDF no tiene paginas.")
    except ExtractionError:
        raise
    except (PdfReadError, ValueError, OSError) as exc:
        raise ExtractionError(f"No se pudo leer el PDF: {exc}") from exc
    return reader


def count_pdf_pages(pdf_bytes: bytes) -> int:
    """Numero de paginas de un PDF (lanza ExtractionError si no se puede leer)."""
    return len(_open_pdf(pdf_bytes).pages)


def parse_page_ranges(spec: str, total_pages: int) -> list[int]:
    """Convierte '5-12, 15' en indices de pagina (base 0), sin repetidos y en orden."""
    normalized = re.sub(r"\s*-\s*", "-", spec.strip())
    indices: list[int] = []
    for part in re.split(r"[,;\s]+", normalized):
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ExtractionError(f"Rango de paginas invalido: '{part}'. Usa por ejemplo 5-12 o 3,7,10-12.")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start or end > total_pages:
            raise ExtractionError(f"El rango {part} esta fuera del PDF (tiene {total_pages} paginas).")
        for index in range(start - 1, end):
            if index not in indices:
                indices.append(index)
    if not indices:
        raise ExtractionError("No se indico ninguna pagina.")
    return sorted(indices)


def select_pdf_pages(pdf_bytes: bytes, spec: str) -> bytes:
    """Devuelve un PDF nuevo solo con las paginas indicadas (p. ej. '12-19')."""
    reader = _open_pdf(pdf_bytes)
    indices = parse_page_ranges(spec, len(reader.pages))
    writer = PdfWriter()
    for index in indices:
        writer.add_page(reader.pages[index])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _prepare_pdf(pdf_bytes: bytes, pages: str | None) -> tuple[bytes, int]:
    """Recorta el PDF si se indicaron paginas y valida los limites de la API."""
    reader = _open_pdf(pdf_bytes)
    total = len(reader.pages)
    if pages and pages.strip():
        pdf_bytes = select_pdf_pages(pdf_bytes, pages)
        total = len(parse_page_ranges(pages, total))
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ExtractionError("El PDF supera el limite de 32 MB de la API de Anthropic. Indica menos paginas.")
    if total > MAX_PDF_PAGES:
        raise ExtractionError(
            f"Son {total} paginas y el maximo es {MAX_PDF_PAGES}. "
            "Indica solo las paginas de la carrera (por ejemplo 12-19)."
        )
    return pdf_bytes, total


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("La respuesta no contiene un objeto JSON.")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("El JSON de la respuesta no es un objeto.")
    return data


def _normalize_result(data: dict[str, Any]) -> dict[str, Any]:
    horses_raw = data.get("horses")
    if not isinstance(horses_raw, list) or not horses_raw:
        raise ValueError("El JSON no contiene la lista 'horses' o esta vacia.")
    race_raw = data.get("race") if isinstance(data.get("race"), dict) else {}

    surface = race_raw.get("surface")
    surface = next((s for s in SURFACES if str(surface).strip().lower() == s.lower()), None)
    try:
        race_number = int(race_raw["race_number"]) if race_raw.get("race_number") is not None else None
    except (TypeError, ValueError):
        race_number = None

    race = {
        "race_name": (str(race_raw.get("race_name")).strip() if race_raw.get("race_name") else None),
        "race_number": race_number,
        "track": (str(race_raw.get("track")).strip() if race_raw.get("track") else None),
        "race_date": (str(race_raw.get("race_date")).strip() if race_raw.get("race_date") else None),
        "surface": surface,
        "distance": (str(race_raw.get("distance")).strip() if race_raw.get("distance") else None),
    }
    horses = [normalize_horse(item) for item in horses_raw if isinstance(item, dict)]
    horses = [h for h in horses if h["name"]]
    if not horses:
        raise ValueError("Ningun caballo tiene nombre en el JSON devuelto.")
    return {"race": race, "horses": horses}


def detect_file_kind(data: bytes) -> str:
    """Tipo de archivo segun sus primeros bytes: 'pdf', 'png', 'jpeg', 'webp', 'gif' o 'unknown'."""
    head = data[:12]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "unknown"


def _encode_image(image: Image.Image, image_format: str, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    if image_format == "JPEG":
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    else:
        image.save(buffer, format=image_format, optimize=True)
    return buffer.getvalue()


def prepare_image(data: bytes, name: str = "imagen") -> tuple[bytes, str]:
    """Normaliza una foto o captura para la API.

    Corrige la rotacion EXIF (fotos de celular), reemplaza la transparencia por blanco,
    reduce las imagenes muy grandes y las comprime si superan el limite de 5 MB.
    Devuelve (bytes, media_type).
    """
    try:
        with Image.open(io.BytesIO(data)) as source:
            source_format = (source.format or "").upper()
            image = ImageOps.exif_transpose(source)
            image.load()
    except Exception as exc:  # noqa: BLE001 - Pillow lanza varios tipos de error
        raise ExtractionError(f"No se pudo leer la imagen '{name}': {exc}") from exc

    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
        image = flat
    elif image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    resample = getattr(Image, "Resampling", Image).LANCZOS
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), resample)

    if source_format == "PNG":
        encoded = _encode_image(image, "PNG")
        if len(encoded) <= MAX_IMAGE_BYTES:
            return encoded, "image/png"

    while True:
        for quality in (90, 80, 70):
            encoded = _encode_image(image, "JPEG", quality)
            if len(encoded) <= MAX_IMAGE_BYTES:
                return encoded, "image/jpeg"
        if max(image.size) <= 1000:
            raise ExtractionError(f"La imagen '{name}' es demasiado pesada.")
        image = image.resize((int(image.width * 0.8), int(image.height * 0.8)), resample)


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def extract_race_data_from_files(
    files: list[tuple[str, bytes]],
    race_hint: str | None = None,
    track: str | None = None,
    pages: str | None = None,
) -> dict[str, Any]:
    """Envia PDFs y/o imagenes a Claude y devuelve {"race": {...}, "horses": [...]}.

    Args:
        files: lista de (nombre, contenido). Admite PDF, PNG, JPG, WEBP y GIF, mezclados.
            Si hay varias imagenes se tratan como paginas de la misma carrera, en orden.
        race_hint: nombre/numero de la carrera a extraer si el material trae varias.
        track: hipodromo (para evaluar la aptitud a la pista), p. ej. "Keeneland".
        pages: paginas a enviar de un PDF, p. ej. "12-19". Solo se puede usar con un unico PDF.

    Raises:
        ExtractionError: archivo invalido, falta la API key, error de API o JSON no parseable.
    """
    if not files:
        raise ExtractionError("No se recibio ningun archivo.")

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ExtractionError("Falta ANTHROPIC_API_KEY. Configurala en el archivo .env.")
    model = os.getenv("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL

    kinds = [detect_file_kind(data) for _, data in files]
    pdf_count = kinds.count("pdf")
    image_total = len(kinds) - pdf_count
    use_pages = bool(pages and pages.strip())
    if use_pages and pdf_count != 1:
        raise ExtractionError("El rango de paginas solo se puede usar cuando subes un unico PDF.")
    if image_total > MAX_IMAGES:
        raise ExtractionError(f"Maximo {MAX_IMAGES} imagenes por extraccion (subiste {image_total}).")

    content: list[dict[str, Any]] = []
    total_pdf_pages = 0
    total_bytes = 0
    for position, ((name, data), kind) in enumerate(zip(files, kinds), start=1):
        if kind == "unknown":
            raise ExtractionError(f"'{name}' no es un PDF ni una imagen compatible (PNG, JPG o WEBP).")
        if kind == "pdf":
            payload, page_count = _prepare_pdf(data, pages if use_pages else None)
            total_pdf_pages += page_count
            if total_pdf_pages > MAX_PDF_PAGES:
                raise ExtractionError(
                    f"Son mas de {MAX_PDF_PAGES} paginas de PDF en total. "
                    "Indica solo las paginas de la carrera (por ejemplo 12-19)."
                )
            media_type, block_type = "application/pdf", "document"
        else:
            payload, media_type = prepare_image(data, name)
            block_type = "image"
        total_bytes += len(payload)
        if total_bytes > MAX_REQUEST_BYTES:
            raise ExtractionError("Los archivos pesan demasiado en total. Sube menos paginas o imagenes.")
        if len(files) > 1:
            content.append({"type": "text", "text": f"Archivo {position} de {len(files)}: {name}"})
        content.append(
            {
                "type": block_type,
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(payload).decode("utf-8"),
                },
            }
        )

    parts = []
    if pdf_count:
        parts.append(f"{_plural(pdf_count, 'PDF', 'PDFs')} ({_plural(total_pdf_pages, 'pagina', 'paginas')})")
    if image_total:
        parts.append(_plural(image_total, "imagen", "imagenes"))
    instructions = [f"Extrae los datos de handicap de este material de Past Performances: {' y '.join(parts)}."]
    if image_total:
        instructions.append(
            "Las imagenes son fotos o capturas de paginas del programa. Si hay varias, pertenecen a la "
            "misma carrera: leelas en orden y combina la informacion. Si un dato no se lee con claridad, usa null."
        )
    if track:
        instructions.append(f"Hipodromo de la carrera: {track} (usalo para evaluar track_aptitude).")
    if race_hint:
        instructions.append(f"Si el material contiene varias carreras, extrae SOLO esta: {race_hint}.")
    instructions.append("Responde unicamente con el objeto JSON.")
    prompt = "\n".join(instructions)

    client = anthropic.Anthropic(api_key=api_key)
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        user_text = prompt
        if attempt > 1:
            user_text += "\nIMPORTANTE: tu respuesta anterior no fue un JSON valido. Devuelve SOLO JSON valido."
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content + [{"type": "text", "text": user_text}]}],
            )
        except anthropic.APIError as exc:
            raise ExtractionError(f"Error de la API de Anthropic: {exc}") from exc

        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        if response.stop_reason == "max_tokens":
            last_error = ValueError("La respuesta se corto por limite de tokens (demasiados caballos).")
            continue
        try:
            return _normalize_result(_parse_json(text))
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc

    raise ExtractionError(f"No se pudo obtener un JSON valido tras {MAX_ATTEMPTS} intentos: {last_error}")


def extract_race_data_from_pdf(
    pdf_bytes: bytes,
    race_hint: str | None = None,
    track: str | None = None,
    pages: str | None = None,
) -> dict[str, Any]:
    """Compatibilidad: extrae los datos de un unico PDF (o imagen) ya en memoria."""
    return extract_race_data_from_files(
        [("documento", pdf_bytes)], race_hint=race_hint, track=track, pages=pages
    )