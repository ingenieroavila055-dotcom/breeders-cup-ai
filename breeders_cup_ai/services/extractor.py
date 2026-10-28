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
    BLINKERS_CHANGES,
    CLASS_RATINGS,
    FIT_LEVELS,
    FOREIGN_LEVELS,
    FOREIGN_ORIGINS,
    FORM_TRENDS,
    GOING_PREFS,
    JOCKEY_SWITCHES,
    RATING_KINDS,
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
MAX_OUTPUT_TOKENS = 20_000  # un JSON de carrera usa ~3,000; margen amplio (la API exige streaming por encima de ~21,000)
MAX_ATTEMPTS = 2


class ExtractionError(RuntimeError):
    """Error controlado durante la extraccion del PDF."""


_PROMPT_TEMPLATE = """Eres un analista experto en handicapping de carreras de caballos de pura sangre \
(Breeders' Cup, Keeneland en arena y grama) y lees Past Performances de Daily Racing Form (DRF), \
ya sea en PDF o en fotos y capturas de pantalla.
Tu unica salida es UN objeto JSON valido, minificado, sin texto antes o despues, sin markdown, sin comentarios.

FUENTE: usa UNICAMENTE el material adjunto. No uses conocimiento externo ni datos que no esten en el \
documento. Si un dato no aparece o no se lee con claridad, usa null (o el valor por defecto indicado). \
Nunca inventes ni estimes cifras. Los nombres de claves y los valores de catalogo deben coincidir EXACTAMENTE.
MORNING LINE: es un dato aparte. NO lo uses para calificar al caballo: la forma, la clase, la aptitud, el \
estilo y el pedigri se deducen solo de sus carreras y estadisticas, nunca de si es el favorito.

ESTRUCTURA DE SALIDA:
{
  "race": {"race_name": string|null, "race_number": integer|null, "track": string|null,
           "race_date": "YYYY-MM-DD"|null, "surface": one of @SURFACES@|null, "distance": string|null},
  "horses": [
    {
      "program_number": string, "name": string, "morning_line": string|null,
      "figures": [{"v": number, "s": "D"|"T"|"S"|null, "d": number|null}],
      "intl_ratings": [{"v": number, "s": "D"|"T"|"S"|null, "d": number|null}],
      "intl_rating_kind": one of @RATING_KINDS@,
      "foreign_origin": one of @FOREIGN_ORIGINS@, "foreign_level": one of @FOREIGN_LEVELS@,
      "form_trend": one of @FORM_TRENDS@, "class_rating": one of @CLASS_RATINGS@, "earnings_per_start": number|null,
      "running_style": one of @RUNNING_STYLES@, "quirin_points": number|null,
      "jockey": string|null, "is_regular_jockey": boolean, "jockey_switch": one of @JOCKEY_SWITCHES@,
      "jockey_win_pct": number|null,
      "trainer": string|null, "trainer_win_pct": number|null, "trainer_situation_pct": number|null,
      "combo_win_pct": number|null,
      "sire": string|null, "dam_sire": string|null,
      "ped_surface_fit": one of @FIT_LEVELS@, "ped_distance_fit": one of @FIT_LEVELS@,
      "ped_wet_fit": one of @FIT_LEVELS@,
      "track_aptitude": one of @TRACK_APTITUDES@, "mud_proven": boolean, "going_pref": one of @GOING_PREFS@,
      "dirt_starts": integer|null, "dirt_wins": integer|null, "turf_starts": integer|null, "turf_wins": integer|null,
      "dist_starts": integer|null, "dist_wins": integer|null, "wet_starts": integer|null, "wet_wins": integer|null,
      "days_since_last": integer|null, "starts_90d": integer|null,
      "workouts_30d": integer|null, "bullet_work": boolean,
      "blinkers_change": one of @BLINKERS_CHANGES@, "lasix_first": boolean,
      "post_position": integer|null, "trouble_last": boolean, "first_us_start": boolean,
      "weight_lbs": number|null, "last_weight_lbs": number|null, "scratched": boolean
    }
  ]
}

DEFINICION DE CADA CAMPO:
- program_number / name / jockey / trainer / sire / dam_sire: tal como aparecen en el programa. Incluye todos \
los caballos, tambien los scratched o also-eligible (scratched=true los retirados).
- morning_line: linea matutina en fraccion, p. ej. "5/2", "9-2", "even". Si no aparece, null.
- figures: TODAS las Beyer Speed Figures que muestre el DRF de sus carreras anteriores (hasta 10), de la MAS \
RECIENTE a la mas antigua, no solo la ultima. v = la figura; s = superficie de esa carrera ("D" arena, "T" grama, \
"S" sintetica); d = distancia en furlongs (1 milla = 8; 1 1/16 millas = 8.5; 1 1/8 = 9). Omite s o d \
(null) si no se leen. Una carrera sin figura no se incluye. Sin Beyers, [].
- intl_ratings / intl_rating_kind: los caballos que corrieron en Europa, Japon u otro pais y traen un rating \
(Timeform, RPR u otro) en vez de Beyer: sus ratings, con el mismo formato y orden que figures \
(v = el rating), y el tipo ("Timeform", "RPR", "Other"). Si no hay rating, [] y "None".
- foreign_origin: "Europe" o "Japan" si su ultima salida o su base de campana es en Europa/Japon; si no, "None". \
foreign_level: nivel de su mejor resultado reciente en el extranjero: "G1_Winner", "G1_Placed" (2do o 3ro en un \
G1), "Other". "None" si foreign_origin es "None".
- form_trend: analiza sus ultimas carreras (figuras y posiciones de llegada). "Ascending" = mejorando; \
"Plateau" = estable; "Descending" = empeorando.
- class_rating: "G1_Winner" = ha ganado un Grado 1; "Graded_Placed" = ha sido colocado en stakes graduados o \
corre en su nivel; "Class_Riser" = sube claramente de categoria. earnings_per_start: ganancias totales divididas \
entre sus salidas, en dolares, si el DRF las trae.
- running_style: "E" = puntero, "P" = presser/perseguidor, "S" = rematador (closer), segun el simbolo de estilo \
del DRF (E, E/P, P, S). quirin_points: los puntos de velocidad temprana (Quirin) de 0 a 8 si aparecen.
- jockey_switch: "Upgrade" si cambia a un jinete claramente mejor; "Downgrade" si a uno claramente inferior; \
"Neutral" en los demas casos. is_regular_jockey: true si monta quien lo monto en la mayoria de sus ultimas salidas. \
jockey_win_pct: porcentaje de victorias del jinete segun el DRF (numero, p. ej. 18 para 18%).
- trainer_win_pct: porcentaje de victorias del entrenador segun el DRF. trainer_situation_pct: porcentaje del \
entrenador en la situacion de este caballo que muestre el DRF (primera salida, tras descanso, grama, 2 anos...). \
combo_win_pct: porcentaje de la dupla jinete-entrenador si aparece.
- ped_surface_fit / ped_distance_fit / ped_wet_fit: adecuacion del pedigri (padre y abuelo materno) a LA \
SUPERFICIE, LA DISTANCIA y al terreno HUMEDO de hoy, segun las estadisticas de pedigri del DRF. "Strong" si son \
buenas, "Weak" si son malas, "Neutral" si son normales, "Unknown" si el DRF no trae esas estadisticas.
- track_aptitude: rendimiento en este hipodromo segun su record: "Proven_Winner" (ha ganado aqui), "Positive" \
(buenos resultados sin victoria), "Negative" (malos resultados), "First_Time" (nunca ha corrido aqui).
- mud_proven: true si tiene victorias o actuaciones fuertes en pista humeda/suave/pesada. going_pref: terreno que \
prefiere segun sus resultados: "Firm", "Soft", "Any" o "Unknown".
- dirt_/turf_/dist_/wet_ starts y wins: sus records (salidas y victorias) en arena, grama, a la distancia de hoy \
y en pista humeda, tal como los muestra el DRF.
- days_since_last: dias entre su ultima carrera y la fecha de hoy. starts_90d: salidas en los ultimos 90 dias.
- workouts_30d: numero de trabajos (works) de los ultimos 30 dias. bullet_work: true si algun trabajo reciente \
esta marcado como el mejor del dia (bullet).
- blinkers_change: "First_Time" si usa blinkers por primera vez, "Off" si se los quita, "None" en los demas \
casos. lasix_first: true si corre con Lasix por primera vez.
- post_position: puesto de salida. trouble_last: true si el comentario de su ultima carrera indica problemas \
(cerrado, abierto, golpeado, tropezo, frenado).
- first_us_start: true si es su primera carrera en EE.UU.
- weight_lbs: peso asignado en esta carrera. last_weight_lbs: peso en su ultima salida.

AHORRO DE TOKENS (muy importante): OMITE toda clave cuyo valor sea null, false, [], "None" o "Unknown". \
Solo escribe las claves con un dato real. Siempre incluye program_number y name de cada caballo.

REGLAS: no agregues claves extra; numeros como numeros (no strings); booleanos como true/false; conserva el \
orden del programa."""

_PROMPT_CATALOGS = {
    "@SURFACES@": SURFACES,
    "@RATING_KINDS@": RATING_KINDS,
    "@FOREIGN_ORIGINS@": FOREIGN_ORIGINS,
    "@FOREIGN_LEVELS@": FOREIGN_LEVELS,
    "@FORM_TRENDS@": FORM_TRENDS,
    "@CLASS_RATINGS@": CLASS_RATINGS,
    "@RUNNING_STYLES@": RUNNING_STYLES,
    "@JOCKEY_SWITCHES@": JOCKEY_SWITCHES,
    "@FIT_LEVELS@": FIT_LEVELS,
    "@TRACK_APTITUDES@": TRACK_APTITUDES,
    "@GOING_PREFS@": GOING_PREFS,
    "@BLINKERS_CHANGES@": BLINKERS_CHANGES,
}
SYSTEM_PROMPT = _PROMPT_TEMPLATE
for _token, _options in _PROMPT_CATALOGS.items():
    SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_token, str(list(_options)))


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


def _split_objects(text: str, start: int) -> list[str]:
    """Objetos {...} completos de nivel 1 dentro de una lista que empieza en `start` (tolera cortes)."""
    objects, depth, in_str, escape, begin = [], 0, False, False, None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and begin is not None:
                objects.append(text[begin : i + 1])
                begin = None
            elif depth < 0:
                break
        elif ch == "]" and depth == 0:
            break
    return objects


def _salvage(text: str) -> dict[str, Any] | None:
    """Recupera los caballos completos de una respuesta cortada o con JSON roto en alguna parte."""
    marker = re.search(r'"horses"\s*:\s*\[', text)
    if not marker:
        return None
    horses = []
    for chunk in _split_objects(text, marker.end()):
        try:
            item = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("name"):
            horses.append(item)
    if not horses:
        return None
    race: dict[str, Any] = {}
    race_match = re.search(r'"race"\s*:\s*(\{[^{}]*\})', text[: marker.start()])
    if race_match:
        try:
            race = json.loads(race_match.group(1))
        except json.JSONDecodeError:
            race = {}
    return {"race": race, "horses": horses}


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
    conditions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Envia PDFs y/o imagenes a Claude y devuelve {"race": {...}, "horses": [...]}.

    Args:
        files: lista de (nombre, contenido). Admite PDF, PNG, JPG, WEBP y GIF, mezclados.
            Si hay varias imagenes se tratan como paginas de la misma carrera, en orden.
        race_hint: nombre/numero de la carrera a extraer si el material trae varias.
        track: hipodromo (para evaluar la aptitud a la pista), p. ej. "Keeneland".
        pages: paginas a enviar de un PDF, p. ej. "12-19". Solo se puede usar con un unico PDF.
        conditions: condiciones de la carrera de hoy (surface, distance, track_condition, race_date) para
            que el pedigri, el descanso y las distancias se evaluen contra ellas.

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
    if conditions:
        labels = (("surface", "superficie"), ("distance", "distancia"), ("track_condition", "condicion de pista"),
                  ("race_date", "fecha"))
        today = ", ".join(f"{label} {conditions[key]}" for key, label in labels if conditions.get(key))
        if today:
            instructions.append(
                f"Condiciones de la carrera de HOY: {today}. Evalua el pedigri, las distancias, el descanso y la "
                "aptitud contra estas condiciones."
            )
    if race_hint:
        instructions.append(f"Si el material contiene varias carreras, extrae SOLO esta: {race_hint}.")
    instructions.append("Responde unicamente con el objeto JSON.")
    prompt = "\n".join(instructions)

    client = anthropic.Anthropic(api_key=api_key)
    last_error: Exception | None = None

    thinking_off = True  # evita que el razonamiento interno consuma el limite de tokens de salida
    for attempt in range(1, MAX_ATTEMPTS + 1):
        user_text = prompt
        if attempt > 1:
            user_text += (
                "\nIMPORTANTE: tu respuesta anterior no fue un JSON valido o se corto por ser demasiado larga. "
                "Devuelve SOLO el JSON minificado, en una sola linea, con un unico objeto por caballo "
                "y sin ningun texto adicional."
            )
        request = {
            "model": model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content + [{"type": "text", "text": user_text}]}],
        }
        try:
            if thinking_off:
                try:
                    response = client.messages.create(thinking={"type": "disabled"}, **request)
                except anthropic.BadRequestError as exc:
                    if "thinking" not in str(exc).lower():
                        raise
                    thinking_off = False  # el modelo no admite el parametro: se envia sin el
                    response = client.messages.create(**request)
            else:
                response = client.messages.create(**request)
        except anthropic.APIError as exc:
            raise ExtractionError(f"Error de la API de Anthropic: {exc}") from exc

        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        truncated = response.stop_reason == "max_tokens"
        if not truncated:
            try:
                return _normalize_result(_parse_json(text))
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        else:
            last_error = ValueError("La respuesta se corto por limite de tokens.")
        # Rescata los caballos completos en vez de perder lo ya pagado.
        recovered = _salvage(text)
        if recovered is not None:
            try:
                result = _normalize_result(recovered)
            except ValueError as exc:
                last_error = exc
            else:
                result["warnings"] = [
                    f"La respuesta llego incompleta o con un error de formato; se recuperaron {len(result['horses'])} "
                    "caballo(s). Compara con el programa y agrega los que falten en la tabla."
                ]
                return result
    sample = (text or "")[:300].replace("\n", " ")
    raise ExtractionError(
        f"No se pudo obtener un JSON valido: {last_error}. Inicio de la respuesta: {sample!r}"
    )


def extract_race_data_from_pdf(
    pdf_bytes: bytes,
    race_hint: str | None = None,
    track: str | None = None,
    pages: str | None = None,
    conditions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compatibilidad: extrae los datos de un unico PDF (o imagen) ya en memoria."""
    return extract_race_data_from_files(
        [("documento", pdf_bytes)], race_hint=race_hint, track=track, pages=pages, conditions=conditions
    )
