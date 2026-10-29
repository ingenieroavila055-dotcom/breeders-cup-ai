"""Criterio experto: un handicapper (Claude) lee el resultado del modelo y los datos del DRF y opina.

El modelo cuantitativo calcula; este paso razona. Puede coincidir o discrepar del modelo, leer lo que
los numeros no capturan (tendencia de las figuras, forma de correr, ritmo, distancia, recorrido) y proponer
un pequeno ajuste por caballo (maximo +/- 3 puntos) con su explicacion. El Morning Line se le entrega como
una senal de mercado mas: sirve para detectar discrepancias, no para calificar. El usuario decide si aplica
el ajuste y vuelve a simular.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic

from services import figures as fg
from services.simulator import ADJUSTMENT_COLUMNS, ADJUSTMENT_LABELS, INTUITION_LIMIT

MAX_OUTPUT_TOKENS = 4_000
DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """Eres un handicapper veterano de carreras de pura sangre, experto en Keeneland (arena y grama), \
en cifras Beyer, ratings Timeform, ritmo (pace), recorrido y distancias. Hablas en espanol, directo y con criterio.

Recibes el resultado de un modelo cuantitativo (ranking, probabilidades, ajustes por factor), los datos del DRF de \
cada caballo y la proyeccion de ritmo. Tu trabajo es RAZONAR como lo haria un experto, no repetir la tabla:
- Lee las series de figuras: tendencia, picos aislados, regularidad, si una figura fue en otra superficie o distancia.
- Evalua el ritmo y el recorrido: quien queda bien ubicado, quien se queda sin pista, quien depende de que alguien \
  apure. Considera el puesto de salida y la distancia.
- Para extranjeros: interpreta su rating y nivel (grupo/grado) frente a los americanos y su terreno preferido.
- Usa tu intuicion sobre lo que los numeros no capturan, pero sin inventar datos: solo hablas de lo que esta en \
  la informacion recibida. Si un dato falta, dilo.
- Puedes discrepar del modelo. El Morning Line es una senal del mercado mas: sirve para decir "el mercado ve \
  distinto, y por esto" o "aqui hay valor", pero no es argumento para calificar a un caballo.
- Se honesto con la incertidumbre: nadie acierta siempre; sugiere cuando una carrera es abierta.

Devuelve UN objeto JSON minificado, sin markdown ni texto extra:
{"thesis": string (4 a 6 frases: como se corre la carrera, quien gana y por que, quien amenaza),
 "pace_plan": string (2 a 3 frases sobre el planteamiento tactico),
 "top4": [numeros de programa en tu orden de llegada probable, hasta 4],
 "horses": [{"program_number": string, "adjust": number entre -3 y 3 (0 si estas de acuerdo con el modelo; \
   positivo si lo ves mejor que el modelo, negativo si peor), "stance": "A favor"|"Neutral"|"En contra", \
   "comment": string (1 o 2 frases con la razon concreta)}],
 "value_note": string (1 o 2 frases: donde ves valor o sobreapuesta frente al Morning Line, o "" si no aplica),
 "caveats": [strings cortos con limites de los datos]}
Incluye un objeto en "horses" por cada caballo recibido. Usa ajustes pequenos: solo +/-2 o mas si tienes una razon \
clara y concreta."""


class ExpertError(RuntimeError):
    """Error controlado al pedir el criterio experto."""


def build_payload(
    table: Any, horses: list[dict[str, Any]], race: dict[str, Any], pace: dict[str, Any] | None
) -> dict[str, Any]:
    """Resumen compacto y sin ruido de la simulacion para el handicapper."""
    by_key = {int(k): h for k, h in enumerate(horses)}
    rows = []
    for record in table.to_dict("records"):
        horse = by_key.get(int(record["horse_key"]), {})
        adjustments = {
            ADJUSTMENT_LABELS[c]: round(float(record[c]), 1)
            for c in ADJUSTMENT_COLUMNS
            if c != "adj_intuition" and abs(float(record.get(c) or 0.0)) >= 0.05
        }
        rows.append(
            {
                "numero": record["program_number"],
                "nombre": record["name"],
                "modelo": {
                    "puesto": int(record["rank"]),
                    "prob_ganar": round(float(record["win_probability"]), 3),
                    "base": record["base_speed_figure"],
                    "rating_final": record["final_adjusted_rating"],
                    "veredicto": record["verdict"],
                    "confianza": record["confidence_level"],
                    "ajustes": adjustments,
                },
                "morning_line": record["morning_line"] or None,
                "figuras_beyer": fg.format_figures(fg.parse_figures(horse.get("figures"))) or None,
                "ratings_intl": fg.format_figures(fg.parse_figures(horse.get("intl_ratings"))) or None,
                "tipo_rating_intl": horse.get("intl_rating_kind"),
                "pace_early": horse.get("early_pace"),
                "pace_late": horse.get("late_pace"),
                "quirin": horse.get("quirin_points"),
                "estilo": (pace or {}).get("styles", {}).get(int(record["horse_key"])) or horse.get("running_style"),
                "puesto_salida": horse.get("post_position"),
                "jinete": horse.get("jockey"),
                "jinete_pct": horse.get("jockey_win_pct"),
                "entrenador": horse.get("trainer"),
                "entrenador_pct": horse.get("trainer_win_pct"),
                "dias_descanso": horse.get("days_since_last"),
                "origen": horse.get("foreign_origin"),
                "primera_en_eeuu": horse.get("first_us_start"),
                "problemas_ultima": horse.get("trouble_last"),
                "no_disponible": record.get("missing") or None,
                "notas_modelo": record.get("notes") or None,
            }
        )
    return {
        "carrera": {k: race.get(k) for k in ("race_name", "surface", "distance", "track_condition", "bias_speed", "bias_rail")},
        "ritmo_proyectado": (pace or {}).get("title"),
        "caballos": rows,
    }


def _parse(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("La respuesta no contiene un objeto JSON.")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("El JSON no es un objeto.")
    return data


def normalize_opinion(data: dict[str, Any], valid_numbers: set[str]) -> dict[str, Any]:
    """Limpia la respuesta: ajustes acotados, solo caballos validos, textos como texto."""
    horses: dict[str, dict[str, Any]] = {}
    for item in data.get("horses") or []:
        if not isinstance(item, dict):
            continue
        number = str(item.get("program_number", "")).strip()
        if number not in valid_numbers:
            continue
        try:
            adjust = float(item.get("adjust", 0.0))
        except (TypeError, ValueError):
            adjust = 0.0
        adjust = max(-INTUITION_LIMIT, min(INTUITION_LIMIT, adjust))
        stance = item.get("stance") if item.get("stance") in ("A favor", "Neutral", "En contra") else "Neutral"
        horses[number] = {"adjust": round(adjust, 1), "stance": stance, "comment": str(item.get("comment") or "").strip()}
    top4 = [str(n).strip() for n in (data.get("top4") or []) if str(n).strip() in valid_numbers][:4]
    caveats = [str(c).strip() for c in (data.get("caveats") or []) if str(c).strip()][:5]
    return {
        "thesis": str(data.get("thesis") or "").strip(),
        "pace_plan": str(data.get("pace_plan") or "").strip(),
        "value_note": str(data.get("value_note") or "").strip(),
        "top4": top4,
        "horses": horses,
        "caveats": caveats,
    }


def ask_expert(
    table: Any, horses: list[dict[str, Any]], race: dict[str, Any], pace: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Pide el criterio experto. Devuelve la opinion normalizada (con 'usage')."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ExpertError("Falta ANTHROPIC_API_KEY.")
    payload = build_payload(table, horses, race, pace)
    valid = {str(h["numero"]) for h in payload["caballos"]}
    client = anthropic.Anthropic(api_key=api_key)
    request = {
        "model": model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
    }
    try:
        try:
            response = client.messages.create(thinking={"type": "disabled"}, **request)
        except anthropic.BadRequestError as exc:
            if "thinking" not in str(exc).lower():
                raise
            response = client.messages.create(**request)
    except anthropic.APIError as exc:
        raise ExpertError(f"Error de la API de Anthropic: {exc}") from exc
    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    try:
        opinion = normalize_opinion(_parse(text), valid)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ExpertError(f"No se pudo leer la respuesta del experto: {exc}") from exc
    if not opinion["thesis"] and not opinion["horses"]:
        raise ExpertError("La respuesta del experto vino vacia.")
    usage = getattr(response, "usage", None)
    opinion["usage"] = {
        "input": int(getattr(usage, "input_tokens", 0) or 0),
        "output": int(getattr(usage, "output_tokens", 0) or 0),
    }
    return opinion


def intuition_adjustments(opinion: dict[str, Any]) -> dict[str, float]:
    """Ajustes por numero de programa listos para simulate_race(intuition=...)."""
    return {number: info["adjust"] for number, info in opinion.get("horses", {}).items() if info["adjust"]}
