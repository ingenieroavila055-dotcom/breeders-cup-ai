"""Series de figuras de velocidad (Beyer) y ratings internacionales (Timeform, RPR...).

Convierte el texto del editor en listas de figuras, calcula la base de rating de un caballo
(promedio ponderado por recencia, pico reciente y mejor figura en las condiciones de hoy),
su consistencia, su tendencia y el riesgo de "rebote".

Formato de texto de una serie, de la mas reciente a la mas antigua:
    "92 T 8; 88 D 8.5; 95 T 7"      figura, superficie (D/T/S) y distancia en furlongs, opcionales
"""

from __future__ import annotations

import re
import statistics
from typing import Any

SURFACE_CODES = {"D": "Dirt", "T": "Turf", "S": "Synthetic"}
SURFACE_LETTERS = {value: key for key, value in SURFACE_CODES.items()}

MAX_SERIES = 10
RECENCY_WINDOW = 8
PEAK_WINDOW = 6
TREND_WINDOW = 5
TREND_SLOPE_POINTS = 1.5  # puntos por carrera para considerar una tendencia ascendente/descendente
BOUNCE_POINTS = 6.0

_TOKEN = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*([DTSdts])?\s*(\d{1,2}(?:\.\d+)?)?")


def _surface_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    letter = text[:1].upper()
    return SURFACE_CODES.get(letter)


def parse_figures(source: Any) -> list[dict[str, Any]]:
    """Texto o lista de dicts {v, s, d} -> lista de figuras [{"v", "s", "d"}] (mas reciente primero)."""
    figures: list[dict[str, Any]] = []
    if source is None:
        return figures
    if isinstance(source, (list, tuple)):
        for item in source:
            if isinstance(item, dict):
                try:
                    value = float(item.get("v", item.get("value", item.get("figure"))))
                except (TypeError, ValueError):
                    continue
                distance = item.get("d", item.get("distance_f"))
                try:
                    distance = float(distance) if distance not in (None, "") else None
                except (TypeError, ValueError):
                    distance = None
                figures.append({"v": value, "s": _surface_name(item.get("s", item.get("surface"))), "d": distance})
            elif isinstance(item, (int, float)):
                figures.append({"v": float(item), "s": None, "d": None})
        return figures[:MAX_SERIES]

    text = str(source).strip()
    if not text or text.lower() in {"nan", "none", "n/d"}:
        return figures
    for part in re.split(r"[;,\n|]+", text):
        match = _TOKEN.fullmatch(part.strip())
        if not match:
            continue
        value = float(match.group(1))
        surface = SURFACE_CODES.get((match.group(2) or "").upper())
        distance = float(match.group(3)) if match.group(3) else None
        figures.append({"v": value, "s": surface, "d": distance})
    return figures[:MAX_SERIES]


def _num(value: float) -> str:
    return f"{value:g}"


def format_figures(figures: list[dict[str, Any]]) -> str:
    parts = []
    for figure in figures[:MAX_SERIES]:
        piece = _num(figure["v"])
        if figure.get("s"):
            piece += f" {SURFACE_LETTERS[figure['s']]}"
        if figure.get("d") is not None:
            piece += f" {_num(figure['d'])}"
        parts.append(piece)
    return "; ".join(parts)


def distance_to_furlongs(text: Any) -> float | None:
    """'6 furlongs' -> 6.0 ; '1 1/16 miles' -> 8.5 ; '1 mile' -> 8.0 ; '5.5f' -> 5.5."""
    if text is None:
        return None
    raw = str(text).strip().lower()
    if not raw:
        return None
    mixed = re.search(r"(\d+)\s+(\d+)\s*/\s*(\d+)", raw)
    fraction = re.search(r"(?<!\d)(\d+)\s*/\s*(\d+)", raw)
    plain = re.search(r"\d+(?:\.\d+)?", raw)
    if mixed:
        amount = int(mixed.group(1)) + int(mixed.group(2)) / int(mixed.group(3))
    elif fraction:
        amount = int(fraction.group(1)) / int(fraction.group(2))
    elif plain:
        amount = float(plain.group(0))
    else:
        return None
    if re.search(r"mile|\bmi\b|\d\s*m\b", raw):
        return round(amount * 8.0, 3)
    if re.search(r"furlong|\bf\b|\d\s*f\b|fur", raw):
        return round(amount, 3)
    return round(amount * 8.0, 3) if amount <= 3 else round(amount, 3)


def convert_international(figures: list[dict[str, Any]], factor: float, offset: float) -> list[dict[str, Any]]:
    """Rating internacional -> equivalente en escala Beyer: valor * factor + offset."""
    return [{**figure, "v": figure["v"] * factor + offset} for figure in figures]


def combined_series(
    beyers: list[dict[str, Any]], international: list[dict[str, Any]], factor: float, offset: float
) -> list[dict[str, Any]]:
    """Serie unica en escala Beyer: primero los Beyers y despues los ratings internacionales convertidos."""
    return list(beyers) + convert_international(international, factor, offset)


def _slope(values_chronological: list[float]) -> float:
    count = len(values_chronological)
    mean_x = (count - 1) / 2.0
    mean_y = sum(values_chronological) / count
    numerator = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(values_chronological))
    denominator = sum((i - mean_x) ** 2 for i in range(count))
    return numerator / denominator if denominator else 0.0


def analyze_series(
    series: list[dict[str, Any]],
    surface: str,
    distance_f: float | None,
    decay: float,
    blend: tuple[float, float, float],
) -> dict[str, Any]:
    """Analiza una serie (mas reciente primero, ya en escala Beyer).

    Devuelve base (None si no hay figuras), n, std, slope, trend, bounce y las tres componentes.
    """
    values = [figure["v"] for figure in series]
    count = len(values)
    if count == 0:
        return {"base": None, "n": 0, "std": None, "slope": None, "trend": None, "bounce": False}

    window = values[:RECENCY_WINDOW]
    weights = [decay**k for k in range(len(window))]
    recency = sum(w * v for w, v in zip(weights, window)) / sum(weights)

    best = sorted(values[:PEAK_WINDOW], reverse=True)[:2]
    peak = sum(best) / len(best)

    matching = [
        figure["v"]
        for figure in series
        if figure.get("s") == surface
        and (distance_f is None or figure.get("d") is None or abs(figure["d"] - distance_f) <= 1.0)
    ]
    condition = max(matching) if matching else peak

    total = sum(blend) or 1.0
    base = (blend[0] * recency + blend[1] * peak + blend[2] * condition) / total

    std = statistics.pstdev(values[:PEAK_WINDOW]) if count >= 3 else None
    slope = _slope(list(reversed(values[:TREND_WINDOW]))) if count >= 3 else None
    trend = None
    if slope is not None:
        trend = "Ascending" if slope >= TREND_SLOPE_POINTS else "Descending" if slope <= -TREND_SLOPE_POINTS else "Plateau"
    bounce = False
    if count >= 3:
        previous = values[1:4]
        bounce = values[0] >= sum(previous) / len(previous) + BOUNCE_POINTS and values[0] >= max(values[1:])

    return {
        "base": base,
        "n": count,
        "std": std,
        "slope": slope,
        "trend": trend,
        "bounce": bounce,
        "recency": recency,
        "peak": peak,
        "condition": condition,
    }


def distance_fit_points(series: list[dict[str, Any]], distance_f: float | None) -> float:
    """+1 si sus mejores figuras se lograron cerca de la distancia de hoy; -1 si lejos; 0 si no se sabe."""
    if distance_f is None:
        return 0.0
    known = [f for f in sorted(series, key=lambda f: f["v"], reverse=True)[:3] if f.get("d") is not None]
    if not known:
        return 0.0
    nearest = min(abs(f["d"] - distance_f) for f in known)
    if nearest <= 0.75:
        return 1.0
    if nearest >= 1.5:
        return -1.0
    return 0.0
