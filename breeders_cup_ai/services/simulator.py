"""Motor de handicap multifactorial + simulacion Monte Carlo.

Flujo:
    1. normalize_horse()      -> limpia/valida los datos de cada caballo.
    2. compute_adjustments()  -> calcula cada factor de handicap.
    3. run_monte_carlo()      -> 10,000 iteraciones con distribucion normal.
    4. simulate_race()        -> une todo y devuelve un DataFrame con probabilidad y EV.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes y catalogos
# ---------------------------------------------------------------------------
DEFAULT_ITERATIONS = 10_000
DEFAULT_SIGMA = 4.0
DEFAULT_BASE_FIGURE = 60.0

SURFACES = ("Dirt", "Turf", "Synthetic")
TRACK_CONDITIONS = ("Fast", "Firm", "Good", "Yielding", "Soft", "Muddy", "Sloppy", "Heavy")
WET_CONDITIONS = frozenset({"Yielding", "Soft", "Muddy", "Sloppy", "Heavy"})

FOREIGN_ORIGINS = ("None", "Europe", "Japan")
FOREIGN_LEVELS = ("None", "G1_Winner", "G1_Placed", "Other")
FORM_TRENDS = ("Ascending", "Plateau", "Descending")
CLASS_RATINGS = ("G1_Winner", "Graded_Placed", "Class_Riser")
JOCKEY_SWITCHES = ("Upgrade", "Neutral", "Downgrade")
TRACK_APTITUDES = ("Proven_Winner", "Positive", "Negative", "First_Time")
RUNNING_STYLES = ("E", "P", "S")

# Puntos de cada factor -------------------------------------------------------
FOREIGN_LEVEL_POINTS = {"G1_Winner": 4.0, "G1_Placed": 3.25, "Other": 2.5}
FORM_TREND_POINTS = {"Ascending": 2.0, "Plateau": 0.0, "Descending": -3.0}
CLASS_POINTS = {"G1_Winner": 1.5, "Graded_Placed": 0.0, "Class_Riser": -2.5}
JOCKEY_POINTS = {"Upgrade": 1.5, "Neutral": 0.0, "Downgrade": -2.0}
TRACK_APTITUDE_POINTS = {"Proven_Winner": 2.0, "Positive": 1.0, "Negative": -2.0, "First_Time": 0.0}
WET_TRACK_BONUS = 2.5
PACE_CROWDED_THRESHOLD = 3
PACE_CROWDED_E_PENALTY = -3.5
PACE_CROWDED_S_BONUS = 3.0
PACE_LONE_E_BONUS = 2.5
WEIGHT_STEP_LBS = 3
WEIGHT_STEP_POINTS = -0.5

# Jinetes de elite (inicial del nombre, apellido) en minusculas, sin acentos.
ELITE_JOCKEYS = frozenset(
    {
        ("i", "ortiz"),  # Irad Ortiz Jr.
        ("f", "prat"),  # Flavien Prat
        ("r", "moore"),  # Ryan Moore
        ("w", "buick"),  # William Buick
    }
)

ADJUSTMENT_COLUMNS = (
    "adj_foreign",
    "adj_form",
    "adj_class",
    "adj_jockey",
    "adj_track",
    "adj_condition",
    "adj_pace",
    "adj_weight",
)

HORSE_FIELDS = (
    "program_number",
    "name",
    "morning_line",
    "base_speed_figure",
    "foreign_origin",
    "foreign_level",
    "dirt_debut",
    "form_trend",
    "class_rating",
    "jockey",
    "is_regular_jockey",
    "jockey_switch",
    "track_aptitude",
    "mud_proven",
    "running_style",
    "weight_lbs",
    "last_weight_lbs",
    "scratched",
)

_ENUM_ALIASES = {
    "form_trend": {
        "improving": "Ascending",
        "up": "Ascending",
        "rising": "Ascending",
        "flat": "Plateau",
        "steady": "Plateau",
        "declining": "Descending",
        "down": "Descending",
        "falling": "Descending",
    },
    "running_style": {
        "early": "E",
        "speed": "E",
        "front": "E",
        "frontrunner": "E",
        "presser": "P",
        "stalker": "P",
        "pressing": "P",
        "closer": "S",
        "sustained": "S",
        "deepcloser": "S",
        "come from behind": "S",
    },
    "foreign_origin": {
        "europa": "Europe",
        "eu": "Europe",
        "uk": "Europe",
        "ireland": "Europe",
        "france": "Europe",
        "england": "Europe",
        "japon": "Japan",
        "jp": "Japan",
        "no": "None",
        "ninguno": "None",
        "usa": "None",
        "us": "None",
    },
}


# ---------------------------------------------------------------------------
# Utilidades de normalizacion
# ---------------------------------------------------------------------------
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def _as_float(value: Any, low: float | None = None, high: float | None = None) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    else:
        match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
        if not match:
            return None
        number = float(match.group(0).replace(",", "."))
    if math.isnan(number) or math.isinf(number):
        return None
    if low is not None:
        number = max(low, number)
    if high is not None:
        number = min(high, number)
    return number


def _as_bool(value: Any, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "si", "sí", "1", "x"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return default


def _key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _as_enum(value: Any, options: tuple[str, ...], default: str, field: str | None = None) -> str:
    if _is_missing(value):
        return default
    lookup = {_key(option): option for option in options}
    raw = str(value).strip()
    candidate = _key(raw)
    if candidate in lookup:
        return lookup[candidate]
    aliases = _ENUM_ALIASES.get(field or "", {})
    for alias, target in aliases.items():
        if candidate == _key(alias):
            return target
    if field == "running_style" and raw[:1].upper() in options:
        return raw[:1].upper()
    return default


def _as_text(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()


def normalize_horse(raw: dict[str, Any]) -> dict[str, Any]:
    """Convierte un diccionario (de Claude o del editor) en un caballo valido."""
    horse = {
        "program_number": _as_text(raw.get("program_number")),
        "name": _as_text(raw.get("name")),
        "morning_line": _as_text(raw.get("morning_line")),
        "base_speed_figure": _as_float(raw.get("base_speed_figure"), 0.0, 120.0),
        "foreign_origin": _as_enum(raw.get("foreign_origin"), FOREIGN_ORIGINS, "None", "foreign_origin"),
        "foreign_level": _as_enum(raw.get("foreign_level"), FOREIGN_LEVELS, "None"),
        "dirt_debut": _as_bool(raw.get("dirt_debut")),
        "form_trend": _as_enum(raw.get("form_trend"), FORM_TRENDS, "Plateau", "form_trend"),
        "class_rating": _as_enum(raw.get("class_rating"), CLASS_RATINGS, "Graded_Placed"),
        "jockey": _as_text(raw.get("jockey")),
        "is_regular_jockey": _as_bool(raw.get("is_regular_jockey"), default=True),
        "jockey_switch": _as_enum(raw.get("jockey_switch"), JOCKEY_SWITCHES, "Neutral"),
        "track_aptitude": _as_enum(raw.get("track_aptitude"), TRACK_APTITUDES, "First_Time"),
        "mud_proven": _as_bool(raw.get("mud_proven")),
        "running_style": _as_enum(raw.get("running_style"), RUNNING_STYLES, "P", "running_style"),
        "weight_lbs": _as_float(raw.get("weight_lbs"), 0.0),
        "last_weight_lbs": _as_float(raw.get("last_weight_lbs"), 0.0),
        "scratched": _as_bool(raw.get("scratched")),
    }
    if horse["foreign_origin"] != "None" and horse["foreign_level"] == "None":
        horse["foreign_level"] = "Other"
    return horse


# ---------------------------------------------------------------------------
# Morning line -> dividendo decimal y probabilidad implicita
# ---------------------------------------------------------------------------
def parse_morning_line(value: Any) -> float | None:
    """Convierte el Morning Line a dividendo decimal (retorno total por 1 unidad).

    Formatos aceptados: "5/2", "9-2", "5:2", "3" (se interpreta como 3-1),
    3.5 (3.5-1), "even"/"evs". Ejemplo: 5/2 -> 3.5 ; 9-5 -> 2.8 ; even -> 2.0.
    """
    if _is_missing(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number + 1.0 if number > 0 else None
    text = str(value).strip().lower().replace(" ", "")
    if text in {"even", "evens", "evs", "ev"}:
        return 2.0
    ratio = re.fullmatch(r"(\d+(?:\.\d+)?)[/\-:](\d+(?:\.\d+)?)", text)
    if ratio:
        numerator, denominator = float(ratio.group(1)), float(ratio.group(2))
        if denominator == 0:
            return None
        return numerator / denominator + 1.0
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        number = float(text)
        return number + 1.0 if number > 0 else None
    return None


def implied_probability(decimal_odds: float | None) -> float | None:
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    return 1.0 / decimal_odds


def expected_value(win_probability: float, decimal_odds: float | None) -> float | None:
    """EV = (Probabilidad_Simulada * Dividendo_Decimal) - 1."""
    if decimal_odds is None:
        return None
    return win_probability * decimal_odds - 1.0


# ---------------------------------------------------------------------------
# Factores de handicap
# ---------------------------------------------------------------------------
def _jockey_key(name: str) -> tuple[str, str] | None:
    cleaned = unicodedata.normalize("NFKD", name or "")
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch)).lower()
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    tokens = [t for t in cleaned.split() if t not in {"jr", "sr", "ii", "iii"}]
    if not tokens:
        return None
    if len(tokens) >= 2 and len(tokens[-1]) == 1:  # formato "Ortiz, I."
        return tokens[-1], tokens[0]
    if len(tokens) == 1:
        return "", tokens[0]
    return tokens[0][0], tokens[-1]


def is_elite_jockey(name: str) -> bool:
    key = _jockey_key(name)
    return key in ELITE_JOCKEYS if key else False


def effective_jockey_switch(horse: dict[str, Any]) -> str:
    """Califica el cambio de jinete.

    - 'Upgrade' solo suma si el jinete que monta es de elite (Ortiz, Prat, Moore, Buick).
    - Si no es el jinete habitual y el que monta es de elite, se califica 'Upgrade'
      aunque la extraccion haya marcado 'Neutral'.
    """
    switch = horse["jockey_switch"]
    elite = is_elite_jockey(horse["jockey"])
    if switch == "Upgrade" and not elite:
        return "Neutral"
    if switch == "Neutral" and not horse["is_regular_jockey"] and elite:
        return "Upgrade"
    return switch


def count_early_speed(horses: list[dict[str, Any]]) -> int:
    return sum(1 for h in horses if h["running_style"] == "E")


def compute_adjustments(horse: dict[str, Any], surface: str, track_condition: str, early_count: int) -> dict[str, float]:
    """Devuelve los puntos de cada factor de handicap para un caballo."""
    adjustments = {column: 0.0 for column in ADJUSTMENT_COLUMNS}

    # 2. Factor extranjero (Europa / Japon)
    if horse["foreign_origin"] in {"Europe", "Japan"}:
        points = FOREIGN_LEVEL_POINTS.get(horse["foreign_level"], FOREIGN_LEVEL_POINTS["Other"])
        if surface == "Turf":
            adjustments["adj_foreign"] = points
        elif surface == "Dirt" and horse["dirt_debut"]:
            adjustments["adj_foreign"] = -points

    # 3. Tendencia de forma
    adjustments["adj_form"] = FORM_TREND_POINTS[horse["form_trend"]]

    # 4. Nivel de lote
    adjustments["adj_class"] = CLASS_POINTS[horse["class_rating"]]

    # 5. Jinete y cambio
    adjustments["adj_jockey"] = JOCKEY_POINTS[effective_jockey_switch(horse)]

    # 6. Aptitud a la pista / trazado
    adjustments["adj_track"] = TRACK_APTITUDE_POINTS[horse["track_aptitude"]]

    # 7. Condicion de pista: bonus unico (no se acumula) en pista suave/humeda
    if track_condition in WET_CONDITIONS:
        european_on_turf = surface == "Turf" and horse["foreign_origin"] == "Europe"
        if european_on_turf or horse["mud_proven"]:
            adjustments["adj_condition"] = WET_TRACK_BONUS

    # 8. Planteamiento de carrera (pace scenario)
    style = horse["running_style"]
    if early_count >= PACE_CROWDED_THRESHOLD:
        if style == "E":
            adjustments["adj_pace"] = PACE_CROWDED_E_PENALTY
        elif style == "S":
            adjustments["adj_pace"] = PACE_CROWDED_S_BONUS
    elif early_count == 1 and style == "E":
        adjustments["adj_pace"] = PACE_LONE_E_BONUS

    # 9. Diferencial de peso: -0.5 por cada 3 lbs adicionales vs la ultima carrera
    weight, last_weight = horse["weight_lbs"], horse["last_weight_lbs"]
    if weight and last_weight and weight > last_weight:
        steps = int((weight - last_weight) // WEIGHT_STEP_LBS)
        adjustments["adj_weight"] = steps * WEIGHT_STEP_POINTS

    return adjustments


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def _draw_performances(
    ratings: np.ndarray | list[float],
    sigma: float,
    iterations: int,
    seed: int | None,
) -> np.ndarray:
    """Valida los parametros y devuelve la matriz (iteraciones x caballos) de rendimientos."""
    values = np.asarray(ratings, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Se necesitan al menos 2 caballos para simular.")
    if sigma <= 0:
        raise ValueError("sigma debe ser mayor que 0.")
    if iterations < 1:
        raise ValueError("iterations debe ser al menos 1.")

    rng = np.random.default_rng(seed)
    return rng.normal(loc=values, scale=sigma, size=(int(iterations), values.size))


def win_probabilities(performances: np.ndarray) -> np.ndarray:
    """Probabilidad de ganar de cada caballo (el mayor rendimiento gana cada iteracion)."""
    winners = performances.argmax(axis=1)
    wins = np.bincount(winners, minlength=performances.shape[1])
    return wins / float(performances.shape[0])


def finish_statistics(performances: np.ndarray, places: int = 4, top_combos: int = 3) -> dict[str, Any]:
    """Distribucion del orden de llegada a partir de las mismas iteraciones.

    Devuelve:
        position_probs: matriz (caballos x places) con P(llegar en cada posicion).
        avg_finish: posicion media de llegada de cada caballo (1 = gana).
        combos: las combinaciones exactas de los primeros `places` mas frecuentes,
            como lista de (indices_de_caballos, frecuencia).
    """
    iterations, count = performances.shape
    places = min(places, count)
    # Orden de llegada por iteracion: indices de caballos de mayor a menor rendimiento.
    order = np.argsort(-performances, axis=1, kind="stable")
    position_probs = np.zeros((count, places))
    for place in range(places):
        position_probs[:, place] = np.bincount(order[:, place], minlength=count) / float(iterations)

    finish_position = np.empty_like(order)
    rows = np.arange(iterations)[:, None]
    finish_position[rows, order] = np.arange(1, count + 1)[None, :]
    avg_finish = finish_position.mean(axis=0)

    # Clave de cada combinacion: numero en base `count` de los primeros `places` caballos.
    keys = np.zeros(iterations, dtype=np.int64)
    for place in range(places):
        keys = keys * count + order[:, place]
    unique_keys, key_counts = np.unique(keys, return_counts=True)
    best = np.argsort(-key_counts, kind="stable")[:top_combos]
    combos: list[tuple[tuple[int, ...], float]] = []
    for position in best:
        key = int(unique_keys[position])
        members: list[int] = []
        for _ in range(places):
            members.append(key % count)
            key //= count
        combos.append((tuple(reversed(members)), float(key_counts[position]) / iterations))
    return {"position_probs": position_probs, "avg_finish": avg_finish, "combos": combos}


def run_monte_carlo(
    ratings: np.ndarray | list[float],
    sigma: float = DEFAULT_SIGMA,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int | None = None,
) -> np.ndarray:
    """Simula `iterations` carreras; cada caballo rinde Normal(rating, sigma).

    Gana el caballo con mayor rendimiento en cada iteracion. Devuelve la
    probabilidad de ganar de cada caballo (suma = 1).
    """
    return win_probabilities(_draw_performances(ratings, sigma, iterations, seed))


def simulate_race(
    horses: list[dict[str, Any]],
    surface: str,
    track_condition: str,
    iterations: int = DEFAULT_ITERATIONS,
    sigma: float = DEFAULT_SIGMA,
    seed: int | None = None,
) -> pd.DataFrame:
    """Ejecuta el handicap completo y la simulacion. Devuelve un DataFrame ordenado
    por probabilidad de ganar (descendente).

    `horse_key` es la posicion del caballo en la lista original recibida.
    """
    if surface not in SURFACES:
        raise ValueError(f"Superficie invalida: {surface!r}. Opciones: {SURFACES}")
    if track_condition not in TRACK_CONDITIONS:
        raise ValueError(f"Condicion invalida: {track_condition!r}. Opciones: {TRACK_CONDITIONS}")

    normalized = [(index, normalize_horse(h)) for index, h in enumerate(horses)]
    active = [(index, h) for index, h in normalized if not h["scratched"] and h["name"]]
    if len(active) < 2:
        raise ValueError("Se necesitan al menos 2 caballos activos (con nombre y sin scratch).")

    early_count = count_early_speed([h for _, h in active])
    known_figures = [h["base_speed_figure"] for _, h in active if h["base_speed_figure"] is not None]
    fallback_figure = float(np.mean(known_figures)) if known_figures else DEFAULT_BASE_FIGURE

    rows: list[dict[str, Any]] = []
    for index, horse in active:
        adjustments = compute_adjustments(horse, surface, track_condition, early_count)
        base = horse["base_speed_figure"] if horse["base_speed_figure"] is not None else fallback_figure
        rows.append(
            {
                "horse_key": index,
                "program_number": horse["program_number"],
                "name": horse["name"],
                "morning_line": horse["morning_line"],
                "base_speed_figure": round(base, 2),
                **{column: round(value, 2) for column, value in adjustments.items()},
                "final_adjusted_rating": round(base + sum(adjustments.values()), 2),
                "decimal_odds": parse_morning_line(horse["morning_line"]),
            }
        )

    performances = _draw_performances(
        [row["final_adjusted_rating"] for row in rows], sigma, iterations, seed
    )
    probabilities = win_probabilities(performances)
    finish = finish_statistics(performances)
    position_probs = finish["position_probs"]

    for position, (row, probability) in enumerate(zip(rows, probabilities)):
        row["p_top2"] = float(position_probs[position, :2].sum())
        row["p_top3"] = float(position_probs[position, :3].sum())
        row["p_top4"] = float(position_probs[position, :4].sum())
        row["avg_finish"] = float(finish["avg_finish"][position])
        odds = row["decimal_odds"]
        row["win_probability"] = float(probability)
        row["implied_probability"] = implied_probability(odds)
        ev = expected_value(float(probability), odds)
        row["expected_value"] = ev
        row["is_positive_ev"] = bool(ev is not None and ev > 0)

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["win_probability", "final_adjusted_rating"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)

    # Orden de llegada probable (posicion media) y combinaciones exactas mas frecuentes.
    horse_keys = [row["horse_key"] for row in rows]
    predicted = result.sort_values(["avg_finish", "win_probability"], ascending=[True, False], kind="mergesort")
    result.attrs["finish_order"] = [int(k) for k in predicted["horse_key"].head(4)]
    result.attrs["top_combos"] = [
        ([int(horse_keys[i]) for i in members], frequency) for members, frequency in finish["combos"]
    ]
    result.attrs["early_count"] = int(early_count)
    return result