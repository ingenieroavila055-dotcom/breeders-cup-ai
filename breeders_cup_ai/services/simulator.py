"""Motor de handicap multifactorial + simulacion Monte Carlo.

Flujo:
    1. normalize_horse()      -> limpia/valida los datos de cada caballo.
    2. analyze / compute_adjustments() -> series de figuras + cada factor de handicap.
    3. run_monte_carlo()      -> 10,000 iteraciones con distribucion normal (dispersion por caballo).
    4. simulate_race()        -> une todo: probabilidad, EV, confianza y veredicto.

El Morning Line NO interviene en la calificacion del caballo: solo se usa al final para el EV.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from services import figures as fg
from services import pace as pace_mod

# ---------------------------------------------------------------------------
# Constantes y catalogos
# ---------------------------------------------------------------------------
DEFAULT_ITERATIONS = 10_000
DEFAULT_SIGMA = 4.0
DEFAULT_BASE_FIGURE = 60.0

SURFACES = ("Dirt", "Turf", "Synthetic")
TRACK_CONDITIONS = ("Fast", "Firm", "Good", "Yielding", "Soft", "Muddy", "Sloppy", "Heavy")
WET_CONDITIONS = frozenset({"Yielding", "Soft", "Muddy", "Sloppy", "Heavy"})

FOREIGN_ORIGINS = ("None", "Europe", "Japan", "Other")
FOREIGN_LEVELS = ("None", "G1_Winner", "G1_Placed", "Other")
FORM_TRENDS = ("Ascending", "Plateau", "Descending")
CLASS_RATINGS = ("G1_Winner", "Graded_Placed", "Class_Riser")
JOCKEY_SWITCHES = ("Upgrade", "Neutral", "Downgrade")
TRACK_APTITUDES = ("Proven_Winner", "Positive", "Negative", "First_Time")
RUNNING_STYLES = ("E", "P", "S")
FIT_LEVELS = ("Unknown", "Strong", "Neutral", "Weak")
BLINKERS_CHANGES = ("None", "First_Time", "Off")
GOING_PREFS = ("Unknown", "Firm", "Soft", "Any")
RATING_KINDS = ("None", "Timeform", "RPR", "Other")
BIAS_SPEED_OPTIONS = ("Neutral", "Speed", "Closers")  # sesgo del dia (lo ingresa el usuario)
BIAS_RAIL_OPTIONS = ("Neutral", "Inside", "Outside")

# Puntos de cada factor -------------------------------------------------------
FOREIGN_LEVEL_POINTS = {"G1_Winner": 4.0, "G1_Placed": 3.25, "Other": 2.5}
FORM_TREND_POINTS = {"Ascending": 2.0, "Plateau": 0.0, "Descending": -3.0}
BOUNCE_PENALTY = -1.5
CLASS_POINTS = {"G1_Winner": 1.5, "Graded_Placed": 0.0, "Class_Riser": -2.5}
EARNINGS_HIGH_RATIO, EARNINGS_LOW_RATIO = 1.5, 0.5
EARNINGS_HIGH_POINTS, EARNINGS_LOW_POINTS = 0.75, -0.5
JOCKEY_POINTS = {"Upgrade": 1.5, "Neutral": 0.0, "Downgrade": -2.0}
TRACK_APTITUDE_POINTS = {"Proven_Winner": 2.0, "Positive": 1.0, "Negative": -2.0, "First_Time": 0.0}
WET_TRACK_BONUS = 2.5
GOING_MISMATCH_POINTS = -1.5
PACE_CROWDED_THRESHOLD = 3
PACE_CROWDED_E_PENALTY = -3.5
PACE_CROWDED_S_BONUS = 3.0
PACE_LONE_E_BONUS = 2.5
QUIRIN_EARLY, QUIRIN_LATE = 6, 2  # puntos de velocidad temprana (0-8)
WEIGHT_STEP_LBS = 3
WEIGHT_STEP_POINTS = -0.5
PED_SURFACE_POINTS = {"Strong": 1.5, "Weak": -1.5}
PED_DISTANCE_POINTS = {"Strong": 1.0, "Weak": -1.0}
PED_WET_POINTS = {"Strong": 1.0, "Weak": -1.0}
RECORD_MIN_STARTS = 3
TRAINER_PCT_POINTS = ((25.0, 1.5), (18.0, 0.75), (8.0, 0.0))  # (minimo, puntos); por debajo del ultimo: -0.75
TRAINER_LOW_POINTS = -0.75
SITUATION_HIGH, SITUATION_LOW = 25.0, 8.0
JOCKEY_PCT_HIGH, JOCKEY_PCT_LOW = 20.0, 8.0
COMBO_PCT_HIGH = 25.0
BIAS_STYLE_POINTS = 1.0
BIAS_POST_POINTS = 0.75
TROUBLE_POINTS = 1.0

# Veredicto -------------------------------------------------------------------
FIJO_MIN_WIN = 0.40  # probabilidad minima de ganar para ser "fijo"
FIJO_MIN_RATIO = 1.5  # veces la probabilidad del segundo
CONTENDER_SHARE = 0.7  # fraccion de la probabilidad "justa" (1/N) para ser contender
VERDICTS = ("FIJO", "CONTENDER", "TIRAR")

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
    "adj_pedigree",
    "adj_distance",
    "adj_layoff",
    "adj_trainer",
    "adj_workouts",
    "adj_equipment",
    "adj_bias",
    "adj_trip",
    "adj_surface",
    "adj_intuition",
)

ADJUSTMENT_LABELS = {
    "adj_foreign": "Extranjero",
    "adj_form": "Forma",
    "adj_class": "Lote",
    "adj_jockey": "Jinete",
    "adj_track": "Pista",
    "adj_condition": "Condicion",
    "adj_pace": "Ritmo",
    "adj_weight": "Peso",
    "adj_pedigree": "Pedigri",
    "adj_distance": "Distancia",
    "adj_layoff": "Descanso",
    "adj_trainer": "Entrenador",
    "adj_workouts": "Trabajos",
    "adj_equipment": "Equipo",
    "adj_bias": "Sesgo/Posicion",
    "adj_trip": "Recorrido",
    "adj_surface": "Superficie",
    "adj_intuition": "Criterio experto",
}

# Pesos del sistema (editables en la app): multiplican los puntos de cada factor.
DEFAULT_WEIGHTS: dict[str, float] = {
    **{column: 1.0 for column in ADJUSTMENT_COLUMNS},
    "base_recency": 0.45,  # mezcla de la base: promedio ponderado por recencia
    "base_peak": 0.30,  # mezcla de la base: pico (2 mejores de las ultimas 6)
    "base_condition": 0.25,  # mezcla de la base: mejor figura en superficie/distancia de hoy
    "decay": 0.75,  # cuanto pesa cada carrera mas antigua (0.75 = pierde 25% por carrera)
    "intl_factor": 0.90,  # rating internacional -> escala Beyer: valor * factor + offset
    "intl_offset": 0.0,
}

TEXT_FIELDS = ("program_number", "name", "morning_line", "jockey", "sire", "dam_sire", "trainer")
FLOAT_FIELDS = {
    "base_speed_figure": (0.0, 120.0),
    "weight_lbs": (0.0, None),
    "last_weight_lbs": (0.0, None),
    "trainer_win_pct": (0.0, 100.0),
    "trainer_situation_pct": (0.0, 100.0),
    "jockey_win_pct": (0.0, 100.0),
    "combo_win_pct": (0.0, 100.0),
    "quirin_points": (0.0, 8.0),
    "early_pace": (0.0, 130.0),
    "late_pace": (0.0, 130.0),
    "earnings_per_start": (0.0, None),
}
INT_FIELDS = (
    "post_position", "days_since_last", "starts_90d", "workouts_30d",
    "dirt_starts", "dirt_wins", "turf_starts", "turf_wins",
    "dist_starts", "dist_wins", "wet_starts", "wet_wins",
)
BOOL_FIELDS = {
    "is_regular_jockey": True, "mud_proven": False, "bullet_work": False,
    "lasix_first": False, "trouble_last": False, "first_us_start": False, "scratched": False,
}
ENUM_FIELDS = {
    "foreign_origin": (FOREIGN_ORIGINS, "None"),
    "foreign_level": (FOREIGN_LEVELS, "None"),
    "form_trend": (FORM_TRENDS, "Plateau"),
    "class_rating": (CLASS_RATINGS, "Graded_Placed"),
    "jockey_switch": (JOCKEY_SWITCHES, "Neutral"),
    "track_aptitude": (TRACK_APTITUDES, "First_Time"),
    "running_style": (RUNNING_STYLES, "P"),
    "ped_surface_fit": (FIT_LEVELS, "Unknown"),
    "ped_distance_fit": (FIT_LEVELS, "Unknown"),
    "ped_wet_fit": (FIT_LEVELS, "Unknown"),
    "blinkers_change": (BLINKERS_CHANGES, "None"),
    "going_pref": (GOING_PREFS, "Unknown"),
    "intl_rating_kind": (RATING_KINDS, "None"),
}

HORSE_FIELDS = (
    "program_number", "name", "morning_line",
    "figures", "intl_ratings", "intl_rating_kind", "base_speed_figure",
    "foreign_origin", "foreign_level", "form_trend", "class_rating", "earnings_per_start",
    "running_style", "quirin_points", "early_pace", "late_pace",
    "jockey", "is_regular_jockey", "jockey_switch", "jockey_win_pct",
    "trainer", "trainer_win_pct", "trainer_situation_pct", "combo_win_pct",
    "sire", "dam_sire", "ped_surface_fit", "ped_distance_fit", "ped_wet_fit",
    "track_aptitude", "mud_proven", "going_pref",
    "dirt_starts", "dirt_wins", "turf_starts", "turf_wins",
    "dist_starts", "dist_wins", "wet_starts", "wet_wins",
    "days_since_last", "starts_90d", "workouts_30d", "bullet_work",
    "blinkers_change", "lasix_first", "post_position", "trouble_last", "first_us_start",
    "weight_lbs", "last_weight_lbs", "scratched",
)
# Campos que se guardan en la columna jsonb "profile" de la base de datos.
PROFILE_FIELDS = tuple(
    field for field in HORSE_FIELDS
    if field not in (
        "program_number", "name", "morning_line", "base_speed_figure", "foreign_origin", "foreign_level",
        "form_trend", "class_rating", "jockey", "is_regular_jockey", "jockey_switch", "track_aptitude",
        "mud_proven", "running_style", "weight_lbs", "last_weight_lbs", "scratched",
    )
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


def _as_int(value: Any, low: int = 0, high: int | None = None) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    result = max(low, int(round(number)))
    return min(high, result) if high is not None else result


def normalize_horse(raw: dict[str, Any]) -> dict[str, Any]:
    """Convierte un diccionario (de Claude, del editor o de la base de datos) en un caballo valido.

    Los campos numericos desconocidos quedan en None (no se rellenan con ceros), y las series de
    figuras se guardan como texto del editor ("92 T 8; 88 D 8.5").
    """
    horse: dict[str, Any] = {}
    for field in TEXT_FIELDS:
        horse[field] = _as_text(raw.get(field))
    for field, (low, high) in FLOAT_FIELDS.items():
        horse[field] = _as_float(raw.get(field), low, high)
    for field in INT_FIELDS:
        horse[field] = _as_int(raw.get(field), 0, 20 if field == "post_position" else None)
    for field, default in BOOL_FIELDS.items():
        horse[field] = _as_bool(raw.get(field), default=default)
    for field, (options, default) in ENUM_FIELDS.items():
        horse[field] = _as_enum(raw.get(field), options, default, field)
    horse["figures"] = fg.format_figures(fg.parse_figures(raw.get("figures")))
    horse["intl_ratings"] = fg.format_figures(fg.parse_figures(raw.get("intl_ratings")))
    if horse["foreign_origin"] != "None" and horse["foreign_level"] == "None":
        horse["foreign_level"] = "Other"
    if horse["intl_ratings"] and horse["intl_rating_kind"] == "None":
        horse["intl_rating_kind"] = "Other"
    return {field: horse[field] for field in HORSE_FIELDS}


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



# ---------------------------------------------------------------------------
# Contexto, figuras y confianza
# ---------------------------------------------------------------------------
def resolve_weights(weights: dict[str, float] | None) -> dict[str, float]:
    merged = dict(DEFAULT_WEIGHTS)
    for key, value in (weights or {}).items():
        if key in merged and value is not None:
            merged[key] = float(value)
    return merged


def effective_style(horse: dict[str, Any]) -> str:
    """Estilo de carrera: si hay puntos de velocidad (Quirin) mandan; si no, el estilo declarado."""
    quirin = horse.get("quirin_points")
    if quirin is not None:
        if quirin >= QUIRIN_EARLY:
            return "E"
        if quirin <= QUIRIN_LATE:
            return "S"
        return "P"
    return horse["running_style"]


def count_early_speed(horses: list[dict[str, Any]]) -> int:
    return sum(1 for h in horses if effective_style(h) == "E")


def build_series(horse: dict[str, Any], weights: dict[str, float]) -> list[dict[str, Any]]:
    """Serie de figuras del caballo en escala Beyer (Beyers + ratings internacionales convertidos)."""
    beyers = fg.parse_figures(horse["figures"])
    international = fg.parse_figures(horse["intl_ratings"])
    series = fg.combined_series(beyers, international, weights["intl_factor"], weights["intl_offset"])
    if not series and horse["base_speed_figure"] is not None:
        series = [{"v": float(horse["base_speed_figure"]), "s": None, "d": None}]
    return series


def analyze_horse(horse: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    weights = ctx["weights"]
    series = build_series(horse, weights)
    blend = (weights["base_recency"], weights["base_peak"], weights["base_condition"])
    analysis = fg.analyze_series(series, ctx["surface"], ctx["distance_f"], weights["decay"], blend)
    analysis["series"] = series
    return analysis


DOMESTIC_CONFIDENCE_LEVELS = (70, 40)  # (puntaje para Alta, puntaje para Media)
FOREIGN_CONFIDENCE_LEVELS = (50, 25)

CONFIDENCE_PARTS = (
    ("Beyers/ratings", 30),
    ("pedigri", 10),
    ("entrenador", 8),
    ("jinete (%)", 6),
    ("trabajos", 6),
    ("records", 10),
    ("puntos de velocidad", 8),
    ("dias de descanso", 6),
    ("posicion de salida", 4),
    ("peso", 4),
    ("ganancias", 3),
    ("distancia/superficie de las figuras", 5),
)


def compute_confidence(horse: dict[str, Any], analysis: dict[str, Any]) -> tuple[int, str, list[str]]:
    """Puntaje 0-100 segun cuantos datos se pudieron leer, su nivel y lo que falta."""
    n = analysis["n"]
    labeled = any(f.get("s") or f.get("d") is not None for f in analysis["series"])
    present = {
        "Beyers/ratings": 30 if n >= 5 else 20 if n >= 3 else 10 if n >= 1 else 0,
        "pedigri": 10 if any(horse[f] != "Unknown" for f in ("ped_surface_fit", "ped_distance_fit", "ped_wet_fit")) else 0,
        "entrenador": 8 if horse["trainer_win_pct"] is not None else 0,
        "jinete (%)": 6 if horse["jockey_win_pct"] is not None else 0,
        "trabajos": 6 if horse["workouts_30d"] is not None else 0,
        "records": 10 if any(horse[f] is not None for f in ("dirt_starts", "turf_starts", "dist_starts", "wet_starts")) else 0,
        "puntos de velocidad": 8 if horse["quirin_points"] is not None else 0,
        "dias de descanso": 6 if horse["days_since_last"] is not None else 0,
        "posicion de salida": 4 if horse["post_position"] is not None else 0,
        "peso": 4 if horse["weight_lbs"] is not None else 0,
        "ganancias": 3 if horse["earnings_per_start"] is not None else 0,
        "distancia/superficie de las figuras": 5 if labeled else 0,
    }
    score = sum(present.values())
    # El DRF trae menos datos de los caballos extranjeros (sin records ni estadisticas de EE.UU.),
    # asi que se les exige menos puntaje para no castigarlos por la fuente.
    high, medium = (FOREIGN_CONFIDENCE_LEVELS if horse["foreign_origin"] != "None" else DOMESTIC_CONFIDENCE_LEVELS)
    level = "Alta" if score >= high else "Media" if score >= medium else "Baja"
    missing = [label for label, maximum in CONFIDENCE_PARTS if present[label] < maximum and (
        label != "Beyers/ratings" or n < 3)]
    return score, level, missing


def sigma_factor(analysis: dict[str, Any], level: str) -> float:
    """Dispersion relativa del caballo: mas consistente = menos incertidumbre."""
    factor = 1.0
    std = analysis["std"]
    if std is not None:
        factor = 0.9 if std < 2.5 else 1.0 if std <= 5.0 else 1.15
    factor *= {"Alta": 1.0, "Media": 1.1, "Baja": 1.25}[level]
    return round(min(1.4, max(0.8, factor)), 3)


# ---------------------------------------------------------------------------
# Factores de handicap
# ---------------------------------------------------------------------------
def _pct_points(value: float | None, high: float, low: float, high_pts: float, low_pts: float) -> float:
    if value is None:
        return 0.0
    if value >= high:
        return high_pts
    if value < low:
        return low_pts
    return 0.0


def _record_points(starts: int | None, wins: int | None) -> float:
    if starts is None or wins is None or starts < RECORD_MIN_STARTS:
        return 0.0
    rate = wins / starts
    if rate >= 0.33:
        return 1.0
    if rate >= 0.20:
        return 0.5
    if wins == 0 and starts >= RECORD_MIN_STARTS + 1:
        return -1.0
    return 0.0


def compute_adjustments(
    horse: dict[str, Any], ctx: dict[str, Any], analysis: dict[str, Any], early_count: int
) -> dict[str, float]:
    """Devuelve los puntos de cada factor de handicap (ya multiplicados por los pesos del sistema)."""
    surface, condition = ctx["surface"], ctx["track_condition"]
    wet = condition in WET_CONDITIONS
    raw = {column: 0.0 for column in ADJUSTMENT_COLUMNS}

    # Extranjero (Europa / Japon): bonus en grama segun su nivel; sin castigo en arena.
    if horse["foreign_origin"] in {"Europe", "Japan", "Other"} and surface == "Turf":
        raw["adj_foreign"] = FOREIGN_LEVEL_POINTS.get(horse["foreign_level"], FOREIGN_LEVEL_POINTS["Other"])

    # Forma: la tendencia calculada de las figuras manda; si no hay, la declarada.
    trend = analysis["trend"] or horse["form_trend"]
    raw["adj_form"] = FORM_TREND_POINTS[trend] + (BOUNCE_PENALTY if analysis["bounce"] else 0.0)

    # Lote: nivel de las carreras + ganancias por salida relativas al campo.
    raw["adj_class"] = CLASS_POINTS[horse["class_rating"]]
    median_eps, eps = ctx.get("median_eps"), horse["earnings_per_start"]
    if median_eps and eps is not None:
        if eps >= median_eps * EARNINGS_HIGH_RATIO:
            raw["adj_class"] += EARNINGS_HIGH_POINTS
        elif eps <= median_eps * EARNINGS_LOW_RATIO:
            raw["adj_class"] += EARNINGS_LOW_POINTS

    # Jinete: cambio de jinete + porcentaje de victorias.
    raw["adj_jockey"] = JOCKEY_POINTS[effective_jockey_switch(horse)] + _pct_points(
        horse["jockey_win_pct"], JOCKEY_PCT_HIGH, JOCKEY_PCT_LOW, 0.75, -0.5
    )

    # Aptitud en esta pista.
    raw["adj_track"] = TRACK_APTITUDE_POINTS[horse["track_aptitude"]]

    # Condicion de pista: bonus unico (no se acumula) en pista suave/humeda.
    if wet:
        european_on_turf = surface == "Turf" and horse["foreign_origin"] == "Europe"
        if european_on_turf or horse["mud_proven"] or horse["going_pref"] == "Soft":
            raw["adj_condition"] = WET_TRACK_BONUS
        elif horse["going_pref"] == "Firm":
            raw["adj_condition"] = GOING_MISMATCH_POINTS
    elif horse["going_pref"] == "Soft":
        raw["adj_condition"] = -1.0

    # Ritmo: cuantos punteros hay y cual es el estilo de este caballo.
    style = effective_style(horse)
    if early_count >= PACE_CROWDED_THRESHOLD:
        if style == "E":
            raw["adj_pace"] = PACE_CROWDED_E_PENALTY
        elif style == "S":
            raw["adj_pace"] = PACE_CROWDED_S_BONUS
    elif early_count == 1 and style == "E":
        raw["adj_pace"] = PACE_LONE_E_BONUS

    # Peso: -0.5 por cada 3 lbs adicionales vs la ultima carrera.
    weight, last_weight = horse["weight_lbs"], horse["last_weight_lbs"]
    if weight and last_weight and weight > last_weight:
        raw["adj_weight"] = int((weight - last_weight) // WEIGHT_STEP_LBS) * WEIGHT_STEP_POINTS

    # Pedigri para la superficie, la distancia y el barro. Con 3+ salidas en la superficie pesa la mitad.
    surface_starts = {"Dirt": horse["dirt_starts"], "Turf": horse["turf_starts"]}.get(surface)
    surface_factor = 0.5 if (surface_starts or 0) >= RECORD_MIN_STARTS else 1.0
    pedigree = PED_SURFACE_POINTS.get(horse["ped_surface_fit"], 0.0) * surface_factor
    pedigree += PED_DISTANCE_POINTS.get(horse["ped_distance_fit"], 0.0)
    if wet:
        pedigree += PED_WET_POINTS.get(horse["ped_wet_fit"], 0.0)
    raw["adj_pedigree"] = pedigree

    # Distancia: donde logro sus mejores figuras + su record a esta distancia.
    raw["adj_distance"] = fg.distance_fit_points(analysis["series"], ctx["distance_f"]) + _record_points(
        horse["dist_starts"], horse["dist_wins"]
    )

    # Descanso.
    days = horse["days_since_last"]
    if days is not None:
        raw["adj_layoff"] = (
            -2.0 if days > 180 else -1.25 if days > 120 else -0.5 if days > 60 else 0.25 if days >= 15 else -0.25
        )

    # Entrenador (general, en esta situacion y en combinacion con el jinete).
    trainer = 0.0
    pct = horse["trainer_win_pct"]
    if pct is not None:
        trainer += next((points for minimum, points in TRAINER_PCT_POINTS if pct >= minimum), TRAINER_LOW_POINTS)
    trainer += _pct_points(horse["trainer_situation_pct"], SITUATION_HIGH, SITUATION_LOW, 0.75, -0.75)
    if horse["combo_win_pct"] is not None and horse["combo_win_pct"] >= COMBO_PCT_HIGH:
        trainer += 0.75
    raw["adj_trainer"] = trainer

    # Trabajos.
    workouts = 0.0
    if horse["workouts_30d"] is not None:
        if horse["workouts_30d"] >= 3:
            workouts += 0.5
        elif horse["workouts_30d"] == 0 and (days is None or days > 30):
            workouts -= 1.0
    if horse["bullet_work"]:
        workouts += 0.5
    raw["adj_workouts"] = workouts

    # Equipo.
    equipment = {"First_Time": 0.5, "Off": -0.25}.get(horse["blinkers_change"], 0.0)
    if horse["lasix_first"]:
        equipment += 0.5
    raw["adj_equipment"] = equipment

    # Sesgo del dia (lo ingresa el usuario) y posicion de salida.
    bias = 0.0
    if ctx["bias_speed"] == "Speed":
        bias += BIAS_STYLE_POINTS if style == "E" else 0.5 * BIAS_STYLE_POINTS if style == "P" else -BIAS_STYLE_POINTS
    elif ctx["bias_speed"] == "Closers":
        bias += BIAS_STYLE_POINTS if style == "S" else -BIAS_STYLE_POINTS if style == "E" else 0.0
    post = horse["post_position"]
    if post is not None:
        if ctx["bias_rail"] == "Inside":
            bias += BIAS_POST_POINTS if post <= 3 else -BIAS_POST_POINTS if post >= 9 else 0.0
        elif ctx["bias_rail"] == "Outside":
            bias += BIAS_POST_POINTS if post >= 8 else -BIAS_POST_POINTS if post <= 2 else 0.0
    raw["adj_bias"] = bias

    # Recorrido: problemas en su ultima carrera suelen subestimar su rendimiento.
    raw["adj_trip"] = TROUBLE_POINTS if horse["trouble_last"] else 0.0

    # Record en la superficie de hoy.
    if surface == "Dirt":
        raw["adj_surface"] = _record_points(horse["dirt_starts"], horse["dirt_wins"])
    elif surface == "Turf":
        raw["adj_surface"] = _record_points(horse["turf_starts"], horse["turf_wins"])

    weights = ctx["weights"]
    return {column: raw[column] * weights[column] for column in ADJUSTMENT_COLUMNS}


INTUITION_LIMIT = 3.0  # el criterio experto puede mover un caballo como maximo +/- 3 puntos


def _intuition_points(intuition: dict[str, float] | None, horse: dict[str, Any]) -> float:
    if not intuition:
        return 0.0
    try:
        value = float(intuition.get(str(horse["program_number"]), 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(-INTUITION_LIMIT, min(INTUITION_LIMIT, value))


def horse_notes(horse: dict[str, Any], analysis: dict[str, Any]) -> list[str]:
    notes = []
    if analysis["n"] == 0:
        notes.append("Sin Beyers ni rating: su base es el promedio del campo (confianza baja)")
    if analysis["bounce"]:
        notes.append("Posible rebote tras una figura muy alta")
    if analysis["std"] is not None and analysis["std"] > 5.0:
        notes.append(f"Figuras irregulares (desviacion {analysis['std']:.1f})")
    if analysis["trend"] and analysis["trend"] != horse["form_trend"]:
        notes.append(f"Tendencia calculada de las figuras: {analysis['trend']}")
    if horse["first_us_start"]:
        notes.append("Primera carrera en EE.UU.")
    if horse.get("_pace_note"):
        notes.append(horse["_pace_note"])
    if horse["intl_ratings"] and not fg.parse_figures(horse["figures"]):
        notes.append("Base a partir de rating internacional convertido a escala Beyer")
    return notes


# ---------------------------------------------------------------------------
# Veredicto
# ---------------------------------------------------------------------------
def verdict_for(win_probability: float, top_probability: float, second_probability: float, count: int, level: str) -> str:
    """FIJO / CONTENDER / TIRAR a partir de la simulacion (nunca del Morning Line)."""
    is_top = abs(win_probability - top_probability) < 1e-12
    ratio_ok = second_probability <= 0 or win_probability >= FIJO_MIN_RATIO * second_probability
    if is_top and win_probability >= FIJO_MIN_WIN and ratio_ok and level != "Baja":
        return "FIJO"
    if win_probability >= CONTENDER_SHARE / max(count, 1):
        return "CONTENDER"
    return "TIRAR"



# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def _draw_performances(
    ratings: np.ndarray | list[float],
    sigma: float,
    iterations: int,
    seed: int | None,
    scales: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Valida los parametros y devuelve la matriz (iteraciones x caballos) de rendimientos.

    `scales` es un factor de dispersion por caballo (1.0 = sigma tal cual).
    """
    values = np.asarray(ratings, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Se necesitan al menos 2 caballos para simular.")
    if sigma <= 0:
        raise ValueError("sigma debe ser mayor que 0.")
    if iterations < 1:
        raise ValueError("iterations debe ser al menos 1.")
    factors = np.ones(values.size) if scales is None else np.asarray(scales, dtype=float)
    if factors.shape != values.shape or np.any(factors <= 0):
        raise ValueError("scales debe tener un valor positivo por caballo.")

    rng = np.random.default_rng(seed)
    return rng.normal(loc=values, scale=sigma * factors, size=(int(iterations), values.size))

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
    distance: str | None = None,
    bias_speed: str = "Neutral",
    bias_rail: str = "Neutral",
    weights: dict[str, float] | None = None,
    intuition: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Ejecuta el handicap completo y la simulacion. Devuelve un DataFrame ordenado
    por probabilidad de ganar (descendente).

    `horse_key` es la posicion del caballo en la lista original recibida. El Morning Line
    no influye en el rating ni en la probabilidad: solo se usa para el EV.
    """
    if surface not in SURFACES:
        raise ValueError(f"Superficie invalida: {surface!r}. Opciones: {SURFACES}")
    if track_condition not in TRACK_CONDITIONS:
        raise ValueError(f"Condicion invalida: {track_condition!r}. Opciones: {TRACK_CONDITIONS}")
    if bias_speed not in BIAS_SPEED_OPTIONS or bias_rail not in BIAS_RAIL_OPTIONS:
        raise ValueError("Sesgo de pista invalido.")

    normalized = [(index, normalize_horse(h)) for index, h in enumerate(horses)]
    active = [(index, h) for index, h in normalized if not h["scratched"] and h["name"]]
    if len(active) < 2:
        raise ValueError("Se necesitan al menos 2 caballos activos (con nombre y sin scratch).")

    resolved = resolve_weights(weights)
    earnings = sorted(h["earnings_per_start"] for _, h in active if h["earnings_per_start"])
    ctx = {
        "surface": surface,
        "track_condition": track_condition,
        "distance_f": fg.distance_to_furlongs(distance),
        "bias_speed": bias_speed,
        "bias_rail": bias_rail,
        "weights": resolved,
        "median_eps": earnings[len(earnings) // 2] if earnings else None,
    }

    pace = pace_mod.project_pace(active, ctx["distance_f"], surface)
    early_count = len(pace["leaders"])
    analyses = {index: analyze_horse(h, ctx) for index, h in active}
    known_bases = [a["base"] for a in analyses.values() if a["base"] is not None]
    fallback_base = float(np.mean(known_bases)) if known_bases else DEFAULT_BASE_FIGURE

    style_names = {"E": "puntero", "P": "perseguidor", "S": "rematador"}
    for index, horse in active:
        horse["_pace_note"] = (
            f"Ritmo: corre de {style_names[pace['styles'][index]]} en una carrera de tipo "
            f"'{pace['title'].lower()}' ({pace['adjust'][index]:+.1f})"
        )

    rows: list[dict[str, Any]] = []
    factors: list[float] = []
    for index, horse in active:
        analysis = analyses[index]
        adjustments = compute_adjustments(horse, ctx, analysis, early_count)
        adjustments["adj_pace"] = pace["adjust"][index] * resolved["adj_pace"]
        adjustments["adj_intuition"] = _intuition_points(intuition, horse) * resolved["adj_intuition"]
        base = analysis["base"] if analysis["base"] is not None else fallback_base
        score, level, missing = compute_confidence(horse, analysis)
        factor = sigma_factor(analysis, level)
        factors.append(factor)
        rows.append(
            {
                "horse_key": index,
                "program_number": horse["program_number"],
                "name": horse["name"],
                "morning_line": horse["morning_line"],
                "base_speed_figure": round(base, 2),
                "n_figures": analysis["n"],
                **{column: round(value, 2) for column, value in adjustments.items()},
                "final_adjusted_rating": round(base + sum(adjustments.values()), 2),
                "confidence": score,
                "confidence_level": level,
                "sigma_factor": factor,
                "missing": ", ".join(missing),
                "notes": " | ".join(horse_notes(horse, analysis)),
                "decimal_odds": parse_morning_line(horse["morning_line"]),
            }
        )

    performances = _draw_performances(
        [row["final_adjusted_rating"] for row in rows], sigma, iterations, seed, scales=factors
    )
    probabilities = win_probabilities(performances)
    finish = finish_statistics(performances)
    position_probs = finish["position_probs"]

    ordered = sorted(float(p) for p in probabilities)[::-1]
    top_probability, second_probability = ordered[0], ordered[1]
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
        row["verdict"] = verdict_for(
            float(probability), top_probability, second_probability, len(rows), row["confidence_level"]
        )

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
    result.attrs["pace"] = pace
    result.attrs["distance_f"] = ctx["distance_f"]
    return result
