"""Proyeccion del ritmo de la carrera (pace projector).

A partir de los numeros de velocidad temprana de cada caballo (TimeformUS Pace Early/Late o puntos
Quirin), de su estilo y de la distancia, decide quien manda, quien persigue y quien viene de atras, que
tipo de ritmo se espera (disputado, honesto, solo en la punta, lento) y cuantos puntos gana o pierde cada
caballo por ese escenario. No usa el Morning Line.
"""

from __future__ import annotations

from typing import Any

EARLY_WINDOW = 6.0  # un caballo es puntero si su Early esta a 6 puntos o menos del mas rapido
CLOSER_GAP = 8.0  # rematador si su Late supera a su Early por 8 o mas
QUIRIN_EARLY, QUIRIN_LATE = 6, 2
SPRINT_MAX_F = 6.5
ROUTE_MIN_F = 9.0

SCENARIOS = {
    "slow": "Ritmo lento, sin velocidad clara",
    "lone": "Punta sin oposicion",
    "honest": "Ritmo honesto, duelo moderado",
    "contested": "Punta muy disputada, ritmo rapido",
}

# puntos por escenario y estilo (antes del ajuste por distancia)
POINTS = {
    "slow": {"E": 0.0, "P": 1.5, "S": -1.5},
    "lone": {"E": 2.5, "P": 0.5, "S": -1.0},
    "honest": {"E": -0.5, "P": 0.5, "S": 0.5},
    "contested": {"E": -3.5, "P": 0.5, "S": 3.0},
}


def _declared_style(horse: dict[str, Any]) -> str:
    quirin = horse.get("quirin_points")
    if quirin is not None:
        if quirin >= QUIRIN_EARLY:
            return "E"
        if quirin <= QUIRIN_LATE:
            return "S"
        return "P"
    return horse.get("running_style") or "P"


def assign_styles(active: list[tuple[int, dict[str, Any]]]) -> dict[int, str]:
    """Estilo de cada caballo: relativo al campo si hay Early/Late; si no, Quirin o el estilo declarado."""
    earlies = [h["early_pace"] for _, h in active if h.get("early_pace") is not None]
    styles: dict[int, str] = {}
    if len(earlies) >= max(2, len(active) // 2):
        top = max(earlies)
        ordered = sorted(earlies)
        median = ordered[len(ordered) // 2]
        for index, horse in active:
            early, late = horse.get("early_pace"), horse.get("late_pace")
            if early is None:
                styles[index] = _declared_style(horse)
            elif early >= top - EARLY_WINDOW and (late is None or early >= late - 2):
                styles[index] = "E"
            elif (late is not None and late - early >= CLOSER_GAP) or early <= median - 10:
                styles[index] = "S"
            else:
                styles[index] = "P"
        return styles
    return {index: _declared_style(horse) for index, horse in active}


def project_pace(
    active: list[tuple[int, dict[str, Any]]], distance_f: float | None, surface: str
) -> dict[str, Any]:
    """Escenario de ritmo y ajuste de puntos por caballo."""
    styles = assign_styles(active)
    leaders = [i for i, _ in active if styles[i] == "E"]
    stalkers = [i for i, _ in active if styles[i] == "P"]
    closers = [i for i, _ in active if styles[i] == "S"]

    count = len(leaders)
    key = "slow" if count == 0 else "lone" if count == 1 else "honest" if count == 2 else "contested"
    sprint = distance_f is not None and distance_f <= SPRINT_MAX_F
    route = distance_f is not None and distance_f >= ROUTE_MIN_F

    adjust: dict[int, float] = {}
    for index, _ in active:
        points = POINTS[key][styles[index]]
        if styles[index] == "S" and points > 0:
            points *= 0.7 if sprint else 1.2 if route else 1.0  # en sprint al rematador le falta pista
        if styles[index] == "E" and sprint and key in ("lone", "honest"):
            points += 0.5  # en distancias cortas la velocidad se sostiene mas
        if styles[index] == "S" and points < 0 and route:
            points *= 0.8  # en rutas hay mas tiempo para remontar aunque el ritmo sea lento
        adjust[index] = round(points, 2)

    return {
        "scenario": key,
        "title": SCENARIOS[key],
        "styles": styles,
        "leaders": leaders,
        "stalkers": stalkers,
        "closers": closers,
        "adjust": adjust,
        "sprint": sprint,
        "route": route,
        "surface": surface,
        "uses_pace_numbers": any(h.get("early_pace") is not None for _, h in active),
    }
