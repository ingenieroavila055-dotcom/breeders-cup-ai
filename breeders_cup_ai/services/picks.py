"""Constructor de Pick 3 / 4 / 5 / 6 a partir de las simulaciones de varias carreras.

Para cada tipo de pick y cada grupo de carreras consecutivas, reparte los caballos de cada
pierna (leg) para maximizar la probabilidad de acertar sin pasar del presupuesto, y luego
elige el mejor tiquete segun el objetivo.

Modelo de pago (estimado): en un pool de picks, un tiquete ganador paga aproximadamente
    stake * (1 - takeout) / prod(prob. de mercado de cada ganador)
La prob. de mercado sale del Morning Line normalizado. Como las piernas se tratan como
independientes, el retorno esperado de un tiquete se factoriza:
    retorno = stake * (1 - takeout) * prod_legs( sum_{h en la pierna} prob_modelo / prob_mercado )
Es una estimacion con el Morning Line, no con los pagos reales del pool.
"""

from __future__ import annotations

import math
from typing import Any

PICK_SIZES = (3, 4, 5, 6)
DEFAULT_BASE_COSTS = {3: 0.50, 4: 0.50, 5: 0.50, 6: 0.20}
DEFAULT_TAKEOUT = 0.20
MAX_RATIO = 4.0  # tope de prob_modelo / prob_mercado para no depender de un solo outsider
GOALS = ("value", "prob")
GOAL_LABELS = {
    "value": "Mayor valor esperado (EV)",
    "prob": "Mayor probabilidad de acertar",
}


class PickError(ValueError):
    """Datos insuficientes o invalidos para construir el pick."""


def _valid(value: Any) -> bool:
    try:
        return value is not None and float(value) == float(value) and float(value) > 0
    except (TypeError, ValueError):
        return False


def build_leg(race_number: int, race_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Convierte los resultados de una carrera en una pierna del pick.

    `records` son filas del DataFrame de simulacion (win_probability, program_number, name,
    morning_line, implied_probability).
    """
    if race_number < 1:
        raise PickError("La carrera necesita un numero (1 o mayor) para poder armar el pick.")
    if not records:
        raise PickError("La carrera no tiene caballos simulados.")

    known = [r for r in records if _valid(r.get("implied_probability"))]
    implied_total = sum(float(r["implied_probability"]) for r in known)
    model_known_total = sum(float(r["win_probability"]) for r in known)

    entries = []
    for record in records:
        model = float(record["win_probability"])
        market = None
        if _valid(record.get("implied_probability")) and implied_total > 0:
            # Se normaliza para quitar el overround y se conserva la masa de los que tienen ML.
            market = float(record["implied_probability"]) / implied_total * model_known_total
        ratio = min(model / market, MAX_RATIO) if market and market > 0 else 1.0
        entries.append(
            {
                "program_number": str(record["program_number"]),
                "name": str(record["name"]),
                "morning_line": str(record.get("morning_line") or ""),
                "model": model,
                "market": market,
                "ratio": ratio,
            }
        )
    entries.sort(key=lambda e: e["model"], reverse=True)
    return {"race_number": int(race_number), "race_name": race_name, "entries": entries}


def _optimize_window(
    legs: list[dict[str, Any]], stake: float, max_combos: int, goal: str, takeout: float
) -> dict[str, Any]:
    """Reparte caballos por pierna con un algoritmo voraz y devuelve el mejor tiquete."""
    count = len(legs)
    chosen = [1] * count

    def totals(selection: list[int]) -> tuple[float, float, int]:
        hit, ratio, combos = 1.0, 1.0, 1
        for leg, size in zip(legs, selection):
            top = leg["entries"][:size]
            hit *= sum(e["model"] for e in top)
            ratio *= sum(e["ratio"] for e in top)
            combos *= size
        return hit, ratio, combos

    def snapshot(selection: list[int]) -> dict[str, Any]:
        hit, ratio, combos = totals(selection)
        cost = stake * combos
        expected_return = stake * (1.0 - takeout) * ratio
        return {
            "selection": list(selection),
            "combos": combos,
            "cost": cost,
            "hit": hit,
            "expected_return": expected_return,
            "ev": expected_return - cost,
        }

    path = [snapshot(chosen)]
    while True:
        _, _, combos = totals(chosen)
        best_leg, best_score = None, 0.0
        for index, leg in enumerate(legs):
            size = chosen[index]
            if size >= len(leg["entries"]) or leg["entries"][size]["model"] <= 0:
                continue
            if combos // size * (size + 1) > max_combos:
                continue
            current = sum(e["model"] for e in leg["entries"][:size])
            gain = math.log((current + leg["entries"][size]["model"]) / current)
            score = gain / math.log((size + 1) / size)
            if score > best_score:
                best_leg, best_score = index, score
        if best_leg is None:
            break
        chosen[best_leg] += 1
        path.append(snapshot(chosen))

    if goal == "prob":
        best = path[-1]
    else:
        best = max(path, key=lambda s: (s["ev"], -s["cost"]))
    return best


def build_picks(
    legs: list[dict[str, Any]],
    sizes: list[int],
    base_costs: dict[int, float],
    multiplier: int,
    budget: float,
    goal: str = "value",
    takeout: float = DEFAULT_TAKEOUT,
) -> list[dict[str, Any]]:
    """Evalua cada tipo de pick y cada grupo de carreras consecutivas.

    Devuelve las opciones ordenadas de mejor a peor segun `goal`.
    """
    if goal not in GOALS:
        raise PickError(f"Objetivo invalido: {goal!r}.")
    if budget <= 0 or multiplier < 1:
        raise PickError("El presupuesto debe ser mayor que 0 y el multiplicador al menos 1.")
    if not 0 <= takeout < 1:
        raise PickError("La deduccion (takeout) debe estar entre 0 y 1.")
    if not legs:
        raise PickError("Agrega al menos una carrera simulada.")

    by_number = {leg["race_number"]: leg for leg in legs}
    numbers = sorted(by_number)
    options: list[dict[str, Any]] = []
    too_expensive: list[int] = []

    for size in sorted(set(sizes)):
        stake = float(base_costs[size]) * int(multiplier)
        if stake <= 0:
            raise PickError(f"El costo base del Pick {size} debe ser mayor que 0.")
        max_combos = int(budget // stake + 1e-9)
        if max_combos < 1:
            too_expensive.append(size)
            continue
        for start in numbers:
            window = list(range(start, start + size))
            if not all(number in by_number for number in window):
                continue
            window_legs = [by_number[number] for number in window]
            best = _optimize_window(window_legs, stake, max_combos, goal, takeout)
            options.append(
                {
                    "size": size,
                    "races": window,
                    "stake": stake,
                    "combos": best["combos"],
                    "cost": best["cost"],
                    "hit": best["hit"],
                    "expected_return": best["expected_return"],
                    "ev": best["ev"],
                    "roi": best["expected_return"] / best["cost"] - 1.0,
                    "legs": [
                        {
                            "race_number": leg["race_number"],
                            "race_name": leg["race_name"],
                            "horses": leg["entries"][:picked],
                        }
                        for leg, picked in zip(window_legs, best["selection"])
                    ],
                }
            )

    if not options:
        if too_expensive and len(too_expensive) == len(set(sizes)):
            cheapest = min(float(base_costs[s]) * int(multiplier) for s in sizes)
            raise PickError(
                f"Con un presupuesto de ${budget:,.2f} no alcanza ni para una combinacion "
                f"(la mas barata cuesta ${cheapest:,.2f}). Sube el presupuesto o baja el multiplicador."
            )
        raise PickError(
            "No hay carreras consecutivas suficientes. Para un Pick N necesitas N carreras seguidas "
            "(por ejemplo 7, 8, 9 y 10 para un Pick 4)."
        )

    key = (lambda o: (o["ev"], o["hit"])) if goal == "value" else (lambda o: (o["hit"], o["ev"]))
    options.sort(key=key, reverse=True)
    return options


def sorted_horses(horses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Caballos de una pierna ordenados por numero de programa (1, 1A, 2, ...)."""
    def key(horse: dict[str, Any]) -> tuple[int, str]:
        digits = "".join(ch for ch in horse["program_number"] if ch.isdigit())
        return (int(digits) if digits else 999, horse["program_number"])
    return sorted(horses, key=key)


def ticket_text(option: dict[str, Any], multiplier: int) -> str:
    """Texto listo para copiar: una linea por carrera."""
    first, last = option["races"][0], option["races"][-1]
    lines = [f"PICK {option['size']} - carreras {first} a {last}"]
    for leg in option["legs"]:
        numbers = "-".join(h["program_number"] for h in sorted_horses(leg["horses"]))
        lines.append(f"Carrera {leg['race_number']}) {numbers}")
    lines.append(
        f"{option['combos']} combinaciones x ${option['stake']:,.2f} (base x{multiplier}) = ${option['cost']:,.2f}"
    )
    return "\n".join(lines)
