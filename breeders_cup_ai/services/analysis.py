"""Comentario de la carrera, planteamiento probable y orden de llegada.

Todo se genera localmente a partir del resultado de la simulacion (sin costo de API).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

STYLE_NAMES = {"E": "velocista", "P": "perseguidor", "S": "sostenido (corre atras)"}

# (columna de ajuste, frase si suma, frase si resta)
FACTOR_PHRASES = {
    "adj_foreign": ("viene del exterior con nivel de grupo y la superficie lo favorece", "su primera salida en EE.UU. le resta"),
    "adj_form": ("llega en forma ascendente", "su forma viene en descenso"),
    "adj_class": ("sube su nivel de clase", "sube de nivel de lote"),
    "adj_jockey": ("tiene jinete de primer nivel", "cambia a un jinete de menor nivel"),
    "adj_track": ("tiene aptitud comprobada en la pista", "no rinde bien en esta pista"),
    "adj_condition": ("la condicion de la pista le favorece", "la condicion de la pista no le favorece"),
    "adj_pace": ("el ritmo esperado le favorece", "el ritmo esperado lo perjudica"),
    "adj_weight": ("", "carga mas peso que en su ultima salida"),
    "adj_pedigree": ("su pedigri encaja con superficie y distancia", "su pedigri no encaja con la carrera"),
    "adj_distance": ("sus mejores figuras salieron a esta distancia", "sus mejores figuras salieron a otra distancia"),
    "adj_layoff": ("llega con descanso y actividad adecuados", "el descanso o la inactividad lo perjudican"),
    "adj_trainer": ("el entrenador tiene buen record en esta situacion", "el entrenador rinde poco en esta situacion"),
    "adj_workouts": ("sus trabajos recientes son buenos", "sus trabajos recientes son flojos"),
    "adj_equipment": ("el cambio de equipo le favorece", "el equipo le resta"),
    "adj_bias": ("el sesgo de la pista le favorece", "el sesgo de la pista lo perjudica"),
    "adj_trip": ("tuvo problemas en su ultima y puede mejorar", "no se espera mejora de su viaje"),
    "adj_surface": ("tiene antecedentes en esta superficie", "tiene poco o ningun antecedente en esta superficie"),
}


# Umbrales para detectar un caballo claramente superior (se evaluan sobre la simulacion).
DOMINANT_WIN = 0.45  # gana al menos 45% de las simulaciones...
DOMINANT_RATIO = 1.6  # ...y tiene 1.6x la probabilidad del segundo
SOLID_WIN = 0.33
SOLID_RATIO = 1.3
ANCHOR_TOP3 = 0.80  # base para exoticas: llega entre los 3 primeros 80% o mas


def _standout(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Detecta si hay un caballo muy superior, aunque no ofrezca valor (+EV)."""
    if len(records) < 2:
        return None
    first, second = records[0], records[1]
    win = first["win_probability"]
    ratio = win / second["win_probability"] if second["win_probability"] > 0 else float("inf")
    if first.get("verdict") == "FIJO":
        level, title = "dominant", "FIJO del sistema"
    elif win >= DOMINANT_WIN and ratio >= DOMINANT_RATIO and first.get("confidence_level") != "Baja":
        level, title = "dominant", "Caballo dominante"
    elif win >= SOLID_WIN and ratio >= SOLID_RATIO:
        level, title = "solid", "Favorito solido"
    else:
        return None

    fair = (1.0 / win - 1.0) if win > 0 else float("inf")
    ev = first.get("expected_value")
    has_ev = ev is not None and pd.notna(ev)
    odds_text = str(first["morning_line"]) if first["morning_line"] else "sin ML"
    if has_ev and ev > 0:
        verdict = (
            f"Ademas ofrece valor: el ML {odds_text} paga mas que su cuota justa ({fair:.1f}-1), "
            f"EV {ev:+.0%}. Es la jugada mas fuerte de la carrera."
        )
    elif has_ev:
        verdict = (
            f"Es el mejor caballo, pero a ML {odds_text} no hay valor (EV {ev:+.0%}; su cuota justa es "
            f"{fair:.1f}-1). Sirve como base de exoticas o para place/show, no como apuesta a ganar con ventaja."
        )
    else:
        verdict = f"Sin Morning Line no se puede calcular el EV. Su cuota justa seria {fair:.1f}-1."
    return {
        "level": level,
        "title": title,
        "label": _label(first),
        "win_probability": win,
        "p_top3": first["p_top3"],
        "ratio": ratio if ratio <= 20 else None,
        "second": _label(second),
        "verdict": verdict,
        "is_positive_ev": bool(has_ev and ev > 0),
    }


def _horse(horses: list[dict[str, Any]], key: int) -> dict[str, Any]:
    return horses[key] if 0 <= key < len(horses) else {}


def _label(record: dict[str, Any]) -> str:
    return f"#{record['program_number']} {record['name']}"


def _pace_scenario(table: pd.DataFrame, horses: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [int(k) for k in table["horse_key"]]
    by_key = {int(r["horse_key"]): r for r in table.to_dict("records")}
    styles = {k: _horse(horses, k).get("running_style", "P") for k in keys}
    speed = [by_key[k] for k in keys if styles[k] == "E"]
    closers = [by_key[k] for k in keys if styles[k] == "S"]
    speed_names = ", ".join(_label(r) for r in speed)
    closer_names = ", ".join(_label(r) for r in closers[:3])

    if not speed:
        title = "Sin velocidad clara"
        text = (
            "Ningun caballo esta marcado como velocista, asi que se espera un ritmo lento y sin pelea por "
            "la punta. En ese escenario ganan terreno los que corren cerca de la punta (perseguidores) y "
            "les cuesta mas a los que vienen de atras."
        )
    elif len(speed) == 1:
        title = "Punta sin oposicion"
        text = (
            f"{speed_names} es el unico velocista y puede mandar la carrera con parciales comodos. "
            "Un velocista solo es una ventaja grande: si nadie lo presiona, es dificil de alcanzar."
        )
    elif len(speed) == 2:
        title = "Duelo moderado por la punta"
        text = (
            f"Hay dos velocistas ({speed_names}). Se espera un ritmo vivo pero sostenible; "
            "el primero en tomar la delantera tiene ligera ventaja y los perseguidores quedan bien ubicados."
        )
    else:
        title = "Punta muy disputada"
        text = (
            f"Hay {len(speed)} velocistas ({speed_names}) que van a pelear por la punta. "
            "Es probable un ritmo rapido que desgasta a los de adelante y favorece a los que vienen de atras"
            + (f" ({closer_names})." if closers else ".")
        )
    return {"title": title, "text": text, "early_count": len(speed)}


def _key_factors(record: dict[str, Any], limit: int = 2) -> list[str]:
    """Las frases de los factores que mas pesan en el rating de un caballo."""
    found = []
    for column, (positive, negative) in FACTOR_PHRASES.items():
        value = record.get(column) or 0.0
        if value > 0 and positive:
            found.append((abs(value), f"{positive} ({value:+.1f})"))
        elif value < 0 and negative:
            found.append((abs(value), f"{negative} ({value:+.1f})"))
    found.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in found[:limit]]


def build_analysis(
    table: pd.DataFrame,
    horses: list[dict[str, Any]],
    race: dict[str, Any],
    iterations: int = 10_000,
) -> dict[str, Any]:
    """Arma el orden de llegada probable y los comentarios de la carrera."""
    by_key = {int(r["horse_key"]): r for r in table.to_dict("records")}
    records = table.to_dict("records")
    count = len(records)

    # --- Orden de llegada probable ------------------------------------------------
    order = [by_key[k] for k in table.attrs.get("finish_order", []) if k in by_key]
    if not order:
        order = sorted(records, key=lambda r: r["avg_finish"])[:4]
    finish = []
    for place, record in enumerate(order, start=1):
        reasons = _key_factors(record)
        finish.append(
            {
                "place": place,
                "label": _label(record),
                "name": record["name"],
                "program_number": record["program_number"],
                "win_probability": record["win_probability"],
                "p_top3": record["p_top3"],
                "p_top4": record["p_top4"],
                "avg_finish": record["avg_finish"],
                "reasons": reasons,
            }
        )
    combos = []
    for members, frequency in table.attrs.get("top_combos", []):
        if all(m in by_key for m in members):
            combos.append(([by_key[m]["program_number"] for m in members], frequency))

    # --- Comentarios generales ------------------------------------------------------
    leader = records[0]
    top3_share = sum(r["win_probability"] for r in records[:3])
    surface = race.get("surface", "")
    condition = race.get("track_condition", "")
    distance = race.get("distance")
    setting = " ".join(p for p in (str(distance) if distance else "", surface, f"({condition})" if condition else "") if p)

    if leader["win_probability"] >= 0.40:
        openness = f"Hay un favorito claro: {_label(leader)} gana {leader['win_probability']:.0%} de las simulaciones."
    elif leader["win_probability"] >= 0.25:
        openness = (
            f"{_label(leader)} lidera con {leader['win_probability']:.0%}, pero la carrera tiene margen "
            "para sorpresas."
        )
    else:
        openness = (
            f"Carrera muy abierta: el mejor ({_label(leader)}) solo gana {leader['win_probability']:.0%} "
            "de las simulaciones."
        )
    summary = [
        f"Campo de {count} caballos en {setting}. {openness}".replace("  ", " ").strip(),
        f"Los tres primeros del ranking concentran {top3_share:.0%} de la probabilidad de ganar.",
    ]

    pace = _pace_scenario(table, horses)

    # --- Alertas de calidad de datos y de valor --------------------------------------
    alerts: list[str] = []
    low_conf = [_label(r) for r in records if r.get("confidence_level") == "Baja"]
    if low_conf:
        shown = ", ".join(low_conf[:4]) + ("..." if len(low_conf) > 4 else "")
        alerts.append(
            f"{len(low_conf)} caballo(s) con confianza Baja por datos faltantes en el DRF ({shown}). "
            "Nunca reciben veredicto FIJO; revisa la tabla de datos leidos."
        )
    no_fig = [_label(r) for r in records if not r.get("n_figures")]
    if no_fig:
        alerts.append(f"Sin figuras ni ratings: {', '.join(no_fig[:4])}. Se les asigno el promedio del campo.")
    missing_ml = [_label(r) for r in records if not r["morning_line"]]
    if missing_ml:
        alerts.append(f"Sin Morning Line: {', '.join(missing_ml[:4])}. No se puede calcular su EV.")
    with_ev = [r for r in records if r["expected_value"] is not None and pd.notna(r["expected_value"])]
    if with_ev:
        market = min(with_ev, key=lambda r: r["decimal_odds"])
        model_rank = int(market["rank"])
        if model_rank > 2:
            alerts.append(
                f"El mercado favorece a {_label(market)} (ML {market['morning_line']}), pero el modelo lo ubica "
                f"{model_rank}.o con {market['win_probability']:.0%}. Posible favorito sobreapostado."
            )
        value = sorted((r for r in with_ev if r["is_positive_ev"]), key=lambda r: r["expected_value"], reverse=True)
        if value:
            best = value[0]
            alerts.append(
                f"Mejor valor: {_label(best)} con ML {best['morning_line']} y EV {best['expected_value']:+.0%}."
            )
    if any(_horse(horses, k).get("foreign_origin") not in (None, "", "None", "USA") for k in by_key):
        alerts.append(
            "Hay caballos del exterior: se evaluan con su rating internacional convertido a escala Beyer "
            "(factor editable en Pesos del sistema). Esa conversion es una aproximacion."
        )
    descending = [_label(r) for k, r in by_key.items()
                  if _horse(horses, k).get("form_trend") == "Descending" and r["rank"] <= 4]
    if descending:
        alerts.append(f"Forma en descenso entre los 4 primeros del ranking: {', '.join(descending)}.")
    alerts.append(
        f"El orden de llegada sale de {iterations:,} carreras simuladas; es una estimacion estadistica, no una prediccion segura."
    )

    standout = _standout(records)
    anchor_pool = records[1:] if standout else records
    anchors = [_label(r) + f" ({r['p_top3']:.0%} top 3)" for r in anchor_pool if r["p_top3"] >= ANCHOR_TOP3]

    return {
        "finish": finish, "combos": combos, "summary": summary, "pace": pace, "alerts": alerts,
        "standout": standout, "anchors": anchors,
    }
