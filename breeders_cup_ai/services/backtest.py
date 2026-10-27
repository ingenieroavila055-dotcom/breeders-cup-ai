"""Medicion del desempeno del sistema contra resultados reales (backtesting).

Cada fila de entrada es una carrera con resultado real:
    {"race_name", "finish": ["3", "5", ...], "win_payout": 7.4 | None,
     "horses": [{"program_number", "win_probability", "verdict", "decimal_odds"}]}
"""

from __future__ import annotations

import math
from typing import Any

CALIBRATION_EDGES = (0.0, 0.10, 0.20, 0.30, 0.45, 1.01)
MIN_RACES_RELIABLE = 30


def _top(horses: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(horses, key=lambda h: h["win_probability"]) if horses else None


def compute_backtest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    races = [r for r in rows if r.get("finish") and r.get("horses")]
    n = len(races)
    stats: dict[str, Any] = {
        "races": n,
        "reliable": n >= MIN_RACES_RELIABLE,
        "top_win": 0, "top_top3": 0,
        "fijo_count": 0, "fijo_wins": 0, "fijo_top3": 0,
        "tirar_count": 0, "tirar_wins": 0,
        "brier": None, "log_loss": None, "calibration": [],
        "fijo_roi": None, "fijo_bets": 0,
        "detail": [],
    }
    if n == 0:
        return stats

    brier_terms: list[float] = []
    log_terms: list[float] = []
    buckets = [{"low": lo, "high": hi, "count": 0, "prob_sum": 0.0, "wins": 0}
               for lo, hi in zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:])]
    fijo_profit, fijo_bets = 0.0, 0

    for race in races:
        winner = str(race["finish"][0])
        podium = [str(p) for p in race["finish"][:3]]
        horses = race["horses"]
        top = _top(horses)
        top_won = bool(top and str(top["program_number"]) == winner)
        stats["top_win"] += top_won
        stats["top_top3"] += bool(top and str(top["program_number"]) in podium)

        winner_prob = None
        for horse in horses:
            p = float(horse["win_probability"])
            won = str(horse["program_number"]) == winner
            brier_terms.append((p - (1.0 if won else 0.0)) ** 2)
            if won:
                winner_prob = p
            for bucket in buckets:
                if bucket["low"] <= p < bucket["high"]:
                    bucket["count"] += 1
                    bucket["prob_sum"] += p
                    bucket["wins"] += won
                    break
            verdict = horse.get("verdict")
            if verdict == "FIJO":
                stats["fijo_count"] += 1
                stats["fijo_wins"] += won
                stats["fijo_top3"] += str(horse["program_number"]) in podium
                if race.get("win_payout"):
                    fijo_bets += 1
                    fijo_profit += (float(race["win_payout"]) - 2.0) if won else -2.0
            elif verdict == "TIRAR":
                stats["tirar_count"] += 1
                stats["tirar_wins"] += won
        if winner_prob is not None:
            log_terms.append(-math.log(max(winner_prob, 1e-4)))
        stats["detail"].append(
            {
                "race": race.get("race_name", ""),
                "winner": winner,
                "top_pick": top["program_number"] if top else "",
                "top_prob": top["win_probability"] if top else None,
                "top_won": top_won,
                "winner_prob": winner_prob,
            }
        )

    stats["brier"] = sum(brier_terms) / len(brier_terms)
    stats["log_loss"] = sum(log_terms) / len(log_terms) if log_terms else None
    stats["calibration"] = [
        {
            "range": f"{b['low']:.0%} - {min(b['high'], 1.0):.0%}",
            "count": b["count"],
            "predicted": b["prob_sum"] / b["count"],
            "actual": b["wins"] / b["count"],
        }
        for b in buckets if b["count"]
    ]
    if fijo_bets:
        stats["fijo_bets"] = fijo_bets
        stats["fijo_roi"] = fijo_profit / (2.0 * fijo_bets)
    return stats
