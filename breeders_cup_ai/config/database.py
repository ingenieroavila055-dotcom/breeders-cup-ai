"""Cliente de Supabase y funciones CRUD para races, horses y simulation_results."""

from __future__ import annotations

import math
import os
import uuid
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

from services.simulator import ADJUSTMENT_COLUMNS, PROFILE_FIELDS, parse_morning_line

load_dotenv()

RACES_TABLE = "races"
HORSES_TABLE = "horses"
RESULTS_TABLE = "simulation_results"
RACE_RESULTS_TABLE = "race_results"

# Listas blancas: solo estas columnas se envian a la base de datos.
RACE_COLUMNS = (
    "race_name",
    "race_number",
    "track",
    "race_date",
    "surface",
    "distance",
    "track_condition",
    "source_filename",
)
HORSE_COLUMNS = (
    "program_number",
    "name",
    "morning_line",
    "base_speed_figure",
    "foreign_origin",
    "foreign_level",
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


class DatabaseError(RuntimeError):
    """Error controlado de acceso a Supabase."""


# ---------------------------------------------------------------------------
# Conexion
# ---------------------------------------------------------------------------
def is_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL", "").strip() and os.getenv("SUPABASE_KEY", "").strip())


@lru_cache(maxsize=1)
def get_client() -> Client:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise DatabaseError("Faltan SUPABASE_URL o SUPABASE_KEY en el archivo .env.")
    try:
        return create_client(url, key)
    except Exception as exc:  # noqa: BLE001 - la libreria lanza varios tipos de error
        raise DatabaseError(f"No se pudo crear el cliente de Supabase: {exc}") from exc


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _clean(value: Any) -> Any:
    """Convierte tipos numpy/pandas y NaN a valores JSON validos."""
    if value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _pick(source: dict[str, Any], columns: Iterable[str]) -> dict[str, Any]:
    return {column: _clean(source.get(column)) for column in columns}


MIGRATION_HINT = (
    " Falta actualizar la base de datos: ejecuta el archivo migration_v2.sql en Supabase "
    "(SQL Editor > New query > Run)."
)


def _execute(builder: Any, action: str) -> list[dict[str, Any]]:
    try:
        response = builder.execute()
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        hint = MIGRATION_HINT if any(
            marker in text for marker in ("PGRST204", "PGRST205", "does not exist", "schema cache", "42703", "42P01")
        ) else ""
        raise DatabaseError(f"Supabase fallo al {action}: {exc}.{hint}") from exc
    return list(response.data or [])


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------
def save_race(race: dict[str, Any]) -> str:
    """Inserta una carrera y devuelve su id (uuid)."""
    row = _pick(race, RACE_COLUMNS)
    if not row.get("race_name"):
        raise DatabaseError("La carrera necesita 'race_name'.")
    if not row.get("surface"):
        raise DatabaseError("La carrera necesita 'surface'.")
    row["id"] = str(uuid.uuid4())
    if not row.get("track"):
        row["track"] = "Keeneland"
    if not row.get("track_condition"):
        row["track_condition"] = "Fast"
    _execute(get_client().table(RACES_TABLE).insert(row), "guardar la carrera")
    return row["id"]


def save_horses(race_id: str, horses: list[dict[str, Any]]) -> list[str]:
    """Inserta los caballos de una carrera. Devuelve los ids en el mismo orden."""
    if not horses:
        return []
    rows: list[dict[str, Any]] = []
    for horse in horses:
        row = _pick(horse, HORSE_COLUMNS)
        if not row.get("name"):
            raise DatabaseError("Todos los caballos necesitan 'name'.")
        row["program_number"] = row.get("program_number") or "?"
        row["id"] = str(uuid.uuid4())
        row["race_id"] = race_id
        row["morning_line_decimal"] = parse_morning_line(horse.get("morning_line"))
        row["profile"] = {field: _clean(horse.get(field)) for field in PROFILE_FIELDS}
        rows.append(row)
    _execute(get_client().table(HORSES_TABLE).insert(rows), "guardar los caballos")
    return [row["id"] for row in rows]


def save_simulation_results(
    race_id: str,
    horse_ids: list[str],
    results: pd.DataFrame,
    iterations: int,
    sigma: float,
) -> str:
    """Guarda el resultado de una simulacion. `horse_ids` sigue el orden de la lista
    original de caballos (la columna `horse_key` del DataFrame de resultados).
    Devuelve el run_id que agrupa todas las filas.
    """
    if results.empty:
        raise DatabaseError("No hay resultados para guardar.")
    run_id = str(uuid.uuid4())
    rows: list[dict[str, Any]] = []
    for record in results.to_dict("records"):
        key = int(record["horse_key"])
        if key >= len(horse_ids):
            raise DatabaseError("Los resultados no coinciden con los caballos guardados.")
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "run_id": run_id,
                "race_id": race_id,
                "horse_id": horse_ids[key],
                "final_adjusted_rating": _clean(record["final_adjusted_rating"]),
                "adjustments": {column: _clean(record.get(column)) or 0.0 for column in ADJUSTMENT_COLUMNS},
                "win_probability": _clean(record["win_probability"]),
                "implied_probability": _clean(record.get("implied_probability")),
                "decimal_odds": _clean(record.get("decimal_odds")),
                "expected_value": _clean(record.get("expected_value")),
                "is_positive_ev": bool(record.get("is_positive_ev", False)),
                "iterations": int(iterations),
                "sigma": float(sigma),
                "verdict": _clean(record.get("verdict")),
                "confidence": _clean(record.get("confidence")),
                "confidence_level": _clean(record.get("confidence_level")),
            }
        )
    _execute(get_client().table(RESULTS_TABLE).insert(rows), "guardar los resultados")
    return run_id


def save_full_analysis(
    race: dict[str, Any],
    horses: list[dict[str, Any]],
    results: pd.DataFrame,
    iterations: int,
    sigma: float,
) -> str:
    """Guarda carrera + caballos + resultados. Si algo falla, elimina lo ya insertado
    (el borrado en cascada limpia las tablas hijas). Devuelve el race_id.
    """
    race_id = save_race(race)
    try:
        horse_ids = save_horses(race_id, horses)
        save_simulation_results(race_id, horse_ids, results, iterations, sigma)
    except DatabaseError:
        try:
            delete_race(race_id)
        except DatabaseError:
            pass
        raise
    return race_id


# ---------------------------------------------------------------------------
# Lectura y borrado
# ---------------------------------------------------------------------------
def list_races(limit: int = 50) -> list[dict[str, Any]]:
    builder = (
        get_client()
        .table(RACES_TABLE)
        .select("*")
        .order("created_at", desc=True)
        .limit(max(1, min(int(limit), 200)))
    )
    return _execute(builder, "listar carreras")


def get_horses(race_id: str) -> list[dict[str, Any]]:
    """Caballos de una carrera, con el perfil (jsonb) ya mezclado en cada fila."""
    builder = get_client().table(HORSES_TABLE).select("*").eq("race_id", race_id).order("created_at")
    rows = _execute(builder, "leer los caballos")
    return [{**{k: v for k, v in row.items() if k != "profile"}, **(row.get("profile") or {})} for row in rows]


def get_latest_results(race_id: str) -> list[dict[str, Any]]:
    """Resultados de la simulacion mas reciente de una carrera."""
    client = get_client()
    latest = _execute(
        client.table(RESULTS_TABLE)
        .select("run_id")
        .eq("race_id", race_id)
        .order("created_at", desc=True)
        .limit(1),
        "leer la ultima simulacion",
    )
    if not latest:
        return []
    builder = (
        client.table(RESULTS_TABLE)
        .select("*")
        .eq("run_id", latest[0]["run_id"])
        .order("win_probability", desc=True)
    )
    return _execute(builder, "leer los resultados")


def delete_race(race_id: str) -> None:
    """Elimina una carrera; sus caballos y resultados se borran en cascada."""
    _execute(get_client().table(RACES_TABLE).delete().eq("id", race_id), "eliminar la carrera")


# ---------------------------------------------------------------------------
# Resultados reales y backtesting
# ---------------------------------------------------------------------------
def save_race_result(race_id: str, finish: list[str], win_payout: float | None = None) -> None:
    """Guarda (o reemplaza) el orden de llegada real: numeros de programa del 1.o al 4.o."""
    places = [str(p).strip() for p in finish if str(p).strip()]
    if not places:
        raise DatabaseError("Indica al menos el numero del ganador.")
    if len(set(places)) != len(places):
        raise DatabaseError("Hay numeros de programa repetidos en el orden de llegada.")
    row = {
        "race_id": race_id,
        "first": places[0],
        "second": places[1] if len(places) > 1 else None,
        "third": places[2] if len(places) > 2 else None,
        "fourth": places[3] if len(places) > 3 else None,
        "win_payout": _clean(win_payout) if win_payout else None,
    }
    _execute(get_client().table(RACE_RESULTS_TABLE).upsert(row, on_conflict="race_id"), "guardar el resultado real")


def list_race_results(limit: int = 500) -> list[dict[str, Any]]:
    builder = get_client().table(RACE_RESULTS_TABLE).select("*").order("created_at", desc=True).limit(limit)
    return _execute(builder, "leer los resultados reales")


def get_backtest_rows() -> list[dict[str, Any]]:
    """Una fila por carrera con resultado real: orden de llegada y la ultima simulacion guardada.

    Cada fila: {race_id, race_name, finish: [...], win_payout, horses: [{program_number, win_probability,
    verdict, decimal_odds}]}.
    """
    results = list_race_results()
    if not results:
        return []
    race_ids = [r["race_id"] for r in results]
    client = get_client()
    races = {r["id"]: r for r in _execute(client.table(RACES_TABLE).select("id,race_name,race_number").in_("id", race_ids), "leer las carreras")}
    horses = _execute(client.table(HORSES_TABLE).select("id,race_id,program_number").in_("race_id", race_ids), "leer los caballos")
    program = {h["id"]: h["program_number"] for h in horses}
    sims = _execute(
        client.table(RESULTS_TABLE)
        .select("race_id,run_id,horse_id,win_probability,verdict,decimal_odds,created_at")
        .in_("race_id", race_ids),
        "leer las simulaciones",
    )
    latest: dict[str, tuple[str, str]] = {}
    for row in sims:  # simulacion mas reciente de cada carrera
        current = latest.get(row["race_id"])
        if current is None or row["created_at"] > current[0]:
            latest[row["race_id"]] = (row["created_at"], row["run_id"])

    rows = []
    for result in results:
        race_id = result["race_id"]
        if race_id not in latest:
            continue
        run_id = latest[race_id][1]
        entries = [
            {
                "program_number": program.get(s["horse_id"], "?"),
                "win_probability": float(s["win_probability"]),
                "verdict": s.get("verdict"),
                "decimal_odds": s.get("decimal_odds"),
            }
            for s in sims
            if s["race_id"] == race_id and s["run_id"] == run_id
        ]
        rows.append(
            {
                "race_id": race_id,
                "race_name": races.get(race_id, {}).get("race_name", ""),
                "finish": [p for p in (result["first"], result.get("second"), result.get("third"), result.get("fourth")) if p],
                "win_payout": float(result["win_payout"]) if result.get("win_payout") else None,
                "horses": entries,
            }
        )
    return rows
