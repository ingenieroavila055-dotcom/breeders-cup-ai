"""SISTEMA TYLER PARA BREEDERS CUP - Pronosticos con Monte Carlo (Streamlit, responsive)."""

from __future__ import annotations

import base64
import io
import os
import uuid
from collections import OrderedDict
from datetime import date
from html import escape

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import database as db  # noqa: E402
from services import simulator as sim  # noqa: E402
from services.analysis import build_analysis  # noqa: E402
from services.extractor import (  # noqa: E402
    MAX_IMAGES,
    MAX_PDF_PAGES,
    ExtractionError,
    count_pdf_pages,
    detect_file_kind,
    extract_race_data_from_files,
)

APP_TITLE = "SISTEMA TYLER PARA BREEDERS CUP"
TRACK_NAME = "Keeneland"


def make_favicon():
    """Icono de pestana generado en memoria (cuadrado dorado con una T)."""
    try:
        from PIL import Image, ImageDraw

        icon = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon)
        draw.rounded_rectangle((0, 0, 127, 127), radius=24, fill="#b8995a")
        draw.rectangle((26, 30, 102, 52), fill="#1a1038")
        draw.rectangle((52, 30, 76, 102), fill="#1a1038")
        return icon
    except Exception:  # noqa: BLE001 - el icono es solo decorativo
        return None


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=make_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Sistema de diseno
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --ink:#1c1233;--muted:#6b6285;--line:#e2dcf0;--page:#f4f2f9;--panel:#ffffff;
  --purple:#3d2478;--deep:#1a1038;--gold:#b8995a;--gold-soft:#f5eedb;--pos:#17794a;--neg:#b3261e;
}
html,body,.stApp,.stApp p,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp li,
.stApp [data-baseweb]{font-family:'Inter','Segoe UI',system-ui,-apple-system,Roboto,sans-serif}
.stApp{background:var(--page);color:var(--ink)}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],footer,#MainMenu{display:none!important}
[data-testid="stMain"],[data-testid="stAppViewContainer"]{overflow-x:hidden}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1180px;padding:0 1.25rem 4rem!important}
[data-testid="stVerticalBlock"]{gap:.9rem}
[data-testid="stElementContainer"]:has(style){display:none}

/* Barra de marca */
.bc-bar{margin-left:calc(50% - 50vw);width:100vw;background:var(--deep);border-bottom:3px solid var(--gold)}
.bc-bar-in{max-width:1180px;margin:0 auto;padding:.85rem 1.25rem;display:flex;align-items:center;
justify-content:space-between;gap:1rem}
.bc-brand{display:flex;align-items:center;gap:.8rem;min-width:0}
.bc-mono{flex:0 0 auto;width:42px;height:42px;background:var(--gold);border-radius:7px;position:relative}
.bc-mono::before{content:"";position:absolute;left:10px;right:10px;top:11px;height:8px;background:var(--deep)}
.bc-mono::after{content:"";position:absolute;left:18px;width:8px;top:11px;bottom:9px;background:var(--deep)}
.bc-name{color:#fff;font-weight:800;font-size:1.05rem;letter-spacing:.12em;line-height:1.1}
.bc-sub{color:var(--gold);font-weight:600;font-size:.68rem;letter-spacing:.26em;margin-top:.28rem}
.bc-venue{text-align:right;flex:0 0 auto;border-left:1px solid rgba(255,255,255,.18);padding-left:1rem}
.bc-venue-l{color:#a99fc7;font-size:.62rem;font-weight:600;letter-spacing:.2em}
.bc-venue-v{color:#fff;font-weight:700;font-size:1rem;letter-spacing:.06em}

/* Contexto de la carrera */
.ctx{display:flex;flex-wrap:wrap;gap:.4rem;margin:1.1rem 0 .2rem}
.ctx>span{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:.28rem .75rem;
font-size:.78rem;font-weight:600;color:var(--deep)}
.ctx .k{color:var(--muted);font-weight:500;margin-right:.4rem}

/* Encabezados de paso */
.step{display:flex;align-items:center;gap:.8rem;margin:2.1rem 0 .4rem;padding-bottom:.7rem;
border-bottom:1px solid var(--line)}
.step-no{background:var(--deep);color:var(--gold);font-weight:700;font-size:.78rem;letter-spacing:.08em;
padding:.32rem .55rem;border-radius:4px}
.step-t{font-weight:700;font-size:1.18rem;color:var(--deep);line-height:1.2}
.step-s{color:var(--muted);font-size:.82rem;margin-top:.1rem}
.sub-h{font-weight:700;font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;color:var(--purple);
margin:1.4rem 0 .5rem}

/* Widgets */
[data-testid="stWidgetLabel"] p{font-size:.7rem!important;font-weight:600!important;letter-spacing:.07em;
text-transform:uppercase;color:var(--muted)!important}
[data-testid="stTextInputRootElement"],[data-testid="stNumberInputContainer"],[data-testid="stDateInputField"],
[data-testid="stSelectbox"] div:has(>input),[data-baseweb="select"]>div,[data-baseweb="input"]{
background:#fff!important;border:1px solid #cfc7e3!important;border-radius:6px!important}
[data-testid="stTextInputRootElement"]:focus-within,[data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stDateInputField"]:focus-within,[data-testid="stSelectbox"] div:has(>input):focus-within{
border-color:var(--purple)!important;box-shadow:0 0 0 3px rgba(61,36,120,.12)}
[data-testid="stElementContainer"]:has(>[data-testid="stButton"]){width:100%!important}
[data-testid="stButton"]{width:100%}
.stButton button{width:100%;min-height:2.9rem;border-radius:6px;font-weight:600;letter-spacing:.01em;
border:1px solid var(--purple);transition:background .15s,border-color .15s}
.stButton button[data-testid="stBaseButton-primary"],.stButton>button[kind="primary"]{background:var(--purple)}
.stButton button[data-testid="stBaseButton-primary"] p,.stButton>button[kind="primary"] p{color:#fff!important}
.stButton button[data-testid="stBaseButton-primary"]:hover,.stButton>button[kind="primary"]:hover{
background:var(--deep);border-color:var(--deep)}
.stButton button[data-testid="stBaseButton-secondary"],.stButton>button[kind="secondary"]{background:#fff}
.stButton button[data-testid="stBaseButton-secondary"] p,.stButton>button[kind="secondary"] p{color:var(--purple)!important}
.stButton button[data-testid="stBaseButton-secondary"]:hover,.stButton>button[kind="secondary"]:hover{background:var(--page)}
.stButton button:disabled{opacity:.45}
[data-testid="stExpander"]{border:none!important;background:transparent}
[data-testid="stExpander"] details{background:var(--panel);border:1px solid var(--line)!important;border-radius:8px;
box-shadow:0 1px 2px rgba(26,16,56,.05)}
[data-testid="stExpander"] summary{padding:.7rem 1rem}
[data-testid="stExpander"] summary p{font-weight:600;color:var(--deep);font-size:.95rem}
[data-testid="stFileUploaderDropzone"]{background:var(--panel);border:1.5px dashed #c5bbe0;border-radius:8px}
[data-testid="stAlert"]{border-radius:6px!important;border:1px solid var(--line)!important;
border-left:4px solid var(--gold)!important;background:var(--panel)!important;color:var(--ink)}
[data-testid="stCaptionContainer"]{color:var(--muted);margin-top:.5rem}

/* Vista previa de fotos */
.thumbs{display:flex;flex-wrap:wrap;gap:.8rem}
.thumbs figure{margin:0;width:140px}
.thumbs img{display:block;width:100%;border:1px solid var(--line);border-radius:4px;background:#fff}
.thumbs figcaption{font-size:.7rem;color:var(--muted);margin-top:.25rem;overflow-wrap:anywhere}

/* Indicadores */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.7rem;margin:.3rem 0 .4rem}
.kpi{background:var(--panel);border:1px solid var(--line);border-top:3px solid var(--gold);border-radius:6px;
padding:.8rem 1rem}
.kpi-l{font-size:.66rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.kpi-v{font-size:1.45rem;font-weight:700;color:var(--deep);font-variant-numeric:tabular-nums;line-height:1.2;
overflow-wrap:anywhere;margin-top:.15rem}
.kpi-s{font-size:.8rem;color:var(--muted);margin-top:.1rem}

/* Tablero de ranking */
.lb{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden;
box-shadow:0 1px 2px rgba(26,16,56,.05)}
.lb-head,.lb-row{display:grid;grid-template-columns:3rem minmax(0,1fr) 5.2rem 13rem 4.6rem 5.4rem;gap:0 1rem;
align-items:center;padding:.7rem 1rem}
.lb-head{background:var(--deep);color:#cfc7e6;font-size:.66rem;font-weight:600;letter-spacing:.12em;
text-transform:uppercase;padding-top:.6rem;padding-bottom:.6rem}
.lb-row{border-top:1px solid var(--line)}
.lb-row.value{background:linear-gradient(90deg,var(--gold-soft),#fff 40%);box-shadow:inset 4px 0 0 var(--gold)}
.lb-pos{font-weight:800;font-size:1.15rem;color:var(--purple);font-variant-numeric:tabular-nums}
.lb-name{font-weight:700;color:var(--deep);overflow-wrap:anywhere}
.lb-num{display:inline-block;min-width:1.5rem;text-align:center;background:var(--page);border:1px solid var(--line);
border-radius:4px;font-size:.75rem;font-weight:700;padding:.02rem .3rem;margin-right:.5rem;color:var(--purple)}
.lb-meta{font-size:.78rem;color:var(--muted);margin-top:.1rem}
.lb-m{display:none;font-size:.78rem;color:var(--muted);margin-top:.1rem}
.lb-chips{display:flex;flex-wrap:wrap;gap:.25rem;margin-top:.4rem}
.chip{font-size:.66rem;font-weight:600;padding:.04rem .45rem;border-radius:4px;border:1px solid var(--line);
background:var(--page);color:var(--purple)}
.chip.neg{color:var(--neg);border-color:#efc9c5;background:#fdf3f2}
.lb-rating,.lb-ml{font-variant-numeric:tabular-nums;font-weight:600;color:var(--ink)}
.lb-probval{font-weight:700;font-size:1.05rem;color:var(--deep);font-variant-numeric:tabular-nums}
.lb-bar{height:6px;background:#ebe6f5;border-radius:3px;margin-top:.3rem;overflow:hidden}
.lb-bar span{display:block;height:100%;background:linear-gradient(90deg,var(--purple),var(--gold))}
.evp{display:inline-block;font-weight:700;font-size:.8rem;padding:.2rem .5rem;border-radius:4px;
font-variant-numeric:tabular-nums}
.evp.pos{background:#e4f4ea;color:var(--pos)}.evp.neg{background:#fbeceb;color:var(--neg)}.evp.na{background:var(--page);color:var(--muted)}
.lb-ev{text-align:right}

/* Jugadas con valor */
.vp{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.7rem}
.vp-card{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--pos);border-radius:6px;
padding:.85rem 1rem}
.vp-top{display:flex;justify-content:space-between;align-items:flex-start;gap:.8rem}
.vp-name{font-weight:700;color:var(--deep);overflow-wrap:anywhere}
.vp-ev{font-weight:800;font-size:1.25rem;color:var(--pos);font-variant-numeric:tabular-nums;white-space:nowrap}
.vp-dl{display:grid;grid-template-columns:1fr 1fr;gap:.25rem .8rem;margin-top:.6rem;font-size:.78rem}
.vp-dl div span{display:block;color:var(--muted);font-size:.64rem;letter-spacing:.08em;text-transform:uppercase}
.vp-dl div b{color:var(--ink);font-variant-numeric:tabular-nums}

/* Orden de llegada y comentarios */
.fo{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.7rem}
.fo-card{background:var(--panel);border:1px solid var(--line);border-top:4px solid var(--purple);border-radius:6px;padding:.85rem 1rem}
.fo-card.p1{border-top-color:var(--gold)}
.fo-place{font-size:.64rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700}
.fo-name{font-weight:800;color:var(--deep);font-size:1.02rem;margin-top:.15rem;overflow-wrap:anywhere}
.fo-stats{display:grid;grid-template-columns:1fr 1fr;gap:.2rem .6rem;margin-top:.55rem;font-size:.76rem;color:var(--muted)}
.fo-stats b{color:var(--ink);font-variant-numeric:tabular-nums}
.fo-why{margin-top:.55rem;font-size:.76rem;color:var(--ink);line-height:1.4}
.fo-combo{margin-top:.6rem;font-size:.78rem;color:var(--muted)}
.so{background:var(--gold-soft);border:1px solid var(--gold);border-left:5px solid var(--gold);border-radius:6px;padding:.9rem 1.1rem}
.so.pos{background:#e4f4ea;border-color:var(--pos)}
.so-t{font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;font-weight:700;color:var(--purple)}
.so-n{font-weight:800;font-size:1.15rem;color:var(--deep);margin:.1rem 0;overflow-wrap:anywhere}
.so-s{font-size:.86rem;color:var(--ink)}.so-v{font-size:.86rem;color:var(--ink);margin-top:.4rem;font-weight:600}
.so-a{margin-top:.6rem;font-size:.82rem;color:var(--ink)}
.cm{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.9rem 1.1rem;margin-top:.1rem}
.cm p{margin:.15rem 0 .5rem;font-size:.9rem;line-height:1.5;color:var(--ink)}
.cm-t{font-weight:700;color:var(--deep);font-size:.95rem}
.cm ul{margin:.2rem 0 0 1.1rem;padding:0}.cm li{font-size:.84rem;line-height:1.45;margin-bottom:.3rem;color:var(--ink)}

@media (max-width:760px){
.block-container,[data-testid="stMainBlockContainer"]{padding:0 .8rem 3rem!important}
.bc-bar-in{padding:.7rem .8rem}
.bc-name{font-size:.92rem;letter-spacing:.09em}
.bc-sub{font-size:.6rem;letter-spacing:.2em}
.bc-mono{width:36px;height:36px}.bc-mono::before{left:8px;right:8px;top:9px;height:7px}
.bc-mono::after{left:15px;width:7px;top:9px;bottom:7px}
.bc-venue{display:none}
.step{margin-top:1.6rem}
.kpis{grid-template-columns:1fr 1fr;gap:.5rem}
.kpi{padding:.65rem .75rem}.kpi-v{font-size:1.1rem}.kpi-l{font-size:.6rem}
.lb-head{display:none}
.lb-row{grid-template-columns:2rem minmax(0,1fr) auto;grid-template-areas:"pos horse ev" "prob prob prob";
gap:.4rem .7rem;padding:.75rem .8rem}
.lb-pos{grid-area:pos}.lb-horse{grid-area:horse}.lb-ev{grid-area:ev}.lb-prob{grid-area:prob}
.lb-rating,.lb-ml{display:none}.lb-m{display:block}
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Catalogos
# ---------------------------------------------------------------------------
BREEDERS_CUP_RACES = [
    {"label": "Juvenile Turf Sprint (G1)", "name": "Juvenile Turf Sprint", "surface": "Turf", "distance": "5 furlongs"},
    {"label": "Juvenile Fillies Turf (G1)", "name": "Juvenile Fillies Turf", "surface": "Turf", "distance": "1 mile"},
    {"label": "Juvenile Turf (G1)", "name": "Juvenile Turf", "surface": "Turf", "distance": "1 mile"},
    {"label": "Juvenile Fillies (G1)", "name": "Juvenile Fillies", "surface": "Dirt", "distance": "1 1/16 miles"},
    {"label": "Juvenile (G1)", "name": "Juvenile", "surface": "Dirt", "distance": "1 1/16 miles"},
    {"label": "Filly & Mare Sprint (G1)", "name": "Filly & Mare Sprint", "surface": "Dirt", "distance": "7 furlongs"},
    {"label": "Distaff (G1)", "name": "Distaff", "surface": "Dirt", "distance": "1 1/8 miles"},
    {"label": "Filly & Mare Turf (G1)", "name": "Filly & Mare Turf", "surface": "Turf", "distance": "1 3/8 miles"},
    {"label": "Turf Sprint (G1)", "name": "Turf Sprint", "surface": "Turf", "distance": "5 furlongs"},
    {"label": "Dirt Mile (G1)", "name": "Dirt Mile", "surface": "Dirt", "distance": "1 mile"},
    {"label": "Mile (G1)", "name": "Mile", "surface": "Turf", "distance": "1 mile"},
    {"label": "Sprint (G1)", "name": "Sprint", "surface": "Dirt", "distance": "6 furlongs"},
    {"label": "Turf (G1)", "name": "Turf", "surface": "Turf", "distance": "1 1/2 miles"},
    {"label": "Classic (G1)", "name": "Classic", "surface": "Dirt", "distance": "1 1/4 miles"},
    {"label": "Otra carrera (manual)", "name": "", "surface": "Dirt", "distance": ""},
]

DEMO_FIELD = [
    {"program_number": "1", "name": "Demo Rapido", "morning_line": "5/2", "base_speed_figure": 100,
     "running_style": "E", "form_trend": "Plateau", "class_rating": "Graded_Placed",
     "jockey": "Irad Ortiz Jr.", "is_regular_jockey": True, "jockey_switch": "Neutral",
     "track_aptitude": "Positive", "weight_lbs": 126, "last_weight_lbs": 126},
    {"program_number": "2", "name": "Demo Europeo", "morning_line": "7/2", "base_speed_figure": 101,
     "foreign_origin": "Europe", "foreign_level": "G1_Winner", "running_style": "P",
     "form_trend": "Plateau", "class_rating": "G1_Winner", "jockey": "Ryan Moore",
     "is_regular_jockey": True, "track_aptitude": "First_Time", "mud_proven": True,
     "weight_lbs": 126, "last_weight_lbs": 126},
    {"program_number": "3", "name": "Demo Japones", "morning_line": "9/2", "base_speed_figure": 102,
     "foreign_origin": "Japan", "foreign_level": "G1_Placed", "dirt_debut": True, "running_style": "S",
     "form_trend": "Ascending", "class_rating": "Graded_Placed", "jockey": "Y. Take",
     "is_regular_jockey": True, "track_aptitude": "First_Time", "weight_lbs": 126, "last_weight_lbs": 124},
    {"program_number": "4", "name": "Demo Presser", "morning_line": "6/1", "base_speed_figure": 99,
     "running_style": "P", "form_trend": "Plateau", "class_rating": "Graded_Placed",
     "jockey": "Flavien Prat", "is_regular_jockey": True, "track_aptitude": "Positive",
     "weight_lbs": 126, "last_weight_lbs": 126},
    {"program_number": "5", "name": "Demo Cerrador", "morning_line": "8/1", "base_speed_figure": 98,
     "running_style": "S", "form_trend": "Ascending", "class_rating": "Graded_Placed",
     "jockey": "J. Smith", "is_regular_jockey": True, "track_aptitude": "Positive",
     "weight_lbs": 126, "last_weight_lbs": 126},
    {"program_number": "6", "name": "Demo Subida", "morning_line": "12/1", "base_speed_figure": 96,
     "running_style": "P", "form_trend": "Descending", "class_rating": "Class_Riser",
     "jockey": "A. Perez", "is_regular_jockey": True, "track_aptitude": "Negative",
     "weight_lbs": 126, "last_weight_lbs": 120},
    {"program_number": "7", "name": "Demo Lodo", "morning_line": "15/1", "base_speed_figure": 95,
     "running_style": "S", "form_trend": "Plateau", "class_rating": "Graded_Placed",
     "jockey": "L. Saez", "is_regular_jockey": True, "track_aptitude": "First_Time", "mud_proven": True,
     "weight_lbs": 126, "last_weight_lbs": 126},
    {"program_number": "8", "name": "Demo Retirado", "morning_line": "20/1", "base_speed_figure": 90,
     "running_style": "E", "scratched": True},
]

EDITOR_COLUMNS = {
    "program_number": st.column_config.TextColumn("#", width="small"),
    "name": st.column_config.TextColumn("Caballo", required=True),
    "morning_line": st.column_config.TextColumn("ML", help="Morning Line. Ej: 5/2, 9-2, even", width="small"),
    "base_speed_figure": st.column_config.NumberColumn("Speed Fig", min_value=0, max_value=120, step=1, width="small"),
    "foreign_origin": st.column_config.SelectboxColumn("Origen", options=list(sim.FOREIGN_ORIGINS)),
    "foreign_level": st.column_config.SelectboxColumn("Nivel extranjero", options=list(sim.FOREIGN_LEVELS)),
    "dirt_debut": st.column_config.CheckboxColumn("Debut en Dirt"),
    "form_trend": st.column_config.SelectboxColumn("Forma", options=list(sim.FORM_TRENDS)),
    "class_rating": st.column_config.SelectboxColumn("Lote", options=list(sim.CLASS_RATINGS)),
    "jockey": st.column_config.TextColumn("Jinete"),
    "is_regular_jockey": st.column_config.CheckboxColumn("Jinete habitual"),
    "jockey_switch": st.column_config.SelectboxColumn("Cambio jinete", options=list(sim.JOCKEY_SWITCHES)),
    "track_aptitude": st.column_config.SelectboxColumn("Aptitud pista", options=list(sim.TRACK_APTITUDES)),
    "mud_proven": st.column_config.CheckboxColumn("Probado en fango"),
    "running_style": st.column_config.SelectboxColumn("Estilo", options=list(sim.RUNNING_STYLES), width="small"),
    "weight_lbs": st.column_config.NumberColumn("Peso (lbs)", min_value=0, step=1),
    "last_weight_lbs": st.column_config.NumberColumn("Peso ult. (lbs)", min_value=0, step=1),
    "scratched": st.column_config.CheckboxColumn("Retirado"),
}
COMPACT_COLUMNS = [
    "program_number", "name", "morning_line", "base_speed_figure",
    "running_style", "form_trend", "class_rating", "jockey", "scratched",
]
CHIP_LABELS = {
    "adj_foreign": "Extranjero",
    "adj_form": "Forma",
    "adj_class": "Lote",
    "adj_jockey": "Jinete",
    "adj_track": "Pista",
    "adj_condition": "Condicion",
    "adj_pace": "Ritmo",
    "adj_weight": "Peso",
}
STYLE_LABELS = {"E": "Puntero", "P": "Presser", "S": "Rematador"}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def horses_to_dataframe(horses: list[dict]) -> pd.DataFrame:
    """DataFrame con todas las columnas del editor y tipos consistentes."""
    cleaned = [sim.normalize_horse(h) for h in horses]
    frame = pd.DataFrame(cleaned, columns=list(sim.HORSE_FIELDS))
    for column in ("base_speed_figure", "weight_lbs", "last_weight_lbs"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def set_horses(horses: list[dict]) -> None:
    st.session_state["horses_df"] = horses_to_dataframe(horses)
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state.pop("results", None)
    st.session_state.pop("saved_race_id", None)


def build_results_table(results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pos": results["rank"],
            "#": results["program_number"],
            "Caballo": results["name"],
            "ML": results["morning_line"],
            "Base": results["base_speed_figure"],
            "Extranjero": results["adj_foreign"],
            "Forma": results["adj_form"],
            "Lote": results["adj_class"],
            "Jinete": results["adj_jockey"],
            "Pista": results["adj_track"],
            "Condicion": results["adj_condition"],
            "Ritmo": results["adj_pace"],
            "Peso": results["adj_weight"],
            "Rating final": results["final_adjusted_rating"],
            "Prob. simulada": results["win_probability"],
            "Prob. implicita": results["implied_probability"],
            "Cuota decimal": results["decimal_odds"],
            "EV": results["expected_value"],
        }
    )


def style_results(table: pd.DataFrame):
    def highlight(row: pd.Series) -> list[str]:
        positive = pd.notna(row["EV"]) and row["EV"] > 0
        color = "background-color: rgba(184, 153, 90, 0.22); font-weight: 600" if positive else ""
        return [color] * len(row)

    return table.style.apply(highlight, axis=1).format(
        {
            "Base": "{:.1f}",
            "Extranjero": "{:+.2f}",
            "Forma": "{:+.2f}",
            "Lote": "{:+.2f}",
            "Jinete": "{:+.2f}",
            "Pista": "{:+.2f}",
            "Condicion": "{:+.2f}",
            "Ritmo": "{:+.2f}",
            "Peso": "{:+.2f}",
            "Rating final": "{:.2f}",
            "Prob. simulada": "{:.1%}",
            "Prob. implicita": "{:.1%}",
            "Cuota decimal": "{:.2f}",
            "EV": "{:+.1%}",
        },
        na_rep="n/d",
    )


# ---------------------------------------------------------------------------
# Componentes HTML (sin saltos de linea ni sangrias: Markdown los rompe)
# ---------------------------------------------------------------------------
def _has_value(value) -> bool:
    return value is not None and pd.notna(value)


def step_header(number: str, title: str, subtitle: str = "") -> None:
    sub = f'<div class="step-s">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="step"><span class="step-no">{escape(number)}</span>'
        f'<div><div class="step-t">{escape(title)}</div>{sub}</div></div>',
        unsafe_allow_html=True,
    )


def brand_bar_html(track: str) -> str:
    return (
        '<div class="bc-bar"><div class="bc-bar-in">'
        '<div class="bc-brand"><div class="bc-mono"></div><div>'
        '<div class="bc-name">SISTEMA TYLER</div><div class="bc-sub">PARA BREEDERS CUP</div></div></div>'
        '<div class="bc-venue"><div class="bc-venue-l">HIPODROMO</div>'
        f'<div class="bc-venue-v">{escape(track.upper())}</div></div>'
        "</div></div>"
    )


def context_html(items: list[tuple[str, str]]) -> str:
    return '<div class="ctx">' + "".join(
        f'<span><span class="k">{escape(k)}</span>{escape(v)}</span>' for k, v in items
    ) + "</div>"


def _chips(record: dict) -> str:
    chips = []
    for column, label in CHIP_LABELS.items():
        value = float(record[column])
        if abs(value) >= 0.005:
            css = "chip" if value > 0 else "chip neg"
            decimals = 1 if abs(value * 10 - round(value * 10)) < 1e-9 else 2
            chips.append(f'<span class="{css}">{escape(label)} {value:+.{decimals}f}</span>')
    return f'<div class="lb-chips">{"".join(chips)}</div>' if chips else ""


def _ev_pill(record: dict) -> str:
    ev = record["expected_value"]
    if not _has_value(ev):
        return '<span class="evp na">n/d</span>'
    return f'<span class="evp {"pos" if ev > 0 else "neg"}">{ev:+.1%}</span>'


def kpis_html(records: list[dict], iterations: int, sigma: float) -> str:
    favorite = records[0]
    with_ev = [r for r in records if _has_value(r["expected_value"])]
    best = max(with_ev, key=lambda r: r["expected_value"]) if with_ev else None
    positives = sum(1 for r in records if r["is_positive_ev"])
    tiles = [
        ("Favorito del modelo", escape(str(favorite["name"])), f'{favorite["win_probability"]:.1%} de probabilidad'),
        (
            "Mejor valor (EV)",
            escape(str(best["name"])) if best else "n/d",
            f'EV {best["expected_value"]:+.1%}' if best else "Falta el Morning Line",
        ),
        ("Jugadas +EV", str(positives), f"de {len(records)} caballos activos"),
        ("Simulacion", f"{iterations:,}", f"iteraciones · sigma {sigma:.2f}"),
    ]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="kpi-l">{label}</div><div class="kpi-v">{value}</div>'
        f'<div class="kpi-s">{sub}</div></div>'
        for label, value, sub in tiles
    ) + "</div>"


def leaderboard_html(records: list[dict], horses: list[dict]) -> str:
    max_probability = max(r["win_probability"] for r in records) or 1.0
    rows = [
        '<div class="lb"><div class="lb-head"><span>Pos</span><span>Caballo</span><span>Rating</span>'
        "<span>Prob. de ganar</span><span>ML</span><span style=\"text-align:right\">EV</span></div>"
    ]
    for record in records:
        horse = horses[int(record["horse_key"])] if int(record["horse_key"]) < len(horses) else {}
        style = STYLE_LABELS.get(horse.get("running_style", ""), "")
        meta_parts = [p for p in (horse.get("jockey", ""), style) if p]
        meta = escape(" · ".join(meta_parts)) if meta_parts else "&nbsp;"
        morning_line = escape(str(record["morning_line"] or "s/ML"))
        width = record["win_probability"] / max_probability * 100
        rows.append(
            f'<div class="lb-row{" value" if record["is_positive_ev"] else ""}">'
            f'<div class="lb-pos">{record["rank"]}</div>'
            f'<div class="lb-horse"><div class="lb-name"><span class="lb-num">{escape(str(record["program_number"]))}</span>'
            f'{escape(str(record["name"]))}</div><div class="lb-meta">{meta}</div>'
            f'<div class="lb-m">ML {morning_line} · Rating {record["final_adjusted_rating"]:.1f}</div>'
            f"{_chips(record)}</div>"
            f'<div class="lb-rating">{record["final_adjusted_rating"]:.1f}</div>'
            f'<div class="lb-prob"><div class="lb-probval">{record["win_probability"]:.1%}</div>'
            f'<div class="lb-bar"><span style="width:{width:.1f}%"></span></div></div>'
            f'<div class="lb-ml">{morning_line}</div>'
            f'<div class="lb-ev">{_ev_pill(record)}</div></div>'
        )
    rows.append("</div>")
    return "".join(rows)


def value_cards_html(records: list[dict]) -> str:
    cards = []
    for record in records:
        probability = record["win_probability"]
        fair = (1.0 / probability - 1.0) if probability > 0 else float("inf")
        implied = record["implied_probability"]
        implied_text = f"{implied:.1%}" if _has_value(implied) else "n/d"
        cards.append(
            '<div class="vp-card"><div class="vp-top">'
            f'<div class="vp-name">#{escape(str(record["program_number"]))} {escape(str(record["name"]))}</div>'
            f'<div class="vp-ev">{record["expected_value"]:+.1%}</div></div>'
            '<div class="vp-dl">'
            f'<div><span>Prob. simulada</span><b>{probability:.1%}</b></div>'
            f'<div><span>Prob. implicita</span><b>{implied_text}</b></div>'
            f'<div><span>Morning Line</span><b>{escape(str(record["morning_line"]))}</b></div>'
            f'<div><span>Cuota justa</span><b>{fair:.1f}-1</b></div>'
            "</div></div>"
        )
    return '<div class="vp">' + "".join(cards) + "</div>"



def finish_order_html(analysis: dict) -> str:
    cards = []
    for item in analysis["finish"]:
        reasons = "".join(f"<div>- {escape(r)}</div>" for r in item["reasons"]) or "<div>Sin factores de ajuste relevantes.</div>"
        cards.append(
            f'<div class="fo-card{" p1" if item["place"] == 1 else ""}">'
            f'<div class="fo-place">{item["place"]}.o lugar</div>'
            f'<div class="fo-name">{escape(item["label"])}</div>'
            '<div class="fo-stats">'
            f'<span>Gana <b>{item["win_probability"]:.0%}</b></span>'
            f'<span>Top 3 <b>{item["p_top3"]:.0%}</b></span>'
            f'<span>Top 4 <b>{item["p_top4"]:.0%}</b></span>'
            f'<span>Llegada media <b>{item["avg_finish"]:.1f}</b></span></div>'
            f'<div class="fo-why">{reasons}</div></div>'
        )
    combo = ""
    if analysis["combos"]:
        numbers, frequency = analysis["combos"][0]
        combo = (
            f'<div class="fo-combo">Combinacion exacta mas frecuente: <b>{" - ".join(escape(str(n)) for n in numbers)}</b> '
            f'({frequency:.1%} de las simulaciones). Ninguna combinacion exacta es probable por si sola; '
            "usa el orden como guia, no como certeza.</div>"
        )
    return '<div class="fo">' + "".join(cards) + "</div>" + combo


def standout_html(analysis: dict) -> str:
    standout = analysis.get("standout")
    anchors = analysis.get("anchors") or []
    if not standout and not anchors:
        return ""
    parts = []
    if standout:
        ratio = f" Tiene {standout['ratio']:.1f}x la probabilidad del segundo ({escape(standout['second'])})." if standout["ratio"] else ""
        parts.append(
            f'<div class="so{" pos" if standout["is_positive_ev"] else ""}">'
            f'<div class="so-t">{escape(standout["title"])}</div>'
            f'<div class="so-n">{escape(standout["label"])}</div>'
            f'<div class="so-s">Gana {standout["win_probability"]:.0%} de las simulaciones y llega entre los 3 primeros '
            f'{standout["p_top3"]:.0%}.{ratio}</div>'
            f'<div class="so-v">{escape(standout["verdict"])}</div></div>'
        )
    if anchors:
        parts.append(
            '<div class="so-a"><b>Bases para exoticas:</b> '
            + escape(", ".join(anchors))
            + " (llegan casi siempre entre los 3 primeros).</div>"
        )
    return "".join(parts)


def commentary_html(analysis: dict) -> str:
    summary = "".join(f"<p>{escape(line)}</p>" for line in analysis["summary"])
    pace = analysis["pace"]
    alerts = "".join(f"<li>{escape(a)}</li>" for a in analysis["alerts"])
    return (
        f'<div class="cm">{summary}<div class="cm-t">Planteamiento probable: {escape(pace["title"])}</div>'
        f'<p>{escape(pace["text"])}</p><div class="cm-t">Puntos a tener en cuenta</div><ul>{alerts}</ul></div>'
    )


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
api_ok = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
db_ok = db.is_configured()
track = TRACK_NAME

# ---------------------------------------------------------------------------
# Recuperacion de sesion: si el navegador (p. ej. el celular) recarga la pagina,
# el trabajo se restaura desde la memoria del servidor usando el parametro ?s=
# de la URL. Si la aplicacion se reinicia en el servidor, la memoria se pierde.
# ---------------------------------------------------------------------------
SESSION_STORE_MAX = 40
PERSISTED_PREFIXES = ("w_", "name_", "distance_", "surface_", "cond_")
PERSISTED_KEYS = ("horses_df", "editor_version", "results", "source_filename", "saved_race_id")


@st.cache_resource
def _session_store() -> OrderedDict:
    return OrderedDict()


session_token = st.query_params.get("s")
if not session_token:
    session_token = uuid.uuid4().hex[:12]
    st.query_params["s"] = session_token

if not st.session_state.get("_session_restored"):
    st.session_state["_session_restored"] = True
    snapshot = _session_store().get(session_token)
    if snapshot:
        for state_key, state_value in snapshot.items():
            st.session_state[state_key] = state_value


def persist_session(current_df) -> None:
    snapshot = {
        key: value
        for key, value in st.session_state.items()
        if key in PERSISTED_KEYS or (isinstance(key, str) and key.startswith(PERSISTED_PREFIXES))
    }
    if current_df is not None:
        snapshot["horses_df"] = current_df.reset_index(drop=True).copy()
    store = _session_store()
    store[session_token] = snapshot
    store.move_to_end(session_token)
    while len(store) > SESSION_STORE_MAX:
        store.popitem(last=False)


st.markdown(brand_bar_html(track), unsafe_allow_html=True)
context_slot = st.empty()

# ---------------------------------------------------------------------------
# 01 Configuracion
# ---------------------------------------------------------------------------
step_header("01", "Configuración de la carrera", "Carrera oficial, superficie y condición de la pista")
with st.expander("Datos de la carrera", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        race_labels = [r["label"] for r in BREEDERS_CUP_RACES]
        st.session_state.setdefault("w_race", race_labels[13])
        race_label = st.selectbox("Carrera oficial", race_labels, key="w_race")
        race_def = next(r for r in BREEDERS_CUP_RACES if r["label"] == race_label)
        st.session_state.setdefault(f"name_{race_label}", race_def["name"])
        race_name = st.text_input("Nombre de la carrera", key=f"name_{race_label}").strip()
        st.session_state.setdefault("w_number", 0)
        race_number = st.number_input(
            "Numero de carrera (0 = sin numero)", min_value=0, max_value=20, step=1, key="w_number"
        )
    with col_b:
        st.session_state.setdefault(f"surface_{race_label}", race_def["surface"])
        surface = st.selectbox("Superficie", list(sim.SURFACES), key=f"surface_{race_label}")
        st.session_state.setdefault(f"distance_{race_label}", race_def["distance"])
        distance = st.text_input("Distancia", key=f"distance_{race_label}").strip()
        st.session_state.setdefault("w_date", date.today())
        race_date = st.date_input("Fecha", key="w_date")

    st.session_state.setdefault(f"cond_{surface}", "Firm" if surface == "Turf" else "Fast")
    track_condition = st.selectbox(
        "Condicion de pista",
        list(sim.TRACK_CONDITIONS),
        key=f"cond_{surface}",
        help="Fast/Firm = rapida. Yielding/Soft/Muddy/Sloppy/Heavy = suave/humeda (activa el bonus de +2.5).",
    )
    if track_condition in sim.WET_CONDITIONS:
        st.warning("Pista suave/humeda: bonus a europeos en grama y a caballos probados en fango.")

with st.expander("Ajustes de simulación"):
    st.session_state.setdefault("w_iter", sim.DEFAULT_ITERATIONS)
    iterations = st.number_input("Iteraciones", min_value=1_000, max_value=200_000, step=1_000, key="w_iter")
    st.session_state.setdefault("w_sigma", sim.DEFAULT_SIGMA)
    sigma = st.slider(
        "Dispersion (sigma, puntos)", min_value=1.0, max_value=10.0, step=0.25, key="w_sigma",
        help="Mayor sigma = carrera mas impredecible (mas probabilidad a los outsiders).",
    )
    st.session_state.setdefault("w_seed", 0)
    seed_value = st.number_input("Semilla (0 = aleatoria)", min_value=0, step=1, key="w_seed")

context_slot.markdown(
    context_html(
        [
            ("Carrera", race_name or "Sin nombre"),
            ("Superficie", surface),
            ("Distancia", distance or "n/d"),
            ("Pista", track_condition),
            ("Fecha", race_date.strftime("%d/%m/%Y")),
        ]
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 02 Past Performances
# ---------------------------------------------------------------------------
if "horses_df" not in st.session_state:
    st.session_state["horses_df"] = horses_to_dataframe([])
    st.session_state["editor_version"] = 0

step_header(
    "02", "Past Performances",
    "Sube el PDF o fotos del programa (DRF / Brisnet), o carga un campo de ejemplo",
)
uploaded_files = st.file_uploader(
    "Past Performances (PDF o fotos)",
    type=["pdf", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    help=f"PDF, PNG, JPG o WEBP. Puedes subir varias fotos de la misma carrera (máximo {MAX_IMAGES}).",
)


@st.cache_data(show_spinner=False)
def cached_page_count(file_id: str, _data: bytes) -> int:
    return count_pdf_pages(_data)


@st.cache_data(show_spinner=False)
def cached_thumbnail(file_id: str, _data: bytes) -> bytes | None:
    """Miniatura liviana para la vista previa (las fotos de celular pesan varios MB)."""
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(_data)) as source:
            thumb = ImageOps.exif_transpose(source).convert("RGB")
        thumb.thumbnail((360, 360))
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()
    except Exception:  # noqa: BLE001 - la vista previa es opcional
        return None


pdf_pages_spec = ""
needs_page_range = False
upload_blocked = False
upload_files: list[tuple[str, bytes]] = [(f.name, f.getvalue()) for f in (uploaded_files or [])]
if upload_files:
    kinds = [detect_file_kind(data) for _, data in upload_files]
    pdf_positions = [i for i, kind in enumerate(kinds) if kind == "pdf"]
    image_positions = [i for i, kind in enumerate(kinds) if kind not in ("pdf", "unknown")]

    for position, kind in enumerate(kinds):
        if kind == "unknown":
            st.error(f"'{upload_files[position][0]}' no es un PDF ni una imagen compatible (PNG, JPG o WEBP).")
            upload_blocked = True
    if len(image_positions) > MAX_IMAGES:
        st.error(f"Subiste {len(image_positions)} imágenes y el máximo es {MAX_IMAGES}.")
        upload_blocked = True

    page_counts: list[int] = []
    try:
        page_counts = [
            cached_page_count(uploaded_files[i].file_id, upload_files[i][1]) for i in pdf_positions
        ]
    except ExtractionError as exc:
        st.error(str(exc))
        upload_blocked = True

    summary = []
    if pdf_positions and page_counts:
        summary.append(f"{len(pdf_positions)} PDF ({sum(page_counts)} páginas)")
    if image_positions:
        summary.append(f"{len(image_positions)} imagen(es)")
    if summary:
        st.caption("Material cargado: " + " y ".join(summary) + ".")

    if image_positions:
        with st.expander("Vista previa de las imágenes"):
            thumbs = [
                (cached_thumbnail(uploaded_files[i].file_id, upload_files[i][1]), upload_files[i][0])
                for i in image_positions
            ]
            shown = [(data, name) for data, name in thumbs if data]
            if shown:
                st.markdown(
                    '<div class="thumbs">'
                    + "".join(
                        f'<figure><img alt="{escape(name)}" src="data:image/jpeg;base64,'
                        f'{base64.b64encode(data).decode("ascii")}"/><figcaption>{escape(name)}</figcaption></figure>'
                        for data, name in shown
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

    if len(pdf_positions) == 1 and page_counts:
        st.caption(
            "Si el PDF trae toda la jornada, indica solo las páginas de la carrera que vas a analizar: "
            "es más rápido, más barato y más preciso."
        )
        pdf_pages_spec = st.text_input(
            "Páginas de esta carrera (opcional)",
            placeholder="Ejemplo: 12-19",
            help="Acepta rangos y listas: 12-19, o 3,7,10-12. Vacío = se envía el PDF completo.",
        ).strip()
        needs_page_range = page_counts[0] > MAX_PDF_PAGES and not pdf_pages_spec
        if needs_page_range:
            st.warning(
                f"El PDF supera el máximo de {MAX_PDF_PAGES} páginas: indica el rango de la carrera para continuar."
            )
    elif len(pdf_positions) > 1 and sum(page_counts) > MAX_PDF_PAGES:
        st.warning(
            f"Los PDF suman más de {MAX_PDF_PAGES} páginas. Sube solo el PDF de la carrera "
            "(o recórtalo antes) para continuar."
        )
        needs_page_range = True

col_extract, col_demo = st.columns(2)
with col_extract:
    extract_clicked = st.button(
        "Extraer datos",
        type="primary",
        disabled=not upload_files or not api_ok or needs_page_range or upload_blocked,
    )
with col_demo:
    demo_clicked = st.button("Cargar campo de ejemplo")

if extract_clicked:
    with st.spinner("Leyendo los archivos y extrayendo las variables..."):
        try:
            extracted = extract_race_data_from_files(
                upload_files,
                race_hint=race_name or None,
                track=track,
                pages=pdf_pages_spec or None,
            )
        except ExtractionError as exc:
            st.error(str(exc))
        else:
            set_horses(extracted["horses"])
            st.session_state["source_filename"] = ", ".join(name for name, _ in upload_files)[:250]
            info = extracted["race"]
            st.success(f"Se extrajeron {len(extracted['horses'])} caballos. Revisa y corrige la tabla.")
            detected = [f"{k}: {v}" for k, v in info.items() if v]
            if detected:
                st.caption("Datos detectados en el material (manda tu configuracion): " + " · ".join(detected))
if demo_clicked:
    set_horses(DEMO_FIELD)
    st.session_state["source_filename"] = None
    st.rerun()
if not api_ok:
    st.caption("La extraccion no esta configurada (revisa el archivo .env).")

if db_ok:
    with st.expander("Cargar una carrera guardada"):
        try:
            saved_races = db.list_races(30)
        except db.DatabaseError as exc:
            st.error(str(exc))
            saved_races = []
        if saved_races:
            options = {
                f"{r['race_name']} · {r['track']} · {str(r['created_at'])[:16]}": r["id"] for r in saved_races
            }
            choice = st.selectbox("Carrera guardada", list(options))
            if st.button("Cargar caballos"):
                try:
                    set_horses(db.get_horses(options[choice]))
                    st.rerun()
                except db.DatabaseError as exc:
                    st.error(str(exc))
        else:
            st.caption("Todavia no hay carreras guardadas.")

# ---------------------------------------------------------------------------
# 03 Revision
# ---------------------------------------------------------------------------
step_header(
    "03", "Revisión de datos",
    "Toca una celda para editarla. El ML acepta 5/2, 9-2 o even. Marca Retirado para excluir un caballo.",
)
show_all = st.toggle("Editar todas las variables", value=False, help="Activa para ver las 18 columnas.")
edited_df = st.data_editor(
    st.session_state["horses_df"],
    column_config=EDITOR_COLUMNS,
    column_order=list(sim.HORSE_FIELDS) if show_all else COMPACT_COLUMNS,
    num_rows="dynamic",
    hide_index=True,
    key=f"horse_editor_{st.session_state['editor_version']}",
)

# ---------------------------------------------------------------------------
# 04 Simulacion
# ---------------------------------------------------------------------------
step_header("04", "Simulación y resultados", "Monte Carlo sobre el rating ajustado de cada caballo")
if st.button("Ejecutar simulacion Monte Carlo", type="primary"):
    horses_list = edited_df.to_dict("records")
    try:
        results = sim.simulate_race(
            horses_list,
            surface=surface,
            track_condition=track_condition,
            iterations=int(iterations),
            sigma=float(sigma),
            seed=int(seed_value) or None,
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state["results"] = {
            "table": results,
            "horses": [sim.normalize_horse(h) for h in horses_list],
            "iterations": int(iterations),
            "sigma": float(sigma),
            "race": {
                "race_name": race_name or "Carrera sin nombre",
                "race_number": int(race_number) or None,
                "track": track,
                "race_date": race_date.isoformat(),
                "surface": surface,
                "distance": distance or None,
                "track_condition": track_condition,
                "source_filename": st.session_state.get("source_filename"),
            },
        }
        st.session_state.pop("saved_race_id", None)

saved = st.session_state.get("results")
if saved and "avg_finish" not in saved["table"].columns:
    # Resultado de una version anterior de la app: se descarta para volver a simular.
    st.session_state.pop("results", None)
    saved = None
    st.info("La aplicacion se actualizo. Vuelve a ejecutar la simulacion para ver el orden de llegada.")
if saved:
    results = saved["table"]
    records = results.to_dict("records")
    duplicated = results["program_number"].duplicated(keep=False) & (results["program_number"] != "")
    if duplicated.any():
        st.warning("Hay numeros de programa repetidos; revisa la tabla.")

    st.markdown(kpis_html(records, saved["iterations"], saved["sigma"]), unsafe_allow_html=True)

    st.markdown('<div class="sub-h">Ranking por probabilidad de ganar</div>', unsafe_allow_html=True)
    st.markdown(leaderboard_html(records, saved["horses"]), unsafe_allow_html=True)

    analysis = build_analysis(results, saved["horses"], saved["race"], saved["iterations"])
    standout_block = standout_html(analysis)
    if standout_block:
        st.markdown('<div class="sub-h">Caballo superior de la carrera</div>', unsafe_allow_html=True)
        st.markdown(standout_block, unsafe_allow_html=True)
    st.markdown('<div class="sub-h">Orden de llegada probable (primeros 4)</div>', unsafe_allow_html=True)
    st.markdown(finish_order_html(analysis), unsafe_allow_html=True)
    st.markdown('<div class="sub-h">Comentario de la carrera</div>', unsafe_allow_html=True)
    st.markdown(commentary_html(analysis), unsafe_allow_html=True)

    st.markdown('<div class="sub-h">Jugadas con valor esperado positivo (+EV)</div>', unsafe_allow_html=True)
    value_records = sorted(
        (r for r in records if r["is_positive_ev"]), key=lambda r: r["expected_value"], reverse=True
    )
    if value_records:
        st.markdown(value_cards_html(value_records), unsafe_allow_html=True)
    else:
        st.info("Ningun caballo muestra valor esperado positivo (+EV) con el Morning Line ingresado.")
    st.caption(
        "EV = (probabilidad simulada × cuota decimal del Morning Line) − 1. "
        "Es una herramienta analitica; las apuestas conllevan riesgo de perdida."
    )

    with st.expander("Tabla detallada con todos los ajustes"):
        st.dataframe(style_results(build_results_table(results)), hide_index=True)

    step_header("05", "Guardar análisis", "Conserva la carrera, los caballos y los resultados")
    if not db_ok:
        st.caption("El guardado no esta configurado (revisa el archivo .env).")
    elif st.session_state.get("saved_race_id"):
        st.success(f"Analisis guardado (race_id: {st.session_state['saved_race_id']}).")
    elif st.button("Guardar carrera, caballos y resultados"):
        try:
            race_id = db.save_full_analysis(
                saved["race"], saved["horses"], results, saved["iterations"], saved["sigma"]
            )
        except db.DatabaseError as exc:
            st.error(str(exc))
        else:
            st.session_state["saved_race_id"] = race_id
            st.rerun()

persist_session(edited_df)
