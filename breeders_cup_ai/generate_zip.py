#!/usr/bin/env python3
"""Empaqueta todo el proyecto en breeders_cup_ai_project.zip.

Uso (desde cualquier carpeta):
    python generate_zip.py

Genera el ZIP junto a este script, con una carpeta raiz "breeders_cup_ai/".
Excluye secretos (.env), cache de Python, entornos virtuales y el propio ZIP.
Solo usa la libreria estandar.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ZIP_NAME = "breeders_cup_ai_project.zip"
ROOT_FOLDER = "breeders_cup_ai"

EXCLUDED_DIRS = {"__pycache__", ".venv", "venv", "env", ".git", ".idea", ".vscode", ".streamlit", "node_modules"}
EXCLUDED_NAMES = {".env", ".DS_Store", ZIP_NAME}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}


def should_include(path: Path) -> bool:
    relative = path.relative_to(PROJECT_DIR)
    if any(part in EXCLUDED_DIRS for part in relative.parts[:-1]):
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    return path.suffix not in EXCLUDED_SUFFIXES


def main() -> int:
    files = sorted(p for p in PROJECT_DIR.rglob("*") if p.is_file() and should_include(p))
    if not files:
        print("No se encontraron archivos para empaquetar.", file=sys.stderr)
        return 1

    zip_path = PROJECT_DIR / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive_name = f"{ROOT_FOLDER}/{path.relative_to(PROJECT_DIR).as_posix()}"
            archive.write(path, archive_name)
            print(f"  + {archive_name}")

    size_kb = zip_path.stat().st_size / 1024
    print(f"\nListo: {zip_path} ({len(files)} archivos, {size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
