# Breeders' Cup AI

Pronosticos con extraccion de Past Performances (Claude), motor de handicap multifactorial,
simulacion Monte Carlo (10,000 iteraciones) y persistencia en Supabase. Interfaz en Streamlit.

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env  (y completa las claves)
```

1. En Supabase abre **SQL Editor** y ejecuta `schema.sql`.
2. Inicia la app: `streamlit run app.py`

Sin claves puedes probar el flujo con el boton **Cargar campo de ejemplo**.
La extraccion necesita `ANTHROPIC_API_KEY`; el guardado necesita `SUPABASE_URL` y `SUPABASE_KEY`.

## Empaquetar

```bash
python generate_zip.py
```
