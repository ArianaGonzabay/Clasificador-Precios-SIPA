# Clasificador de Precios Mayoristas SIPA

Sistema de extraccion de datos de precios mayoristas agricolas del boletin SIPA (Ecuador) usando OCR.

## Requisitos

- Python 3.12
- pip

## Instalacion

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion

```bash
.\.venv\Scripts\activate
streamlit run app.py
```

La interfaz se abre en `http://localhost:8501`. Suba un boletin SIPA en formato PDF y haga clic en **Procesar**.
