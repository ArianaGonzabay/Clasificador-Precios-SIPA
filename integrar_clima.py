"""
Script de integración del clima histórico (Open-Meteo) con el dataset de precios SIPA.
Carga clima_historico.csv, normaliza la columna 'provincia', 
y hace un LEFT JOIN con el dataset preprocesado por (año, mes, provincia).
"""

import pandas as pd
import os

# ─── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIMA_PATH = os.path.join(BASE_DIR, 'data', 'clima_historico.csv')
PREPROC_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_preprocesado_sipa.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_preprocesado_sipa.csv')

# ─── Cargar datos ─────────────────────────────────────────────────────────────
df_clima = pd.read_csv(CLIMA_PATH)
df_preproc = pd.read_csv(PREPROC_PATH)

print(f"Clima: {len(df_clima)} filas, provincias: {sorted(df_clima['provincia'].unique())}")
print(f"Preprocesado: {len(df_preproc)} filas")

# Limpiar columnas climáticas previas si ya existen (para evitar duplicados)
cols_clima_previas = [c for c in df_preproc.columns if c.startswith('clima_')]
if cols_clima_previas:
    df_preproc = df_preproc.drop(columns=cols_clima_previas)
    print(f"Se eliminaron columnas climáticas previas: {cols_clima_previas}")

if 'periodo' in df_preproc.columns:
    df_preproc['_año_ext'] = df_preproc['periodo'].str.split('-').str[0].astype(int)
    df_preproc['_mes_ext'] = df_preproc['periodo'].str.split('-').str[1].astype(int)
elif 'año' in df_preproc.columns and 'mes' in df_preproc.columns:
    df_preproc['_año_ext'] = df_preproc['año'].astype(int)
    df_preproc['_mes_ext'] = df_preproc['mes'].astype(int)
else:
    raise ValueError("El dataset preprocesado no tiene columna 'periodo', 'año' o 'mes'.")

# ─── Normalizar nombre de provincia en clima ──────────────────────────────────
df_clima['provincia_norm'] = df_clima['provincia'].str.strip().str.upper()
df_preproc['_prov_norm'] = df_preproc['provincia'].str.strip().str.upper()

rename_clima = {
    'CANAR': 'CAÑAR',
    'PERU': 'PERU',  # Keep as is, some products come from Perú
}
df_clima['provincia_norm'] = df_clima['provincia_norm'].replace(rename_clima)

# ─── Calcular rezagos climáticos en el dataset de clima (por provincia) ───────
df_clima_sorted = df_clima.sort_values(['provincia_norm', 'año', 'mes']).copy()
df_clima_sorted['temp_lag1'] = df_clima_sorted.groupby('provincia_norm')['temp'].shift(1)
df_clima_sorted['temp_lag2'] = df_clima_sorted.groupby('provincia_norm')['temp'].shift(2)
df_clima_sorted['precip_lag1'] = df_clima_sorted.groupby('provincia_norm')['precip'].shift(1)
df_clima_sorted['precip_lag2'] = df_clima_sorted.groupby('provincia_norm')['precip'].shift(2)

# ─── Merge ────────────────────────────────────────────────────────────────────
df_merged = df_preproc.merge(
    df_clima_sorted[['año', 'mes', 'provincia_norm', 'temp', 'precip', 'temp_lag1', 'temp_lag2', 'precip_lag1', 'precip_lag2']].rename(columns={
        'año': '_año_ext',
        'mes': '_mes_ext',
        'provincia_norm': '_prov_norm'
    }),
    on=['_año_ext', '_mes_ext', '_prov_norm'],
    how='left'
)

df_merged = df_merged.rename(columns={
    'temp': 'clima_temp_media',
    'precip': 'clima_precipitacion_mm',
    'temp_lag1': 'clima_temp_lag1',
    'temp_lag2': 'clima_temp_lag2',
    'precip_lag1': 'clima_precip_lag1',
    'precip_lag2': 'clima_precip_lag2',
})

df_merged = df_merged.drop(columns=['_año_ext', '_mes_ext', '_prov_norm'], errors='ignore')

for col in ['clima_temp_media', 'clima_precipitacion_mm', 'clima_temp_lag1', 'clima_temp_lag2', 'clima_precip_lag1', 'clima_precip_lag2']:
    df_merged[col] = df_merged.groupby('provincia')[col].transform(lambda x: x.fillna(x.median()))
    df_merged[col] = df_merged[col].fillna(df_merged[col].median())

match_pct = float(df_merged['clima_temp_media'].notna().mean()) * 100
print(f"\nCobertura clima: {match_pct:.1f}% de filas tienen datos climáticos.")
print(f"Dataset final: {len(df_merged)} filas, {len(df_merged.columns)} columnas")
print(f"Columnas climáticas: clima_temp_media, clima_precipitacion_mm, clima_temp_lag1, clima_temp_lag2, clima_precip_lag1, clima_precip_lag2")

# ─── Guardar ──────────────────────────────────────────────────────────────────
df_merged.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"Guardado en: {OUTPUT_PATH}")
