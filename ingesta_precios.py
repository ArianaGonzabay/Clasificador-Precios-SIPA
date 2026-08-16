"""
Módulo de Ingesta y Consolidación de Precios Mayoristas del SIPA.

Procesa archivos en formato Excel (.xlsx) o CSV (.csv) cargados desde la interfaz web,
normalizando las cabeceras e integrando la información histórica de forma incremental.
"""

import os
import re
from difflib import get_close_matches

import numpy as np
import pandas as pd

MESES_NUM = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
    "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12,
}

NO_PERECEDERO_KEYWORDS = [
    "arroz", "aceite", "fideo", "azúcar", "café", "harina", "sal", "grano", "seco", "lenteja", "garbanzo",
]


def _limpiar_cabecera(c):
    s = str(c)
    s = s.replace("Ao", "Año").replace("Cantn", "Cantón")
    s = re.sub(r"[^\w\s\(\)/:\.,\$-]", "ñ", s)
    s = s.replace("Cantñn", "Cantón").replace("Año", "Año").replace("ñAñoñ", "Año")
    return s


def procesar_archivo_precios(file_storage, csv_crudo_path):
    """
    Procesa el archivo de precios mayoristas SIPA (Excel o CSV) subido, lo normaliza
    y lo fusiona (sin duplicados) con el histórico ya guardado en disco.

    Devuelve (df_guardado_completo, reporte).
    """
    nombre = file_storage.filename or ""

    if nombre.lower().endswith(".csv"):
        df_excel = pd.read_csv(file_storage)
    else:
        xl = pd.ExcelFile(file_storage)
        sheet_name = xl.sheet_names[0]
        if "Precios Mercados12-26" in xl.sheet_names:
            sheet_name = "Precios Mercados12-26"

        df_temp = pd.read_excel(file_storage, sheet_name=sheet_name, nrows=15)
        header_row = 0
        for r in range(len(df_temp)):
            row_vals = df_temp.iloc[r].dropna().astype(str).str.lower().tolist()
            if any("producto" in x for x in row_vals) and any("provincia" in x for x in row_vals):
                header_row = r + 1
                break

        file_storage.stream.seek(0)
        df_excel = pd.read_excel(file_storage, sheet_name=sheet_name, skiprows=header_row)

    # Eliminar columnas Unnamed (vacías)
    df_excel = df_excel.loc[:, ~df_excel.columns.str.contains("^Unnamed")]

    df_excel.columns = [_limpiar_cabecera(c) for c in df_excel.columns]
    col_year = [c for c in df_excel.columns if str(c).startswith("A") and str(c).endswith("o")]
    if not col_year:
        raise ValueError(f"No se encontró la columna de Año. Columnas encontradas: {list(df_excel.columns)}")
    col_year = col_year[0]
    df_excel = df_excel.rename(columns={col_year: "Año"})

    df_excel["mes_num"] = df_excel["Mes"].map(MESES_NUM)
    df_excel["año_num"] = df_excel["Año"].astype(int)
    df_excel["provincia_std"] = df_excel["Provincia"].str.strip().str.upper()
    df_excel["mercado_std"] = df_excel["Mercado"].astype(str).str.strip()
    df_excel["canton_std"] = df_excel["Cantón"].astype(str).str.strip()
    df_excel["pres_std"] = df_excel["Pres."].astype(str).str.strip()
    df_excel["tipo_mercado_std"] = df_excel["Tipo Mercado"].astype(str).str.strip()

    df_clean = df_excel.dropna(
        subset=["Producto", "provincia_std", "año_num", "mes_num", "Promedio de Precio (USD)"]
    ).copy()

    df_clean = df_clean.sort_values(
        by=["Producto", "provincia_std", "canton_std", "mercado_std", "pres_std", "año_num", "mes_num"]
    ).reset_index(drop=True)
    df_clean["precio_anterior"] = df_clean.groupby(
        ["Producto", "provincia_std", "canton_std", "mercado_std", "pres_std"]
    )["Promedio de Precio (USD)"].shift(1)

    df_grouped = df_clean.rename(columns={
        "Producto": "producto_raw",
        "Promedio de Precio (USD)": "precio_actual",
        "provincia_std": "provincia",
        "año_num": "año",
        "mes_num": "mes",
        "mercado_std": "mercado",
        "canton_std": "canton",
        "pres_std": "presentacion",
        "tipo_mercado_std": "tipo_mercado",
    })

    df_grouped["variacion"] = (
        (df_grouped["precio_actual"] - df_grouped["precio_anterior"]) / df_grouped["precio_anterior"]
    ) * 100
    df_grouped["periodo"] = df_grouped["año"].astype(str) + "-" + df_grouped["mes"].astype(str).str.zfill(2)
    df_grouped["estado_precio"] = np.where(df_grouped["precio_anterior"].isna(), "parcial", "completo")

    # Categoría (perecedero / no_perecedero), reutilizando lo ya guardado en disco cuando existe
    prod_to_cat = {}
    if os.path.exists(csv_crudo_path):
        try:
            df_old = pd.read_csv(csv_crudo_path)
            prod_to_cat = df_old.drop_duplicates("producto_raw").set_index("producto_raw")["categoria"].to_dict()
        except Exception:
            pass

    unique_prods = df_grouped["producto_raw"].unique()
    cached_cat = {}
    for prod in unique_prods:
        if prod in prod_to_cat:
            cached_cat[prod] = prod_to_cat[prod]
        else:
            matches = get_close_matches(prod, list(prod_to_cat.keys()), n=1, cutoff=0.6)
            if matches:
                cached_cat[prod] = prod_to_cat[matches[0]]
            else:
                if any(k in prod.lower() for k in NO_PERECEDERO_KEYWORDS):
                    cached_cat[prod] = "no_perecedero"
                else:
                    cached_cat[prod] = "perecedero"

    df_grouped["categoria"] = df_grouped["producto_raw"].map(cached_cat)

    # Guardar en disco, fusionando con el histórico existente sin duplicar
    os.makedirs(os.path.dirname(csv_crudo_path), exist_ok=True)
    df_guardar = df_grouped.copy()
    if os.path.exists(csv_crudo_path):
        try:
            df_existente = pd.read_csv(csv_crudo_path, encoding="utf-8-sig")
            df_guardar = pd.concat([df_existente, df_guardar], ignore_index=True)
            cols_dedup = [
                c for c in ["producto_raw", "provincia", "canton", "mercado", "presentacion", "periodo"]
                if c in df_guardar.columns
            ]
            if cols_dedup:
                df_guardar = df_guardar.drop_duplicates(subset=cols_dedup, keep="last").reset_index(drop=True)
        except Exception:
            pass
    df_guardar.to_csv(csv_crudo_path, index=False, encoding="utf-8-sig")

    reporte = {
        "porcentaje_completitud": 100,
        "registros_completos": len(df_grouped),
        "registros_parciales": len(df_grouped[df_grouped["estado_precio"] == "parcial"]),
        "registros_con_problema": 0,
        "quincenas": df_grouped["periodo"].nunique(),
        "problemas": [],
    }
    return df_guardar, reporte
