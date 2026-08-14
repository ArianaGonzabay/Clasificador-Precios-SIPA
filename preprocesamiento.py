import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def limpiar_nombre_producto(texto):
    import re
    if not isinstance(texto, str):
        return texto
    t = texto.strip()
    t = re.sub(r"^[+\s*#]+|[+\s*#]+$", "", t)
    t = re.sub(r"\s+\d+(?:[.,]\d+)?$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def preprocesar_datos(df):
    from difflib import get_close_matches
    estadisticas = {
        "registros_originales": len(df),
        "productos_originales": df["producto_raw"].nunique(),
        "provincias": df["provincia"].nunique(),
    }

    df_limpio = df[df["estado_precio"] == "completo"].copy()
    
    # 1. Limpieza de artefactos OCR y normalizacion de productos
    df_limpio["producto_limpio"] = df_limpio["producto_raw"].apply(limpiar_nombre_producto)
    conteo = df_limpio["producto_limpio"].value_counts()
    canonicos = list(conteo[conteo >= 5].index)

    def normalizar(nombre):
        if nombre in canonicos:
            return nombre
        matches = get_close_matches(nombre, canonicos, n=1, cutoff=0.75)
        return matches[0] if matches else nombre

    df_limpio["producto"] = df_limpio["producto_limpio"].apply(normalizar)

    estadisticas["registros_completos"] = len(df_limpio)
    estadisticas["registros_parciales"] = len(df[df["estado_precio"] == "parcial"])
    estadisticas["registros_invalidos"] = len(df[df["estado_precio"] == "invalido"])
    mapa_categoria = df_limpio.drop_duplicates("producto").set_index("producto")["categoria"].to_dict()

    if "quincena_id" in df_limpio.columns:
        df_limpio["periodo"] = df_limpio["quincena_id"]
    elif "año" in df_limpio.columns and "quincena" in df_limpio.columns:
        df_limpio["periodo"] = df_limpio["año"].astype(str) + "-" + df_limpio["quincena"].astype(str).str.zfill(2)
    else:
        df_limpio["periodo"] = "desconocido"
        

    df_pivot = df_limpio.pivot_table(
        index=["producto", "provincia"],
        columns="periodo",
        values=["precio_anterior", "precio_actual"],
        aggfunc="first"
    )

    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    periodos_unicos = sorted(df_limpio["periodo"].unique())
    cols_ordenadas = ["producto", "provincia"]
    for p in periodos_unicos:
        col_ant = f"precio_anterior_{p}"
        col_act = f"precio_actual_{p}"
        if col_ant in df_pivot.columns:
            cols_ordenadas.append(col_ant)
        if col_act in df_pivot.columns:
            cols_ordenadas.append(col_act)

    cols_existentes = [c for c in cols_ordenadas if c in df_pivot.columns]
    df_pivot = df_pivot[cols_existentes]

    productos_descartados = []
    productos_mantener = []

    for idx, row in df_pivot.iterrows():
        cols_precio = [c for c in df_pivot.columns if c.startswith("precio_actual_")]
        valores = row[cols_precio]
        total_periodos = len(valores)
        faltantes = valores.isna().sum()
        porcentaje_faltantes = faltantes / total_periodos if total_periodos > 0 else 1

        if porcentaje_faltantes > 0.30:
            productos_descartados.append({
                "producto": row["producto"],
                "provincia": row["provincia"],
                "porcentaje_faltantes": round(porcentaje_faltantes * 100, 1)
            })
        else:
            productos_mantener.append(idx)

    df_pivot = df_pivot.loc[productos_mantener]

    cols_precio = [c for c in df_pivot.columns if c.startswith("precio_")]
    df_pivot[cols_precio] = df_pivot[cols_precio].interpolate(method="linear", axis=1)
    df_pivot[cols_precio] = df_pivot[cols_precio].ffill(axis=1).bfill(axis=1)

    estadisticas["registros_despues_filtro"] = len(df_pivot)
    estadisticas["productos_descartados"] = len(productos_descartados)
    estadisticas["lista_descartados"] = productos_descartados

    df_modelo, le_producto, le_provincia = _crear_features(df_pivot, periodos_unicos, mapa_categoria)

    estadisticas["registros_modelo"] = len(df_modelo)
    estadisticas["columnas_modelo"] = list(df_modelo.columns)

    return {
        "dataset_final": df_modelo,
        "dataset_wide": df_pivot,
        "estadisticas": estadisticas,
        "productos_descartados": productos_descartados,
        "le_producto": le_producto,
        "le_provincia": le_provincia,
    }

def _crear_features(df_pivot, periodos_unicos, mapa_categoria=None):
    
    registros = []

    for idx, row in df_pivot.iterrows():
        producto = row["producto"]
        provincia = row["provincia"]

        # Construir serie de precios temporal ordenada por este producto+provincia
        serie_precios = []
        for periodo in periodos_unicos:
            col_actual = f"precio_actual_{periodo}"
            if col_actual in df_pivot.columns:
                serie_precios.append((periodo, row.get(col_actual)))
            else:
                serie_precios.append((periodo, np.nan))

        precios_series = pd.Series([p[1] for p in serie_precios], index=[p[0] for p in serie_precios])
        pm2 = precios_series.shift(1).rolling(window=2, min_periods=2).mean()
        pm3 = precios_series.shift(1).rolling(window=3, min_periods=3).mean()
        vol3 = precios_series.shift(1).rolling(window=3, min_periods=2).std()
        momentum = (precios_series.shift(1) - precios_series.shift(3)) / (precios_series.shift(3) + 1e-5)

        for i, periodo in enumerate(periodos_unicos):
            col_actual = f"precio_actual_{periodo}"
            if col_actual not in df_pivot.columns:
                continue

            precio_actual = row.get(col_actual)
            if pd.isna(precio_actual):
                continue

            precio_t1 = None
            if i > 0:
                col_t1 = f"precio_actual_{periodos_unicos[i-1]}"
                if col_t1 in df_pivot.columns:
                    precio_t1 = row.get(col_t1)

            precio_t2 = None
            if i > 1:
                col_t2 = f"precio_actual_{periodos_unicos[i-2]}"
                if col_t2 in df_pivot.columns:
                    precio_t2 = row.get(col_t2)

            if precio_t1 is not None and not pd.isna(precio_t1):
                registro = {
                    "producto": producto,
                    "provincia": provincia,
                    "periodo": periodo,
                    "precio_actual": precio_actual,
                    "precio_t1": precio_t1,
                    "precio_t2": precio_t2 if precio_t2 is not None else np.nan,
                }

                if precio_t2 is not None and not pd.isna(precio_t2) and precio_t2 > 0:
                    registro["variacion_t2_t1"] = round(((precio_t1 - precio_t2) / precio_t2) * 100, 2)
                else:
                    registro["variacion_t2_t1"] = np.nan

                registro["promedio_movil_2q"] = round(pm2.loc[periodo], 4) if pd.notna(pm2.loc[periodo]) else np.nan
                registro["promedio_movil_3q"] = round(pm3.loc[periodo], 4) if pd.notna(pm3.loc[periodo]) else np.nan
                registro["volatilidad_3q"] = round(vol3.loc[periodo], 4) if pd.notna(vol3.loc[periodo]) else 0.0
                registro["momentum"] = round(momentum.loc[periodo], 4) if pd.notna(momentum.loc[periodo]) else 0.0

                try:
                    partes = periodo.split("-")
                    if len(partes) == 3 and partes[2].startswith("Q"):
                        registro["año"] = int(partes[0])
                        registro["mes"] = int(partes[1])
                        registro["quincena"] = int(partes[2][1:])
                    else:
                        registro["año"] = int(partes[0])
                        registro["quincena"] = int(partes[1])
                        registro["mes"] = 1 if int(partes[1]) == 1 else 2
                except (ValueError, IndexError):
                    registro["mes"] = np.nan
                    registro["año"] = np.nan
                    registro["quincena"] = np.nan

                variacion_precio = ((precio_actual - precio_t1) / precio_t1) * 100 if precio_t1 > 0 else 0
                registro["variacion_real"] = round(variacion_precio, 2)

                if variacion_precio > 7:
                    registro["comportamiento"] = "Alza"
                elif variacion_precio < -7:
                    registro["comportamiento"] = "Caída"
                else:
                    registro["comportamiento"] = "Estable"

                registro["categoria"] = mapa_categoria.get(producto, "desconocido") if mapa_categoria else "desconocido"

                registros.append(registro)

    df_modelo = pd.DataFrame(registros)
    df_modelo, le_producto, le_provincia = _codificar_categoricas(df_modelo)
    return df_modelo, le_producto, le_provincia

def _codificar_categoricas(df):
    from sklearn.preprocessing import LabelEncoder
    le_producto = LabelEncoder()
    le_provincia = LabelEncoder()
    df["producto_encoded"] = le_producto.fit_transform(df["producto"])
    df["provincia_encoded"] = le_provincia.fit_transform(df["provincia"])
    # NUEVO: categoria como binaria (0=no_perecedero, 1=perecedero)
    df["categoria_perecedero"] = (df["categoria"] == "perecedero").astype(int)
    return df, le_producto, le_provincia

def obtener_resumen(df_modelo):
    resumen = {
        "total_registros": len(df_modelo),
        "productos_unicos": df_modelo["producto"].nunique(),
        "provincias": df_modelo["provincia"].nunique(),
        "periodos": df_modelo["periodo"].nunique(),
        "distribucion_comportamiento": df_modelo["comportamiento"].value_counts().to_dict(),
        "valores_faltantes": df_modelo.isnull().sum().to_dict(),
    }
    return resumen



import pandas as pd

# Cargar el dataset crudo (ajusta la ruta si es necesario)
df_crudo = pd.read_csv('data/processed/dataset_crudo_sipa.csv')

# Ejecutar el preprocesamiento real
resultado = preprocesar_datos(df_crudo)
df_modelo = resultado['dataset_final']

print(f"Registros en el dataset final: {len(df_modelo)}")
df_modelo.head()

print(df_modelo['comportamiento'].value_counts())
print()
print(df_modelo['comportamiento'].value_counts(normalize=True).round(3) * 100)

print(df_crudo['quincena_id'].nunique())
print(sorted(df_crudo['quincena_id'].unique()))

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

