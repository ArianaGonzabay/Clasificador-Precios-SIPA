import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── Ruta al CSV de clima histórico ──────────────────────────────────────────
_CLIMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'clima_historico.csv')

def _cargar_clima():
    """Carga el CSV climático y devuelve un dict {(provincia_upper, año, mes): (temp, precip)}."""
    if not os.path.exists(_CLIMA_PATH):
        return {}
    df = pd.read_csv(_CLIMA_PATH)
    # Normalizar nombre de provincia
    df['provincia'] = df['provincia'].str.strip().str.upper()
    # Rezagos dentro del CSV (ordenados por provincia, año, mes)
    df = df.sort_values(['provincia', 'año', 'mes']).copy()
    df['temp_lag1']   = df.groupby('provincia')['temp'].shift(1)
    df['temp_lag2']   = df.groupby('provincia')['temp'].shift(2)
    df['precip_lag1'] = df.groupby('provincia')['precip'].shift(1)
    df['precip_lag2'] = df.groupby('provincia')['precip'].shift(2)
    
    # NUEVO: Lluvia acumulada de los últimos 3 meses (actual + lag1 + lag2)
    df['precip_acc3'] = df['precip'] + df['precip_lag1'].fillna(0) + df['precip_lag2'].fillna(0)
    
    lookup = {}
    for _, row in df.iterrows():
        key = (row['provincia'], int(row['año']), int(row['mes']))
        lookup[key] = {
            'clima_temp_media':       row['temp'],
            'clima_precipitacion_mm': row['precip'],
            'clima_temp_lag1':        row['temp_lag1'],
            'clima_temp_lag2':        row['temp_lag2'],
            'clima_precip_lag1':      row['precip_lag1'],
            'clima_precip_lag2':      row['precip_lag2'],
            'clima_precip_acc3':      row['precip_acc3'],
        }
    return lookup

_CLIMA_LOOKUP = None   # cargado una sola vez en memoria

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

    if "periodo" in df_limpio.columns:
        df_limpio["periodo"] = df_limpio["periodo"].astype(str)
    elif "quincena_id" in df_limpio.columns:
        df_limpio["periodo"] = df_limpio["quincena_id"]
    elif "año" in df_limpio.columns and "quincena" in df_limpio.columns:
        df_limpio["periodo"] = df_limpio["año"].astype(str) + "-" + df_limpio["quincena"].astype(str).str.zfill(2)
    else:
        df_limpio["periodo"] = "desconocido"
        

    index_cols = ["producto", "provincia"]
    for col in ["canton", "mercado", "presentacion", "tipo_mercado"]:
        if col in df_limpio.columns:
            index_cols.append(col)

    df_pivot = df_limpio.pivot_table(
        index=index_cols,
        columns="periodo",
        values=["precio_anterior", "precio_actual"],
        aggfunc="first"
    )

    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    periodos_unicos = sorted(df_limpio["periodo"].unique())
    cols_ordenadas = index_cols.copy()
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

        if porcentaje_faltantes > 0.70:
            productos_descartados.append({
                "producto": row["producto"],
                "provincia": row["provincia"],
                "canton": row.get("canton", "desconocido"),
                "mercado": row.get("mercado", "desconocido"),
                "presentacion": row.get("presentacion", "desconocido"),
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

    df_modelo, encoders = _crear_features(df_pivot, periodos_unicos, mapa_categoria)

    estadisticas["registros_modelo"] = len(df_modelo)
    estadisticas["columnas_modelo"] = list(df_modelo.columns)

    return {
        "dataset_final": df_modelo,
        "dataset_wide": df_pivot,
        "estadisticas": estadisticas,
        "productos_descartados": productos_descartados,
        "encoders": encoders,
        "le_producto": encoders["producto"],
        "le_provincia": encoders["provincia"],
    }

def _crear_features(df_pivot, periodos_unicos, mapa_categoria=None):
    global _CLIMA_LOOKUP
    if _CLIMA_LOOKUP is None:
        _CLIMA_LOOKUP = _cargar_clima()

    registros = []

    for idx, row in df_pivot.iterrows():
        producto = row["producto"]
        provincia = row["provincia"]
        canton = row.get("canton", "desconocido")
        mercado = row.get("mercado", "desconocido")
        presentacion = row.get("presentacion", "desconocido")
        tipo_mercado = row.get("tipo_mercado", "desconocido")
        prov_upper = str(provincia).strip().upper()

        # ── Construir serie de precios temporal ──────────────────────────────
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

        # ── Umbral dinámico de variación ─────────────────────────────────────
        precios_array = precios_series.dropna().values
        variaciones = []
        for j in range(1, len(precios_array)):
            p_act = precios_array[j]
            p_ant = precios_array[j-1]
            if p_ant > 0:
                variaciones.append(((p_act - p_ant) / p_ant) * 100)
        std_var = np.std(variaciones) if len(variaciones) >= 3 else 7.0
        if pd.isna(std_var) or std_var == 0:
            std_var = 7.0
        # Reducimos multiplicador dinámico a 0.35x para suavizar la variable objetivo y capturar mejor tendencias reales
        threshold = max(3.0, min(15.0, 0.35 * std_var))

        # ── Mapa periodo → índice para búsqueda rápida ───────────────────────
        periodo_to_idx = {p: k for k, p in enumerate(periodos_unicos)}

        for i, periodo in enumerate(periodos_unicos):
            col_actual = f"precio_actual_{periodo}"
            if col_actual not in df_pivot.columns:
                continue

            precio_actual = row.get(col_actual)
            if pd.isna(precio_actual):
                continue

            # ── Rezagos de precio t1 … t6 ────────────────────────────────────
            def _get_precio(offset):
                j = i - offset
                if j < 0:
                    return np.nan
                col = f"precio_actual_{periodos_unicos[j]}"
                v = row.get(col, np.nan)
                return v if pd.notna(v) else np.nan

            precio_t1 = _get_precio(1)
            precio_t2 = _get_precio(2)
            precio_t3 = _get_precio(3)
            precio_t4 = _get_precio(4)
            precio_t5 = _get_precio(5)
            precio_t6 = _get_precio(6)

            if pd.isna(precio_t1):
                continue  # necesitamos al menos el precio inmediatamente anterior

            # ── Variaciones ──────────────────────────────────────────────────
            def _var_pct(p_now, p_ref):
                if pd.notna(p_now) and pd.notna(p_ref) and p_ref > 0:
                    return round(((p_now - p_ref) / p_ref) * 100, 2)
                return np.nan

            # ── Comparación año a año (mismo mes, 24 quincenas atrás ≈ 1 año) ─
            # Los periodos son quincenales: 24 quincenas = 1 año
            precio_yoy = _get_precio(24)
            ratio_yoy = _var_pct(precio_actual, precio_yoy)

            val_pm2 = pm2.loc[periodo] if pd.notna(pm2.loc[periodo]) else np.nan
            val_pm3 = pm3.loc[periodo] if pd.notna(pm3.loc[periodo]) else np.nan

            registro = {
                "producto": producto,
                "provincia": provincia,
                "canton": canton,
                "mercado": mercado,
                "presentacion": presentacion,
                "tipo_mercado": tipo_mercado,
                "periodo": periodo,
                "precio_actual": precio_actual,
                # Rezagos de precio
                "precio_t1": precio_t1,
                "precio_t2": precio_t2,
                "precio_t3": precio_t3,
                "precio_t4": precio_t4,
                "precio_t5": precio_t5,
                "precio_t6": precio_t6,
                # Variaciones
                "variacion_t2_t1": _var_pct(precio_t1, precio_t2),
                "variacion_t3_t1": _var_pct(precio_t1, precio_t3),
                # Comparación año a año
                "variacion_yoy": ratio_yoy,
                # Rolling
                "promedio_movil_2q": round(val_pm2, 4) if pd.notna(val_pm2) else np.nan,
                "promedio_movil_3q": round(val_pm3, 4) if pd.notna(val_pm3) else np.nan,
                "volatilidad_3q": round(vol3.loc[periodo], 4) if pd.notna(vol3.loc[periodo]) else 0.0,
                "momentum": round(momentum.loc[periodo], 4) if pd.notna(momentum.loc[periodo]) else 0.0,
                # NUEVAS: Interacción temporal
                "precio_vs_tendencia": round(((precio_t1 - val_pm3) / val_pm3) * 100, 2) if pd.notna(val_pm3) and val_pm3 > 0 else 0.0,
                "cruce_medias": round(val_pm2 - val_pm3, 4) if pd.notna(val_pm2) and pd.notna(val_pm3) else 0.0,
            }

            # ── Extraer año / mes ─────────────────────────────────
            try:
                partes = periodo.split("-")
                if len(partes) >= 2:
                    registro["año"] = int(partes[0])
                    registro["mes"] = int(partes[1])
                else:
                    registro["año"] = np.nan
                    registro["mes"] = np.nan
            except (ValueError, IndexError):
                registro["mes"] = np.nan
                registro["año"] = np.nan

            # ── Variables climáticas (lookup por provincia/año/mes) ──────────
            anio = registro.get("año")
            mes  = registro.get("mes")
            clima_default = {
                'clima_temp_media': np.nan,
                'clima_precipitacion_mm': np.nan,
                'clima_temp_lag1': np.nan,
                'clima_temp_lag2': np.nan,
                'clima_precip_lag1': np.nan,
                'clima_precip_lag2': np.nan,
                'clima_precip_acc3': np.nan,
            }
            if _CLIMA_LOOKUP and pd.notna(anio) and pd.notna(mes):
                clima_vals = _CLIMA_LOOKUP.get((prov_upper, int(anio), int(mes)), {})
                registro.update(clima_vals if clima_vals else clima_default)
            else:
                registro.update(clima_default)

            # ── Variable objetivo ────────────────────────────────────────────
            variacion_precio = ((precio_actual - precio_t1) / precio_t1) * 100 if precio_t1 > 0 else 0
            registro["variacion_real"] = round(variacion_precio, 2)

            if variacion_precio > threshold:
                registro["comportamiento"] = "Alza"
            elif variacion_precio < -threshold:
                registro["comportamiento"] = "Caída"
            else:
                registro["comportamiento"] = "Estable"

            registro["categoria"] = mapa_categoria.get(producto, "desconocido") if mapa_categoria else "desconocido"

            registros.append(registro)

    df_modelo = pd.DataFrame(registros)
    df_modelo, encoders = _codificar_categoricas(df_modelo)
    return df_modelo, encoders

class TargetEncoder:
    def __init__(self, default_value=0.0):
        self.mapping = {}
        self.default_value = default_value

    def fit(self, categories, target):
        df_temp = pd.DataFrame({'cat': categories, 'target': target})
        self.mapping = df_temp.groupby('cat')['target'].mean().to_dict()
        self.default_value = df_temp['target'].mean() if not df_temp.empty else 0.0
        return self

    def transform(self, categories):
        # Convert category to list if it is a pandas Series/Index/ndarray or list
        cats = list(categories) if hasattr(categories, '__iter__') and not isinstance(categories, str) else [categories]
        res = np.array([self.mapping.get(cat, self.default_value) for cat in cats], dtype=np.float64)
        if isinstance(categories, str):
            return res[0]
        return res

    def fit_transform(self, categories, target):
        return self.fit(categories, target).transform(categories)

    @property
    def classes_(self):
        return list(self.mapping.keys())

def _codificar_categoricas(df):
    te_producto = TargetEncoder()
    te_provincia = TargetEncoder()
    df["producto_encoded"] = te_producto.fit_transform(df["producto"], df["variacion_real"])
    df["provincia_encoded"] = te_provincia.fit_transform(df["provincia"], df["variacion_real"])
    
    encoders = {
        "producto": te_producto,
        "provincia": te_provincia
    }
    # Target encode canton, mercado, and presentacion
    for col in ["canton", "mercado", "presentacion", "tipo_mercado"]:
        if col in df.columns:
            te_col = TargetEncoder()
            df[f"{col}_encoded"] = te_col.fit_transform(df[col], df["variacion_real"])
            encoders[col] = te_col
            
    # NUEVO: categoria como binaria (0=no_perecedero, 1=perecedero)
    df["categoria_perecedero"] = (df["categoria"] == "perecedero").astype(int)
    return df, encoders

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



if __name__ == "__main__":
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

    col_id = 'periodo' if 'periodo' in df_crudo.columns else 'quincena_id'
    print(df_crudo[col_id].nunique())
    print(sorted(df_crudo[col_id].unique()))

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)

