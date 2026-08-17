import os
import joblib
import pandas as pd
import numpy as np


def cargar_modelo_y_artefactos(models_dir=None):
    """
    Carga el mejor modelo guardado en data/models/ y sus encoders asociados.
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(__file__), "data", "models")

    modelo_path = os.path.join(models_dir, "mejor_modelo.pkl")
    le_target_path = os.path.join(models_dir, "le_target.pkl")
    le_prod_path = os.path.join(models_dir, "le_producto.pkl")
    le_prov_path = os.path.join(models_dir, "le_provincia.pkl")
    features_path = os.path.join(models_dir, "features.pkl")

    if not (os.path.exists(modelo_path) and os.path.exists(le_target_path)):
        return None, None, None, None, None

    modelo = joblib.load(modelo_path)
    le_target = joblib.load(le_target_path)
    le_prod = joblib.load(le_prod_path) if os.path.exists(le_prod_path) else None
    le_prov = joblib.load(le_prov_path) if os.path.exists(le_prov_path) else None
    features = joblib.load(features_path) if os.path.exists(features_path) else None

    return modelo, le_target, le_prod, le_prov, features


def obtener_ultimo_registro(df, producto, provincia):
    """
    Busca en el dataset el registro más reciente para la combinación de producto y provincia.
    Si no existe la combinación exacta con esa provincia, busca el último registro válido de ese producto.
    """
    if df is None or df.empty:
        return None

    df_sub = df[(df["producto"] == producto) & (df["provincia"] == provincia)]

    if df_sub.empty:
        df_sub = df[df["producto"] == producto]

    if df_sub.empty:
        return None
    
    if "precio_t1" in df_sub.columns:
        df_validos = df_sub[df_sub["precio_t1"].notna()]
        if not df_validos.empty:
            return df_validos.iloc[-1].to_dict()

    return df_sub.iloc[-1].to_dict()


def predecir_registro(precio_t1, precio_t2, mes, producto, provincia, le_prod, le_prov, le_target, modelo, features, categoria_perecedero=0, canton_encoded=0.0, mercado_encoded=0.0, presentacion_encoded=0.0, tipo_mercado_encoded=0.0, registro_dict=None):
    """
    Recibe datos de un producto y sus precios anteriores para realizar la predicción de comportamiento.
    """
    variacion_t2_t1 = round(((precio_t1 - precio_t2) / precio_t2) * 100, 2) if (precio_t2 and precio_t2 > 0) else 0.0
    promedio_movil_2q = round((precio_t1 + precio_t2) / 2.0, 4)
    promedio_movil_3q = promedio_movil_2q

    prod_enc = le_prod.transform([producto])[0] if (le_prod and producto in le_prod.classes_) else 0
    prov_enc = le_prov.transform([provincia])[0] if (le_prov and provincia in le_prov.classes_) else 0

    row_dict = {
        "precio_t1": precio_t1,
        "variacion_t2_t1": variacion_t2_t1,
        "promedio_movil_2q": promedio_movil_2q,
        "promedio_movil_3q": promedio_movil_3q,
        "volatilidad_3q": 0.0,
        "momentum": 0.0,
        "mes": mes,
        "producto_encoded": prod_enc,
        "provincia_encoded": prov_enc,
        "canton_encoded": canton_encoded,
        "mercado_encoded": mercado_encoded,
        "presentacion_encoded": presentacion_encoded,
        "tipo_mercado_encoded": tipo_mercado_encoded,
        "categoria_perecedero": int(categoria_perecedero),
    }


    if registro_dict is not None:
        for k, v in registro_dict.items():
            if pd.notna(v):
                row_dict[k] = v

    mes_actual = row_dict.get("mes", 1)
    row_dict["mes_seno"] = np.sin(2 * np.pi * mes_actual / 12)
    row_dict["mes_coseno"] = np.cos(2 * np.pi * mes_actual / 12)

    pm2 = row_dict.get("promedio_movil_2q", 0.0)
    pm3 = row_dict.get("promedio_movil_3q", 0.0)

    row_dict["distancia_pm2_pct"] = (precio_t1 - pm2) / pm2 if pm2 != 0 else 0.0
    row_dict["distancia_pm3_pct"] = (precio_t1 - pm3) / pm3 if pm3 != 0 else 0.0

    row_dict.setdefault("volatilidad_3q", 0.0)
    row_dict.setdefault("momentum", 0.0)
    for i in range(1, 7):
        row_dict.setdefault(f"var_lag_{i}", 0.0)

    lags_cols = [f"var_lag_{i}" for i in range(1, 7)]
    full_features = features + lags_cols

    for feat in full_features:
        row_dict.setdefault(feat, 0.0)

    X_single = pd.DataFrame([row_dict])[full_features]
    
    pred_idx = modelo.predict(X_single)[0]
    pred_label = le_target.inverse_transform([pred_idx])[0]

    probs = None
    if hasattr(modelo, "predict_proba"):
        probs_raw = modelo.predict_proba(X_single)[0]
        probs = {cls: round(prob * 100, 1) for cls, prob in zip(le_target.classes_, probs_raw)}

    return pred_label, probs, row_dict


def predecir_dataframe(df, modelo, le_target, features):
    """
    Clasifica un DataFrame completo que contenga las features requeridas.
    """
    df_eval = df.dropna(subset=["precio_t2", "variacion_t2_t1", "promedio_movil_2q", "promedio_movil_3q"]).copy()
    if "categoria_perecedero" not in df_eval.columns:
        df_eval["categoria_perecedero"] = 0

    if "mes_seno" not in df_eval.columns and "mes" in df_eval.columns:
        df_eval["mes_seno"] = np.sin(2 * np.pi * df_eval["mes"] / 12)
    if "mes_coseno" not in df_eval.columns and "mes" in df_eval.columns:
        df_eval["mes_coseno"] = np.cos(2 * np.pi * df_eval["mes"] / 12)

    lags_cols = [f"var_lag_{i}" for i in range(1, 7)]
    full_features = list(features)
    for col in lags_cols:
        if col not in full_features:
            full_features.append(col)

    for col in full_features:
        if col not in df_eval.columns:
            df_eval[col] = 0.0

    X = df_eval[full_features]

    preds_idx = modelo.predict(X)
    df_eval["prediccion"] = le_target.inverse_transform(preds_idx)

    if hasattr(modelo, "predict_proba"):
        probs_matrix = modelo.predict_proba(X)
        for i, cls_name in enumerate(le_target.classes_):
            df_eval[f"prob_{cls_name}"] = np.round(probs_matrix[:, i] * 100, 1)

    return df_eval