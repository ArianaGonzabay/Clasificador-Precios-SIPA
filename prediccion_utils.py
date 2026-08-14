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

    # 1. Coincidencia exacta producto + provincia
    df_sub = df[(df["producto"] == producto) & (df["provincia"] == provincia)]

    # 2. Si no hay coincidencia exacta de esa provincia, buscar por producto en cualquier provincia
    if df_sub.empty:
        df_sub = df[df["producto"] == producto]

    if df_sub.empty:
        return None

    # Preferir registros con precio_t1 válido
    if "precio_t1" in df_sub.columns:
        df_validos = df_sub[df_sub["precio_t1"].notna()]
        if not df_validos.empty:
            return df_validos.iloc[-1].to_dict()

    return df_sub.iloc[-1].to_dict()


def predecir_registro(precio_t1, precio_t2, mes, producto, provincia, le_prod, le_prov, le_target, modelo, features, categoria_perecedero=0):
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
        "precio_t2": precio_t2,
        "variacion_t2_t1": variacion_t2_t1,
        "promedio_movil_2q": promedio_movil_2q,
        "promedio_movil_3q": promedio_movil_3q,
        "volatilidad_3q": 0.0,
        "momentum": 0.0,
        "mes": mes,
        "producto_encoded": prod_enc,
        "provincia_encoded": prov_enc,
        "categoria_perecedero": int(categoria_perecedero),
    }

    X_single = pd.DataFrame([row_dict])[features]
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
    X = df_eval[features]

    preds_idx = modelo.predict(X)
    df_eval["prediccion"] = le_target.inverse_transform(preds_idx)

    if hasattr(modelo, "predict_proba"):
        probs_matrix = modelo.predict_proba(X)
        for i, cls_name in enumerate(le_target.classes_):
            df_eval[f"prob_{cls_name}"] = np.round(probs_matrix[:, i] * 100, 1)

    return df_eval
