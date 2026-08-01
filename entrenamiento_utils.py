import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

FEATURES = [
    "precio_t1", "precio_t2", "variacion_t2_t1",
    "promedio_movil_2q", "promedio_movil_3q",
    "mes", "producto_encoded", "provincia_encoded"
]
TARGET = "comportamiento"
FEATURES_REZAGO = ["precio_t2", "variacion_t2_t1", "promedio_movil_2q", "promedio_movil_3q"]


def ejecutar_entrenamiento_y_evaluacion(df_final, le_producto=None, le_provincia=None):
    """
    Ejecuta la Fase 2: Limpieza de filas sin rezago, entrenamiento de los 6 modelos con TimeSeriesSplit,
    cálculo de métricas, matrices de confusión, importancia de variables y guardado del mejor modelo.
    """
    # 1. Limpieza de filas sin rezago suficiente (NaN)
    filas_antes = len(df_final)
    df_limpio = df_final.dropna(subset=FEATURES_REZAGO).copy()
    filas_despues = len(df_limpio)
    filas_eliminadas = filas_antes - filas_despues

    # 2. X y y
    X = df_limpio[FEATURES].copy()
    y = df_limpio[TARGET].copy()

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # 3. TimeSeriesSplit (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)

    # 4. Configuración de modelos (Random Forest y XGBoost)
    modelos_config = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(learning_rate=0.05, tree_method="hist", random_state=42, eval_metric="mlogloss", n_jobs=-1),
    }

    resultados = {}
    candidatos = []

    for nombre, modelo_base in modelos_config.items():
        fold_metrics = []
        fold_matrices = []
        ultimo_modelo = None

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            modelo = clone(modelo_base)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=range(len(le_target.classes_)))

            fold_metrics.append({"accuracy": acc, "f1_macro": f1, "precision": prec, "recall": rec})
            fold_matrices.append(cm)
            ultimo_modelo = modelo

        metrics_df = pd.DataFrame(fold_metrics)
        promedios = metrics_df.mean().to_dict()
        cm_ultimo = fold_matrices[-1]

        falsos = cm_ultimo.sum() - np.trace(cm_ultimo)
        cumple = promedios["f1_macro"] >= 0.75 and promedios["accuracy"] >= 0.80

        feat_imp = None
        if hasattr(ultimo_modelo, "feature_importances_"):
            feat_imp = ultimo_modelo.feature_importances_

        resultados[nombre] = {
            "metricas": promedios,
            "metricas_por_fold": metrics_df,
            "matriz_confusion": cm_ultimo,
            "modelo_entrenado": ultimo_modelo,
            "feature_importances": feat_imp,
            "cumple": cumple,
            "falsos": falsos,
        }

        candidatos.append({
            "Modelo": nombre,
            "F1-Score (Macro)": round(promedios["f1_macro"], 4),
            "Accuracy": round(promedios["accuracy"], 4),
            "Precision (Macro)": round(promedios["precision"], 4),
            "Recall (Macro)": round(promedios["recall"], 4),
            "Falsos (último fold)": falsos,
            "Cumple Criterios": "SI" if cumple else "NO",
        })

    tabla_comparativa = pd.DataFrame(candidatos)

    # Selección del mejor modelo
    candidatos_validos = [c for c in candidatos if c["Cumple Criterios"] == "SI"]
    if candidatos_validos:
        mejor = min(candidatos_validos, key=lambda x: x["Falsos (último fold)"])
    else:
        mejor = max(candidatos, key=lambda x: x["F1-Score (Macro)"])

    mejor_nombre = mejor["Modelo"]
    mejor_modelo = resultados[mejor_nombre]["modelo_entrenado"]

    # Guardar artefactos
    models_dir = os.path.join(os.path.dirname(__file__), "data", "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(mejor_modelo, os.path.join(models_dir, "mejor_modelo.pkl"))
    joblib.dump(le_target, os.path.join(models_dir, "le_target.pkl"))
    if le_producto is not None:
        joblib.dump(le_producto, os.path.join(models_dir, "le_producto.pkl"))
    if le_provincia is not None:
        joblib.dump(le_provincia, os.path.join(models_dir, "le_provincia.pkl"))
    joblib.dump(FEATURES, os.path.join(models_dir, "features.pkl"))

    return {
        "filas_antes": filas_antes,
        "filas_despues": filas_despues,
        "filas_eliminadas": filas_eliminadas,
        "resultados": resultados,
        "tabla_comparativa": tabla_comparativa,
        "mejor_nombre": mejor_nombre,
        "mejor_metricas": mejor,
        "classes": list(le_target.classes_),
        "features": FEATURES,
        "models_dir": models_dir,
    }
