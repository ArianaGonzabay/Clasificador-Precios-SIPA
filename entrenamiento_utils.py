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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone

# FEATURES restauradas: se vuelve a incluir precio_t1, precio_t2 y promedios móviles,
# y se agrega categoria_perecedero (requiere el cambio en preprocesamiento.ipynb)
FEATURES = [
    "variacion_t2_t1",
    "distancia_pm2_pct", "distancia_pm3_pct",
    "mes", "producto_encoded", "provincia_encoded",
    "categoria_perecedero",
]
TARGET = "comportamiento"
FEATURES_REZAGO = ["variacion_t2_t1", "distancia_pm2_pct", "distancia_pm3_pct"]

# Modelos que necesitan features escaladas (sensibles a la magnitud de las variables)
MODELOS_QUE_NECESITAN_ESCALADO = {"Logistic Regression", "SVM"}


def ejecutar_entrenamiento_y_evaluacion(df_final, le_producto=None, le_provincia=None):
    """
    Ejecuta la Fase 2: Limpieza de filas sin rezago, entrenamiento de los 6 modelos con TimeSeriesSplit,
    cálculo de métricas, matrices de confusión, importancia de variables y guardado del mejor modelo.
    """
    # 1. Limpieza de filas sin rezago suficiente (NaN)
    filas_antes = len(df_final)

    # Si la columna categoria_perecedero no existe todavía (no se ha actualizado preprocesamiento),
    # se crea con 0 para no romper el pipeline, pero avisa por consola.
    if "categoria_perecedero" not in df_final.columns:
        print("[AVISO] 'categoria_perecedero' no está en el dataset. "
              "Actualiza preprocesamiento.ipynb. Se usará 0 para todos los registros.")
        df_final = df_final.copy()
        df_final["categoria_perecedero"] = 0

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

    # 4. Configuración de modelos (los 6 propuestos en la Tarea 4)
    # 4. Configuración de modelos (Regularizados para evitar el sobreajuste)
    modelos_config = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, 
            max_depth=5,              # Reducido de 10 a 5 para evitar memorización
            min_samples_split=15,     # Requiere más datos para crear una rama nueva
            random_state=42, 
            n_jobs=-1, 
            class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            learning_rate=0.02,       # Aprendizaje más lento y cuidadoso
            max_depth=4,              # Reducido drásticamente de 10 a 4
            n_estimators=300, 
            subsample=0.8,            # Usa solo el 80% de los datos por árbol (evita ruido)
            colsample_bytree=0.8,     # Usa solo el 80% de las columnas por árbol
            tree_method="hist", 
            random_state=42, 
            eval_metric="mlogloss", 
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42, class_weight="balanced"),
        "Logistic Regression": LogisticRegression(C=0.5, solver="lbfgs", max_iter=1000, random_state=42, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"), # Aumentamos vecinos para suavizar predicción
        "SVM": SVC(C=0.8, kernel="rbf", random_state=42, class_weight="balanced", probability=True),
    }

    resultados = {}
    candidatos = []

    for nombre, modelo_base in modelos_config.items():
        fold_metrics = []
        fold_matrices = []
        ultimo_modelo = None
        scaler_usado = None  # se guarda el scaler si el modelo lo necesitó

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Escalado solo para modelos sensibles a la magnitud (LR, SVM)
            if nombre in MODELOS_QUE_NECESITAN_ESCALADO:
                scaler = StandardScaler()
                X_train_fit = scaler.fit_transform(X_train)
                X_test_fit = scaler.transform(X_test)
                scaler_usado = scaler
            else:
                X_train_fit = X_train
                X_test_fit = X_test

            modelo = clone(modelo_base)

            if nombre == "XGBoost":
                pesos_train = compute_sample_weight("balanced", y_train)
                modelo.fit(X_train_fit, y_train, sample_weight=pesos_train)
            else:
                modelo.fit(X_train_fit, y_train)

            y_pred = modelo.predict(X_test_fit)

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
            "scaler": scaler_usado,  # None si no aplicó escalado
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
    mejor_scaler = resultados[mejor_nombre]["scaler"]

    # Guardar artefactos
    models_dir = os.path.join(os.path.dirname(__file__), "data", "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(mejor_modelo, os.path.join(models_dir, "mejor_modelo.pkl"))
    joblib.dump(le_target, os.path.join(models_dir, "le_target.pkl"))
    if mejor_scaler is not None:
        joblib.dump(mejor_scaler, os.path.join(models_dir, "scaler.pkl"))
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