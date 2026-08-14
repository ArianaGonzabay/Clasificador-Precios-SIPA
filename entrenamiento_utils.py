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
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone, BaseEstimator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# FEATURES actualizadas con variables informativas de series temporales reales + Lags temporales
FEATURES = [
    "precio_t1", 
    "variacion_t2_t1",
    # --- CAMBIO CLAVE: De valores crudos a distancias relativas ---
    "distancia_pm2_pct", # Qué tan lejos está el precio de su promedio de 2 quincenas
    "distancia_pm3_pct", # Qué tan lejos está el precio de su promedio de 3 quincenas
    # --------------------------------------------------------------
    "volatilidad_3q", 
    "momentum",
    "mes_seno", 
    "mes_coseno", 
    "producto_encoded", 
    "provincia_encoded", 
    "categoria_perecedero",
]
TARGET = "comportamiento"
FEATURES_REZAGO = ["precio_t2", "variacion_t2_t1", "promedio_movil_2q", "promedio_movil_3q"]

# Modelos que necesitan features escaladas (sensibles a la magnitud de las variables)
MODELOS_QUE_NECESITAN_ESCALADO = {"Logistic Regression", "SVM"}


def ejecutar_entrenamiento_y_evaluacion(df_final, le_producto=None, le_provincia=None):
    """
    Ejecuta la Fase 2: Ordenamiento temporal cronológico, limpieza de filas sin rezago,
    entrenamiento de los modelos, cálculo de métricas y guardado del mejor modelo.
    """
    
    # 1. VALIDACIÓN Y CÁLCULOS AL VUELO (Ejecutado una sola vez y de forma limpia)
    if "categoria_perecedero" not in df_final.columns:
        print("[AVISO] 'categoria_perecedero' no está en el dataset. Usando 0.")
        df_final = df_final.copy()
        df_final["categoria_perecedero"] = 0

    if "volatilidad_3q" not in df_final.columns:
        df_final = df_final.copy()
        df_final["volatilidad_3q"] = df_final.groupby(["producto", "provincia"])["variacion_t2_t1"].transform(lambda x: x.rolling(window=3, min_periods=1).std().fillna(0.0))

    if "momentum" not in df_final.columns:
        df_final = df_final.copy()
        df_final["momentum"] = 0.0

    if "distancia_pm2_pct" not in df_final.columns:
        df_final = df_final.copy()
        df_final["distancia_pm2_pct"] = np.where(
            df_final["promedio_movil_2q"] != 0, 
            (df_final["precio_t1"] - df_final["promedio_movil_2q"]) / df_final["promedio_movil_2q"], 
            0.0
        )
        
    if "distancia_pm3_pct" not in df_final.columns:
        df_final = df_final.copy()
        df_final["distancia_pm3_pct"] = np.where(
            df_final["promedio_movil_3q"] != 0, 
            (df_final["precio_t1"] - df_final["promedio_movil_3q"]) / df_final["promedio_movil_3q"], 
            0.0
        )

    # 2. ORDENAMIENTO ESTRICTO
    if "periodo" in df_final.columns:
        df_final = df_final.sort_values(by=["periodo", "producto", "provincia"]).reset_index(drop=True)
    elif "quincena_id" in df_final.columns:
        df_final = df_final.sort_values(by=["quincena_id", "producto", "provincia"]).reset_index(drop=True)

    # 3. LIMPIEZA DE FILAS SIN REZAGO
    filas_antes = len(df_final)
    df_limpio = df_final.dropna(subset=FEATURES_REZAGO).copy()
    
    # 4. CODIFICACIÓN CÍCLICA DEL TIEMPO
    df_limpio["mes_seno"] = np.sin(2 * np.pi * df_limpio["mes"] / 12)
    df_limpio["mes_coseno"] = np.cos(2 * np.pi * df_limpio["mes"] / 12)
    
    filas_despues = len(df_limpio)
    filas_eliminadas = filas_antes - filas_despues

    # 5. CREACIÓN DE LAGS TEMPORALES
    df_limpio = df_limpio.copy()
    lags_cols = []
    for i in range(1, 7):
        col = f"var_lag_{i}"
        df_limpio[col] = df_limpio.groupby(["producto", "provincia"])["variacion_t2_t1"].shift(i).fillna(0.0)
        lags_cols.append(col)

    feature_cols = FEATURES + lags_cols
    X = df_limpio[feature_cols].copy()
    y = df_limpio[TARGET].copy()

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # --- CLASES LSTM ---
    class LSTMRegresorModule(nn.Module):
        def __init__(self, input_dim=17, hidden_units=64):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_units, num_layers=2, batch_first=True)
            self.fc = nn.Linear(hidden_units, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    class LSTMModeloProfesor(BaseEstimator):
        def __init__(self, look_back=2, epochs=35, lr=0.01):
            self.look_back = look_back
            self.epochs = epochs
            self.lr = lr
            self.scaler = StandardScaler()
            self.model = None

        def fit(self, X, y):
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
            X_scaled = self.scaler.fit_transform(X_arr)
            y_var = X["variacion_t2_t1"].values.astype(np.float32) if isinstance(X, pd.DataFrame) else X_arr[:, 2].astype(np.float32)
            X_t = torch.FloatTensor(X_scaled).unsqueeze(1)
            y_t = torch.FloatTensor(y_var).unsqueeze(1)
            self.model = LSTMRegresorModule(input_dim=X_scaled.shape[1], hidden_units=64)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                output = self.model(X_t)
                loss = criterion(output, y_t)
                loss.backward()
                optimizer.step()
            return self

        def predict(self, X):
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
            X_scaled = self.scaler.transform(X_arr)
            X_t = torch.FloatTensor(X_scaled).unsqueeze(1)
            self.model.eval()
            with torch.no_grad():
                var_preds = self.model(X_t).squeeze().numpy()
            if var_preds.ndim == 0:
                var_preds = np.array([var_preds.item()])
            clases = []
            for v in var_preds:
                if v > 7.0:
                    clases.append(0) # Alza
                elif v < -7.0:
                    clases.append(1) # Caida
                else:
                    clases.append(2) # Estable
            return np.array(clases)

    # 6. CONFIGURACIÓN DE MODELOS (Restaurada a la versión ganadora para evitar subajuste)
    modelos_config = {
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1, class_weight="balanced"),
        "XGBoost": XGBClassifier(learning_rate=0.08, max_depth=9, n_estimators=400, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42, class_weight="balanced"),
        "Logistic Regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "SVM": SVC(C=1.0, kernel="rbf", random_state=42, class_weight="balanced", probability=True),
        "LSTM": LSTMModeloProfesor(look_back=2, epochs=30, lr=0.01),
    }

    # 7. ENTRENAMIENTO Y EVALUACIÓN
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {}
    candidatos = []

    for nombre, modelo_base in modelos_config.items():
        fold_metrics = []
        fold_matrices = []
        ultimo_modelo = None
        scaler_usado = None

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            if nombre in MODELOS_QUE_NECESITAN_ESCALADO:
                scaler = StandardScaler()
                X_train_fit = scaler.fit_transform(X_train)
                X_test_fit = scaler.transform(X_test)
                scaler_usado = scaler
            else:
                X_train_fit = X_train
                X_test_fit = X_test

            if hasattr(modelo_base, "__sklearn_clone__") or hasattr(modelo_base, "get_params"):
                try:
                    modelo = clone(modelo_base)
                except Exception:
                    modelo = LSTMModeloProfesor(look_back=2, epochs=20, lr=0.01)
            else:
                modelo = LSTMModeloProfesor(look_back=2, epochs=20, lr=0.01)

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
        feat_imp = ultimo_modelo.feature_importances_ if hasattr(ultimo_modelo, "feature_importances_") else None

        resultados[nombre] = {
            "metricas": promedios,
            "metricas_por_fold": metrics_df,
            "matriz_confusion": cm_ultimo,
            "modelo_entrenado": ultimo_modelo,
            "scaler": scaler_usado,
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

    candidatos_validos = [c for c in candidatos if c["Cumple Criterios"] == "SI"]
    if candidatos_validos:
        mejor = min(candidatos_validos, key=lambda x: x["Falsos (último fold)"])
    else:
        mejor = max(candidatos, key=lambda x: x["F1-Score (Macro)"])

    mejor_nombre = mejor["Modelo"]
    mejor_modelo = resultados[mejor_nombre]["modelo_entrenado"]
    mejor_scaler = resultados[mejor_nombre]["scaler"]

    # 8. GUARDAR ARTEFACTOS
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