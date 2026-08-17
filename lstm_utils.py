"""
lstm_utils.py — Clasificación del comportamiento de precios con LSTM.

A diferencia de los modelos tabulares (Random Forest, XGBoost, etc.), LSTM
necesita SECUENCIAS: para cada producto+provincia, una serie ordenada de
precios en el tiempo, no filas sueltas con rezagos fijos.

Usa el df_pivot (formato wide: una fila por producto+provincia, una columna
por periodo) que ya genera preprocesar_datos() antes de pivotear a formato
largo — es la fuente correcta para armar secuencias.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

VENTANA = 6         
UMBRAL_PCT = 7.0      

def construir_secuencias(df_pivot, periodos_unicos, ventana=VENTANA):
    """
    Convierte el dataset wide (producto+provincia x periodo) en secuencias
    deslizantes de tamaño `ventana` para entrenar el LSTM.

    Devuelve:
        X_seq: array (n_muestras, ventana, 1) — precios normalizados por serie
        y_labels: array (n_muestras,) — 'Alza'/'Estable'/'Caída'
        meta: lista de dicts con producto, provincia, periodo_objetivo (para trazabilidad)
    """
    X_seq, y_labels, meta = [], [], []

    for _, row in df_pivot.iterrows():
        producto = row["producto_raw"]
        provincia = row["provincia"]

        precios = []
        for periodo in periodos_unicos:
            col = f"precio_actual_{periodo}"
            precios.append(row.get(col, np.nan) if col in df_pivot.columns else np.nan)

        serie = pd.Series(precios, index=periodos_unicos)

        for i in range(ventana, len(serie)):
            ventana_precios = serie.iloc[i - ventana:i].values
            precio_objetivo = serie.iloc[i]
            precio_anterior = serie.iloc[i - 1]

            if np.isnan(ventana_precios).any() or pd.isna(precio_objetivo) or pd.isna(precio_anterior):
                continue
            if precio_anterior <= 0:
                continue

            variacion = ((precio_objetivo - precio_anterior) / precio_anterior) * 100
            if variacion > UMBRAL_PCT:
                etiqueta = "Alza"
            elif variacion < -UMBRAL_PCT:
                etiqueta = "Caída"
            else:
                etiqueta = "Estable"

            X_seq.append(ventana_precios)
            y_labels.append(etiqueta)
            meta.append({
                "producto": producto,
                "provincia": provincia,
                "periodo_objetivo": periodos_unicos[i],
            })

    X_seq = np.array(X_seq, dtype=float)
    y_labels = np.array(y_labels)
    return X_seq, y_labels, meta


def entrenar_lstm(df_pivot, periodos_unicos, ventana=VENTANA, epochs=50, batch_size=64):
    """
    Entrena un LSTM clasificador con split temporal (80% más antiguo para
    entrenar, 20% más reciente para test — no aleatorio, respeta el tiempo).
    """
    print("Construyendo secuencias...")
    X_seq, y_labels, meta = construir_secuencias(df_pivot, periodos_unicos, ventana)
    print(f"Secuencias generadas: {len(X_seq)}")
    print(pd.Series(y_labels).value_counts())

    medias = X_seq.mean(axis=1, keepdims=True)
    stds = X_seq.std(axis=1, keepdims=True)
    stds[stds == 0] = 1  
    X_seq_norm = (X_seq - medias) / stds
    X_seq_norm = X_seq_norm.reshape(X_seq_norm.shape[0], X_seq_norm.shape[1], 1)

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_labels)
    y_categorical = to_categorical(y_encoded)

    df_meta = pd.DataFrame(meta)
    orden = df_meta.sort_values("periodo_objetivo").index.values
    n_train = int(len(orden) * 0.8)
    idx_train, idx_test = orden[:n_train], orden[n_train:]

    X_train, X_test = X_seq_norm[idx_train], X_seq_norm[idx_test]
    y_train, y_test = y_categorical[idx_train], y_categorical[idx_test]
    y_train_labels, y_test_labels = y_encoded[idx_train], y_encoded[idx_test]

    print(f"\nTrain: {len(X_train)} secuencias | Test: {len(X_test)} secuencias")

    pesos = compute_class_weight("balanced", classes=np.unique(y_train_labels), y=y_train_labels)
    class_weight_dict = dict(enumerate(pesos))

    modelo = Sequential([
        Masking(mask_value=0.0, input_shape=(ventana, 1)),
        LSTM(32, return_sequences=True),
        Dropout(0.3),
        LSTM(16),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax"),  
    ])

    modelo.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    print("\nEntrenando LSTM...")
    historia = modelo.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred_probs = modelo.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test_labels, y_pred)
    f1 = f1_score(y_test_labels, y_pred, average="macro")
    prec = precision_score(y_test_labels, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test_labels, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test_labels, y_pred)

    print("\n=== RESULTADOS LSTM (test, 20% más reciente) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro): {rec:.4f}")
    print(f"\n¿Cumple criterios? F1>=0.75: {f1>=0.75} | Acc>=0.80: {acc>=0.80}")
    print("\nMatriz de confusión (filas=real, columnas=predicho):")
    print(pd.DataFrame(cm, index=le_target.classes_, columns=le_target.classes_))

    return {
        "modelo": modelo,
        "le_target": le_target,
        "historia": historia,
        "metricas": {"accuracy": acc, "f1_macro": f1, "precision": prec, "recall": rec},
        "matriz_confusion": cm,
        "ventana": ventana,
    }