"""
Clasificador de Precios Mayoristas SIPA — interfaz web sin API (Flask + Jinja2).

Cada ruta renderiza HTML directamente en el servidor: al enviar un formulario
(subir archivo, entrenar, predecir) el navegador recarga la página con el
resultado ya calculado. No hay endpoints JSON ni llamadas fetch() del lado
del cliente — es la arquitectura clásica "servidor primero".

Ejecutar con:  python app.py
Luego abrir:   http://localhost:5000
"""

import io
import os

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from entrenamiento_utils import ejecutar_entrenamiento_y_evaluacion
from ingesta_precios import procesar_archivo_precios
from preprocesamiento import preprocesar_datos
from prediccion_utils import (
    cargar_modelo_y_artefactos,
    obtener_ultimo_registro,
    predecir_dataframe,
    predecir_registro,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")
CSV_CRUDO = os.path.join(PROCESSED_DIR, "dataset_crudo_sipa.csv")
CSV_PREPROC = os.path.join(PROCESSED_DIR, "dataset_preprocesado_sipa.csv")
CLIMA_PATH = os.path.join(DATA_DIR, "clima_historico.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "sipa-clasificador-dev"  # solo se usa para los mensajes flash

# Caché en memoria de resultados que no se guardan tal cual en disco
# (tabla comparativa de modelos, dataset preprocesado recién calculado, etc.)
# Es un único proceso de un solo usuario -- equivalente a lo que hacía
# st.session_state en la versión de Streamlit.
CACHE = {
    "reporte_extraccion": None,
    "resultado_preprocesamiento": None,
    "res_entrenamiento": None,
    "df_predicho": None,
}


def estado_pipeline():
    return {
        "extraido": os.path.exists(CSV_CRUDO),
        "preprocesado": os.path.exists(CSV_PREPROC),
        "entrenado": os.path.exists(os.path.join(MODELS_DIR, "mejor_modelo.pkl")),
    }


@app.context_processor
def inyectar_estado():
    return {"estado": estado_pipeline()}


def _cargar_dataset_preprocesado_disco():
    if os.path.exists(CSV_PREPROC):
        try:
            return pd.read_csv(CSV_PREPROC)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# INICIO
# ---------------------------------------------------------------------------

@app.route("/")
def inicio():
    return render_template("inicio.html")


# ---------------------------------------------------------------------------
# EXTRACCIÓN Y PREPROCESAMIENTO
# ---------------------------------------------------------------------------

@app.route("/extraccion")
def extraccion():
    df_preview = None
    reporte = CACHE.get("reporte_extraccion")

    if os.path.exists(CSV_CRUDO):
        df_full = pd.read_csv(CSV_CRUDO, encoding="utf-8-sig")
        df_preview = df_full.tail(50)
        if reporte is None:
            reporte = {
                "registros_completos": len(df_full),
                "registros_parciales": int((df_full.get("estado_precio", pd.Series(dtype=str)) == "parcial").sum()),
                "quincenas": df_full["periodo"].nunique() if "periodo" in df_full.columns else 0,
            }

    resultado_preproc = CACHE.get("resultado_preprocesamiento")
    df_preproc_preview = None
    stats = None
    if resultado_preproc is not None:
        df_preproc_preview = resultado_preproc["dataset_final"].tail(50)
        stats = resultado_preproc["estadisticas"]
    elif os.path.exists(CSV_PREPROC):
        df_preproc_preview = pd.read_csv(CSV_PREPROC).tail(50)

    return render_template(
        "extraccion.html",
        df_preview=df_preview,
        reporte=reporte,
        df_preproc_preview=df_preproc_preview,
        stats=stats,
    )


@app.route("/extraccion/subir-precios", methods=["POST"])
def subir_precios():
    archivo = request.files.get("archivo_precios")
    if not archivo or archivo.filename == "":
        flash("Selecciona un archivo de precios antes de subir.", "error")
        return redirect(url_for("extraccion"))
    try:
        df, reporte = procesar_archivo_precios(archivo, CSV_CRUDO)
        CACHE["reporte_extraccion"] = reporte
        CACHE["resultado_preprocesamiento"] = None
        flash(f"Archivo procesado correctamente: {len(df)} registros en total.", "exito")
    except Exception as e:
        flash(f"Error procesando el archivo: {e}", "error")
    return redirect(url_for("extraccion"))


@app.route("/extraccion/subir-clima", methods=["POST"])
def subir_clima():
    archivo = request.files.get("archivo_clima")
    if not archivo or archivo.filename == "":
        flash("Selecciona un archivo de clima antes de subir.", "error")
        return redirect(url_for("extraccion"))
    try:
        os.makedirs(os.path.dirname(CLIMA_PATH), exist_ok=True)
        if archivo.filename.lower().endswith(".csv"):
            archivo.save(CLIMA_PATH)
        else:
            df_clima = pd.read_excel(archivo)
            df_clima.to_csv(CLIMA_PATH, index=False, encoding="utf-8-sig")
        flash("Datos climáticos guardados correctamente.", "exito")
    except Exception as e:
        flash(f"Error guardando datos climáticos: {e}", "error")
    return redirect(url_for("extraccion"))


@app.route("/extraccion/preprocesar", methods=["POST"])
def ejecutar_preprocesamiento():
    if not os.path.exists(CSV_CRUDO):
        flash("No hay dataset crudo en disco. Sube un archivo de precios primero.", "error")
        return redirect(url_for("extraccion"))
    try:
        df_crudo = pd.read_csv(CSV_CRUDO)
        resultado = preprocesar_datos(df_crudo)
        resultado["dataset_final"].to_csv(CSV_PREPROC, index=False, encoding="utf-8-sig")
        CACHE["resultado_preprocesamiento"] = resultado
        CACHE["res_entrenamiento"] = None  # invalida entrenamientos previos con datos viejos
        flash("Preprocesamiento completado y guardado en disco.", "exito")
    except Exception as e:
        flash(f"Error en preprocesamiento: {e}", "error")
    return redirect(url_for("extraccion"))


# ---------------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------------

@app.route("/entrenamiento")
def entrenamiento():
    res = CACHE.get("res_entrenamiento")
    return render_template("entrenamiento.html", res=res)


@app.route("/entrenamiento/ejecutar", methods=["POST"])
def ejecutar_entrenamiento():
    if not os.path.exists(CSV_PREPROC):
        flash("Primero debes ejecutar el preprocesamiento en la sección Extracción.", "error")
        return redirect(url_for("entrenamiento"))
    try:
        df_preproc = pd.read_csv(CSV_PREPROC)
        resultado_preproc = CACHE.get("resultado_preprocesamiento") or {}
        le_prod = resultado_preproc.get("le_producto")
        le_prov = resultado_preproc.get("le_provincia")
        encoders = resultado_preproc.get("encoders")

        res = ejecutar_entrenamiento_y_evaluacion(df_preproc, le_prod, le_prov, encoders=encoders)
        CACHE["res_entrenamiento"] = res
        flash(f"Entrenamiento finalizado. Modelo seleccionado: {res['mejor_nombre']}.", "exito")
    except Exception as e:
        flash(f"Error durante el entrenamiento: {e}", "error")
    return redirect(url_for("entrenamiento"))


# ---------------------------------------------------------------------------
# PREDICCIÓN
# ---------------------------------------------------------------------------

@app.route("/prediccion")
def prediccion():
    modelo, le_target, le_prod, le_prov, features = cargar_modelo_y_artefactos(MODELS_DIR)
    df_preproc = _cargar_dataset_preprocesado_disco()

    productos = list(le_prod.classes_) if le_prod is not None and hasattr(le_prod, "classes_") else []
    if not productos and df_preproc is not None and "producto" in df_preproc.columns:
        productos = sorted(df_preproc["producto"].dropna().unique().tolist())

    provincias = list(le_prov.classes_) if le_prov is not None and hasattr(le_prov, "classes_") else []
    if not provincias and df_preproc is not None and "provincia" in df_preproc.columns:
        provincias = sorted(df_preproc["provincia"].dropna().unique().tolist())

    return render_template(
        "prediccion.html",
        modelo_listo=(modelo is not None),
        productos=productos,
        provincias=provincias,
        resultado=None,
        registro=None,
        seleccion=None,
        df_lote=None,
    )


@app.route("/prediccion/individual", methods=["POST"])
def prediccion_individual():
    producto = request.form.get("producto")
    provincia = request.form.get("provincia")

    modelo, le_target, le_prod, le_prov, features = cargar_modelo_y_artefactos(MODELS_DIR)
    df_preproc = _cargar_dataset_preprocesado_disco()

    if modelo is None:
        flash("Todavía no hay un modelo entrenado. Ve a la sección Entrenamiento primero.", "error")
        return redirect(url_for("prediccion"))

    productos = list(le_prod.classes_) if le_prod is not None else (
        sorted(df_preproc["producto"].unique()) if df_preproc is not None else []
    )
    provincias = list(le_prov.classes_) if le_prov is not None else (
        sorted(df_preproc["provincia"].unique()) if df_preproc is not None else []
    )

    registro = obtener_ultimo_registro(df_preproc, producto, provincia) if df_preproc is not None else None
    resultado = None

    if registro is not None:
        val_pt1 = registro.get("precio_t1")
        val_pt2 = registro.get("precio_t2")
        val_mes = int(registro.get("mes", 6)) if pd.notna(registro.get("mes")) else 6

        if pd.notna(val_pt1) and pd.notna(val_pt2):
            pred_label, probs, _ = predecir_registro(
                val_pt1, val_pt2, val_mes,
                producto, provincia,
                le_prod, le_prov, le_target, modelo, features,
                categoria_perecedero=int(registro.get("categoria_perecedero", 0)) if pd.notna(registro.get("categoria_perecedero")) else 0,
                canton_encoded=registro.get("canton_encoded", 0.0),
                mercado_encoded=registro.get("mercado_encoded", 0.0),
                presentacion_encoded=registro.get("presentacion_encoded", 0.0),
                tipo_mercado_encoded=registro.get("tipo_mercado_encoded", 0.0),
            )
            resultado = {"clase": pred_label, "probs": probs}
        else:
            flash("Este registro no cuenta con suficiente historia previa para clasificar.", "error")
    else:
        flash("No se encontraron registros para esa combinación de producto y provincia.", "error")

    return render_template(
        "prediccion.html",
        modelo_listo=True,
        productos=productos,
        provincias=provincias,
        resultado=resultado,
        registro=registro,
        seleccion={"producto": producto, "provincia": provincia},
        df_lote=None,
    )


@app.route("/prediccion/lote", methods=["POST"])
def prediccion_lote():
    modelo, le_target, le_prod, le_prov, features = cargar_modelo_y_artefactos(MODELS_DIR)
    df_preproc = _cargar_dataset_preprocesado_disco()

    if modelo is None or df_preproc is None:
        flash("Necesitas un modelo entrenado y un dataset preprocesado para clasificar en lote.", "error")
        return redirect(url_for("prediccion"))

    df_predicho = predecir_dataframe(df_preproc, modelo, le_target, features)
    CACHE["df_predicho"] = df_predicho
    flash(f"Se clasificaron {len(df_predicho)} registros.", "exito")
    return redirect(url_for("prediccion_resultados_lote"))


@app.route("/prediccion/lote/resultados")
def prediccion_resultados_lote():
    df_predicho = CACHE.get("df_predicho")
    modelo, le_target, le_prod, le_prov, features = cargar_modelo_y_artefactos(MODELS_DIR)
    productos = list(le_prod.classes_) if le_prod is not None else []
    provincias = list(le_prov.classes_) if le_prov is not None else []
    return render_template(
        "prediccion.html",
        modelo_listo=(modelo is not None),
        productos=productos,
        provincias=provincias,
        resultado=None,
        registro=None,
        seleccion=None,
        df_lote=df_predicho.head(200) if df_predicho is not None else None,
    )


@app.route("/prediccion/lote/descargar")
def descargar_lote():
    df_predicho = CACHE.get("df_predicho")
    if df_predicho is None:
        flash("Primero ejecuta la clasificación en lote.", "error")
        return redirect(url_for("prediccion"))
    buf = io.BytesIO()
    df_predicho.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.seek(0)
    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name="predicciones_sipa.csv",
    )


if __name__ == "__main__":
    # use_reloader=False es importante: el entrenamiento puede tardar varios
    # minutos, y PyTorch a veces modifica sus propios archivos de configuración
    # internos (torch/_dynamo/config.py, torch/_functorch/config.py) al usarse.
    # Con el reloader activo, Flask interpreta eso como "cambió el código" y
    # reinicia el servidor a mitad del entrenamiento, matando la petición en curso.
    # Si vas a editar el código y quieres que se recargue solo, cambia esto a
    # use_reloader=True mientras programas (pero no mientras entrenas modelos).
    app.run(debug=True, use_reloader=False)