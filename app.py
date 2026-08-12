"""
Interfaz web con Streamlit para el Clasificador de Precios SIPA.

Uso: streamlit run app.py
"""

import os
import tempfile
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import nbformat
from entrenamiento_utils import ejecutar_entrenamiento_y_evaluacion
from prediccion_utils import cargar_modelo_y_artefactos, predecir_registro, predecir_dataframe, obtener_ultimo_registro


st.set_page_config(
    page_title="Clasificador Precios SIPA",
    page_icon="S",
    layout="wide",
)
CSS_TEMA = """
<style>
    :root {
        --verde-campo: #2d6a4f;
        --verde-claro: #52b788;
        --verde-oscuro: #1b4332;
        --naranja-tierra: #e07a5f;
        --fondo-app: #0f1a14;
        --fondo-tarjeta: #16241c;
    }

    /* Fondo general de la app */
    .stApp {
        background: linear-gradient(180deg, #0f1a14 0%, #10201a 100%);
    }

    /* Fondo de la barra lateral */
    section[data-testid="stSidebar"] {
        background-color: #0d1712;
        border-right: 1px solid rgba(82, 183, 136, 0.15);
    }

    /* Header principal con degradado agrícola */
    .encabezado-sipa {
        background: linear-gradient(120deg, #1b4332 0%, #2d6a4f 55%, #52b788 100%);
        padding: 2rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }
    .encabezado-sipa h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
    }
    .encabezado-sipa p {
        color: #d8f3dc;
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
    }
    .encabezado-sipa .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: white;
        padding: 0.15rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-top: 0.6rem;
        letter-spacing: 0.03em;
    }

    /* Tarjetas de métricas */
    div[data-testid="stMetric"] {
        background: var(--fondo-tarjeta);
        border: 1px solid rgba(82, 183, 136, 0.25);
        border-left: 4px solid var(--verde-campo);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        opacity: 0.85;
    }

    /* Contenedores de dataframes y expanders con tarjeta oscura verde */
    div[data-testid="stDataFrame"], div[data-testid="stExpander"] {
        background: var(--fondo-tarjeta);
        border-radius: 10px;
        border: 1px solid rgba(82, 183, 136, 0.15);
    }

    /* Pestañas principales */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid rgba(82, 183, 136, 0.25);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        color: #b7e4c7;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(82, 183, 136, 0.12) !important;
        color: white !important;
    }

    /* Botones primarios */
    div.stButton > button[kind="primary"] {
        background-color: var(--verde-campo);
        border: none;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--verde-claro);
    }

    /* Barra de progreso */
    div[data-testid="stProgress"] > div > div {
        background-color: var(--verde-claro);
    }

    /* Divisores más sutiles */
    hr {
        border-color: rgba(82, 183, 136, 0.2) !important;
    }

    /* Encabezados de sección (st.header / st.subheader) */
    h1, h2, h3 {
        color: #eaf5ee;
    }
</style>
"""

st.markdown(CSS_TEMA, unsafe_allow_html=True)

@st.cache_resource
def load_extractor(notebook_fingerprint):
    notebook_path = os.path.join(os.path.dirname(__file__), "extractor.ipynb")
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    namespace = {"__name__": "__main__"}
    skip_patterns = [
        "procesar_boletin(PDF_PATH)",
        "df.to_csv",
        "PDF_PATH =",
        "CSV_PATH =",
        "df = pd.DataFrame(registros)",
        "df.head(",
    ]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            source = cell.source
            if any(p in source for p in skip_patterns):
                continue
            exec(source, namespace)

    return namespace["procesar_boletin"], namespace["validar_calidad"]


@st.cache_resource
def load_preprocesamiento(notebook_fingerprint):
    notebook_path = os.path.join(os.path.dirname(__file__), "preprocesamiento.ipynb")
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    namespace = {"__name__": "__main__"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            # Filtra las líneas que inician con % (mágicos) o ! (comandos de terminal)
            clean_source = "\n".join(
                line for line in cell.source.splitlines()
                if not line.strip().startswith(("%", "!"))
            )
            try:
                exec(clean_source, namespace)
            except Exception:
                pass

    return namespace["preprocesar_datos"], namespace["obtener_resumen"]


# Cargar notebooks
extractor_path = os.path.join(os.path.dirname(__file__), "extractor.ipynb")
with open(extractor_path, "rb") as f:
    extractor_fp = hashlib.sha256(f.read()).hexdigest()

preproc_path = os.path.join(os.path.dirname(__file__), "preprocesamiento.ipynb")
with open(preproc_path, "rb") as f:
    preproc_fp = hashlib.sha256(f.read()).hexdigest()

procesar_boletin, validar_calidad = load_extractor(extractor_fp)
preprocesar_datos, obtener_resumen = load_preprocesamiento(preproc_fp)


def cargar_dataset_preprocesado():
    """
    Obtiene el dataset preprocesado desde session_state o lo carga automáticamente desde el CSV guardado en disco.
    Si solo existe el dataset crudo, ejecuta el preprocesamiento automáticamente.
    """
    if "resultado_preprocesamiento" in st.session_state:
        return st.session_state["resultado_preprocesamiento"].get("dataset_final")

    csv_preproc = os.path.join(os.path.dirname(__file__), "data", "processed", "dataset_preprocesado_sipa.csv")
    if os.path.exists(csv_preproc):
        try:
            return pd.read_csv(csv_preproc)
        except Exception:
            pass

    csv_crudo = os.path.join(os.path.dirname(__file__), "data", "processed", "dataset_crudo_sipa.csv")
    if os.path.exists(csv_crudo):
        try:
            df_crudo = pd.read_csv(csv_crudo)
            res = preprocesar_datos(df_crudo)
            st.session_state["resultado_preprocesamiento"] = res
            os.makedirs(os.path.dirname(csv_preproc), exist_ok=True)
            res["dataset_final"].to_csv(csv_preproc, index=False, encoding="utf-8-sig")
            return res["dataset_final"]
        except Exception:
            pass

    return None


def main():
    ENCABEZADO_HTML = """
    <div class="encabezado-sipa">
        <h1>Clasificador de Precios Mayoristas SIPA</h1>
        <p>Predicción del comportamiento de precios agrícolas en Ecuador mediante
        aprendizaje automático supervisado — Random Forest, XGBoost, Decision Tree,
        Logistic Regression, KNN y SVM</p>
        <span class="badge">Fuente: Sistema de Información Pública Agropecuaria (SIPA)</span>
    </div>
    """
    st.markdown(ENCABEZADO_HTML, unsafe_allow_html=True)

    # Pestañas principales
    tab1, tab2, tab3 = st.tabs([
        "1. Extracción y Preprocesamiento",
        "2. Entrenamiento de Modelos",
        "3. Predicción de Precios",
    ])

    with st.sidebar:
        st.markdown("### Proyecto")
        st.caption("Grupo 4 · Inteligencia Artificial · ESPOL")
        st.markdown("---")
    
        st.markdown("### Estado del pipeline")
        estado_df = "df" in st.session_state
        estado_preproc = "resultado_preprocesamiento" in st.session_state
        estado_modelo = "res_entrenamiento" in st.session_state
    
        st.markdown(f"{'OK' if estado_df else '--'} Boletines extraídos")
        st.markdown(f"{'OK' if estado_preproc else '--'} Datos preprocesados")
        st.markdown(f"{'OK' if estado_modelo else '--'} Modelo entrenado")
    
        st.markdown("---")
        st.markdown("### Sobre el proyecto")
        st.caption(
            "Clasifica el comportamiento futuro de precios mayoristas agrícolas "
            "(Alza / Estable / Caída) a partir de boletines quincenales del SIPA."
        )

    # =========================================================================
    # PESTAÑA 1: EXTRACCIÓN Y PREPROCESAMIENTO
    # =========================================================================
    with tab1:
        st.header("1. Subir boletines PDF")
        
        # CAMBIO 1: accept_multiple_files=True habilitado
        uploaded_files = st.file_uploader("Seleccione los archivos PDF de boletines SIPA", type=["pdf"], accept_multiple_files=True, key="uploader_pdf")

        if uploaded_files: # Si la lista de archivos no está vacía
            st.divider()

            if st.button(f"Procesar {len(uploaded_files)} boletines en lote", type="primary", use_container_width=True, key="btn_procesar_pdf"):
                
                # Listas para acumular los datos de todos los PDFs
                todos_los_registros = []
                todos_los_encabezados_fallidos = []
                
                # Elementos visuales de progreso
                barra_progreso = st.progress(0)
                texto_progreso = st.empty()

                with st.spinner("Procesando boletines con OCR... Esto puede tardar varios minutos."):
                    # CAMBIO 2: Bucle para procesar cada PDF subido
                    for i, file in enumerate(uploaded_files):
                        texto_progreso.text(f"Extrayendo datos de: {file.name} ({i+1}/{len(uploaded_files)})")
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name

                        try:
                            # Llama a procesar_boletin por cada archivo
                            registros, encabezados = procesar_boletin(tmp_path)
                            
                            if registros:
                                todos_los_registros.extend(registros) # Acumula las filas
                            if encabezados:
                                todos_los_encabezados_fallidos.extend(encabezados) # Acumula los errores
                                
                        except Exception as e:
                            st.error(f"Error procesando {file.name}: {e}")
                        finally:
                            os.unlink(tmp_path)
                            
                        # Actualizar barra
                        barra_progreso.progress((i + 1) / len(uploaded_files))

                    texto_progreso.text("¡Extracción de todos los documentos finalizada! Consolidando dataset...")

                    # CAMBIO 3: Crear el DataFrame unificado y evaluarlo
                    if todos_los_registros:
                        df = pd.DataFrame(todos_los_registros)
                        reporte = validar_calidad(df, todos_los_encabezados_fallidos)

                        if "orden" in df.columns:
                            df = df.sort_values("orden").drop(columns=["orden"])

                        st.session_state["df"] = df
                        st.session_state["reporte"] = reporte
                        st.session_state["filename"] = f"Lote de {len(uploaded_files)} archivos"

                        # Autoguardado dataset crudo consolidado (concatenar con existente)
                        csv_crudo_path = os.path.join(os.path.dirname(__file__), "data", "processed", "dataset_crudo_sipa.csv")
                        os.makedirs(os.path.dirname(csv_crudo_path), exist_ok=True)

                        if os.path.exists(csv_crudo_path):
                            df_existente = pd.read_csv(csv_crudo_path, encoding="utf-8-sig")
                            df = pd.concat([df_existente, df], ignore_index=True)
                            n_antes = len(df)

                            cols_dedup = [c for c in ["producto_raw", "provincia", "quincena_id"] if c in df.columns]
                            if cols_dedup:
                                df = df.drop_duplicates(subset=cols_dedup, keep="last").reset_index(drop=True)

                            n_duplicados = n_antes - len(df)
                            if n_duplicados > 0:
                                st.info(f"Se eliminaron {n_duplicados} registros duplicados del dataset consolidado.")
                            st.success(f"Dataset consolidado: {len(df)} registros totales (datos anteriores conservados).")

                        df.to_csv(csv_crudo_path, index=False, encoding="utf-8-sig")
                    else:
                        st.error("No se extrajo ningún registro válido de los archivos subidos.")

        if "df" in st.session_state:
            df = st.session_state["df"]
            reporte = st.session_state["reporte"]

            st.divider()
            st.header("2. Resultados de Extracción")

            completitud = reporte["porcentaje_completitud"]
            if completitud >= 95:
                st.success(f"Boletín procesado exitosamente — Completitud: {completitud}%")
            elif completitud >= 80:
                st.warning(f"Boletín procesado con advertencias — Completitud: {completitud}%")
            else:
                st.error(f"Boletín con problemas significativos — Completitud: {completitud}%")

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Registros", len(df))
            col2.metric("Completos", reporte["registros_completos"])
            col3.metric("Parciales", reporte["registros_parciales"])
            col4.metric("Productos", df["producto_raw"].nunique())
            col5.metric("Quincenas", reporte["quincenas"])
            col6.metric("Calidad", f"{completitud}%")

            hay_problemas = reporte["registros_con_problema"] > 0
            hay_parciales = reporte["registros_parciales"] > 0

            # Registros Parciales
            if hay_parciales:
                st.divider()
                st.header("3. Registros Parciales (Productos Nuevos)")
                st.info(f"Se encontraron {reporte['registros_parciales']} productos nuevos sin precio anterior.")

                df_parciales = df[df["estado_precio"] == "parcial"][["producto_raw", "precio_anterior", "precio_actual", "provincia", "quincena_id"]]
                st.dataframe(df_parciales, use_container_width=True, height=200, hide_index=True)

                accion_parciales = st.radio(
                    "¿Qué desea hacer con los productos nuevos?",
                    ["Conservar todos (recomendado para entrenar modelos)", "Excluir productos nuevos"],
                    horizontal=True,
                    key="accion_parciales"
                )

                if accion_parciales.startswith("Excluir"):
                    df = df[df["estado_precio"] != "parcial"]
                    st.info(f"Se excluyeron {reporte['registros_parciales']} productos nuevos")

            # Registros con Problemas
            if hay_problemas:
                st.divider()
                st.header("4. Registros con Problemas")

                n_invalidos = len(df[df["estado_precio"] == "invalido"])
                if n_invalidos > 0:
                    df = df[df["estado_precio"] != "invalido"]
                    st.warning(f"Se excluyeron {n_invalidos} registros sin precios.")

                st.subheader("Detalle de problemas")
                df_problemas = pd.DataFrame(reporte["problemas"])
                if not df_problemas.empty:
                    if "precio_anterior" not in df_problemas.columns:
                        df_problemas["precio_anterior"] = None
                    if "precio_actual" not in df_problemas.columns:
                        df_problemas["precio_actual"] = None
                    cols_inv = ["tipo", "producto", "precio_anterior", "precio_actual", "provincia", "quincena", "detalle"]
                    cols_inv = [c for c in cols_inv if c in df_problemas.columns]
                    st.dataframe(df_problemas[cols_inv], use_container_width=True, height=250, hide_index=True)

            # Confirmación de revisión
            if hay_problemas or hay_parciales:
                st.divider()
                st.header("5. Confirmar Revisión")
                st.info("Revise los registros listados arriba.")
                revision_confirmada = st.checkbox("Confirmo que revisé los registros", key="chk_revision")

                if not revision_confirmada:
                    st.stop()

            # Vista previa de datos
            st.divider()
            st.header("Vista previa de datos extraídos")
            st.dataframe(df, use_container_width=True, height=350, hide_index=True)

            # Guardar dataset crudo
            col_save, col_download = st.columns(2)
            with col_save:
                csv_path = os.path.join(os.path.dirname(__file__), "data", "processed", "dataset_crudo_sipa.csv")
                if st.button("Guardar CSV crudo", type="secondary", key="btn_save_crudo"):
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    dfGuardar = df.copy()
                    if os.path.exists(csv_path):
                        dfExistente = pd.read_csv(csv_path, encoding="utf-8-sig")
                        dfGuardar = pd.concat([dfExistente, dfGuardar], ignore_index=True)
                        cols_dedup = [c for c in ["producto_raw", "provincia", "quincena_id"] if c in dfGuardar.columns]
                        if cols_dedup:
                            dfGuardar = dfGuardar.drop_duplicates(subset=cols_dedup, keep="last").reset_index(drop=True)
                    dfGuardar.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    st.success(f"Guardado: {csv_path} ({len(dfGuardar)} registros totales)")

            with col_download:
                csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="Descargar CSV crudo",
                    data=csv_bytes,
                    file_name="dataset_crudo_sipa.csv",
                    mime="text/csv",
                    type="primary",
                    key="btn_dl_crudo"
                )

            # Preprocesamiento
            st.divider()
            st.header("Preprocesamiento de datos")
            st.markdown("Transformar los datos extraídos en un dataset listo para entrenar modelos de IA (generación de rezagos, promedios móviles y etiquetas de comportamiento).")

            if st.button("Ejecutar preprocesamiento", type="primary", use_container_width=True, key="btn_ejecutar_preproc"):
                with st.spinner("Preprocesando datos..."):
                    try:
                        col_requeridas = ["producto_raw", "precio_actual", "precio_anterior", "provincia", "estado_precio"]
                        faltantes = [c for c in col_requeridas if c not in df.columns]
                        if faltantes:
                            st.error(f"Faltan columnas requeridas: {faltantes}")
                        else:
                            resultado = preprocesar_datos(df)
                            st.session_state["resultado_preprocesamiento"] = resultado

                            # Autoguardado del dataset preprocesado en disco (concatenar con existente)
                            csv_preproc_path = os.path.join(os.path.dirname(__file__), "data", "processed", "dataset_preprocesado_sipa.csv")
                            os.makedirs(os.path.dirname(csv_preproc_path), exist_ok=True)

                            df_nuevo_preproc = resultado["dataset_final"]
                            if os.path.exists(csv_preproc_path):
                                df_existente_preproc = pd.read_csv(csv_preproc_path, encoding="utf-8-sig")
                                df_nuevo_preproc = pd.concat([df_existente_preproc, df_nuevo_preproc], ignore_index=True)
                                n_antes = len(df_nuevo_preproc)

                                cols_dedup = [c for c in ["producto", "provincia", "periodo"] if c in df_nuevo_preproc.columns]
                                if cols_dedup:
                                    df_nuevo_preproc = df_nuevo_preproc.drop_duplicates(subset=cols_dedup, keep="last").reset_index(drop=True)

                                n_duplicados = n_antes - len(df_nuevo_preproc)
                                if n_duplicados > 0:
                                    st.info(f"Se eliminaron {n_duplicados} registros duplicados del dataset preprocesado.")
                                st.success(f"Dataset preprocesado consolidado: {len(df_nuevo_preproc)} registros totales.")

                            df_nuevo_preproc.to_csv(csv_preproc_path, index=False, encoding="utf-8-sig")

                            st.success("Preprocesamiento completado exitosamente y guardado en disco")
                    except Exception as e:
                        st.error(f"Error en preprocesamiento: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            if "resultado_preprocesamiento" in st.session_state:
                resultado = st.session_state["resultado_preprocesamiento"]
                stats = resultado["estadisticas"]

                st.subheader("Resultado del Preprocesamiento")
                col_st1, col_st2, col_st3 = st.columns(3)
                col_st1.metric("Entrada (registros válidos)", stats["registros_completos"])
                col_st2.metric("Salida (dataset final)", stats["registros_modelo"])
                col_st3.metric("Productos descartados (>30% faltante)", stats["productos_descartados"])

                if stats["productos_descartados"] > 0:
                    st.warning(f"{stats['productos_descartados']} productos fueron eliminados por tener más del 30% de quincenas sin datos.")
                    df_desc = pd.DataFrame(resultado["productos_descartados"])
                    st.dataframe(df_desc, use_container_width=True, hide_index=True)

                df_modelo = resultado["dataset_final"]
                st.subheader("Vista previa del dataset final preprocesado")
                st.dataframe(df_modelo, use_container_width=True, height=350)

                col_save2, col_download2 = st.columns(2)
                with col_save2:
                    csv_path2 = os.path.join(os.path.dirname(__file__), "data", "processed", "dataset_preprocesado_sipa.csv")
                    if st.button("Guardar CSV preprocesado", type="secondary", key="btn_save_preproc"):
                        os.makedirs(os.path.dirname(csv_path2), exist_ok=True)
                        dfGuardar2 = df_modelo.copy()
                        if os.path.exists(csv_path2):
                            dfExistente2 = pd.read_csv(csv_path2, encoding="utf-8-sig")
                            dfGuardar2 = pd.concat([dfExistente2, dfGuardar2], ignore_index=True)
                            cols_dedup2 = [c for c in ["producto", "provincia", "periodo"] if c in dfGuardar2.columns]
                            if cols_dedup2:
                                dfGuardar2 = dfGuardar2.drop_duplicates(subset=cols_dedup2, keep="last").reset_index(drop=True)
                        dfGuardar2.to_csv(csv_path2, index=False, encoding="utf-8-sig")
                        st.success(f"Guardado: {csv_path2} ({len(dfGuardar2)} registros totales)")

                with col_download2:
                    csv_bytes2 = df_modelo.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button(
                        label="Descargar CSV preprocesado",
                        data=csv_bytes2,
                        file_name="dataset_preprocesado_sipa.csv",
                        mime="text/csv",
                        type="primary",
                        key="btn_dl_preproc"
                    )

    # =========================================================================
    # PESTAÑA 2: ENTRENAMIENTO DE MODELOS
    # =========================================================================
    with tab2:
        st.header("Entrenamiento y Evaluación de Modelos")
        st.markdown("Entrenar y evaluar modelos de clasificación (**Random Forest**, **XGBoost**, **Decision Tree**, **Logistic Regression**, **KNN** y **SVM**).")

        df_preproc_disponible = cargar_dataset_preprocesado()

        if df_preproc_disponible is None:
            st.info("Para entrenar los modelos, primero suba un boletín y ejecute el preprocesamiento en la pestaña **'1. Extracción y Preprocesamiento'**.")
        else:
            if st.button("Ejecutar entrenamiento y evaluación", type="primary", use_container_width=True, key="btn_train_models"):
                with st.spinner("Entrenando modelos con TimeSeriesSplit... Esto tomará unos segundos."):
                    try:
                        le_prod = st.session_state.get("resultado_preprocesamiento", {}).get("le_producto")
                        le_prov = st.session_state.get("resultado_preprocesamiento", {}).get("le_provincia")

                        res_entrenamiento = ejecutar_entrenamiento_y_evaluacion(df_preproc_disponible, le_prod, le_prov)
                        st.session_state["res_entrenamiento"] = res_entrenamiento
                        st.success("Entrenamiento finalizado exitosamente")
                    except Exception as e:
                        st.error(f"Error durante el entrenamiento: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            if "res_entrenamiento" in st.session_state:
                res = st.session_state["res_entrenamiento"]

                st.divider()
                st.subheader("1. Limpieza de filas sin rezago (NaN por historia insuficiente)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Filas iniciales", res["filas_antes"])
                c2.metric("Filas eliminadas (sin rezago)", res["filas_eliminadas"])
                c3.metric("Filas disponibles para entrenamiento", res["filas_despues"])

                st.divider()
                st.subheader("2. Tabla Comparativa de Modelos")
                st.markdown("**Criterios de selección:** F1-Score (Macro) ≥ 0.75 y Accuracy ≥ 0.80")
                st.dataframe(res["tabla_comparativa"], use_container_width=True, hide_index=True)

                st.success(f"**MODELO SELECCIONADO:** {res['mejor_nombre']} | F1-Score: {res['mejor_metricas']['F1-Score (Macro)']} | Accuracy: {res['mejor_metricas']['Accuracy']}")

    # =========================================================================
    # PESTAÑA 3: PREDICCIÓN DE PRECIOS
    # =========================================================================
    with tab3:
        st.header("Clasificación y Predicción de Precios")
        st.markdown("Realizar clasificaciones del comportamiento del precio (*Alza*, *Estable*, *Caída*) utilizando el modelo entrenado guardado.")

        modelo, le_target, le_prod, le_prov, features = cargar_modelo_y_artefactos()
        df_preproc = cargar_dataset_preprocesado()

        if modelo is None:
            st.info("Para realizar predicciones, ejecute primero el entrenamiento del modelo en la pestaña **'2. Entrenamiento de Modelos'**.")
        else:
            st.success("Modelo entrenado cargado exitosamente desde disco.")

            tab_sub_indiv, tab_sub_lote = st.tabs(["Predicción por Producto y Provincia", "Clasificación en Lote (Dataset)"])

            # --- SUB-TAB 1: PREDICCIÓN POR PRODUCTO Y PROVINCIA ---
            with tab_sub_indiv:
                st.subheader("Seleccionar Producto y Provincia")

                prods_lista = list(le_prod.classes_) if le_prod else (list(df_preproc["producto"].unique()) if (df_preproc is not None and "producto" in df_preproc.columns) else ["General"])
                provs_lista = list(le_prov.classes_) if le_prov else (list(df_preproc["provincia"].unique()) if (df_preproc is not None and "provincia" in df_preproc.columns) else ["General"])

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    sel_producto = st.selectbox("Seleccione el Producto", prods_lista, key="pred_prod_tab3")
                with col_p2:
                    sel_provincia = st.selectbox("Seleccione la Provincia", provs_lista, key="pred_prov_tab3")

                rec = obtener_ultimo_registro(df_preproc, sel_producto, sel_provincia) if df_preproc is not None else None

                if rec is not None:
                    st.divider()
                    st.subheader("Datos Históricos Detectados en el Dataset")

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Período / Quincena", str(rec.get("periodo", "N/A")))
                    col_m2.metric("Precio Quincena Anterior (t1)", f"${rec.get('precio_t1', 0):.2f}" if pd.notna(rec.get('precio_t1')) else "N/A")
                    col_m3.metric("Precio Hace 2 Quincenas (t2)", f"${rec.get('precio_t2', 0):.2f}" if pd.notna(rec.get('precio_t2')) else "N/A")
                    col_m4.metric("Precio Registrado en Boletín", f"${rec.get('precio_actual', 0):.2f}" if pd.notna(rec.get('precio_actual')) else "N/A")

                    val_pt1 = rec.get("precio_t1")
                    val_pt2 = rec.get("precio_t2")
                    val_mes = int(rec.get("mes", 6)) if pd.notna(rec.get("mes")) else 6
                    comp_real = rec.get("comportamiento", None)

                    if pd.notna(val_pt1) and pd.notna(val_pt2):
                        if st.button("Clasificar / Predecir Comportamiento", type="primary", use_container_width=True, key="btn_predecir_auto_tab3"):
                            pred_label, probs, inputs_derived = predecir_registro(
                                val_pt1, val_pt2, val_mes,
                                sel_producto, sel_provincia,
                                le_prod, le_prov, le_target, modelo, features,
                                categoria_perecedero=int(rec.get("categoria_perecedero", 0)) if pd.notna(rec.get("categoria_perecedero")) else 0
                            )

                            st.divider()
                            st.markdown("### Resultado de la Clasificación")

                            col_r1, col_r2 = st.columns(2)
                            with col_r1:
                                if pred_label == "Alza":
                                    st.markdown('''
                                    <div style="background:#5c1a1a;border-left:5px solid #e63946;
                                                padding:1rem 1.2rem;border-radius:10px;">
                                        <b style="color:#ffb3ba;font-size:1.1rem;"> ALZA</b><br>
                                        <span style="color:#f1c0c0;">Pronóstico de incremento de precio &gt; +3%</span>
                                    </div>''', unsafe_allow_html=True)
                                elif pred_label == "Caída":
                                    st.markdown('''
                                    <div style="background:#1b4332;border-left:5px solid #52b788;
                                                padding:1rem 1.2rem;border-radius:10px;">
                                        <b style="color:#b7e4c7;font-size:1.1rem;"> CAÍDA</b><br>
                                        <span style="color:#d8f3dc;">Pronóstico de reducción de precio &lt; -3%</span>
                                    </div>''', unsafe_allow_html=True)
                                else:
                                    st.markdown('''
                                    <div style="background:#4a3b1f;border-left:5px solid #e9c46a;
                                                padding:1rem 1.2rem;border-radius:10px;">
                                        <b style="color:#f4dfa8;font-size:1.1rem;"> ESTABLE</b><br>
                                        <span style="color:#f0e6c8;">El precio se mantendrá en el rango de ±3%</span>
                                    </div>''', unsafe_allow_html=True)
                            
                            with col_r2:
                                if comp_real:
                                    st.info(f"**Comportamiento Real Registrado:** {comp_real}")

                            if probs:
                                st.markdown("**Confianza de la Clasificación:**")
                                cols_prob = st.columns(len(probs))
                                for idx, (cls_name, prob_val) in enumerate(probs.items()):
                                    cols_prob[idx].metric(f"Probabilidad {cls_name}", f"{prob_val}%")
                    else:
                        st.warning("Este registro no cuenta con suficiente historia previa (NaN) para clasificar.")
                else:
                    if df_preproc is None:
                        st.info("Para detectar precios automáticamente, suba un boletín y ejecute el preprocesamiento en la Pestaña 1 una primera vez.")
                    else:
                        st.info("No se encontraron registros en el dataset para esta combinación específica de producto y provincia.")

            # --- SUB-TAB 2: CLASIFICACIÓN EN LOTE ---
            with tab_sub_lote:
                st.subheader("Clasificar todo el dataset preprocesado")
                if df_preproc is not None:
                    if st.button("Ejecutar clasificación en lote", type="secondary", use_container_width=True, key="btn_lote_tab3"):
                        df_predicho = predecir_dataframe(df_preproc, modelo, le_target, features)
                        st.session_state["df_predicho"] = df_predicho
                        st.success(f"Clasificados {len(df_predicho)} registros")

                    if "df_predicho" in st.session_state:
                        df_res = st.session_state["df_predicho"]
                        cols_mostrar = ["producto", "provincia", "periodo", "precio_t1", "precio_actual", "comportamiento", "prediccion"]
                        cols_mostrar = [c for c in cols_mostrar if c in df_res.columns]

                        st.dataframe(df_res[cols_mostrar], use_container_width=True, height=350)

                        csv_lote = df_res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                        st.download_button(
                            label="Descargar Predicciones (CSV)",
                            data=csv_lote,
                            file_name="predicciones_sipa.csv",
                            mime="text/csv",
                            type="primary",
                            key="btn_dl_lote_tab3"
                        )
                else:
                    st.info("Debe ejecutar el preprocesamiento en la Pestaña 1 para clasificar el dataset completo.")


if __name__ == "__main__":
    main()
