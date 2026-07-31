"""
Interfaz web con Streamlit para el Clasificador de Precios SIPA.

Uso: streamlit run app.py
"""

import os
import tempfile
import hashlib
import pandas as pd
import streamlit as st
import nbformat


st.set_page_config(
    page_title="Clasificador Precios SIPA",
    page_icon="",
    layout="wide",
)


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
            exec(cell.source, namespace)

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


def main():
    st.title("Clasificador de Precios Mayoristas SIPA")
    st.markdown("**Fase 1:** Extraccion y preprocesamiento de datos de boletines del SIPA")

    st.divider()

    # =====================================================
    # 1. SUBIR BOLETIN
    # =====================================================
    st.header("1. Subir boletin")
    uploaded_file = st.file_uploader("Seleccione un archivo PDF", type=["pdf"])

    if uploaded_file is not None:
        st.divider()

        if st.button("Procesar", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Procesando boletin con OCR... Esto puede tardar varios minutos."):
                try:
                    registros, encabezados_fallidos = procesar_boletin(tmp_path)

                    if registros:
                        df = pd.DataFrame(registros)
                        reporte = validar_calidad(df, encabezados_fallidos)

                        if "orden" in df.columns:
                            df = df.sort_values("orden").drop(columns=["orden"])

                        st.session_state["df"] = df
                        st.session_state["reporte"] = reporte
                        st.session_state["filename"] = uploaded_file.name

                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(tmp_path)

    # =====================================================
    # 2. RESULTADOS DE LA EXTRACCION
    # =====================================================
    if "df" in st.session_state:
        df = st.session_state["df"]
        reporte = st.session_state["reporte"]

        st.divider()
        st.header("2. Resultados")

        completitud = reporte["porcentaje_completitud"]
        if completitud >= 95:
            st.success(f"Boletin procesado exitosamente - Completitud: {completitud}%")
        elif completitud >= 80:
            st.warning(f"Boletin procesado con advertencias - Completitud: {completitud}%")
        else:
            st.error(f"Boletin con problemas significativos - Completitud: {completitud}%")

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
                "Que desea hacer con los productos nuevos?",
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
            n_parciales_en_df = len(df[df["estado_precio"] == "parcial"])

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

        # Confirmacion de revision
        if hay_problemas or hay_parciales:
            st.divider()
            st.header("5. Confirmar Revision")
            st.info("Revise los registros listados arriba.")
            revision_confirmada = st.checkbox("Confirmo que revise los registros")

            if not revision_confirmada:
                st.stop()

        # Vista previa de datos
        st.divider()
        num_vista = "6" if (hay_problemas or hay_parciales) else "3"
        st.header(f"{num_vista}. Vista previa de datos")
        st.dataframe(df, use_container_width=True, height=400, hide_index=True)

        # Guardar dataset crudo
        st.divider()
        num_guardar = "7" if (hay_problemas or hay_parciales) else "4"
        st.header(f"{num_guardar}. Guardar dataset crudo")

        col_save, col_download, col_report = st.columns(3)

        with col_save:
            csv_path = os.path.join(os.path.dirname(__file__), "data", "processed", "dataset_crudo_sipa.csv")
            if st.button("Guardar CSV", type="secondary"):
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                st.success(f"Guardado: {csv_path}")

        with col_download:
            csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="Descargar CSV",
                data=csv_bytes,
                file_name="dataset_crudo_sipa.csv",
                mime="text/csv",
                type="primary",
            )

        with col_report:
            if hay_problemas:
                reporte_data = pd.DataFrame(reporte["problemas"])
                reporte_csv = reporte_data.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="Descargar Reporte",
                    data=reporte_csv,
                    file_name="reporte_calidad_sipa.csv",
                    mime="text/csv",
                )

        # =====================================================
        # 8. PREPROCESAMIENTO
        # =====================================================
        st.divider()
        num_preproc = "8" if (hay_problemas or hay_parciales) else "5"
        st.header(f"{num_preproc}. Preprocesamiento")
        st.markdown("Transformar los datos extraidos en un dataset listo para entrenar modelos de IA.")

        if st.button("Ejecutar preprocesamiento", type="primary", use_container_width=True):
            with st.spinner("Preprocesando datos..."):
                try:
                    col_requeridas = ["producto_raw", "precio_actual", "precio_anterior", "provincia", "estado_precio"]
                    faltantes = [c for c in col_requeridas if c not in df.columns]
                    if faltantes:
                        st.error(f"Faltan columnas requeridas: {faltantes}")
                    else:
                        resultado = preprocesar_datos(df)
                        st.session_state["resultado_preprocesamiento"] = resultado
                        st.success("Preprocesamiento completado exitosamente")
                except Exception as e:
                    st.error(f"Error en preprocesamiento: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Mostrar resultados del preprocesamiento
        if "resultado_preprocesamiento" in st.session_state:
            resultado = st.session_state["resultado_preprocesamiento"]
            stats = resultado["estadisticas"]

            st.subheader("Resultado del Preprocesamiento")

            col1, col2, col3 = st.columns(3)
            col1.metric("Entrada (registros validos)", stats["registros_completos"])
            col2.metric("Salida (dataset final)", stats["registros_modelo"])
            col3.metric("Productos descartados", stats["productos_descartados"])

            if stats["productos_descartados"] > 0:
                st.warning(f"{stats['productos_descartados']} productos fueron eliminados por tener mas del 30% de quincenas sin datos.")
                df_desc = pd.DataFrame(resultado["productos_descartados"])
                st.dataframe(df_desc, use_container_width=True, hide_index=True)

            df_modelo = resultado["dataset_final"]
            st.subheader("Vista previa del dataset final")
            st.dataframe(df_modelo, use_container_width=True, height=400)

            st.subheader("Guardar Dataset Preprocesado")
            col_save2, col_download2 = st.columns(2)

            with col_save2:
                csv_path2 = os.path.join(os.path.dirname(__file__), "data", "processed", "dataset_preprocesado_sipa.csv")
                if st.button("Guardar CSV preprocesado", type="secondary"):
                    os.makedirs(os.path.dirname(csv_path2), exist_ok=True)
                    df_modelo.to_csv(csv_path2, index=False, encoding="utf-8-sig")
                    st.success(f"Guardado: {csv_path2}")

            with col_download2:
                csv_bytes2 = df_modelo.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="Descargar CSV preprocesado",
                    data=csv_bytes2,
                    file_name="dataset_preprocesado_sipa.csv",
                    mime="text/csv",
                    type="primary",
                )


if __name__ == "__main__":
    main()
