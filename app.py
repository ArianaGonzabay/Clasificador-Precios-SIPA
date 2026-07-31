"""
Interfaz web con Streamlit para el Clasificador de Precios SIPA.

Uso: streamlit run app.py
"""

import os
import tempfile
import sys
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
def load_functions(notebook_fingerprint):
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


notebook_path = os.path.join(os.path.dirname(__file__), "extractor.ipynb")
with open(notebook_path, "rb") as notebook_file:
    notebook_fingerprint = hashlib.sha256(notebook_file.read()).hexdigest()

procesar_boletin, validar_calidad = load_functions(notebook_fingerprint)


def main():
    st.title("Clasificador de Precios Mayoristas SIPA")
    st.markdown("**Fase 1:** Extraccion de datos de boletines del SIPA")

    st.divider()

    col_upload = st.columns(1)[0]

    with col_upload:
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

    if "df" in st.session_state:
        df = st.session_state["df"]
        reporte = st.session_state["reporte"]

        st.divider()
        st.header("2. Resultados")

        # Indicador de estado
        completitud = reporte["porcentaje_completitud"]
        if completitud >= 95:
            st.success(f"Boletín procesado exitosamente - Completitud: {completitud}%")
        elif completitud >= 80:
            st.warning(f"Boletín procesado con advertencias - Completitud: {completitud}%")
        else:
            st.error(f"Boletín con problemas significativos - Completitud: {completitud}%")

        # Métricas principales
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Registros", len(df))
        col2.metric("Completos", reporte["registros_completos"])
        col3.metric("Parciales", reporte["registros_parciales"])
        col4.metric("Productos", df["producto_raw"].nunique())
        col5.metric("Quincenas", reporte["quincenas"])
        col6.metric("Calidad", f"{completitud}%")

        # Resumen de problemas
        hay_problemas = reporte["registros_con_problema"] > 0
        hay_parciales = reporte["registros_parciales"] > 0
        revision_confirmada = False

        # Sección de Registros Parciales (productos nuevos sin precio anterior)
        if hay_parciales:
            st.divider()
            st.header("3. Registros Parciales (Productos Nuevos)")
            st.info(f"Se encontraron {reporte['registros_parciales']} productos nuevos sin precio anterior. Solo tienen precio actual.")

            df_parciales = df[df["estado_precio"] == "parcial"][["producto_raw", "precio_anterior", "precio_actual", "provincia", "quincena_id"]]
            st.dataframe(df_parciales, use_container_width=True, height=200)

            accion_parciales = st.radio(
                "¿Qué desea hacer con los productos nuevos?",
                ["Conservar todos (recomendado para entrenar modelos)", "Excluir productos nuevos"],
                horizontal=True,
                key="accion_parciales"
            )

            if accion_parciales.startswith("Excluir"):
                df = df[df["estado_precio"] != "parcial"]
                st.info(f"Se excluyeron {reporte['registros_parciales']} productos nuevos")

        # Sección de Problemas (registros inválidos sin ningún precio)
        if hay_problemas:
            st.divider()
            st.header("4. Registros Inválidos (Sin Precios)")
            st.warning(f"Se encontraron {reporte['registros_con_problema']} registros sin ningún precio detectado. Estos son errores del OCR.")

            # Tabla resumen por tipo
            st.subheader("Resumen por Tipo de Problema")
            resumen_data = []
            for tipo, info in reporte["resumen_tipos"].items():
                resumen_data.append({
                    "Tipo": tipo.replace("_", " ").title(),
                    "Cantidad": info["cantidad"],
                    "Ejemplo": info["ejemplos"][0]["detalle"] if info["ejemplos"] else ""
                })
            st.dataframe(pd.DataFrame(resumen_data), use_container_width=True, hide_index=True)

            # Detalle de problemas
            st.subheader("Detalle de Registros Inválidos")
            df_problemas = pd.DataFrame(reporte["problemas"])
            if "precio_anterior" not in df_problemas.columns:
                df_problemas["precio_anterior"] = None
            if "precio_actual" not in df_problemas.columns:
                df_problemas["precio_actual"] = None
            cols_inv = ["tipo", "producto", "precio_anterior", "precio_actual", "provincia", "quincena", "detalle"]
            cols_inv = [c for c in cols_inv if c in df_problemas.columns]
            st.dataframe(df_problemas[cols_inv], use_container_width=True, height=250)

            accion_problemas = st.radio(
                "¿Qué desea hacer con los registros inválidos?",
                ["Mantener en el dataset (sin precio)", "Excluir del dataset"],
                horizontal=True,
                key="accion_problemas"
            )

            if accion_problemas.startswith("Excluir"):
                df = df[df["estado_precio"] != "invalido"]
                st.info(f"Se excluyeron {reporte['registros_invalidos']} registros inválidos")

        # Confirmación de revisión
        if hay_problemas or hay_parciales:
            st.divider()
            num_revision = "5" if (hay_problemas or hay_parciales) else "3"
            st.header(f"{num_revision}. Confirmar Revisión")
            st.warning("Debe revisar los registros identificados antes de continuar.")
            revision_confirmada = st.checkbox("Confirmo que revisé los registros listados arriba")

            if not revision_confirmada:
                st.info("Por favor revise los registros y marque la casilla para continuar.")
                st.stop()

        # Vista previa de datos
        st.divider()
        num_vista = "6" if (hay_problemas or hay_parciales) else "3"
        st.header(f"{num_vista}. Vista previa de datos")
        st.dataframe(df, use_container_width=True, height=400)

        # Guardar dataset
        st.divider()
        num_guardar = "7" if (hay_problemas or hay_parciales) else "4"
        st.header(f"{num_guardar}. Guardar dataset")

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


if __name__ == "__main__":
    main()
