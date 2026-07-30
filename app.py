"""
Interfaz web con Streamlit para el Clasificador de Precios SIPA - Fase 1.
CU-01: Preprocesar boletin de precios.

Uso: streamlit run app.py
"""

import os
import tempfile
import pandas as pd
import streamlit as st
import nbformat

st.set_page_config(
    page_title="Clasificador Precios SIPA",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def load_functions():
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


procesar_boletin, validar_calidad = load_functions()


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
                        variacion = (df["precio_actual"] - df["precio_anterior"]) / df["precio_anterior"] * 100
                        variacion = variacion.replace([float("inf"), float("-inf")], 0)
                        df["variacion_pct"] = variacion.fillna(0).round(0).astype(int)

                        reporte = validar_calidad(df, encabezados_fallidos)

                        if "orden" in df.columns:
                            df = df.sort_values("orden").drop(columns=["orden"])

                        st.session_state["df"] = df
                        st.session_state["reporte"] = reporte
                        st.session_state["filename"] = uploaded_file.filename

                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(tmp_path)

    if "df" in st.session_state:
        df = st.session_state["df"]
        reporte = st.session_state["reporte"]

        st.divider()
        st.header("2. Resultados")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Registros", len(df))
        col2.metric("Productos", df["producto_raw"].nunique())
        col3.metric("Provincias", df["provincia"].nunique())
        col4.metric("Quincenas", reporte["quincenas"])
        col5.metric("Completitud", f"{reporte['porcentaje_completitud']}%")

        st.divider()
        st.header("3. Vista previa de datos")
        st.dataframe(df, use_container_width=True, height=400)

        st.divider()
        st.header("4. Guardar dataset")

        col_save, col_download = st.columns(2)

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


if __name__ == "__main__":
    main()
