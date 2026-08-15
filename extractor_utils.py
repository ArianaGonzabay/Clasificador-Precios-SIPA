"""
Extracción de boletines SIPA en PDF vía OCR (RapidOCR + pdfplumber).

Este módulo reconstruye, tal cual, las funciones que ya tenían en extractor.ipynb:
parse_header, ocr_region, parsear_tabla_precios, dividir_en_provincias,
procesar_boletin y validar_calidad.

Se importa de forma perezosa desde app.py (solo si el usuario sube un PDF),
porque instanciar RapidOCR() es costoso y no debe pagarse en cada arranque
del servidor si el flujo principal usa Excel/CSV.
"""

import os
import re
import numpy as np
import pdfplumber
from rapidocr_onnxruntime import RapidOCR
import pandas as pd

OCR_LANGUAGE = "es"
OCR_RESOLUTION = 400
OCR_ENGINE = RapidOCR()

MESES = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6,
    "JULIO": 7, "AGOSTO": 8, "SEPTIEMBRE": 9, "OCTUBRE": 10,
    "NOVIEMBRE": 11, "DICIEMBRE": 12
}

PROVINCIAS = ["AZUAY", "GUAYAS", "PICHINCHA"]

BBOX_PERECEDEROS_FULL = (28, 142, 814, 538)
BBOX_NO_PERECEDEROS_FULL = (27, 42, 814, 296)


def parse_header(texto_pagina):
    """Extrae metadata del boletin: numero, mes, ano, quincena."""
    m = re.search(
        r"N\u00ba\s*(\d+)\s+([A-Z\u00c1\u00c9\u00cd\u00d3\u00da]+)\s*-\s*(\d{4})\s+(PRIMERA|SEGUNDA)\s+QUINCENA",
        texto_pagina
    )
    if not m:
        return None
    num, mes, anio, quincena = m.groups()
    if mes not in MESES:
        return None
    return {
        "boletin_num": int(num),
        "mes": MESES[mes],
        "a\u00f1o": int(anio),
        "quincena": 1 if quincena == "PRIMERA" else 2,
        "quincena_id": f"{anio}-{MESES[mes]:02d}-Q{1 if quincena == 'PRIMERA' else 2}",
    }


def ocr_region(pagina_pdfplumber, bbox):
    """Renderiza una region de la pagina y aplica OCR.
    Fusiona bloques cercanos en Y para formar lineas completas."""
    recorte = pagina_pdfplumber.within_bbox(bbox)
    imagen = np.array(recorte.to_image(resolution=OCR_RESOLUTION).original)
    resultados, _ = OCR_ENGINE(imagen)

    if not resultados:
        return ""

    bloques = []
    for det in resultados:
        if not det or len(det) < 2:
            continue
        bbox_det, texto = det[0], str(det[1]).strip()
        if not texto:
            continue
        y_center = (bbox_det[0][1] + bbox_det[2][1]) / 2
        x_left = bbox_det[0][0]
        bloques.append((y_center, x_left, texto))

    bloques.sort(key=lambda b: (b[0], b[1]))

    UMBRAL_Y = 12
    lineas_fusionadas = []
    linea_actual = []
    y_ref = bloques[0][0]

    for y, x, texto in bloques:
        if abs(y - y_ref) > UMBRAL_Y:
            linea_actual.sort(key=lambda b: b[1])
            lineas_fusionadas.append(" ".join(b[2] for b in linea_actual))
            linea_actual = []
            y_ref = y
        linea_actual.append((y, x, texto))

    if linea_actual:
        linea_actual.sort(key=lambda b: b[1])
        lineas_fusionadas.append(" ".join(b[2] for b in linea_actual))

    return "\n".join(lineas_fusionadas)


def parsear_tabla_precios(texto_ocr):
    """Convierte texto OCR de una tabla de precios en registros, respetando el orden de lectura."""
    registros = []
    patron_precio = re.compile(r"^[\d]+(?:[.,]\d{1,2})?$")
    unidades_cantidad = {
        "lb", "l", "lt", "kg", "g", "gr", "ml", "cc",
        "c", "u", "und", "un", "funda", "malla", "caja",
        "bolsa", "paquete", "atajo", "ramo", "paca", "saco",
        "manojo", "botella", "frasco", "litro", "litros", "docena",
        "bandeja", "tarro", "bulto", "envase", "galon", "gal",
    }
    palabras_descartar = {
        "producto",
        "productos perecederos",
        "productos no perecederos",
        "fuente",
        "sistema",
        "usd/presentaci\u00f3n",
        "quincena",
        "boletin",
        "bolet\u00edn",
        "provincia",
        "precio",
        "tabla",
        "mes",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
        "enero",
        "febrero",
        "azuay",
        "guayas",
        "pichincha",
        "at/t-1",
        "at/t",
        "a/t",
        "usd",
        "us$/presentacion",
        "us$/presentaci\u00f3n",
    }

    def to_float(v):
        if v is None:
            return None
        v = str(v).strip().rstrip("%")
        if not v or v == "-":
            return None
        v = v.replace(" ", "").replace(",", ".")
        if "." not in v and len(v) > 4 and v.isdigit():
            v = v[:-2] + "." + v[-2:]
        try:
            return float(v)
        except ValueError:
            return None

    def limpiar_token_precio(token):
        """Corrige errores comunes de OCR en tokens de precio."""
        if token is None or token == "-":
            return token
        t = str(token).strip()
        # Corregir letras confundidas con numeros
        t = re.sub(r"[Oo]", "0", t)  # O -> 0
        t = re.sub(r"[lIi|]", "1", t)  # l, I, i, | -> 1
        t = re.sub(r"[Ss]", "5", t)  # S -> 5
        t = re.sub(r"[Bb]", "8", t)  # B -> 8
        t = re.sub(r"[Zz]", "2", t)  # Z -> 2
        t = re.sub(r",", ".", t)  # coma -> punto
        t = re.sub(r"\s+", "", t)  # espacios
        t = t.rstrip("%")
        return t if t else token

    for linea in texto_ocr.split("\n"):
        linea = linea.strip()
        if not linea:
            continue

        linea_limpia = re.sub(r"\s+", " ", linea.lower()).strip()
        if linea_limpia in palabras_descartar:
            continue
        if any(p in linea_limpia for p in palabras_descartar):
            continue
        if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", linea):
            continue
        if re.fullmatch(r"[\d.,%/-]+", linea):
            continue

        tokens = linea.split()
        if not tokens:
            continue

        precio_tokens = []
        indices_precios = []
        variacion_raw = None
        for i in range(len(tokens) - 1, -1, -1):
            token_original = tokens[i]
            token = token_original.rstrip("%")
            if token_original.endswith("%") and variacion_raw is None:
                variacion_raw = token_original
                continue
            if token == "-":
                precio_tokens.insert(0, token)
                indices_precios.insert(0, i)
                if len(precio_tokens) == 2:
                    break
                continue
            # Aplicar limpieza OCR antes de validar
            token_limpio = limpiar_token_precio(token)
            if not patron_precio.match(token_limpio):
                continue
            if i + 1 < len(tokens) and re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", tokens[i + 1]):
                continue
            precio_tokens.insert(0, token_limpio)
            indices_precios.insert(0, i)
            if len(precio_tokens) == 2:
                break

        if len(precio_tokens) < 2:
            precio_tokens = []
            indices_precios = []

        if indices_precios:
            nombre_tokens = tokens[:indices_precios[0]]
        else:
            nombre_tokens = tokens

        nombre = " ".join(nombre_tokens).strip(" -\u2022")
        if not nombre:
            nombre = linea.strip(" -\u2022")
        if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", nombre):
            continue

        nombre = re.sub(r"\bIb\b", "lb", nombre)
        nombre = re.sub(r"\blb\b", "lb", nombre)
        nombre = re.sub(r"\b1b\b", "lb", nombre)
        nombre = re.sub(r"\b\|b\b", "lb", nombre)
        nombre = re.sub(r"\bIt\b", "lt", nombre)
        nombre = re.sub(r"\b\|t\b", "lt", nombre)
        nombre = re.sub(r"\[", "(", nombre)
        nombre = re.sub(r"\(", "(", nombre)
        nombre = re.sub(r"\)", ")", nombre)
        nombre = re.sub(r"Invemadero", "Invernadero", nombre)
        nombre = re.sub(r"Tiema", "Tierna", nombre)
        nombre = re.sub(r"Tierma", "Tierna", nombre)
        nombre = re.sub(r"Tiera", "Tierna", nombre)
        nombre = re.sub(r"Fr[e\u00e9]jol", "Frejol", nombre)
        nombre = re.sub(r"aprox[_\s:]+", "aprox. ", nombre)
        nombre = re.sub(r"Se[n\u00f1]o", "Seco", nombre)
        nombre = re.sub(r"Se\x82o", "Seco", nombre)
        nombre = re.sub(r"Se\ufffd[o\u00f3]", "Seco", nombre)
        nombre = re.sub(r"\]$", ")", nombre)
        nombre = re.sub(r"Se[^c\d\s]{1,3}o(?=\s*\()", "Seco", nombre)
        nombre = re.sub(r"de 11\b", "de 1 l", nombre)
        nombre = re.sub(r"de 1 1\b", "de 1 l", nombre)
        nombre = re.sub(r"de 1 I\b", "de 1 l", nombre)
        nombre = re.sub(r"de 1I\)", "de 1 l)", nombre)
        nombre = re.sub(r"de 1l\)", "de 1 l)", nombre)
        nombre = re.sub(r"de 1 /\)", "de 1 l)", nombre)
        nombre = re.sub(r"de 1 \|\)", "de 1 l)", nombre)
        nombre = re.sub(r"de 1 \|", "de 1 l", nombre)
        nombre = re.sub(r"aprox\.\)", "aprox.)", nombre)
        nombre = re.sub(r"\(aprox\.\)", "", nombre)

        while nombre.startswith("(") and nombre.count("(") > nombre.count(")"):
            nombre = nombre[1:]
        nombre = re.sub(r"\)\)$", ")", nombre)
        if "(" in nombre and nombre.count("(") > nombre.count(")"):
            nombre = nombre + ")"
        nombre = re.sub(r"\s+\(\)$", "", nombre)
        nombre = re.sub(r"\(\)\s*\)", ")", nombre)
        nombre = re.sub(r"\(Envase de\)$", "", nombre)
        nombre = nombre.strip()

        if len(nombre) < 2:
            continue

        precio_anterior = to_float(precio_tokens[0]) if len(precio_tokens) >= 2 else None
        precio_actual = to_float(precio_tokens[1]) if len(precio_tokens) >= 2 else None

        variacion = None
        if variacion_raw:
            v = variacion_raw.rstrip("%").replace(" ", "")
            try:
                variacion = round(float(v), 1)
            except ValueError:
                variacion = None

        registros.append({
            "producto_raw": nombre,
            "precio_anterior": precio_anterior,
            "precio_actual": precio_actual,
            "variacion": variacion,
        })

    return registros


def dividir_en_provincias(bbox_full, n_provincias=3):
    x0, top, x1, bottom = bbox_full
    ancho_col = (x1 - x0) / n_provincias
    return [(x0 + i * ancho_col, top, x0 + (i + 1) * ancho_col, bottom) for i in range(n_provincias)]


def procesar_boletin(pdf_path):
    registros_totales = []
    encabezados_fallidos = []
    orden_global = 0

    with pdfplumber.open(pdf_path) as pdf:
        n_paginas = len(pdf.pages)

        for i in range(0, n_paginas, 2):
            pagina_portada = pdf.pages[i]
            texto_portada = pagina_portada.extract_text() or ""
            info = parse_header(texto_portada)

            if info is None:
                encabezados_fallidos.append(i + 1)
                print(f"  [!] No se pudo leer encabezado en pagina {i+1}, se omite.")
                continue

            print(f"  Procesando boletin No{info['boletin_num']} - {info['quincena_id']} ...")

            bboxes_perec = dividir_en_provincias(BBOX_PERECEDEROS_FULL)
            for provincia, bbox in zip(PROVINCIAS, bboxes_perec):
                texto = ocr_region(pagina_portada, bbox)
                filas = parsear_tabla_precios(texto)
                for f in filas:
                    f.update(info)
                    f["provincia"] = provincia
                    f["categoria"] = "perecedero"
                    f["orden"] = orden_global
                    orden_global += 1
                registros_totales.extend(filas)

            if i + 1 < n_paginas:
                pagina_2 = pdf.pages[i + 1]
                bboxes_noperec = dividir_en_provincias(BBOX_NO_PERECEDEROS_FULL)
                for provincia, bbox in zip(PROVINCIAS, bboxes_noperec):
                    texto = ocr_region(pagina_2, bbox)
                    filas = parsear_tabla_precios(texto)
                    for f in filas:
                        f.update(info)
                        f["provincia"] = provincia
                        f["categoria"] = "no_perecedero"
                        f["orden"] = orden_global
                        orden_global += 1
                    registros_totales.extend(filas)

    return registros_totales, encabezados_fallidos


def validar_calidad(df, encabezados_fallidos):
    """Valida la calidad de los datos extraídos del boletín."""
    problemas = []

    # 1. Productos vacíos o nulos
    mask_datos = df["producto_raw"].isna() | df["producto_raw"].str.len().fillna(0).eq(0)
    for _, row in df[mask_datos].iterrows():
        problemas.append({
            "tipo": "producto_vacio", "producto": row["producto_raw"],
            "provincia": row["provincia"], "quincena": row["quincena_id"],
            "detalle": "Nombre de producto vacío"
        })

    # 2. Productos cortos (menos de 3 caracteres)
    mask_corto = (~mask_datos) & (df["producto_raw"].str.len() < 3)
    for _, row in df[mask_corto].iterrows():
        problemas.append({
            "tipo": "producto_corto", "producto": row["producto_raw"],
            "provincia": row["provincia"], "quincena": row["quincena_id"],
            "detalle": f"Nombre con solo {len(str(row['producto_raw']))} caracteres"
        })

    # 3. Clasificar registros por completitud de precios
    # completo: ambos precios presentes
    # parcial: solo precio_actual (producto nuevo sin historial)
    # invalido: ambos precios nulos
    df["estado_precio"] = "completo"
    mask_solo_actual = df["precio_anterior"].isna() & df["precio_actual"].notna()
    df.loc[mask_solo_actual, "estado_precio"] = "parcial"
    mask_ambos_nulos = df["precio_anterior"].isna() & df["precio_actual"].isna()
    df.loc[mask_ambos_nulos, "estado_precio"] = "invalido"

    # Solo marcar como problema los registros inválidos (sin ningún precio)
    for _, row in df[mask_ambos_nulos].iterrows():
        problemas.append({
            "tipo": "precio_invalido", "producto": row["producto_raw"],
            "precio_anterior": None, "precio_actual": None,
            "provincia": row["provincia"], "quincena": row["quincena_id"],
            "detalle": "Sin precios disponibles (OCR no detectó valores)"
        })

    # 4. Precios iguales a cero o negativos
    mask_precio_invalido = (~mask_ambos_nulos) & ((df["precio_actual"] <= 0) | (df["precio_anterior"] <= 0))
    for _, row in df[mask_precio_invalido].iterrows():
        problemas.append({
            "tipo": "precio_invalido", "producto": row["producto_raw"],
            "precio_anterior": row["precio_anterior"], "precio_actual": row["precio_actual"],
            "provincia": row["provincia"], "quincena": row["quincena_id"],
            "detalle": f"Precio ≤ $0: anterior=${row['precio_anterior']}, actual=${row['precio_actual']}"
        })

    # 5. Precios excesivos (> $500)
    mask_precio_excesivo = (~mask_ambos_nulos) & ((df["precio_actual"] > 500) | (df["precio_anterior"] > 500))
    for _, row in df[mask_precio_excesivo].iterrows():
        problemas.append({
            "tipo": "precio_excesivo", "producto": row["producto_raw"],
            "precio_anterior": row["precio_anterior"], "precio_actual": row["precio_actual"],
            "provincia": row["provincia"], "quincena": row["quincena_id"],
            "detalle": f"Precio > $500: anterior=${row['precio_anterior']}, actual=${row['precio_actual']}"
        })

    # 6. Variación extrema (-100% a +500%)
    mask_variacion = df["variacion"].notna() & ((df["variacion"] < -100) | (df["variacion"] > 500))
    for _, row in df[mask_variacion].iterrows():
        problemas.append({
            "tipo": "variacion_extrema", "producto": row["producto_raw"],
            "precio_anterior": row["precio_anterior"], "precio_actual": row["precio_actual"],
            "provincia": row["provincia"], "quincena": row["quincena_id"],
            "detalle": f"Variación de {row['variacion']}% fuera de rango"
        })

    # 7. Provincias inválidas
    provincias_validas = ["AZUAY", "GUAYAS", "PICHINCHA"]
    mask_prov = ~df["provincia"].isin(provincias_validas)
    for _, row in df[mask_prov].iterrows():
        problemas.append({
            "tipo": "provincia_invalida", "producto": row["producto_raw"],
            "precio_anterior": row["precio_anterior"], "precio_actual": row["precio_actual"],
            "provincia": row["provincia"], "quincena": row["quincena_id"],
            "detalle": f"Provincia no reconocida: {row['provincia']}"
        })

    # 8. Categorías inválidas
    cat_validas = ["perecedero", "no_perecedero"]
    if "categoria" in df.columns:
        mask_cat = ~df["categoria"].isin(cat_validas)
        for _, row in df[mask_cat].iterrows():
            problemas.append({
                "tipo": "categoria_invalida", "producto": row["producto_raw"],
                "precio_anterior": row["precio_anterior"], "precio_actual": row["precio_actual"],
                "provincia": row["provincia"], "quincena": row["quincena_id"],
                "detalle": f"Categoría no reconocida: {row['categoria']}"
            })

    # 9. Duplicados exactos
    if "categoria" in df.columns:
        cols_dup = ["producto_raw", "provincia", "quincena_id", "categoria"]
        mask_dup = df.duplicated(subset=cols_dup, keep="first")
        for _, row in df[mask_dup].iterrows():
            problemas.append({
                "tipo": "duplicado", "producto": row["producto_raw"],
                "precio_anterior": row["precio_anterior"], "precio_actual": row["precio_actual"],
                "provincia": row["provincia"], "quincena": row["quincena_id"],
                "detalle": f"Duplicado en categoría {row['categoria']}"
            })

    # Resumen por tipo
    resumen_tipos = {}
    for p in problemas:
        tipo = p["tipo"]
        if tipo not in resumen_tipos:
            resumen_tipos[tipo] = {"cantidad": 0, "ejemplos": []}
        resumen_tipos[tipo]["cantidad"] += 1
        if len(resumen_tipos[tipo]["ejemplos"]) < 3:
            resumen_tipos[tipo]["ejemplos"].append(p)

    # Métricas por provincia
    metricas_provincia = {}
    for prov in df["provincia"].unique():
        df_prov = df[df["provincia"] == prov]
        total_prov = len(df_prov)
        problemas_prov = len([p for p in problemas if p["provincia"] == prov])
        metricas_provincia[prov] = {
            "total": total_prov,
            "completos": total_prov - problemas_prov,
            "problemas": problemas_prov,
            "porcentaje_calidad": round(100 * (1 - problemas_prov / total_prov), 2) if total_prov > 0 else 0
        }

    total = len(df)
    completitud = round(100 * (1 - len(problemas) / total), 2) if total > 0 else 0

    # Contar registros por estado de precio
    completos = len(df[df["estado_precio"] == "completo"])
    parciales = len(df[df["estado_precio"] == "parcial"])
    invalidos = len(df[df["estado_precio"] == "invalido"])

    return {
        "total_registros": total,
        "registros_completos": completos,
        "registros_parciales": parciales,
        "registros_invalidos": invalidos,
        "registros_con_problema": len(problemas),
        "porcentaje_completitud": completitud,
        "productos_unicos": df["producto_raw"].nunique(),
        "quincenas": df["quincena_id"].nunique(),
        "problemas": problemas,
        "resumen_tipos": resumen_tipos,
        "metricas_provincia": metricas_provincia,
        "encabezados_fallidos": encabezados_fallidos,
    }
