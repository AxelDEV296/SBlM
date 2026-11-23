"""Carga y valida los artefactos que acompañan a la aplicación."""
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.comun import COLUMNAS

ROOT = Path(__file__).resolve().parent
ARCHIVO_MODELO = ROOT / "modelo" / "modelo_msft.joblib"
ARCHIVO_METRICAS = ROOT / "resultados" / "metricas.json"
ARCHIVO_PREDICCIONES = ROOT / "resultados" / "predicciones_prueba.csv"


def cargar_modelo(ruta=ARCHIVO_MODELO):
    """Carga únicamente el modelo distribuido con el proyecto."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        modelo = joblib.load(ruta)
    if not isinstance(modelo, Pipeline):
        raise ValueError("El artefacto debe contener un pipeline de scikit-learn.")
    pasos = (SimpleImputer, StandardScaler, LogisticRegression)
    if tuple(type(paso) for _, paso in modelo.steps) != pasos:
        raise ValueError("Los pasos del modelo no coinciden con el pipeline esperado.")
    if list(modelo.feature_names_in_) != COLUMNAS or list(modelo.classes_) != [0, 1]:
        raise ValueError("Las columnas o clases del modelo no coinciden con la aplicación.")
    return modelo


def cargar_resultados(ruta_metricas=ARCHIVO_METRICAS, ruta_csv=ARCHIVO_PREDICCIONES):
    metricas = json.loads(ruta_metricas.read_text(encoding="utf-8"))
    registros = pd.read_csv(ruta_csv)
    necesarias = {"Fecha", "sube_siguiente", "prediccion", "probabilidad_sube"}
    faltantes = necesarias - set(registros.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {', '.join(sorted(faltantes))}.")
    registros["Fecha"] = pd.to_datetime(registros["Fecha"], format="%Y-%m-%d", errors="raise")
    fechas = registros["Fecha"]
    if (registros.empty or fechas.isna().any() or fechas.duplicated().any()
            or not fechas.is_monotonic_increasing):
        raise ValueError("El CSV debe contener fechas válidas, únicas y ordenadas.")
    for columna in ("sube_siguiente", "prediccion", "probabilidad_sube"):
        registros[columna] = pd.to_numeric(registros[columna], errors="raise")
    if not registros[["sube_siguiente", "prediccion"]].isin([0, 1]).all().all():
        raise ValueError("Las etiquetas del CSV deben ser 0 o 1.")
    if not registros["probabilidad_sube"].between(0, 1).all():
        raise ValueError("Las probabilidades del CSV deben estar entre 0 y 1.")
    if (metricas["columnas_predictoras"] != COLUMNAS
            or metricas["filas_test"] != len(registros)
            or metricas["fecha_primera_test"] != fechas.iloc[0].date().isoformat()
            or metricas["fecha_ultima_test"] != fechas.iloc[-1].date().isoformat()):
        raise ValueError("Las métricas y el CSV no describen el mismo experimento.")
    for grupo, nombre in (("metricas_test_modelo", "exactitud"),
                          ("metricas_test_modelo", "f1_macro"),
                          ("metricas_test_baseline", "f1_macro")):
        if not 0 <= metricas[grupo][nombre] <= 1:
            raise ValueError(f"Métrica fuera de rango: {grupo}.{nombre}.")
    if np.shape(metricas["matriz_confusion"]) != (2, 2):
        raise ValueError("La matriz de confusión debe tener dos filas y dos columnas.")
    return metricas, registros


def probabilidad_sube(modelo, retorno_1d, retorno_5d, volatilidad_5d, volumen_relativo_5d):
    valores = [retorno_1d, retorno_5d, volatilidad_5d, volumen_relativo_5d]
    try:
        entrada = np.asarray(valores, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Los predictores deben ser números finitos.") from error
    if entrada.shape != (4,) or not np.isfinite(entrada).all():
        raise ValueError("Los predictores deben ser números finitos.")
    if entrada[2] < 0 or entrada[3] < 0:
        raise ValueError("La volatilidad y el volumen relativo no pueden ser negativos.")
    registro = pd.DataFrame([entrada], columns=COLUMNAS)
    return float(modelo.predict_proba(registro)[0, 1])
