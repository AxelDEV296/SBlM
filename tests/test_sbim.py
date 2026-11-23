"""Pruebas del paquete distribuido y de sus controles de Streamlit."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from streamlit.testing.v1 import AppTest

from artefactos import (
    ARCHIVO_METRICAS, ARCHIVO_MODELO, ARCHIVO_PREDICCIONES, ROOT,
    cargar_modelo, cargar_resultados, probabilidad_sube,
)
from scripts.comun import COLUMNAS


class ArtefactosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modelo = cargar_modelo()
        cls.metricas, cls.registros = cargar_resultados()

    def test_archivos_obligatorios(self):
        for ruta in (ARCHIVO_MODELO, ARCHIVO_METRICAS, ARCHIVO_PREDICCIONES,
                     ROOT / "app.py", ROOT / "requirements.txt", ROOT / "resultados" / "matriz_confusion.png"):
            with self.subTest(archivo=ruta.name):
                self.assertTrue(ruta.is_file())
                self.assertGreater(ruta.stat().st_size, 0)

    def test_datos_y_metricas(self):
        self.assertEqual(len(self.registros), self.metricas["filas_test"])
        y = self.registros["sube_siguiente"]
        pred = self.registros["prediccion"]
        self.assertEqual(confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                         self.metricas["matriz_confusion"])
        for predicciones, clave in ((pred, "metricas_test_modelo"),
                                     (np.ones(len(y)), "metricas_test_baseline")):
            calculadas = {
                "exactitud": accuracy_score(y, predicciones),
                "precision_clase_sube": precision_score(y, predicciones),
                "recall_clase_sube": recall_score(y, predicciones),
                "f1_clase_sube": f1_score(y, predicciones),
                "f1_macro": f1_score(y, predicciones, average="macro"),
            }
            for nombre, valor in calculadas.items():
                with self.subTest(grupo=clave, metrica=nombre):
                    self.assertAlmostEqual(valor, self.metricas[clave][nombre])
        np.testing.assert_array_equal(
            (self.registros["probabilidad_sube"] > 0.5).astype(int), pred
        )

    def test_modelo_y_prediccion(self):
        self.assertEqual(list(self.modelo.feature_names_in_), COLUMNAS)
        self.assertEqual(self.modelo.named_steps["clasificador"].C, self.metricas["mejor_C"])
        valores = [0.003, 0.01, 0.015, 1.0]
        entrada = pd.DataFrame([valores], columns=COLUMNAS)
        probabilidades = self.modelo.predict_proba(entrada)[0]
        self.assertTrue(np.isfinite(probabilidades).all())
        self.assertTrue(((0 <= probabilidades) & (probabilidades <= 1)).all())
        self.assertAlmostEqual(probabilidades.sum(), 1.0)
        self.assertEqual(self.modelo.predict(entrada)[0], probabilidades.argmax())
        self.assertAlmostEqual(probabilidad_sube(self.modelo, *valores), probabilidades[1])

    def test_entradas_invalidas(self):
        for valores in ([np.nan, 0, 0, 1], [0, np.inf, 0, 1], [0, 0, -1, 1],
                        [0, 0, 0, -1], ["texto", 0, 0, 1]):
            with self.subTest(valores=valores), self.assertRaises(ValueError):
                probabilidad_sube(self.modelo, *valores)
        with tempfile.TemporaryDirectory() as directorio:
            csv = Path(directorio) / "invalido.csv"
            csv.write_text("Fecha,prediccion\n2024-09-05,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Faltan columnas"):
                cargar_resultados(ARCHIVO_METRICAS, csv)
            with self.assertRaises(FileNotFoundError):
                cargar_modelo(Path(directorio) / "ausente.joblib")


class StreamlitTest(unittest.TestCase):
    def test_inicio_y_controles(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("Microsoft", app.title[0].value)
        self.assertIn("académico", app.warning[0].value)
        self.assertEqual(len(app.tabs), 3)
        self.assertEqual(app.metric[0].value, "58.2%")
        self.assertEqual(app.metric[1].value, "0.577")
        self.assertTrue(app.get("vega_lite_chart"))
        self.assertEqual(app.dataframe[0].value.shape[0], 15)
        self.assertEqual(app.table[0].value.shape, (2, 2))
        app.selectbox[0].select("2024-09-05").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[3].value, "53.3%")
        app.slider[0].set_value(5.0).run()
        self.assertFalse(app.exception)
        esperado = probabilidad_sube(cargar_modelo(), 0.05, 0.01, 0.015, 1.0)
        self.assertEqual(app.metric[4].value, f"{esperado:.1%}")

    def test_error_de_arranque_detiene_la_interfaz(self):
        with patch("artefactos.ARCHIVO_MODELO", ROOT / "ausente.joblib"):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn("ausente.joblib", app.error[0].value)
        self.assertFalse(app.metric)
