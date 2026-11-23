# Semáforo bursátil académico de Microsoft (MSFT)

Aplicación educativa que estima la probabilidad de que el siguiente cierre de MSFT sea superior al cierre del día seleccionado. Usa un pipeline entrenado con Python 3.13.5 y scikit-learn 1.8.0. Streamlit presenta la evaluación histórica y un simulador de variables. Pandas y NumPy manejan los datos, y Joblib carga el artefacto de entrenamiento.

## Instalación y ejecución en Windows

Abre en PyCharm la carpeta que contiene `app.py`. Si el entorno `.venv-py313` ya está instalado, puedes ejecutar directamente las dos últimas líneas. Para crearlo desde cero, instala Python 3.13 y ejecuta estos comandos en PowerShell desde la carpeta del proyecto:

```powershell
py -3.13 -m venv .venv-py313
.\.venv-py313\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-py313\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-py313\Scripts\python.exe -m streamlit run app.py
```

En PyCharm, elige `.venv-py313\Scripts\python.exe` como intérprete. Crea una configuración Python con tipo **Module name**, módulo `streamlit`, parámetros `run app.py` y la carpeta del proyecto como directorio de trabajo. Abre la URL que muestra Streamlit, normalmente `http://localhost:8501`.

## Archivos

| Archivo | Uso |
| --- | --- |
| `app.py` | Interfaz y controles de Streamlit |
| `artefactos.py` | Carga, validación e inferencia |
| `scripts/comun.py` | Definición de variables y preparación histórica |
| `modelo/modelo_msft.joblib` | Pipeline entrenado |
| `resultados/metricas.json`, `resultados/predicciones_prueba.csv`, `resultados/matriz_confusion.png` | Resultados de la evaluación |
| `scripts/03_entrenar.py` | Código de entrenamiento archivado |
| `datos/FUENTE_DATOS.md` | Procedencia de los datos |
| `requirements.txt`, `tests/` | Dependencias y pruebas |

El arranque usa los artefactos locales. Carga únicamente el modelo de una revisión cuya procedencia hayas verificado. Joblib puede ejecutar código al deserializar.

## Modelo y evaluación

El pipeline usa el retorno de uno y cinco días, la volatilidad de cinco días y el volumen relativo a cinco días. Aplica imputación por mediana, estandarización y regresión logística. La búsqueda de hiperparámetros usó validación cronológica con `TimeSeriesSplit(gap=1)`. El JSON registra 164 observaciones de entrenamiento y 55 de prueba, del 5 de septiembre al 20 de noviembre de 2024.

La exactitud indica la proporción de aciertos. El F1 macro promedia por igual el F1 de las dos clases. El modelo obtuvo 58.2 % de exactitud y 0.577 de F1 macro, frente a 56.4 % y 0.360 para la línea base que predice subida. La matriz cuenta los casos por clase observada (filas) y predicha (columnas).

El selector muestra predicciones guardadas. El simulador calcula probabilidades para valores hipotéticos. El CSV distribuido conserva fechas, etiquetas y probabilidades. Los predictores históricos y el conjunto de entrenamiento son necesarios para repetir las 55 inferencias originales y verificar el hash del conjunto de variables. El entrenamiento está documentado en el código y requiere aportar `datos/msft_variables.csv` para repetirlo.

Las probabilidades carecen de calibración y se basan en una muestra limitada de 2024. El proyecto tiene un fin académico. Consulta [la procedencia y las limitaciones de redistribución](datos/FUENTE_DATOS.md).

## Streamlit Community Cloud

Desde [Streamlit Community Cloud](https://share.streamlit.io/), crea una aplicación con estos valores una vez que la rama de publicación contenga los cambios:

| Campo | Valor |
| --- | --- |
| Repository | `AxelDEV296/SBlM` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version (Advanced settings) | `3.13` |
| Secrets | Vacío |

Comprueba las tres pestañas, cambia una fecha y mueve un control del simulador después de desplegar. [Guía oficial de despliegue](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
