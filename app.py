"""Interfaz Streamlit: modelo histórico y educativo de MSFT."""
from pickle import UnpicklingError

import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

from artefactos import (
    ARCHIVO_METRICAS, ARCHIVO_MODELO, ARCHIVO_PREDICCIONES,
    cargar_modelo, cargar_resultados, probabilidad_sube,
)

st.set_page_config(page_title='Semáforo bursátil MSFT — IA educativa', page_icon='📊', layout='wide')
st.title('📊 Semáforo bursátil de Microsoft (MSFT)')
st.caption('Aprendizaje automático supervisado · Actividad educativa basada en datos históricos')
st.warning('Prototipo académico: NO es asesoramiento financiero ni predicción fiable. El histórico incluido llega hasta noviembre de 2024; las probabilidades no están calibradas.')

modelo_en_cache = st.cache_resource(cargar_modelo)
resultados_en_cache = st.cache_data(cargar_resultados)

try:
    modelo = modelo_en_cache(ARCHIVO_MODELO)
    m, registros = resultados_en_cache(ARCHIVO_METRICAS, ARCHIVO_PREDICCIONES)
except (
    OSError, ValueError, KeyError, TypeError, AttributeError,
    EOFError, ImportError, UnpicklingError, InconsistentVersionWarning,
) as error:
    st.error(f'No se pudieron cargar los artefactos de SBIM. Revisa los archivos del repositorio y las versiones de requirements.txt. Detalle: {error}')
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric('Exactitud en prueba', f"{m['metricas_test_modelo']['exactitud']:.1%}")
col2.metric('F1 macro en prueba', f"{m['metricas_test_modelo']['f1_macro']:.3f}")
col3.metric('F1 macro, línea base', f"{m['metricas_test_baseline']['f1_macro']:.3f}")
st.write(f"Periodo de prueba: **{m['fecha_primera_test']} a {m['fecha_ultima_test']}**. Entrenamiento: **{m['filas_train']}** días; prueba: **{m['filas_test']}** días.")

tab1, tab2, tab3 = st.tabs(['🔎 Probar con datos históricos', '📈 Resultados y método', '🧪 Simulador manual'])
with tab1:
    st.subheader('Demostración usando fechas reservadas para prueba')
    fechas = registros['Fecha'].dt.strftime('%Y-%m-%d').tolist()
    fecha = st.selectbox('Selecciona una fecha histórica (cierre de t)', fechas, index=len(fechas)-1, key='fecha_historica')
    fila = registros.loc[registros.Fecha.dt.strftime('%Y-%m-%d') == fecha].iloc[0]
    prob = float(fila.probabilidad_sube)
    st.metric('Probabilidad estimada de una subida al siguiente cierre', f'{prob:.1%}')
    st.info('Predicción guardada durante la evaluación: ' + ('SUBE' if int(fila.prediccion) else 'NO SUBE'))
    st.write('Resultado observado en el conjunto histórico: **'+('SUBIÓ' if int(fila.sube_siguiente) else 'NO SUBIÓ')+'**')
    st.caption('La etiqueta observada se muestra solo para fines docentes y nunca se utiliza como entrada del modelo.')
with tab2:
    st.subheader('Probabilidad histórica estimada para la clase sube (solo conjunto de prueba)')
    st.line_chart(registros.set_index('Fecha')['probabilidad_sube'],
                  alt='Probabilidad estimada de subida por fecha de prueba')
    st.subheader('Evaluación cronológica')
    st.dataframe(registros.tail(15), width='stretch', hide_index=True, alt='Últimas 15 predicciones de prueba')
    st.write('Matriz de confusión [filas reales 0/1, columnas predichas 0/1]:')
    st.table(pd.DataFrame(m['matriz_confusion'], index=['Real no sube', 'Real sube'], columns=['Predicho no sube', 'Predicho sube']),
             alt='Matriz de confusión de la evaluación')
    st.markdown('**Pipeline:** 4 variables históricas → imputación mediana → StandardScaler → regresión logística. Validación con TimeSeriesSplit y gap=1; conjunto final de prueba separado temporalmente.')
with tab3:
    st.subheader('Cambiar características numéricas manualmente')
    st.caption('Estas cifras son escenarios hipotéticos, no el precio de mercado actual.')
    r1 = st.slider('Retorno del último día (%)', -10.0, 10.0, 0.3, 0.1, key='retorno_1d')
    r5 = st.slider('Retorno de los últimos cinco días (%)', -20.0, 20.0, 1.0, 0.1)
    v5 = st.slider('Volatilidad de cinco días (%)', 0.0, 10.0, 1.5, 0.1)
    vr = st.slider('Volumen / volumen medio de cinco días', 0.1, 5.0, 1.0, 0.1)
    prob = probabilidad_sube(modelo, r1 / 100, r5 / 100, v5 / 100, vr)
    st.metric('Probabilidad estimada clase sube', f'{prob:.1%}')
    st.caption('Fuera del rango de entrenamiento los resultados pueden ser poco fiables. No utilizar para invertir.')

st.divider()
st.caption('Modelo entrenado con muestra de MSFT 2024. El histórico bruto no se redistribuye en este despliegue; procedencia en datos/FUENTE_DATOS.md.')
