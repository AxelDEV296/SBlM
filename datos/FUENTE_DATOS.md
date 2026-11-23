# Fuente y alcance del conjunto de datos incluido

- **Activo**: Microsoft Corporation (`MSFT`).
- **Origen de esta muestra congelada**: archivo público `Microsoft Stock Data.csv` del repositorio [`LSG626/Microsoft-Stock-Model`](https://github.com/LSG626/Microsoft-Stock-Model/blob/main/Microsoft%20Stock%20Data.csv), basado en un histórico de cotizaciones.
- **Referencia puntual consultada**: revisión `812dc823336059429d19f7c570538d8709c3d5ae`; muestra del **2 de enero al 21 de noviembre de 2024**.
- **Campos utilizados en la muestra de entrenamiento**: `Fecha`, `Close` (se toma `Adj Close` del archivo original, redondeado a dos decimales para esta práctica) y `Volume` (volumen original entero).
- **Archivos distribuidos**. El CSV de resultados incluye fechas, etiquetas y probabilidades. La serie original y los volúmenes utilizados para entrenar se conservan en la fuente de referencia.
- Los datos corresponden al periodo histórico indicado. Los registros conservados se usan para la demostración académica.
- **Licencia**: el repositorio de terceros no exhibe una licencia claramente identificada en la revisión realizada. Antes de redistribuir datos completos o explotar comercialmente la muestra, verificar los derechos y condiciones de Yahoo, el proveedor y el repositorio. Para GitHub público, se recomienda obtener y publicar únicamente datos con autorización, o bien publicar el código y el modelo sin la serie.
- **Uso**: análisis docente, no asesoría ni señal de compraventa.

Para descargar directamente una serie nueva de Yahoo Finance a través de `yfinance`, consulta: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
