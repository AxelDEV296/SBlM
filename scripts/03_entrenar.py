"""ETAPA 3. Entrenar modelo con división temporal y sin fuga de datos.

Separa el 75% cronológicamente, excluye el último registro del conjunto
inicial de entrenamiento porque su etiqueta utiliza el primer día de test,
y ajusta StandardScaler solamente con observaciones de entrenamiento.
"""
import json
import time
import hashlib
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from comun import ROOT, DATOS, MODELO, RESULTADOS, COLUMNAS


def puntajes(y, pred):
    return {
        'exactitud': float(accuracy_score(y, pred)),
        'precision_clase_sube': float(precision_score(y, pred, zero_division=0)),
        'recall_clase_sube': float(recall_score(y, pred, zero_division=0)),
        'f1_clase_sube': float(f1_score(y, pred, zero_division=0)),
        'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
    }


def main():
    origen = DATOS / 'msft_variables.csv'
    if not origen.exists():
        raise FileNotFoundError('Prepara datos/msft_variables.csv con los predictores y etiquetas antes de entrenar.')
    df = pd.read_csv(origen, parse_dates=['Fecha'])
    df = df[df.sube_siguiente.notna()].sort_values('Fecha').reset_index(drop=True)
    if len(df) < 90:
        raise ValueError('Insuficientes filas útiles para entrenamiento/validación.')
    corte = int(len(df) * 0.75)
    entrenamiento = df.iloc[:corte-1].copy()  # Embargo del límite train/test
    prueba = df.iloc[corte:].copy()
    X_train, y_train = entrenamiento[COLUMNAS], entrenamiento.sube_siguiente.astype(int)
    X_test, y_test = prueba[COLUMNAS], prueba.sube_siguiente.astype(int)
    if y_train.nunique() != 2:
        raise ValueError('El entrenamiento debe contener las clases 0 y 1.')

    pipeline = Pipeline([
        ('imputacion', SimpleImputer(strategy='median')),
        ('escalado', StandardScaler()),
        ('clasificador', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ])
    grid = GridSearchCV(pipeline,
        param_grid={'clasificador__C': [0.1, 1.0, 10.0]},
        cv=TimeSeriesSplit(n_splits=3, gap=1),
        scoring='f1_macro', refit=True, n_jobs=1)
    print('\n=== 03. ENTRENAMIENTO ===')
    print('Entrenamiento:', len(entrenamiento), 'de', entrenamiento.Fecha.min().date(), 'a', entrenamiento.Fecha.max().date())
    print('Prueba:', len(prueba), 'de', prueba.Fecha.min().date(), 'a', prueba.Fecha.max().date())
    inicio = time.perf_counter()
    grid.fit(X_train, y_train)
    segundos = time.perf_counter() - inicio
    final = grid.best_estimator_
    pred = final.predict(X_test)
    prob = final.predict_proba(X_test)[:, list(final.classes_).index(1)]

    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    puntaje_modelo = puntajes(y_test, pred)
    puntaje_base = puntajes(y_test, baseline_pred)
    RESULTADOS.mkdir(parents=True, exist_ok=True); MODELO.mkdir(parents=True, exist_ok=True)
    archivo_modelo = MODELO / 'modelo_msft.joblib'
    joblib.dump(final, archivo_modelo)
    registro = prueba[['Fecha', 'Close', 'sube_siguiente']].copy()
    registro['prediccion'] = pred
    registro['probabilidad_sube'] = prob
    registro.to_csv(RESULTADOS / 'predicciones_prueba.csv', index=False)
    matriz = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
    resumen = {
        'titulo': 'Semáforo bursátil académico de MSFT',
        'fuente_historial': 'Consultar datos/FUENTE_DATOS.md; muestra 2024 o CSV Yahoo descargado por el alumno',
        'fecha_primera_train': str(entrenamiento.Fecha.min().date()),
        'fecha_ultima_train': str(entrenamiento.Fecha.max().date()),
        'fecha_primera_test': str(prueba.Fecha.min().date()),
        'fecha_ultima_test': str(prueba.Fecha.max().date()),
        'filas_train': len(entrenamiento),
        'filas_test': len(prueba),
        'mejor_C': float(grid.best_params_['clasificador__C']),
        'f1_macro_cv': float(grid.best_score_),
        'metricas_test_modelo': puntaje_modelo,
        'metricas_test_baseline': puntaje_base,
        'matriz_confusion': matriz,
        'tiempo_entrenamiento_segundos': round(segundos, 3),
        'columnas_predictoras': COLUMNAS,
        'python': sys.version.split()[0],
        'scikit_learn': sklearn.__version__,
        'hash_dataset_variables_sha256': hashlib.sha256(origen.read_bytes()).hexdigest(),
        'nota': 'Experimento educativo en series temporales. No predice con fiabilidad el mercado ni recomienda inversiones.'
    }
    (RESULTADOS / 'metricas.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Mejor regularización C:', resumen['mejor_C'])
    print('F1 macro CV temporal:', round(resumen['f1_macro_cv'],4))
    print('Exactitud test:', round(puntaje_modelo['exactitud'],4))
    print('F1 macro test:', round(puntaje_modelo['f1_macro'],4))
    print('F1 macro baseline:', round(puntaje_base['f1_macro'],4))
    print('Matriz de confusión [real 0/1 x pred 0/1]:', matriz)
    print('Modelo guardado:', archivo_modelo)


if __name__ == '__main__':
    main()
