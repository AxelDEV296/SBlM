"""Funciones compartidas por las etapas de la clase."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / 'datos'
RESULTADOS = ROOT / 'resultados'
MODELO = ROOT / 'modelo'
COLUMNAS = ['retorno_1d', 'retorno_5d', 'volatilidad_5d', 'volumen_relativo_5d']


def leer_historial(ruta=None):
    """Lee Fecha, Close y Volume. No exige red en el ejemplo incluido."""
    archivo = Path(ruta) if ruta else DATOS / 'msft_2024_referencia.csv'
    if not archivo.is_absolute():
        archivo = ROOT / archivo
    if not archivo.exists():
        raise FileNotFoundError(f'No se encontró {archivo}')
    df = pd.read_csv(archivo, parse_dates=['Fecha'])
    necesarias = {'Fecha', 'Close', 'Volume'}
    if not necesarias.issubset(df.columns):
        raise ValueError(f'Faltan columnas: {necesarias - set(df.columns)}')
    df = df[['Fecha', 'Close', 'Volume']].copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna().drop_duplicates(subset=['Fecha']).sort_values('Fecha')
    df = df[(df.Close > 0) & (df.Volume >= 0)].reset_index(drop=True)
    if len(df) < 100:
        raise ValueError('Se necesitan al menos 100 registros históricos válidos.')
    return df


def crear_variables(df):
    """Predictores hasta el cierre de t; etiqueta utiliza SOLO el cierre siguiente."""
    d = df.copy().sort_values('Fecha').reset_index(drop=True)
    d['retorno_1d'] = d.Close.pct_change()
    d['retorno_5d'] = d.Close.pct_change(5)
    d['volatilidad_5d'] = d['retorno_1d'].rolling(5, min_periods=5).std()
    d['volumen_relativo_5d'] = d.Volume / d.Volume.rolling(5, min_periods=5).mean().replace(0, np.nan)
    proximo = d.Close.shift(-1)
    d['sube_siguiente'] = np.where(proximo.notna(), (proximo > d.Close).astype(float), np.nan)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=COLUMNAS).reset_index(drop=True)
    return d
