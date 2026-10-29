"""Distâncias geográficas em km, a partir das coordenadas originais dos anúncios.

Graus de latitude e longitude não equivalem a quilômetros (em NYC, 1 grau de longitude vale cerca de 84 km
e 1 de latitude cerca de 111 km). Por isso a distância usa a fórmula de haversine sobre uma esfera de raio
`config.EARTH_RADIUS_KM`, e não a distância Euclidiana entre graus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em linha reta sobre a esfera (km) entre pares de coordenadas em graus."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * config.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def distance_to_times_square_km(frame: pd.DataFrame) -> pd.Series:
    """Distância de cada anúncio até a Times Square, em km (mesmo índice de `frame`)."""
    reference = config.TIMES_SQUARE
    distances = haversine_km(frame["latitude"], frame["longitude"], reference["lat"], reference["lon"])
    return pd.Series(distances, index=frame.index, name=config.DISTANCE_FEATURE)
