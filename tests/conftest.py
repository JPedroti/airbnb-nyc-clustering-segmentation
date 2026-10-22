"""Dados sintéticos pequenos, no mesmo schema do AB_NYC_2019.csv, para testes que não dependem do arquivo real."""
import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture
def synthetic_listings() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "name": [f"anúncio {i}" for i in range(n)],
        "host_id": rng.integers(1, 60, n),
        "host_name": "anfitrião",
        "neighbourhood_group": rng.choice(config.NEIGHBOURHOOD_GROUP_ORDER, n),
        "neighbourhood": rng.choice(["A", "B", "C"], n),
        "latitude": rng.uniform(40.6, 40.8, n),
        "longitude": rng.uniform(-74.0, -73.8, n),
        "room_type": rng.choice(config.ROOM_TYPE_ORDER, n, p=[0.5, 0.45, 0.05]),
        "price": rng.integers(20, 600, n),
        "minimum_nights": rng.choice([1, 2, 3, 30, 500], n, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        "number_of_reviews": rng.integers(0, 100, n),
        "last_review": pd.Timestamp("2019-06-01"),
        "reviews_per_month": rng.uniform(0.01, 5, n),
        "calculated_host_listings_count": rng.integers(1, 20, n),
        "availability_365": rng.integers(0, 366, n),
    })
    no_reviews = frame["number_of_reviews"] == 0
    frame.loc[no_reviews, ["reviews_per_month", "last_review"]] = np.nan
    return frame


@pytest.fixture
def fitted_pipeline(synthetic_listings):
    """Pipeline K-Means (R2, K = 3) ajustado nos dados sintéticos limpos; devolve (pipeline, base limpa)."""
    from src import data
    from src.preprocessing import make_kmeans_pipeline

    frame, _ = data.clean_listings(synthetic_listings)
    return make_kmeans_pipeline(3).fit(frame), frame
