"""Pré-processamento reutilizável: o mesmo objeto é ajustado no treino e aplicado a dados novos.

Etapas das numéricas, nesta ordem:
1. `reviews_per_month` ausente vira 0, porque o anúncio não tem avaliações;
2. winsorização superior de `price` e `reviews_per_month` no p99,5, com o limite aprendido no `fit`,
   e teto fixo de 365 em `minimum_nights`;
3. `log1p` nas variáveis assimétricas;
4. `StandardScaler` (cenário B) ou nada (cenário A).

As categóricas recebem `OneHotEncoder` com categorias fixas (ordem de `config.CATEGORY_ORDERS`) e não são
escalonadas: cada dummy vale 0 ou 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class NumericTreatment(BaseEstimator, TransformerMixin):
    """Preenche ausências estruturais, limita extremos e aplica log1p, preservando a ordem das colunas.

    Os limites de winsorização são aprendidos no `fit` (atributo `caps_`) e reaplicados sem mudança no
    `transform`, inclusive em dados novos.
    """

    def __init__(
        self,
        zero_fill_columns=tuple(config.ZERO_FILL_COLUMNS),
        winsor_columns=tuple(config.WINSOR_COLUMNS),
        winsor_quantile=config.WINSOR_QUANTILE,
        fixed_caps=(("minimum_nights", config.MINIMUM_NIGHTS_CAP),),
        log_columns=tuple(config.LOG1P_COLUMNS),
    ):
        self.zero_fill_columns = zero_fill_columns
        self.winsor_columns = winsor_columns
        self.winsor_quantile = winsor_quantile
        self.fixed_caps = fixed_caps
        self.log_columns = log_columns

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for column in self.zero_fill_columns:
            if column in X.columns:
                X[column] = X[column].fillna(0)
        return X.astype(float)

    def fit(self, X, y=None):
        X = self._fill(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        caps = {column: float(cap) for column, cap in self.fixed_caps if column in X.columns}
        for column in self.winsor_columns:
            if column in X.columns:
                caps[column] = float(X[column].quantile(self.winsor_quantile))
        self.caps_ = caps
        return self

    def transform(self, X):
        X = self._fill(X)
        for column, cap in self.caps_.items():
            X[column] = X[column].clip(upper=cap)
        for column in self.log_columns:
            if column in X.columns:
                X[column] = np.log1p(X[column])
        return X.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


def make_preprocessor(representation: str = config.MAIN_REPRESENTATION, scale: bool = True) -> ColumnTransformer:
    """ColumnTransformer da representação pedida; `scale=False` reproduz o cenário A (sem StandardScaler)."""
    spec = config.REPRESENTATIONS[representation]
    numeric_steps = [("treat", NumericTreatment())]
    numeric_steps.append(("scale", StandardScaler() if scale else "passthrough"))
    transformers = [("num", Pipeline(numeric_steps), list(spec["numeric"]))]
    if spec["categorical"]:
        categories = [config.CATEGORY_ORDERS[column] for column in spec["categorical"]]
        encoder = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", encoder, list(spec["categorical"])))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def make_kmeans_pipeline(
    n_clusters: int,
    representation: str = config.MAIN_REPRESENTATION,
    scale: bool = True,
    random_state: int | None = None,
) -> Pipeline:
    """Pipeline completo (pré-processamento + K-Means) com o protocolo fixo de `config.KMEANS_PARAMS`."""
    params = dict(config.KMEANS_PARAMS)
    if random_state is not None:
        params["random_state"] = random_state
    return Pipeline([
        ("preprocess", make_preprocessor(representation, scale=scale)),
        ("kmeans", KMeans(n_clusters=n_clusters, **params)),
    ])


def input_columns(representation: str = config.MAIN_REPRESENTATION) -> list[str]:
    """Colunas brutas que a representação precisa receber."""
    spec = config.REPRESENTATIONS[representation]
    return list(spec["numeric"]) + list(spec["categorical"])


def feature_block(feature_name: str) -> str:
    """Variável original de uma coluna transformada (ex.: 'cat__room_type_Private room' -> 'room_type')."""
    name = feature_name.split("__", 1)[-1]
    for column in config.CATEGORY_ORDERS:
        if name.startswith(f"{column}_"):
            return column
    return name
