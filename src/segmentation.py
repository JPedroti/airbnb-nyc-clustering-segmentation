"""Aplicação do K-Means salvo a anúncios novos: validação da entrada e atribuição de segmentos sem retreinar.

Cada linha recebe um `status`:
- `ok`: dentro do domínio do treino, segmento atribuído pelo centróide mais próximo (`predict`);
- qualquer outro texto: o motivo pelo qual a linha ficou sem segmento (ex.: preço inválido, coluna
  ausente na linha, categoria nunca vista). Essas linhas saem com `cluster` vazio: nenhum segmento é
  inventado para um anúncio fora do domínio em que o modelo foi ajustado.

`reviews_per_month` ausente é aceito (vira 0, como no treino), porque significa "sem avaliações".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import input_columns

OK = "ok"


def required_columns(representation: str = config.MAIN_REPRESENTATION) -> list[str]:
    """Colunas que o CSV de entrada precisa ter: o id e as colunas brutas da representação."""
    return [config.ID_COL] + input_columns(representation)


def check_columns(frame: pd.DataFrame, representation: str = config.MAIN_REPRESENTATION) -> None:
    missing = [column for column in required_columns(representation) if column not in frame.columns]
    if missing:
        raise ValueError(f"A entrada não tem as colunas obrigatórias: {missing}")


def known_categories(pipeline) -> dict[str, list[str]]:
    """Categorias vistas no treino, lidas do OneHotEncoder do pipeline ajustado."""
    preprocess = pipeline.named_steps["preprocess"]
    if "cat" not in preprocess.named_transformers_:
        return {}
    encoder = preprocess.named_transformers_["cat"]
    columns = next(cols for name, _, cols in preprocess.transformers_ if name == "cat")
    return {column: [str(c) for c in cats] for column, cats in zip(columns, encoder.categories_)}


def row_status(frame: pd.DataFrame, categories: dict[str, list[str]],
               representation: str = config.MAIN_REPRESENTATION) -> pd.Series:
    """Motivo de exclusão de cada linha (ou `ok`), aplicando as mesmas regras de domínio do treino."""
    spec = config.REPRESENTATIONS[representation]
    reasons = pd.Series("", index=frame.index, dtype=object)

    def flag(mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False).astype(bool)
        reasons[mask] = np.where(reasons[mask] == "", reason, reasons[mask] + "; " + reason)

    for column in spec["numeric"]:
        if column not in config.ZERO_FILL_COLUMNS:
            flag(frame[column].isna(), f"{column} ausente")
    if spec.get("distance"):
        for column in config.COORDINATE_COLUMNS:
            flag(frame[column].isna(), f"{column} ausente")
    flag(frame["price"] < config.MIN_VALID_PRICE, f"price < {config.MIN_VALID_PRICE}")
    flag(frame["minimum_nights"] < 1, "minimum_nights < 1")
    flag(~frame["availability_365"].between(0, 365) & frame["availability_365"].notna(), "availability_365 fora de [0, 365]")
    flag(frame["reviews_per_month"] < 0, "reviews_per_month < 0")
    flag(frame["calculated_host_listings_count"] < 1, "calculated_host_listings_count < 1")
    for column in spec["categorical"]:
        known = set(categories.get(column, []))
        values = frame[column].astype("string")
        flag(values.isna(), f"{column} ausente")
        flag(values.notna() & ~values.isin(known), f"{column} desconhecido")
    return reasons.where(reasons != "", OK)


def assign_segments(frame: pd.DataFrame, pipeline, segment_names: dict[int, str],
                    representation: str = config.MAIN_REPRESENTATION) -> pd.DataFrame:
    """Devolve id, cluster, segment_name e status, na ordem das linhas de entrada.

    Só chama `pipeline.predict` nas linhas com status `ok`; o pipeline nunca é reajustado.
    """
    check_columns(frame, representation)
    status = row_status(frame, known_categories(pipeline), representation)
    valid = status == OK
    clusters = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    if valid.any():
        clusters[valid] = pipeline.predict(frame.loc[valid, input_columns(representation)])
    return pd.DataFrame({
        config.ID_COL: frame[config.ID_COL].to_numpy(),
        "cluster": clusters.to_numpy(),
        "segment_name": clusters.map(lambda c: segment_names.get(int(c)) if pd.notna(c) else pd.NA).to_numpy(),
        "status": status.to_numpy(),
    })
