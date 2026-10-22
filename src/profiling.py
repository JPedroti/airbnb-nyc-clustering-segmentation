"""Tabelas de perfil comercial e de composição por grupo (categoria do dataset ou segmento).

As estatísticas usam os valores originais dos anúncios (sem log nem winsorização), para que os perfis
sejam lidos em unidades de negócio: dólares por noite, noites e dias declarados disponíveis.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def _q1(series: pd.Series) -> float:
    return series.quantile(0.25)


def _q3(series: pd.Series) -> float:
    return series.quantile(0.75)


def profile_table(frame: pd.DataFrame, group: str, order: Sequence | None = None) -> pd.DataFrame:
    """Uma linha por grupo: tamanho, participação e medianas/quartis das variáveis comerciais.

    `reviews_per_month` ausente é tratado como 0, porque a ausência ocorre exatamente nos anúncios
    sem nenhuma avaliação (verificado na EDA).
    """
    work = frame.assign(reviews_per_month=frame["reviews_per_month"].fillna(0))
    grouped = work.groupby(group, observed=True)
    table = pd.DataFrame({
        "n_listings": grouped.size(),
        "pct_listings": grouped.size() / len(work) * 100,
        "price_q1": grouped["price"].agg(_q1),
        "price_median": grouped["price"].median(),
        "price_q3": grouped["price"].agg(_q3),
        "minimum_nights_median": grouped["minimum_nights"].median(),
        "pct_minimum_nights_30plus": grouped["minimum_nights"].agg(lambda s: (s >= 30).mean() * 100),
        "availability_365_q1": grouped["availability_365"].agg(_q1),
        "availability_365_median": grouped["availability_365"].median(),
        "availability_365_q3": grouped["availability_365"].agg(_q3),
        "pct_availability_zero": grouped["availability_365"].agg(lambda s: (s == 0).mean() * 100),
        "number_of_reviews_median": grouped["number_of_reviews"].median(),
        "reviews_per_month_median": grouped["reviews_per_month"].median(),
        "host_listings_median": grouped["calculated_host_listings_count"].median(),
    })
    if order is not None:
        table = table.reindex([g for g in order if g in table.index])
    return table


def composition_table(
    frame: pd.DataFrame,
    group: str,
    category: str,
    category_order: Sequence[str],
    group_order: Sequence | None = None,
    overall_label: str = "Base geral",
) -> pd.DataFrame:
    """Percentual de cada categoria dentro de cada grupo, com a composição da base inteira na última linha."""
    shares = pd.crosstab(frame[group], frame[category], normalize="index") * 100
    shares = shares.reindex(columns=list(category_order), fill_value=0.0)
    if group_order is not None:
        shares = shares.reindex([g for g in group_order if g in shares.index])
    overall = frame[category].value_counts(normalize=True).reindex(list(category_order), fill_value=0.0) * 100
    shares.loc[overall_label] = overall.to_numpy()
    return shares


def sample_representativeness(
    population: pd.DataFrame,
    sample: pd.DataFrame,
    numeric: Sequence[str] = config.FEATURE_NUMERIC,
    categorical: dict[str, Sequence[str]] = config.CATEGORY_ORDERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara amostra e base: (participação das categorias, quartis das numéricas).

    As diferenças vêm em pontos percentuais (categóricas) e em % do valor da base (quartis).
    """
    rows = []
    for column, order in categorical.items():
        pop = population[column].value_counts(normalize=True).reindex(list(order), fill_value=0) * 100
        smp = sample[column].value_counts(normalize=True).reindex(list(order), fill_value=0) * 100
        for category in order:
            rows.append({"column": column, "category": category, "pct_base": pop[category],
                         "pct_sample": smp[category], "diff_pp": smp[category] - pop[category]})
    categories = pd.DataFrame(rows).set_index(["column", "category"])

    rows = []
    for column in numeric:
        pop, smp = population[column], sample[column]
        if column in config.ZERO_FILL_COLUMNS:
            pop, smp = pop.fillna(0), smp.fillna(0)
        for q in (0.25, 0.50, 0.75):
            base_value, sample_value = pop.quantile(q), smp.quantile(q)
            rows.append({"column": column, "quantile": q, "base": base_value, "sample": sample_value,
                         "diff_pct": (sample_value - base_value) / base_value * 100 if base_value else 0.0})
    quantiles = pd.DataFrame(rows).set_index(["column", "quantile"])
    return categories, quantiles


NOISE_GROUP = "ruído (-1)"
SMALL_GROUP = "clusters < 1%"


def dbscan_groups(labels, min_share: float = 0.01) -> tuple[np.ndarray, list[str]]:
    """Rótulos de leitura do DBSCAN e sua ordem de exibição.

    Clusters com pelo menos `min_share` da amostra mantêm o id ("cluster 0"); os menores são agrupados em
    "clusters < 1%"; o ruído (-1) fica em grupo próprio, sem identidade de segmento.
    """
    labels = np.asarray(labels)
    sizes = pd.Series(labels[labels != -1]).value_counts()
    big = sorted(int(c) for c in sizes[sizes >= min_share * len(labels)].index)
    groups = np.where(labels == -1, NOISE_GROUP,
                      np.where(np.isin(labels, big), np.char.add("cluster ", labels.astype(str)), SMALL_GROUP))
    order = [f"cluster {c}" for c in big] + [g for g in (SMALL_GROUP, NOISE_GROUP) if g in set(groups)]
    return groups, order


# --------------------------------------------------------------------------------------------------
# Nomes dos segmentos
# --------------------------------------------------------------------------------------------------
def load_segment_names(path: Path = config.SEGMENT_NAMES_PATH) -> dict:
    """Lê configs/segment_names.json: nomes, descrições e a assinatura de tamanhos do K-Means final."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def segment_name_map(names: dict) -> dict[int, str]:
    """{id do cluster: nome curto}."""
    return {int(segment["cluster"]): segment["name"] for segment in names["segments"]}
