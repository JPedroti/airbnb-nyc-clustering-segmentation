"""Métricas internas de clusterização com regras explícitas de validade, varredura de K e diagnósticos de distância.

Regras (seção 6 do enunciado):
- Silhouette, Davies-Bouldin e Calinski-Harabasz exigem de 2 a n-1 grupos na população avaliada. Fora
  disso, a métrica é NaN e a coluna `<métrica>_note` explica o motivo; nenhum valor é inventado.
- As métricas são calculadas no MESMO espaço transformado em que o modelo foi ajustado, nunca no PCA 2D.
- O ruído do DBSCAN (rótulo -1) não é tratado como cluster: as métricas usam só os pontos não ruído, e a
  cobertura avaliada é reportada junto.
- O Silhouette usa uma amostra fixa de índices, a mesma para todas as comparações.
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from src import config
from src.preprocessing import feature_block

NOISE_LABEL = -1


def fixed_sample_indices(n_rows: int, size: int = config.SILHOUETTE_SAMPLE_SIZE,
                         seed: int = config.RANDOM_STATE) -> np.ndarray:
    """Posições sorteadas sem reposição, em ordem crescente (reprodutíveis para o mesmo n e semente)."""
    if size >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=size, replace=False))


def _validity_note(labels: np.ndarray) -> str | None:
    """Motivo de não aplicabilidade, ou None se houver de 2 a n-1 grupos."""
    n_points = len(labels)
    n_groups = len(np.unique(labels)) if n_points else 0
    if n_points < 3:
        return "não aplicável: menos de 3 pontos avaliados"
    if n_groups < 2:
        return "não aplicável: menos de 2 grupos válidos"
    if n_groups > n_points - 1:
        return "não aplicável: um grupo por ponto"
    return None


def internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    silhouette_idx: np.ndarray | None = None,
    exclude_noise: bool = False,
) -> dict:
    """Silhouette (na amostra fixa), Davies-Bouldin e Calinski-Harabasz, com validade e cobertura.

    `silhouette_idx` são posições em `X`. Com `exclude_noise=True`, os pontos de rótulo -1 saem de
    todas as métricas (inclusive da amostra do Silhouette), e `coverage` informa a fração avaliada.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    keep = labels != NOISE_LABEL if exclude_noise else np.ones(len(labels), dtype=bool)
    X_eval, labels_eval = X[keep], labels[keep]
    result = {
        "n_clusters": int(len(np.unique(labels_eval))) if keep.any() else 0,
        "n_evaluated": int(keep.sum()),
        "coverage": float(keep.mean()) if len(labels) else 0.0,
    }

    note = _validity_note(labels_eval)
    for name, scorer in [("davies_bouldin", davies_bouldin_score), ("calinski_harabasz", calinski_harabasz_score)]:
        result[name] = np.nan if note else float(scorer(X_eval, labels_eval))
        result[f"{name}_note"] = note

    if silhouette_idx is None:
        silhouette_idx = np.arange(len(labels))
    sample_mask = np.zeros(len(labels), dtype=bool)
    sample_mask[silhouette_idx] = True
    sample_mask &= keep
    sample_labels = labels[sample_mask]
    sample_note = _validity_note(sample_labels)
    result["silhouette"] = np.nan if sample_note else float(silhouette_score(X[sample_mask], sample_labels))
    result["silhouette_note"] = sample_note if sample_note is None else f"{sample_note} (amostra)"
    result["silhouette_n"] = int(sample_mask.sum())
    return result


def cluster_sizes(labels: np.ndarray) -> pd.Series:
    """Tamanho de cada rótulo, em ordem de rótulo."""
    return pd.Series(labels).value_counts().sort_index()


def kmeans_sweep(
    X: np.ndarray,
    k_values: Sequence[int] = config.K_RANGE,
    silhouette_idx: np.ndarray | None = None,
    params: dict | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Ajusta K-Means para cada K com o mesmo protocolo e registra inertia, métricas e tamanhos.

    Devolve (tabela por K, rótulos por K).
    """
    params = dict(config.KMEANS_PARAMS if params is None else params)
    rows, labels_by_k = [], {}
    for k in k_values:
        model = KMeans(n_clusters=k, **params).fit(X)
        labels = model.labels_
        sizes = cluster_sizes(labels)
        shares = sizes / sizes.sum()
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            **{key: value for key, value in internal_metrics(X, labels, silhouette_idx).items()
               if key in ("silhouette", "davies_bouldin", "calinski_harabasz")},
            "min_cluster_pct": float(shares.min() * 100),
            "max_cluster_pct": float(shares.max() * 100),
            "cluster_pcts": " / ".join(f"{s * 100:.1f}" for s in shares.sort_values(ascending=False)),
            "n_iter": int(model.n_iter_),
        })
        labels_by_k[k] = labels
    return pd.DataFrame(rows).set_index("k"), labels_by_k


def seed_stability(X: np.ndarray, k: int, seeds: Sequence[int] = config.STABILITY_SEEDS,
                   params: dict | None = None) -> dict:
    """ARI entre as partições obtidas com sementes diferentes (1 = partições idênticas)."""
    params = dict(config.KMEANS_PARAMS if params is None else params)
    partitions = []
    for seed in seeds:
        params["random_state"] = seed
        partitions.append(KMeans(n_clusters=k, **params).fit_predict(X))
    scores = [adjusted_rand_score(a, b) for a, b in combinations(partitions, 2)]
    return {"k": k, "ari_mean": float(np.mean(scores)), "ari_min": float(np.min(scores))}


def variance_contribution(X: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    """Participação de cada coluna na distância Euclidiana quadrática média entre pares de pontos.

    Para dois pontos sorteados ao acaso, E[(x_i - x_j)^2] = 2 Var(x) em cada coluna, então a fatia de
    cada coluna na distância quadrática esperada é Var(coluna) / soma das variâncias. Essa conta é exata
    e não exige calcular nenhuma matriz de distâncias.
    """
    variances = np.asarray(X).var(axis=0)
    table = pd.DataFrame({
        "feature": list(feature_names),
        "block": [feature_block(name) for name in feature_names],
        "variance": variances,
    })
    table["pct_of_squared_distance"] = table["variance"] / table["variance"].sum() * 100
    return table


def distance_decomposition(x_a: np.ndarray, x_b: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    """Contribuição de cada coluna para a distância Euclidiana quadrática entre dois pontos."""
    squared = (np.asarray(x_a, dtype=float) - np.asarray(x_b, dtype=float)) ** 2
    table = pd.DataFrame({"feature": list(feature_names), "squared_difference": squared})
    total = squared.sum()
    table["pct_of_squared_distance"] = squared / total * 100 if total > 0 else 0.0
    return table


# --------------------------------------------------------------------------------------------------
# DBSCAN
# --------------------------------------------------------------------------------------------------
def kth_neighbor_distances(X: np.ndarray, k: int) -> np.ndarray:
    """Distância de cada ponto ao seu k-ésimo vizinho (contando o próprio ponto), em ordem crescente.

    É a curva usada para escolher eps: com min_samples = k, um ponto é core quando essa distância é <= eps.
    """
    distances, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    return np.sort(distances[:, -1])


def dbscan_point_types(model: DBSCAN) -> dict:
    """Contagem de pontos core, border e noise de um DBSCAN ajustado."""
    labels = model.labels_
    is_core = np.zeros(len(labels), dtype=bool)
    is_core[model.core_sample_indices_] = True
    noise = labels == NOISE_LABEL
    return {"n_core": int(is_core.sum()), "n_border": int((~is_core & ~noise).sum()), "n_noise": int(noise.sum())}


def dbscan_grid(
    X: np.ndarray,
    eps_values: Sequence[float] = config.DBSCAN_EPS_GRID,
    min_samples_values: Sequence[int] = config.DBSCAN_MIN_SAMPLES_GRID,
    silhouette_idx: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[tuple[float, int], np.ndarray]]:
    """Ajusta DBSCAN em cada par (eps, min_samples) e registra grupos, ruído e métricas sem o ruído.

    As métricas usam só os pontos não ruído; `coverage` é a fração da amostra avaliada. Com menos de
    2 clusters não ruído, as métricas ficam NaN com o motivo nas colunas `*_note`.
    """
    rows, labels_by_params = [], {}
    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = model.labels_
            clustered = labels[labels != NOISE_LABEL]
            sizes = pd.Series(clustered).value_counts()
            metrics = internal_metrics(X, labels, silhouette_idx, exclude_noise=True)
            rows.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": int(sizes.size),
                "n_clusters_ge_1pct": int((sizes >= 0.01 * len(labels)).sum()),
                **dbscan_point_types(model),
                "pct_noise": float((labels == NOISE_LABEL).mean() * 100),
                "largest_cluster_pct": float(sizes.max() / len(labels) * 100) if sizes.size else np.nan,
                "coverage": metrics["coverage"],
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "metrics_note": metrics["davies_bouldin_note"],
            })
            labels_by_params[(eps, min_samples)] = labels
    return pd.DataFrame(rows), labels_by_params
