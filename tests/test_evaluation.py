"""Métricas internas: regras de validade, ruído do DBSCAN, amostra fixa e decomposição da distância."""
import numpy as np

from src import evaluation


def _two_blobs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 0.3, (n, 2)), rng.normal(5, 0.3, (n, 2))])
    labels = np.repeat([0, 1], n)
    return X, labels


def test_metrics_are_valid_for_two_groups():
    X, labels = _two_blobs()
    result = evaluation.internal_metrics(X, labels)
    assert result["n_clusters"] == 2
    assert 0.8 < result["silhouette"] <= 1.0
    assert result["silhouette_note"] is None and result["davies_bouldin_note"] is None


def test_single_group_is_not_applicable():
    X, _ = _two_blobs()
    result = evaluation.internal_metrics(X, np.zeros(len(X), dtype=int))
    assert np.isnan(result["silhouette"]) and np.isnan(result["davies_bouldin"]) and np.isnan(result["calinski_harabasz"])
    assert "não aplicável" in result["davies_bouldin_note"]


def test_one_group_per_point_is_not_applicable():
    X, _ = _two_blobs(n=5)
    result = evaluation.internal_metrics(X, np.arange(len(X)))
    assert np.isnan(result["calinski_harabasz"])
    assert "um grupo por ponto" in result["calinski_harabasz_note"]


def test_noise_is_excluded_and_coverage_reported():
    X, labels = _two_blobs()
    labels = labels.copy()
    labels[:30] = evaluation.NOISE_LABEL
    result = evaluation.internal_metrics(X, labels, exclude_noise=True)
    assert result["coverage"] == (len(labels) - 30) / len(labels)
    assert result["n_clusters"] == 2


def test_only_noise_and_one_cluster_is_not_applicable():
    X, labels = _two_blobs()
    labels = np.where(labels == 0, evaluation.NOISE_LABEL, 0)
    result = evaluation.internal_metrics(X, labels, exclude_noise=True)
    assert result["n_clusters"] == 1
    assert np.isnan(result["silhouette"])


def test_fixed_sample_indices_are_reproducible_unique_and_sorted():
    a = evaluation.fixed_sample_indices(1_000, size=100, seed=42)
    b = evaluation.fixed_sample_indices(1_000, size=100, seed=42)
    np.testing.assert_array_equal(a, b)
    assert len(np.unique(a)) == 100 and np.all(np.diff(a) > 0)


def test_variance_contribution_matches_mean_pairwise_squared_distance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3)) * np.array([1.0, 10.0, 0.1])
    table = evaluation.variance_contribution(X, ["a", "b", "c"])
    assert np.isclose(table["pct_of_squared_distance"].sum(), 100.0)
    # Média sobre todos os pares ordenados (i, j) de (x_i - x_j)^2, coluna a coluna, é 2 x variância.
    pairwise = ((X[:, None, :] - X[None, :, :]) ** 2).mean(axis=(0, 1))
    np.testing.assert_allclose(pairwise, 2 * table["variance"].to_numpy())


def test_dbscan_grid_counts_noise_and_marks_degenerate_solutions():
    X, _ = _two_blobs()
    grid, labels = evaluation.dbscan_grid(X, eps_values=[0.001, 1.0, 50.0], min_samples_values=[5])
    tiny, good, huge = grid.to_dict("records")
    # eps minúsculo: tudo é ruído; eps adequado: dois grupos; eps enorme: um grupo só.
    assert tiny["pct_noise"] == 100.0 and tiny["n_clusters"] == 0 and np.isnan(tiny["silhouette"])
    assert good["n_clusters"] == 2 and good["coverage"] > 0.9 and good["silhouette"] > 0.8
    assert huge["n_clusters"] == 1 and "não aplicável" in huge["metrics_note"]
    for row in (tiny, good, huge):
        assert row["n_core"] + row["n_border"] + row["n_noise"] == len(X)
    assert set(labels) == {(0.001, 5), (1.0, 5), (50.0, 5)}


def test_kth_neighbor_distances_are_sorted_and_one_per_point():
    X, _ = _two_blobs()
    distances = evaluation.kth_neighbor_distances(X, k=4)
    assert len(distances) == len(X) and np.all(np.diff(distances) >= 0)


def test_eta_squared_limits():
    values = np.array([1.0, 1.0, 5.0, 5.0])
    assert evaluation.eta_squared(values, ["a", "a", "b", "b"]) == 1.0  # grupos explicam tudo
    assert np.isclose(evaluation.eta_squared(values, ["a", "b", "a", "b"]), 0.0)  # grupos não separam nada


def test_distance_decomposition_adds_up_to_squared_distance():
    a, b = np.array([0.0, 1.0, 2.0]), np.array([3.0, 1.0, 0.0])
    table = evaluation.distance_decomposition(a, b, ["x", "y", "z"])
    assert np.isclose(table["squared_difference"].sum(), np.sum((a - b) ** 2))
    assert np.isclose(table["pct_of_squared_distance"].sum(), 100.0)
