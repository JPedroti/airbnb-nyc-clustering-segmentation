"""Perfis: valores originais, ausência estrutural de reviews_per_month e agrupamento de leitura do DBSCAN."""
import numpy as np
import pandas as pd

from src import profiling


def test_profile_uses_original_values_and_zero_for_missing_reviews():
    frame = pd.DataFrame({
        "group": ["a", "a", "b"],
        "price": [100, 300, 50],
        "minimum_nights": [1, 30, 2],
        "availability_365": [0, 365, 10],
        "number_of_reviews": [0, 10, 3],
        "reviews_per_month": [np.nan, 2.0, 1.0],
        "calculated_host_listings_count": [1, 3, 1],
    })
    table = profiling.profile_table(frame, "group")
    assert table.loc["a", "price_median"] == 200  # sem log nem winsorização
    assert table.loc["a", "reviews_per_month_median"] == 1.0  # ausente conta como 0
    assert table.loc["a", "pct_minimum_nights_30plus"] == 50.0
    assert np.isclose(table["pct_listings"].sum(), 100.0)


def test_dbscan_groups_keep_big_clusters_and_separate_noise():
    labels = np.array([0] * 50 + [1] * 45 + [2] * 1 + [-1] * 4)
    groups, order = profiling.dbscan_groups(labels, min_share=0.01)
    assert order == ["cluster 0", "cluster 1", "cluster 2", profiling.NOISE_GROUP]
    groups, order = profiling.dbscan_groups(labels, min_share=0.05)
    assert order == ["cluster 0", "cluster 1", profiling.SMALL_GROUP, profiling.NOISE_GROUP]
    assert (groups[labels == -1] == profiling.NOISE_GROUP).all()
    assert (groups[labels == 2] == profiling.SMALL_GROUP).all()
