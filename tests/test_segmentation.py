"""Contrato da segmentação de anúncios novos: ordem, ids, linhas inválidas sem segmento e nenhum reajuste."""
import numpy as np
import pandas as pd
import pytest

from src.segmentation import OK, assign_segments

NAMES = {0: "segmento a", 1: "segmento b", 2: "segmento c"}


def test_output_keeps_order_and_ids_and_matches_predict(fitted_pipeline):
    pipeline, frame = fitted_pipeline
    shuffled = frame.sample(frac=1, random_state=1)
    out = assign_segments(shuffled, pipeline, NAMES)
    assert out.columns.tolist() == ["id", "cluster", "segment_name", "status"]
    assert out["id"].tolist() == shuffled["id"].tolist()
    assert (out["status"] == OK).all()
    np.testing.assert_array_equal(out["cluster"].to_numpy(dtype=int), pipeline.predict(shuffled))
    assert out["segment_name"].tolist() == [NAMES[c] for c in out["cluster"]]


def test_rows_outside_the_training_domain_get_no_segment_and_a_reason(fitted_pipeline):
    pipeline, frame = fitted_pipeline
    new = frame.head(5).copy().reset_index(drop=True)
    new.loc[0, "price"] = 0
    new.loc[1, "price"] = np.nan
    new.loc[2, "room_type"] = "Hotel room"
    new.loc[3, "reviews_per_month"] = np.nan  # permitido: anúncio sem avaliações
    new.loc[4, "availability_365"] = 400
    out = assign_segments(new, pipeline, NAMES)
    assert out.loc[3, "status"] == OK and pd.notna(out.loc[3, "cluster"])
    assert "price < 1" in out.loc[0, "status"]
    assert "price ausente" in out.loc[1, "status"]
    assert "room_type desconhecido" in out.loc[2, "status"]
    assert "availability_365 fora de [0, 365]" in out.loc[4, "status"]
    assert out.loc[[0, 1, 2, 4], "cluster"].isna().all()
    assert out.loc[[0, 1, 2, 4], "segment_name"].isna().all()


def test_missing_required_column_is_an_error(fitted_pipeline):
    pipeline, frame = fitted_pipeline
    with pytest.raises(ValueError, match="room_type"):
        assign_segments(frame.drop(columns="room_type"), pipeline, NAMES)


def test_segmentation_never_refits(fitted_pipeline):
    pipeline, frame = fitted_pipeline
    kmeans = pipeline.named_steps["kmeans"]
    treatment = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["treat"]
    centers, caps = kmeans.cluster_centers_.copy(), dict(treatment.caps_)
    # Uma distribuição bem diferente não pode alterar centróides nem limites aprendidos no treino.
    assign_segments(frame.assign(price=frame["price"] * 10), pipeline, NAMES)
    np.testing.assert_array_equal(kmeans.cluster_centers_, centers)
    assert treatment.caps_ == caps
