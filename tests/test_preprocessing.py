"""Pré-processamento: tratamento das numéricas, limites aprendidos no fit, encoding e colunas usadas."""
import numpy as np
import pandas as pd

from src import config, data
from src.preprocessing import NumericTreatment, feature_block, input_columns, make_preprocessor


def test_reviews_per_month_missing_becomes_zero_before_log():
    frame = pd.DataFrame({"reviews_per_month": [np.nan, 1.0, 3.0]})
    out = NumericTreatment(winsor_columns=(), fixed_caps=()).fit(frame).transform(frame)
    assert out[0, 0] == 0.0  # log1p(0)
    assert np.isclose(out[1, 0], np.log1p(1.0))


def test_caps_are_learned_in_fit_and_reused_on_new_data():
    train = pd.DataFrame({"price": np.arange(1, 1001, dtype=float)})
    treatment = NumericTreatment(winsor_columns=("price",), winsor_quantile=0.9, fixed_caps=(), log_columns=())
    treatment.fit(train)
    cap = treatment.caps_["price"]
    assert np.isclose(cap, train["price"].quantile(0.9))
    new = pd.DataFrame({"price": [50.0, 50_000.0]})
    out = treatment.transform(new)
    assert out[0, 0] == 50.0
    assert out[1, 0] == cap  # o valor novo usa o limite do treino, não um limite recalculado


def test_minimum_nights_fixed_cap():
    frame = pd.DataFrame({"minimum_nights": [1.0, 365.0, 1250.0]})
    out = NumericTreatment(winsor_columns=(), log_columns=()).fit(frame).transform(frame)
    assert out[:, 0].tolist() == [1.0, 365.0, float(config.MINIMUM_NIGHTS_CAP)]


def test_identifiers_and_coordinates_never_enter_the_feature_space(synthetic_listings):
    prep = make_preprocessor("R3", scale=True).fit(synthetic_listings)
    blocks = {feature_block(name) for name in prep.get_feature_names_out()}
    assert blocks == set(config.FEATURE_NUMERIC) | {"room_type", "neighbourhood_group"}
    assert set(input_columns("R2")) == set(config.FEATURE_NUMERIC) | {"room_type"}


def test_unscaled_scenario_keeps_original_scale_and_scaled_has_unit_variance(synthetic_listings):
    frame, _ = data.clean_listings(synthetic_listings)
    x_a = make_preprocessor("R2", scale=False).fit_transform(frame)
    x_b = make_preprocessor("R2", scale=True).fit_transform(frame)
    availability = config.FEATURE_NUMERIC.index("availability_365")
    np.testing.assert_allclose(x_a[:, availability], frame["availability_365"].to_numpy())
    n_numeric = len(config.FEATURE_NUMERIC)
    np.testing.assert_allclose(x_b[:, :n_numeric].std(axis=0), 1.0, atol=1e-9)
    # As dummies do OneHot ficam em 0/1 nos dois cenários.
    assert set(np.unique(x_b[:, n_numeric:])) <= {0.0, 1.0}


def test_unknown_category_is_encoded_as_all_zero(synthetic_listings):
    prep = make_preprocessor("R2", scale=True).fit(synthetic_listings)
    new = synthetic_listings.head(1).assign(room_type="Hotel room")
    dummies = prep.transform(new)[0, len(config.FEATURE_NUMERIC):]
    assert dummies.sum() == 0.0


def test_clean_listings_removes_only_invalid_prices(synthetic_listings):
    frame = synthetic_listings.copy()
    frame.loc[:2, "price"] = 0
    clean, log = data.clean_listings(frame)
    assert len(clean) == len(frame) - 3
    assert int(log["n_removed"].iloc[0]) == 3
    assert (clean["price"] >= config.MIN_VALID_PRICE).all()
