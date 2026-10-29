"""Distância até a Times Square e representação experimental R4."""
import numpy as np
import pandas as pd

from src import config, data
from src.geo import distance_to_times_square_km, haversine_km
from src.preprocessing import DistanceToReference, feature_block, input_columns, make_preprocessor
from src.segmentation import OK, row_status


def test_haversine_zero_and_one_degree_of_latitude():
    assert haversine_km(40.0, -74.0, 40.0, -74.0) == 0.0
    # Um grau de latitude ao longo de um meridiano = 2 * pi * R / 360.
    np.testing.assert_allclose(haversine_km(40.0, -74.0, 41.0, -74.0), np.pi * config.EARTH_RADIUS_KM / 180)


def test_one_degree_of_longitude_is_shorter_than_latitude_in_nyc():
    lon_km = haversine_km(40.73, -74.0, 40.73, -73.0)
    lat_km = haversine_km(40.0, -74.0, 41.0, -74.0)
    assert 80 < lon_km < 90 < lat_km  # cerca de 84 km contra 111 km


def test_distance_to_times_square_keeps_index_and_is_zero_at_the_reference():
    ts = config.TIMES_SQUARE
    frame = pd.DataFrame({"latitude": [ts["lat"], 40.7], "longitude": [ts["lon"], -73.95]}, index=[10, 20])
    distance = distance_to_times_square_km(frame)
    assert distance.index.tolist() == [10, 20] and distance.name == config.DISTANCE_FEATURE
    assert distance.loc[10] == 0.0 and distance.loc[20] > 0


def test_distance_transformer_outputs_one_named_column():
    frame = pd.DataFrame({"latitude": [40.7, 40.8, 40.6], "longitude": [-73.9, -73.95, -74.0]})
    transformer = DistanceToReference().fit(frame)
    out = transformer.transform(frame)
    assert out.shape == (3, 1)
    np.testing.assert_allclose(out[:, 0], distance_to_times_square_km(frame).to_numpy())
    assert transformer.get_feature_names_out().tolist() == [config.DISTANCE_FEATURE]


def test_r4_adds_only_the_distance_and_r2_is_unchanged(synthetic_listings):
    frame, _ = data.clean_listings(synthetic_listings)
    blocks_r2 = {feature_block(n) for n in make_preprocessor("R2").fit(frame).get_feature_names_out()}
    prep_r4 = make_preprocessor("R4").fit(frame)
    names_r4 = list(prep_r4.get_feature_names_out())
    assert {feature_block(n) for n in names_r4} == blocks_r2 | {config.DISTANCE_FEATURE}
    assert "latitude" not in " ".join(names_r4) and "longitude" not in " ".join(names_r4)
    distance_column = prep_r4.transform(frame)[:, names_r4.index(f"geo__{config.DISTANCE_FEATURE}")]
    np.testing.assert_allclose(distance_column.std(), 1.0, atol=1e-9)  # mesma escala das numéricas
    assert set(config.COORDINATE_COLUMNS) <= set(input_columns("R4"))
    assert not set(config.COORDINATE_COLUMNS) & set(input_columns("R2"))


def test_r4_rows_without_coordinates_are_flagged(synthetic_listings):
    frame, _ = data.clean_listings(synthetic_listings)
    frame = frame.head(3).copy()
    frame.loc[frame.index[0], "latitude"] = np.nan
    status = row_status(frame, {"room_type": config.ROOM_TYPE_ORDER}, representation="R4")
    assert "latitude ausente" in status.iloc[0] and (status.iloc[1:] == OK).all()
