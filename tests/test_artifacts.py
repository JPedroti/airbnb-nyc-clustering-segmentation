"""Persistência do pipeline: conferência do SHA-256, recarga em processo novo e reprodução dos rótulos do treino."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import config, data
from src.artifacts import load_model, save_model, write_json
from src.segmentation import OK, assign_segments


def _save_synthetic(tmp_path, pipeline):
    model_path, metadata_path = tmp_path / "pipeline.joblib", tmp_path / "metadata.json"
    sha = save_model(pipeline, model_path)
    write_json(metadata_path, {
        "model_sha256": sha,
        "features": {"representation": config.MAIN_REPRESENTATION},
        "segments": [{"cluster": c, "name": f"segmento {c}"} for c in range(3)],
    })
    return model_path, metadata_path


def test_model_roundtrip_and_tampered_file_is_refused(tmp_path, fitted_pipeline):
    pipeline, frame = fitted_pipeline
    model_path, metadata_path = _save_synthetic(tmp_path, pipeline)
    loaded, _ = load_model(model_path, metadata_path)
    np.testing.assert_array_equal(loaded.predict(frame), pipeline.predict(frame))
    model_path.write_bytes(model_path.read_bytes() + b"alterado")
    with pytest.raises(RuntimeError, match="não confere"):
        load_model(model_path, metadata_path)


def test_segment_script_in_a_new_process_matches_in_process_predict(tmp_path, fitted_pipeline):
    pipeline, frame = fitted_pipeline
    model_path, metadata_path = _save_synthetic(tmp_path, pipeline)
    new = frame.sample(50, random_state=3)
    new.to_csv(tmp_path / "novos.csv", index=False)
    completed = subprocess.run(
        [sys.executable, str(config.PROJECT_ROOT / "scripts" / "segment.py"), "--input", str(tmp_path / "novos.csv"),
         "--output", str(tmp_path / "saida.csv"), "--model", str(model_path), "--metadata", str(metadata_path)],
        capture_output=True, text=True, encoding="utf-8", cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    out = pd.read_csv(tmp_path / "saida.csv")
    assert out["id"].tolist() == new["id"].tolist()
    np.testing.assert_array_equal(out["cluster"].to_numpy(), pipeline.predict(new))


@pytest.mark.skipif(
    not (config.MODEL_PATH.exists() and config.METADATA_PATH.exists() and config.DATA_PATH.exists()
         and config.TRAIN_ASSIGNMENTS_PATH.exists()),
    reason="requer data/AB_NYC_2019.csv e os artefatos de scripts/train.py",
)
def test_saved_pipeline_reproduces_training_labels():
    pipeline, metadata = load_model()
    raw = data.load_raw()
    sample = raw[raw["price"] >= config.MIN_VALID_PRICE].sample(2_000, random_state=0)
    names = {int(s["cluster"]): s["name"] for s in metadata["segments"]}
    out = assign_segments(sample, pipeline, names)
    train = pd.read_csv(config.TRAIN_ASSIGNMENTS_PATH).set_index(config.ID_COL)["cluster"]
    assert (out["status"] == OK).all()
    np.testing.assert_array_equal(out["cluster"].to_numpy(dtype=int), train.loc[sample[config.ID_COL]].to_numpy())
