"""Persistência do pipeline K-Means e dos artefatos JSON.

O pipeline é congelado registrando o SHA-256 do arquivo em `metadata.json` no treino. Quem consome o
modelo (scripts/segment.py, testes) carrega por `load_model`, que recusa um arquivo cujo hash seja
diferente do registrado.
"""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.data import file_sha256


def _to_builtin(obj):
    """Converte tipos do numpy/pathlib em tipos nativos serializáveis em JSON (NaN vira null)."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model(model, path: Path = config.MODEL_PATH) -> str:
    """Grava o pipeline com joblib e devolve o SHA-256 do arquivo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return file_sha256(path)


def load_model(model_path: Path = config.MODEL_PATH, metadata_path: Path = config.METADATA_PATH):
    """Carrega o pipeline depois de conferir o SHA-256 com o registrado em metadata.json."""
    metadata = read_json(metadata_path)
    actual = file_sha256(model_path)
    expected = metadata["model_sha256"]
    if actual != expected:
        raise RuntimeError(f"{model_path} não confere com o modelo registrado (sha256 {actual} != {expected}).")
    return joblib.load(model_path), metadata


def environment_info() -> dict:
    """Versões do ambiente que gerou os artefatos."""
    import matplotlib
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
    }
