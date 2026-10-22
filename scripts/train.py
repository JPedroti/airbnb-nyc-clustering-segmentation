"""Treina e salva o pipeline K-Means escolhido (pré-processamento + modelo), a partir dos dados brutos.

    python scripts/train.py

Etapas:
1. Lê data/AB_NYC_2019.csv (com conferência de schema e SHA-256) e remove as linhas inválidas (price < 1).
2. Ajusta Pipeline(ColumnTransformer R2 com StandardScaler + KMeans K = config.FINAL_K) em todas as linhas
   válidas: os limites de winsorização, o scaler e o encoder são aprendidos no mesmo conjunto clusterizado.
3. Confere se o tamanho de cada cluster bate com a assinatura de configs/segment_names.json. Se a numeração
   mudou (ex.: outra versão de biblioteca ou de dados), o script para e pede a revisão dos nomes, em vez de
   gravar nomes trocados.
4. Grava em artifacts/:
   - kmeans_pipeline.joblib: o pipeline completo, usado por scripts/segment.py;
   - metadata.json: SHA-256 do modelo e dos dados, features, parâmetros, limites aprendidos, tamanhos,
     métricas internas, nomes dos segmentos e versões das bibliotecas;
   - segment_profiles.csv: perfil comercial de cada segmento, com nome e descrição;
   - train_assignments.csv: id e cluster de cada anúncio do treino.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config, data, evaluation, profiling  # noqa: E402
from src.artifacts import environment_info, save_model, write_json  # noqa: E402
from src.preprocessing import input_columns, make_kmeans_pipeline  # noqa: E402


def check_segment_names(names: dict, sizes: dict[int, int], n_rows: int) -> None:
    """Para o treino se os nomes de configs/segment_names.json não correspondem a este ajuste."""
    model = names["model"]
    expected_model = {"representation": config.MAIN_REPRESENTATION, "scaled": True, "k": config.FINAL_K, "n_rows": n_rows}
    signature = {int(s["cluster"]): int(s["n_listings"]) for s in names["segments"]}
    if model != expected_model or signature != sizes:
        raise SystemExit(
            "Os nomes em configs/segment_names.json não correspondem a este ajuste.\n"
            f"  esperado no arquivo: {model} | tamanhos {signature}\n"
            f"  obtido no treino   : {expected_model} | tamanhos {sizes}\n"
            "Revise os perfis (notebook 02, seção 8) e atualize os nomes e a assinatura antes de salvar."
        )


def main() -> None:
    raw = data.load_raw()
    listings, cleaning_log = data.clean_listings(raw)
    print(f"dados: {len(raw)} linhas; {int(cleaning_log['n_removed'].sum())} removidas; {len(listings)} usadas no ajuste")

    pipeline = make_kmeans_pipeline(config.FINAL_K, config.MAIN_REPRESENTATION, scale=True).fit(listings)
    labels = pipeline.named_steps["kmeans"].labels_
    sizes = {int(k): int(v) for k, v in evaluation.cluster_sizes(labels).items()}

    names = profiling.load_segment_names()
    check_segment_names(names, sizes, len(listings))
    name_map = profiling.segment_name_map(names)
    descriptions = {int(s["cluster"]): s["description"] for s in names["segments"]}

    X = pipeline.named_steps["preprocess"].transform(listings)
    metrics = evaluation.internal_metrics(X, labels, evaluation.fixed_sample_indices(len(listings)))
    metrics["inertia"] = float(pipeline.named_steps["kmeans"].inertia_)

    order = list(range(config.FINAL_K))
    profiles = profiling.profile_table(listings.assign(cluster=labels), "cluster", order)
    profiles.insert(0, "description", [descriptions[c] for c in order])
    profiles.insert(0, "segment_name", [name_map[c] for c in order])
    profiles.index.name = "cluster"

    model_sha = save_model(pipeline, config.MODEL_PATH)
    profiles.to_csv(config.SEGMENT_PROFILES_PATH)
    pd.DataFrame({config.ID_COL: listings[config.ID_COL], "cluster": labels}).to_csv(
        config.TRAIN_ASSIGNMENTS_PATH, index=False)

    preprocess = pipeline.named_steps["preprocess"]
    treatment = preprocess.named_transformers_["num"].named_steps["treat"]
    spec = config.REPRESENTATIONS[config.MAIN_REPRESENTATION]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": config.MODEL_PATH.relative_to(config.PROJECT_ROOT),
        "model_sha256": model_sha,
        "data": {
            "path": config.DATA_PATH.relative_to(config.PROJECT_ROOT),
            "sha256": data.file_sha256(config.DATA_PATH),
            "n_rows_raw": len(raw),
            "n_rows_removed": int(cleaning_log["n_removed"].sum()),
            "removal_rule": f"price < {config.MIN_VALID_PRICE}",
            "n_rows_fitted": len(listings),
        },
        "features": {
            "representation": config.MAIN_REPRESENTATION,
            "numeric": spec["numeric"],
            "categorical": spec["categorical"],
            "input_columns": input_columns(config.MAIN_REPRESENTATION),
            "id_column": config.ID_COL,
            "transformed_columns": list(preprocess.get_feature_names_out()),
            "category_orders": {c: config.CATEGORY_ORDERS[c] for c in spec["categorical"]},
        },
        "preprocessing": {
            "zero_fill": config.ZERO_FILL_COLUMNS,
            "upper_caps_learned_in_fit": treatment.caps_,
            "winsor_quantile": config.WINSOR_QUANTILE,
            "log1p": [c for c in config.LOG1P_COLUMNS if c in spec["numeric"]],
            "scaler": "StandardScaler (só numéricas)",
            "encoder": "OneHotEncoder(handle_unknown='ignore'); segment.py recusa categorias desconhecidas",
        },
        "kmeans": {"n_clusters": config.FINAL_K, **config.KMEANS_PARAMS,
                   "n_iter": int(pipeline.named_steps["kmeans"].n_iter_)},
        "cluster_sizes": sizes,
        "internal_metrics": {
            **{k: metrics[k] for k in ("inertia", "silhouette", "davies_bouldin", "calinski_harabasz")},
            "silhouette_sample": f"{metrics['silhouette_n']} linhas, semente {config.RANDOM_STATE}",
            "space": "espaço transformado R2 (StandardScaler), não o PCA",
        },
        "segments": [{"cluster": c, "name": name_map[c], "description": descriptions[c], "n_listings": sizes[c]}
                     for c in order],
        "inference": "predict = centróide mais próximo no espaço transformado; nenhum ajuste em dados novos.",
        "environment": environment_info(),
    }
    write_json(config.METADATA_PATH, metadata)

    print(f"modelo salvo: {config.MODEL_PATH.relative_to(config.PROJECT_ROOT)} (sha256 {model_sha[:12]}...)")
    print(f"Silhouette {metrics['silhouette']:.3f} | Davies-Bouldin {metrics['davies_bouldin']:.3f} | "
          f"Calinski-Harabasz {metrics['calinski_harabasz']:,.0f} | inertia {metrics['inertia']:,.0f}")
    for c in order:
        print(f"  {c} · {name_map[c]}: {sizes[c]} anúncios ({sizes[c] / len(listings):.1%})")


if __name__ == "__main__":
    main()
