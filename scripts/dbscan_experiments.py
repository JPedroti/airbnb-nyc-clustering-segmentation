"""Reproduz os experimentos de DBSCAN do notebook 02 (seção 7) e grava rótulos, parâmetros e diagnóstico de ruído.

    python scripts/dbscan_experiments.py

O DBSCAN do scikit-learn não tem `predict`: os rótulos gravados valem só para os anúncios da amostra usada
no ajuste. Este script não oferece inferência para anúncios novos e não simula uma.

Protocolo (o mesmo do notebook):
1. Base limpa (price >= 1) -> pré-processador R2 com StandardScaler ajustado na base inteira, o mesmo espaço
   do K-Means.
2. Amostra fixa de config.DBSCAN_SAMPLE_SIZE anúncios (semente config.RANDOM_STATE).
3. Grade eps x min_samples em R2 e, como experimento complementar, no espaço só numérico R1. Métricas só
   nos pontos não ruído, com a cobertura reportada; "não aplicável" quando há menos de 2 clusters.
4. Configuração de referência (config.DBSCAN_REFERENCE): rótulos com o tipo de ponto (core/border/noise),
   perfil por grupo e ruído por segmento do K-Means (quando artifacts/train_assignments.csv existir).

Saídas em artifacts/dbscan/: grid_r2.csv, grid_r1_numeric.csv, reference_labels.csv, reference_profile.csv,
noise_by_kmeans_segment.csv (opcional) e params.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from src import config, data, evaluation, preprocessing, profiling  # noqa: E402
from src.artifacts import environment_info, write_json  # noqa: E402

OUT = config.DBSCAN_ARTIFACTS_DIR


def point_types(model: DBSCAN) -> np.ndarray:
    """core, border ou noise para cada ponto do ajuste."""
    types = np.full(len(model.labels_), "border", dtype=object)
    types[model.core_sample_indices_] = "core"
    types[model.labels_ == evaluation.NOISE_LABEL] = "noise"
    return types


def main() -> None:
    start = time.time()
    listings, _ = data.clean_listings(data.load_raw())
    sample_idx = evaluation.fixed_sample_indices(len(listings), config.DBSCAN_SAMPLE_SIZE)
    sample = listings.iloc[sample_idx].reset_index(drop=True)
    silhouette_idx = evaluation.fixed_sample_indices(len(sample), config.SILHOUETTE_SAMPLE_SIZE)
    OUT.mkdir(parents=True, exist_ok=True)

    spaces = {}
    for representation, name in [("R2", "grid_r2.csv"), ("R1", "grid_r1_numeric.csv")]:
        prep = preprocessing.make_preprocessor(representation, scale=True).fit(listings)
        X_sample = prep.transform(sample)
        spaces[representation] = X_sample
        grid, _ = evaluation.dbscan_grid(X_sample, silhouette_idx=silhouette_idx)
        grid.to_csv(OUT / name, index=False)
        print(f"grade {representation}: {len(grid)} combinações -> {(OUT / name).relative_to(config.PROJECT_ROOT)}")

    X_r2 = spaces["R2"]
    reference = DBSCAN(**config.DBSCAN_REFERENCE).fit(X_r2)
    labels = reference.labels_
    groups, order = profiling.dbscan_groups(labels)
    pd.DataFrame({
        config.ID_COL: sample[config.ID_COL],
        "dbscan_label": labels,
        "point_type": point_types(reference),
    }).to_csv(OUT / "reference_labels.csv", index=False)

    frame = sample.assign(dbscan_group=groups)
    profile = profiling.profile_table(frame, "dbscan_group", order).join(
        profiling.composition_table(frame, "dbscan_group", "room_type", config.ROOM_TYPE_ORDER).add_prefix("% "))
    profile.to_csv(OUT / "reference_profile.csv")

    metrics = evaluation.internal_metrics(X_r2, labels, silhouette_idx, exclude_noise=True)
    non_noise = labels != evaluation.NOISE_LABEL
    room_codes = sample["room_type"].map({r: i for i, r in enumerate(config.ROOM_TYPE_ORDER)}).to_numpy()
    summary = {
        "n_clusters": metrics["n_clusters"],
        **evaluation.dbscan_point_types(reference),
        "pct_noise": float((~non_noise).mean() * 100),
        "coverage": metrics["coverage"],
        "silhouette_non_noise": metrics["silhouette"],
        "davies_bouldin_non_noise": metrics["davies_bouldin"],
        "calinski_harabasz_non_noise": metrics["calinski_harabasz"],
        "ari_vs_room_type_non_noise": float(adjusted_rand_score(labels[non_noise], room_codes[non_noise])),
    }

    if config.TRAIN_ASSIGNMENTS_PATH.exists():
        assignments = pd.read_csv(config.TRAIN_ASSIGNMENTS_PATH).set_index(config.ID_COL)["cluster"]
        kmeans_labels = assignments.reindex(sample[config.ID_COL]).to_numpy()
        noise_by_segment = (pd.crosstab(kmeans_labels, ~non_noise, normalize="index")[True] * 100).rename("pct_noise_dbscan")
        noise_by_segment.index.name = "kmeans_cluster"
        noise_by_segment.to_csv(OUT / "noise_by_kmeans_segment.csv")
        summary["ari_vs_kmeans_non_noise"] = float(adjusted_rand_score(labels[non_noise], kmeans_labels[non_noise]))
    else:
        print("aviso: artifacts/train_assignments.csv não existe; rode scripts/train.py para o cruzamento com o K-Means")

    write_json(OUT / "params.json", {
        "note": "DBSCAN não tem predict: estes rótulos valem só para a amostra do ajuste.",
        "space": "R2 com StandardScaler (pré-processador ajustado na base limpa inteira)",
        "sample": {"size": config.DBSCAN_SAMPLE_SIZE, "seed": config.RANDOM_STATE, "n_rows_population": len(listings)},
        "silhouette_sample": {"size": int(len(silhouette_idx)), "seed": config.RANDOM_STATE},
        "grid": {"eps": config.DBSCAN_EPS_GRID, "min_samples": config.DBSCAN_MIN_SAMPLES_GRID},
        "reference": config.DBSCAN_REFERENCE,
        "reference_rationale": "min_samples = 2 x dimensões de R2 (8); eps perto do p90 da distância ao 16º vizinho",
        "reference_summary": summary,
        "environment": environment_info(),
    })
    print(f"referência {config.DBSCAN_REFERENCE}: {summary['n_clusters']} clusters, "
          f"{summary['n_noise']} pontos de ruído ({summary['pct_noise']:.1f}%), "
          f"ARI com room_type {summary['ari_vs_room_type_non_noise']:.3f}")
    print(f"concluído em {time.time() - start:.0f} s -> {OUT.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
