"""Atribui segmentos a anúncios novos com o pipeline K-Means salvo, sem retreinar.

    python scripts/segment.py --input novos_anuncios.csv --output segmentos.csv

A entrada segue o schema do AB_NYC_2019.csv e precisa ter, no mínimo, as colunas id, price, minimum_nights,
availability_365, reviews_per_month, calculated_host_listings_count e room_type. Colunas extras são
ignoradas, e reviews_per_month pode vir vazio (anúncio sem avaliações).

A saída mantém a ordem das linhas de entrada e tem uma linha por anúncio:

    id,cluster,segment_name,status

status = "ok" quando o segmento foi atribuído (centróide mais próximo). Qualquer outro valor explica por
que a linha ficou sem segmento: preço inválido, valor obrigatório ausente, categoria de room_type nunca
vista no treino etc. Nenhum segmento é inventado para essas linhas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.artifacts import load_model  # noqa: E402
from src.segmentation import OK, assign_segments  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="CSV com os anúncios no schema do AB_NYC_2019.csv")
    parser.add_argument("--output", required=True, type=Path, help="onde gravar o CSV com os segmentos")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH, help="pipeline salvo por train.py")
    parser.add_argument("--metadata", type=Path, default=config.METADATA_PATH, help="metadata.json com o SHA-256 do modelo")
    args = parser.parse_args()

    pipeline, metadata = load_model(args.model, args.metadata)
    segment_names = {int(s["cluster"]): s["name"] for s in metadata["segments"]}
    frame = pd.read_csv(args.input)

    result = assign_segments(frame, pipeline, segment_names, metadata["features"]["representation"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    n_ok = int((result["status"] == OK).sum())
    print(f"{len(result)} linhas lidas; {n_ok} segmentadas; {len(result) - n_ok} sem segmento "
          f"(modelo sha256 {metadata['model_sha256'][:12]}...) -> {args.output}")
    if n_ok < len(result):
        print("motivos sem segmento:")
        for reason, count in result.loc[result["status"] != OK, "status"].value_counts().items():
            print(f"  {count:>6}  {reason}")
    print("segmentos:")
    for name, count in result.loc[result["status"] == OK, "segment_name"].value_counts().items():
        print(f"  {count:>6}  {name}")


if __name__ == "__main__":
    main()
