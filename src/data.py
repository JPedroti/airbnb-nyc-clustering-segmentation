"""Leitura do AB_NYC_2019.csv, auditoria de qualidade e limpeza de linhas inválidas.

A limpeza aqui só remove linhas que não podem entrar na análise. Transformações (log1p, winsorização,
imputação, encoding, scaling) moram no pipeline de `src/preprocessing.py`, para que treino e inferência
apliquem exatamente o mesmo tratamento.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pandas as pd

from src import config


def file_sha256(path: Path) -> str:
    """Hash SHA-256 do arquivo, lido em blocos."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_schema(frame: pd.DataFrame, required: list[str] | None = None) -> None:
    """Falha se faltar alguma coluna esperada; colunas extras são permitidas."""
    required = config.EXPECTED_COLUMNS if required is None else required
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no arquivo: {missing}")


def load_raw(path: Path = config.DATA_PATH, check_hash: bool = True) -> pd.DataFrame:
    """Lê o CSV bruto sem nenhuma limpeza.

    `id` e `host_id` são lidos como inteiros apenas para identificação; `last_review` vira data.
    Se `config.DATASET_SHA256` estiver definido e o hash do arquivo for outro, emite um aviso.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. Veja data/README.md para baixar o dataset."
        )
    if check_hash and config.DATASET_SHA256 is not None:
        observed = file_sha256(path)
        if observed != config.DATASET_SHA256:
            warnings.warn(
                f"SHA-256 diferente do registrado ({observed[:12]}... != {config.DATASET_SHA256[:12]}...).",
                stacklevel=2,
            )
    frame = pd.read_csv(path, parse_dates=config.DATE_COLUMNS)
    check_schema(frame)
    return frame


# --------------------------------------------------------------------------------------------------
# Auditoria
# --------------------------------------------------------------------------------------------------
def quality_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Tipo, ausentes e cardinalidade de cada coluna."""
    report = pd.DataFrame({
        "dtype": frame.dtypes.astype(str),
        "n_missing": frame.isna().sum(),
        "pct_missing": frame.isna().mean() * 100,
        "n_unique": frame.nunique(dropna=True),
    })
    report.index.name = "column"
    return report


def range_checks(frame: pd.DataFrame) -> pd.DataFrame:
    """Conta as linhas que violam cada regra de faixa ou de consistência.

    Regras de domínio (impossíveis por definição) e regras de plausibilidade (extremos que pedem
    decisão) ficam na mesma tabela, identificadas pela coluna `kind`.
    """
    bbox = config.NYC_BBOX
    rules = [
        ("price <= 0", "domínio", frame["price"] <= 0),
        ("minimum_nights < 1", "domínio", frame["minimum_nights"] < 1),
        ("availability_365 fora de [0, 365]", "domínio", ~frame["availability_365"].between(0, 365)),
        ("number_of_reviews < 0", "domínio", frame["number_of_reviews"] < 0),
        ("reviews_per_month < 0", "domínio", frame["reviews_per_month"] < 0),
        ("calculated_host_listings_count < 1", "domínio", frame["calculated_host_listings_count"] < 1),
        ("coordenadas fora do retângulo de NYC", "domínio",
         ~(frame["latitude"].between(bbox["lat_min"], bbox["lat_max"])
           & frame["longitude"].between(bbox["lon_min"], bbox["lon_max"]))),
        ("reviews_per_month ausente com number_of_reviews > 0", "consistência",
         frame["reviews_per_month"].isna() & (frame["number_of_reviews"] > 0)),
        ("reviews_per_month presente com number_of_reviews = 0", "consistência",
         frame["reviews_per_month"].notna() & (frame["number_of_reviews"] == 0)),
        ("last_review ausente com number_of_reviews > 0", "consistência",
         frame["last_review"].isna() & (frame["number_of_reviews"] > 0)),
        ("minimum_nights > 365", "plausibilidade", frame["minimum_nights"] > config.MINIMUM_NIGHTS_CAP),
        ("price > 1.000", "plausibilidade", frame["price"] > 1_000),
    ]
    table = pd.DataFrame(
        [(name, kind, int(mask.sum())) for name, kind, mask in rules],
        columns=["rule", "kind", "n_rows"],
    )
    table["pct_rows"] = table["n_rows"] / len(frame) * 100
    return table


# --------------------------------------------------------------------------------------------------
# Limpeza
# --------------------------------------------------------------------------------------------------
def invalid_rows_mask(frame: pd.DataFrame) -> pd.Series:
    """Linhas fora do domínio aceito para clusterização (hoje: price abaixo do mínimo válido)."""
    return frame["price"] < config.MIN_VALID_PRICE


def clean_listings(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove linhas inválidas e devolve (base limpa, registro do que saiu).

    O registro tem uma linha por regra aplicada, com a contagem removida, para que notebook,
    scripts e README reportem o mesmo número.
    """
    invalid = invalid_rows_mask(frame)
    log = pd.DataFrame([{
        "rule": f"price < {config.MIN_VALID_PRICE}",
        "n_removed": int(invalid.sum()),
        "n_before": len(frame),
        "n_after": int((~invalid).sum()),
    }])
    return frame.loc[~invalid].reset_index(drop=True), log
