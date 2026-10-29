"""Constantes do projeto: caminhos, schema do dataset, sementes e protocolos fixados pelo enunciado.

Tudo o que define o experimento mora aqui, para que notebooks, scripts e testes compartilhem uma única
fonte de verdade. O conjunto final de features e o K escolhido só são registrados depois da EDA
(Checkpoint 1) e da análise de K (Checkpoint 2).
"""
from pathlib import Path

# --------------------------------------------------------------------------------------------------
# Caminhos (relativos à raiz do projeto, resolvidos na importação)
# --------------------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "AB_NYC_2019.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Artefatos gravados por scripts/train.py e scripts/dbscan_experiments.py
MODEL_PATH = ARTIFACTS_DIR / "kmeans_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
SEGMENT_PROFILES_PATH = ARTIFACTS_DIR / "segment_profiles.csv"
TRAIN_ASSIGNMENTS_PATH = ARTIFACTS_DIR / "train_assignments.csv"
DBSCAN_ARTIFACTS_DIR = ARTIFACTS_DIR / "dbscan"

# SHA-256 do arquivo usado no projeto (Kaggle dgomonov/new-york-city-airbnb-open-data, snapshot de 2019),
# calculado em 2026-10-20 sobre o arquivo baixado pelo usuário (7.077.973 bytes).
DATASET_SHA256 = "e420db40ff10fcb40efc1b5b1648ee0b18a48f4e4537155cecc59fe95d18783a"

# --------------------------------------------------------------------------------------------------
# Schema do dataset (16 colunas, conforme o enunciado)
# --------------------------------------------------------------------------------------------------
ID_COL = "id"
EXPECTED_COLUMNS = [
    "id", "name", "host_id", "host_name", "neighbourhood_group", "neighbourhood",
    "latitude", "longitude", "room_type", "price", "minimum_nights", "number_of_reviews",
    "last_review", "reviews_per_month", "calculated_host_listings_count", "availability_365",
]
# Identificadores e texto livre: nunca entram como features numéricas.
ID_AND_TEXT_COLUMNS = ["id", "host_id", "name", "host_name"]
DATE_COLUMNS = ["last_review"]
CATEGORICAL_COLUMNS = ["room_type", "neighbourhood_group", "neighbourhood"]
NUMERIC_COLUMNS = [
    "price", "minimum_nights", "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
]
# Ordem fixa das categorias (usada em tabelas e gráficos para que a cor siga a categoria).
ROOM_TYPE_ORDER = ["Entire home/apt", "Private room", "Shared room"]
NEIGHBOURHOOD_GROUP_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

# --------------------------------------------------------------------------------------------------
# Regras de limpeza e tratamento de extremos (EDA, notebook 01; aprovadas no Checkpoint 1 em 2026-10-20)
# --------------------------------------------------------------------------------------------------
# price <= 0 não é um preço comercial válido: a linha sai da base clusterizada.
MIN_VALID_PRICE = 1
# Teto fixo de 1 ano para minimum_nights (valores como 999 e 1.250 não descrevem uma condição de
# estadia plausível). O p99,5 (90 noites) cortaria parte da cauda de estadias longas.
MINIMUM_NIGHTS_CAP = 365
# Winsorização superior aprendida no conjunto clusterizado, para price e reviews_per_month.
WINSOR_QUANTILE = 0.995
WINSOR_COLUMNS = ["price", "reviews_per_month"]
# reviews_per_month ausente = anúncio sem avaliações (verificado): a taxa observada é 0.
ZERO_FILL_COLUMNS = ["reviews_per_month"]
# Variáveis não negativas e assimétricas que recebem log1p.
LOG1P_COLUMNS = ["price", "minimum_nights", "number_of_reviews", "reviews_per_month",
                 "calculated_host_listings_count"]
# Retângulo aproximado que contém os cinco distritos de NYC (referência externa, usada apenas como
# checagem de sanidade das coordenadas).
NYC_BBOX = {"lat_min": 40.47, "lat_max": 40.93, "lon_min": -74.27, "lon_max": -73.68}

# --------------------------------------------------------------------------------------------------
# Representações de features (todas sobre as mesmas linhas). R2 é a proposta principal, aprovada no
# Checkpoint 1; R1 e R3 são representações de comparação.
# --------------------------------------------------------------------------------------------------
FEATURE_NUMERIC = ["price", "minimum_nights", "availability_365", "reviews_per_month",
                   "calculated_host_listings_count"]
CATEGORY_ORDERS = {"room_type": ROOM_TYPE_ORDER, "neighbourhood_group": NEIGHBOURHOOD_GROUP_ORDER}
REPRESENTATIONS = {
    "R1": {"numeric": FEATURE_NUMERIC, "categorical": []},
    "R2": {"numeric": FEATURE_NUMERIC, "categorical": ["room_type"]},
    "R3": {"numeric": FEATURE_NUMERIC, "categorical": ["room_type", "neighbourhood_group"]},
    # Experimento complementar (notebook 02, seção 11): R2 + distância até a Times Square, calculada a
    # partir de latitude/longitude. Não substitui R2.
    "R4": {"numeric": FEATURE_NUMERIC, "categorical": ["room_type"], "distance": True},
}
MAIN_REPRESENTATION = "R2"

# Distância em linha reta (haversine) até a Times Square, em km. A coordenada é uma referência externa
# aproximada do centro da praça. log1p: a assimetria bruta é 1,04 e o maior |z| cai de 6,5 para 3,1
# (anúncios de Staten Island a até 35,9 km), o mesmo critério das demais variáveis não negativas.
TIMES_SQUARE = {"lat": 40.7580, "lon": -73.9855}
EARTH_RADIUS_KM = 6371.0
DISTANCE_FEATURE = "dist_times_square_km"
DISTANCE_LOG1P = True
COORDINATE_COLUMNS = ["latitude", "longitude"]

# --------------------------------------------------------------------------------------------------
# Reprodutibilidade e protocolos de custo computacional (seção 6 e 7.2 do enunciado)
# --------------------------------------------------------------------------------------------------
RANDOM_STATE = 42
K_RANGE = list(range(2, 11))  # K = 2..10, ponto de partida pedido no enunciado
SILHOUETTE_SAMPLE_SIZE = 5_000  # amostra fixa para o Silhouette (faixa permitida: 2.000-5.000)
DBSCAN_SAMPLE_SIZE = 10_000  # amostra reprodutível para o DBSCAN (faixa permitida: 8.000-12.000)

# Protocolo fixo do K-Means (o mesmo para todos os K e todas as representações). O scikit-learn 1.9 usa
# n_init="auto" (uma inicialização com k-means++); fixamos 10 para reduzir a dependência da semente.
KMEANS_PARAMS = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": RANDOM_STATE}
# Sementes extras para a checagem de robustez (ARI entre execuções com sementes diferentes).
STABILITY_SEEDS = [0, 1, 2, 3, 4]
# K fixado ANTES da varredura, só para o experimento controlado com e sem normalização (seção 5.3 do
# enunciado). Não é a escolha final de K.
SCALING_EXPERIMENT_K = 4
# K aprovado no Checkpoint 2 (2026-10-21; notebook 02, seção 5), com R2 e StandardScaler. CANDIDATE_KS
# são os K comparados lado a lado antes da decisão.
FINAL_K = 7
CANDIDATE_KS = [5, 7]

# DBSCAN (seção 7 do enunciado), sempre na amostra fixa de DBSCAN_SAMPLE_SIZE linhas e no espaço R2-B.
# min_samples = 16 segue a regra prática de 2 x número de dimensões (8 em R2). eps = 0,9 fica perto do
# p90 da distância ao 16º vizinho (0,94), na região em que a curva de k-distâncias começa a subir.
DBSCAN_EPS_GRID = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
DBSCAN_MIN_SAMPLES_GRID = [8, 16, 32]
DBSCAN_REFERENCE = {"eps": 0.9, "min_samples": 16}
# Configuração mais fina, usada só para ler os grupos densos pequenos.
DBSCAN_FINE = {"eps": 0.5, "min_samples": 16}

# Nomes e descrições dos segmentos do K-Means final (definidos depois da análise de perfis).
SEGMENT_NAMES_PATH = CONFIGS_DIR / "segment_names.json"
