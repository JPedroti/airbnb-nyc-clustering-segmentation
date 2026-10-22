# Dados

O arquivo de dados **não é versionado** (ver `.gitignore`). Para reproduzir o projeto:

1. Baixe o dataset oficial *New York City Airbnb Open Data* em
   <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data> (é preciso estar logado no Kaggle).
   Use o **snapshot de 2019** indicado no enunciado, e não uma versão mais recente de outra fonte (ex.: Inside Airbnb).
2. Descompacte e coloque o arquivo em `data/AB_NYC_2019.csv`.
3. Confira a integridade. O arquivo usado neste projeto tem 7.077.973 bytes e SHA-256

   ```
   e420db40ff10fcb40efc1b5b1648ee0b18a48f4e4537155cecc59fe95d18783a
   ```

   ```bash
   python -c "import hashlib;print(hashlib.sha256(open('data/AB_NYC_2019.csv','rb').read()).hexdigest())"
   ```

   `src/data.py` (`load_raw`) avisa se o hash for diferente.

## O que o arquivo contém (verificado no notebook `01_eda.ipynb`)

- **Tamanho e unidade:** 48.895 linhas × 16 colunas. Uma linha é **um anúncio** (`id` único), e não uma reserva, uma pessoa ou
  uma hospedagem. Há 37.457 anfitriões (`host_id`) distintos e nenhuma linha duplicada.
- **Período:** snapshot histórico de 2019. A última avaliação registrada (`last_review`) é de 2019-07-08, e o dataset não descreve
  o mercado atual.
- **Ausentes:**
  - `reviews_per_month` e `last_review` têm 10.052 ausentes, todos nos anúncios com `number_of_reviews = 0` (ausência
    estrutural);
  - `name` tem 16 ausentes e `host_name` tem 21.
- **Faixas:**
  - 11 anúncios com `price = 0`, excluídos da base clusterizada;
  - 14 com `minimum_nights > 365` (máximo de 1.250);
  - 239 com `price > 1.000` (máximo de 10.000).
- **Categóricas:** `room_type` tem 3 categorias, `neighbourhood_group` tem 5 distritos e `neighbourhood` tem 221 bairros.
- **`calculated_host_listings_count`:** confere com o número de linhas de cada `host_id` no arquivo.

As decisões de limpeza e de features, com a evidência de cada uma, estão na seção 10 do notebook 01.
