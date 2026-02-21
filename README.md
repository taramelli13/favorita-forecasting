# Favorita Forecasting 2.0

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-18_passing-brightgreen)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-9cf)
![XGBoost](https://img.shields.io/badge/XGBoost-3.0-orange)
![Polars](https://img.shields.io/badge/Polars-1.38-blue)

Previsão de vendas diárias para a Corporación Favorita, rede de supermercados do Equador com 54 lojas e ~4.100 produtos. Prevê `unit_sales` num horizonte de 16 dias usando LightGBM e XGBoost com 90 features engenheiradas.

Baseado na competição [Kaggle Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting).

> Feature engineering levou o NWRMSLE de **0.8006** (baseline, 34 features) para **0.5236** (90 features), uma redução de **34.6%**.
>
> | | Baseline | Modelo final | Variação |
> |---|---|---|---|
> | Features | 34 (sem lags, sem rolling stats) | 90 (lags, rolling, target encoding, feriados, petróleo) | +56 features |
> | CV Média NWRMSLE | 0.8006 | 0.5236 | **-34.6%** |

## Resultados

### Validação cruzada (4 folds, janela expansiva)

| Modelo | CV Média NWRMSLE | Val Final NWRMSLE | Tempo | Device |
|---|---|---|---|---|
| LightGBM | 0.5236 | 0.5304 | 2h 04min | CPU |
| XGBoost | 0.5235 | 0.5309 | 25.5 min | CUDA |

Os dois modelos chegaram em ~0.5236. XGBoost foi ~5x mais rápido na GPU.

![Resultados da validação cruzada por fold](docs/images/results.png)

<details>
<summary>Detalhes por fold</summary>

| Fold | Período de validação | LGB NWRMSLE | LGB Iters | XGB NWRMSLE | XGB Iters |
|---|---|---|---|---|---|
| 1 | Jun 13 - Jun 28 | 0.5207 | 721 | 0.5220 | 1.069 |
| 2 | Jun 29 - Jul 14 | 0.5284 | 546 | 0.5249 | 1.418 |
| 3 | Jul 15 - Jul 30 | 0.5150 | 1.452 | 0.5164 | 1.817 |
| 4 | Jul 31 - Ago 15 | 0.5303 | 899 | 0.5309 | 1.513 |

</details>

### Top 10 features (LightGBM gain)

![Top 10 feature importance](docs/images/feature_importance.png)

| # | Feature | Tipo |
|---|---|---|
| 1 | `store_item_mean_sales` | Target encoding |
| 2 | `item_mean_sales` | Target encoding |
| 3 | `onpromotion` | Feature original |
| 4 | `sales_rolling_mean_14` | Estatística rolling |
| 5 | `sales_rolling_mean_28` | Estatística rolling |
| 6 | `oil_rolling_mean_7` | Externa / rolling |
| 7 | `family` | Categórica |
| 8 | `sales_lag_21` | Lag |
| 9 | `sales_lag_28` | Lag |
| 10 | `sales_rolling_mean_7` | Estatística rolling |

Target encodings e médias móveis dominam o ranking. O histórico de vendas é o que mais importa pra previsão.

## Dataset

- Fonte: competição [Kaggle Corporación Favorita](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
- 125M linhas de dados de treino (2013-2017)
- 54 lojas x ~4.100 itens no Equador
- Target: `unit_sales` (unidades vendidas por dia por loja-item)
- Métrica: NWRMSLE (itens perecíveis com peso 1.25x)
- Horizonte de previsão: 16 dias (16-31 Ago 2017)

## Feature engineering (90 features)

Os dados brutos de vendas mais 6 tabelas auxiliares (lojas, itens, preço do petróleo, feriados, transações) são transformados em 90 features:

| Categoria | Qtd | O que captura |
|---|---|---|
| Lag features | 13 | Vendas defasadas de 16 a 365 dias + diff. Todos os lags >= 16 pra evitar vazamento. |
| Estatísticas rolling | 28 | Média, desvio, min, max em janelas de 7 a 365 dias, deslocadas em 16 dias. |
| Target encoding | 5 | Média expansiva de vendas por loja-item, loja-família, loja, item, família. |
| Features temporais | 14 | Dia da semana/mês/ano, fim de semana, dia de pagamento, limites do mês, sin/cos cíclico. |
| Features de feriado | 8 | Feriados nacionais/regionais/locais, transferências, pontes, eventos (por cidade/estado). |
| Features externas | 10 | Petróleo (bruto + lags + rolling), transações, contagem de promoções por loja. |
| Promoção | 2 | Flag `onpromotion` + `family_promo_pct`. |
| Categóricas | 10 | Tipo de loja, cluster, família de produto, classe, etc. |

## Modelos

Os dois modelos usam o objetivo Tweedie (variance power 1.5), que se ajusta bem à distribuição de vendas com muitos zeros e valores não-negativos.

- **LightGBM**: API nativa (`lgb.train()`), 255 folhas, CPU, early stopping em 100 rounds
- **XGBoost**: API nativa (`xgb.train()`), max depth 8, CUDA, early stopping em 100 rounds

### Estratégia de validação

CV com janela expansiva em 4 folds. Cada fold prevê uma janela de 16 dias que espelha o horizonte da competição:

```
Fold 1: Treina até Jun 12  -> Valida Jun 13-28
Fold 2: Treina até Jun 28  -> Valida Jun 29-Jul 14
Fold 3: Treina até Jul 14  -> Valida Jul 15-30
Fold 4: Treina até Jul 30  -> Valida Jul 31-Ago 15
```

## O que deu trabalho

### Encaixar 87M linhas x 90 features em 16 GB de RAM

O conjunto completo de features não cabe na memória. Três soluções resolveram:

**Feature pipeline chunked** (`features/pipeline.py`). O Polars dava segfault ao rodar operações complexas de `.over()` nas 87M linhas. Processar uma loja por vez (54 lojas) e gravar cada pedaço em Parquet via PyArrow resolveu. Pico de memória caiu de 60+ GB pra cerca de 1-2 GB por loja.

**Leitura de colunas em lotes** (`pipeline/train_pipeline.py`). Converter o Parquet de 17.7M x 90 pra um único array numpy exigiria ~13 GB. Em vez disso, pré-aloco um array float32 e preencho 30 colunas por vez, sem nunca manter o DataFrame Polars e o array completo ao mesmo tempo. Também tive que copiar o Parquet pra um diretório local temporário porque o sync do OneDrive estava adicionando quase 2 horas de overhead de I/O.

**Slices zero-copy no CV** (`pipeline/train_pipeline.py`). O CV original recarregava o DataFrame Polars (~7 GB) enquanto os arrays numpy (~6.4 GB) ainda estavam na memória. OOM. A solução foi rodar o CV direto nos arrays numpy usando `np.searchsorted` na coluna de datas ordenada, que dá slices zero-copy. Pico de memória durante CV: ~7.5 GB.

### Crash do LightGBM na GPU

O modo GPU do LightGBM crasha com `Check failed: (best_split_info.left_count) > (0)` quando usa 90+ features em >7M linhas. Parece ser um bug no histogram binning da GPU. O LightGBM treina na CPU (2h 04min). XGBoost no CUDA funciona sem problemas (25.5 min).

### Paths com Unicode

As bibliotecas C do LightGBM e XGBoost não conseguem gravar em paths com caracteres acentuados (o diretório do projeto fica dentro de `Área de Trabalho`). Os modelos são serializados pra strings/bytes em Python e gravados com I/O nativo do Python.

## Arquitetura

```
src/favorita_forecasting/
├── config.py              # Configurações Pydantic, carrega params.yaml
├── cli.py                 # CLI Typer: prepare | train | predict | tune | evaluate
├── data/                  # Carga, schemas, validação, splits temporais
├── preprocessing/         # Limpeza, transformação log1p, merge de tabelas
├── features/              # Módulos de feature engineering
│   ├── pipeline.py        # Executa todos os builders, chunked por loja
│   ├── lag_features.py    # Features de lag (shift >= 16)
│   ├── rolling_features.py # Média/desvio/min/max móvel
│   ├── target_encoding.py  # Target encodings com média expansiva
│   ├── time_features.py   # Features de calendário e cíclicas
│   ├── holiday_features.py # Feriados por localidade
│   └── external_features.py # Petróleo, transações, promoções
├── models/                # Wrappers LightGBM/XGBoost, ensemble, Optuna
├── evaluation/            # Métrica NWRMSLE, CV com janela expansiva, análise de erro
├── pipeline/              # Pipelines de treino, predição e submissão
├── tracking/              # Wrapper MLflow (no-op se não instalado)
└── utils/                 # Helpers Polars
```

### Fluxo de dados

```
data/raw/*.csv -> loader -> cleaner -> merger -> features -> data/processed/*.parquet
                                                                   |
                                             train_pipeline -> models/*.txt
                                                                   |
                                           predict_pipeline -> data/submissions/*.csv
```

### Pipeline DVC

```
prepare -> train ---------> predict
        -> train-xgboost
        -> tune
```

`dvc repro` re-executa só os stages cujas dependências mudaram.

## Stack

| | |
|---|---|
| Linguagem | Python 3.13 |
| Gerenciador de pacotes | Poetry |
| Processamento de dados | Polars (lazy evaluation) |
| Modelos ML | LightGBM (API nativa), XGBoost (API nativa) |
| Tuning de hiperparâmetros | Optuna |
| Tracking de experimentos | MLflow (backend SQLite) |
| Pipeline | DVC |
| Configuração | Pydantic + params.yaml |
| CLI | Typer |
| Testes | pytest (18 testes unitários) |
| Container | Docker |

## Como rodar

### Pré-requisitos

- Python >= 3.13
- [Poetry](https://python-poetry.org/) >= 2.0
- GPU NVIDIA com CUDA (opcional, pro XGBoost)

### Instalação

```bash
git clone https://github.com/taramelli13/favorita-forecasting.git
cd favorita-forecasting
poetry install
```

### Dados

Baixe os dados da [competição no Kaggle](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) e coloque os CSVs em `data/raw/`:

```
data/raw/
├── train.csv
├── test.csv
├── stores.csv
├── items.csv
├── oil.csv
├── holidays_events.csv
└── transactions.csv
```

### Executando o pipeline

```bash
# Gerar features (chunked por loja, ~18 min)
poetry run favorita prepare

# Treinar LightGBM com validação cruzada
poetry run favorita train

# Treinar XGBoost com validação cruzada
poetry run favorita train --model xgboost

# Gerar predições
poetry run favorita predict

# Tuning de hiperparâmetros com Optuna
poetry run favorita tune --n-trials 100
```

Ou reproduzir tudo com DVC:

```bash
dvc repro
```

### MLflow UI

```bash
poetry run mlflow ui
# http://localhost:5000
```

### Testes

```bash
poetry run pytest tests/unit/ -v
```

## Configuração

Hiperparâmetros e configurações do pipeline ficam em `params.yaml`:

```yaml
data:
  min_train_date: "2017-03-01"   # corta de 87M pra 17.7M linhas (restrição de RAM)

features:
  lags: [16, 17, 18, 19, 20, 21, 28, 30, 60, 90, 180, 365]
  rolling_windows: [7, 14, 28, 60, 90, 180, 365]
  rolling_shift: 16              # evita vazamento de dados

train:
  lightgbm:
    objective: tweedie
    num_leaves: 255
    learning_rate: 0.02
    device: cpu                  # workaround pro bug na GPU
  xgboost:
    objective: reg:tweedie
    max_depth: 8
    learning_rate: 0.02
    device: cuda
```

## Roadmap

- [ ] **Ensemble**: blend ponderado das predições LightGBM + XGBoost
- [ ] **Optuna tuning**: 100+ trials por modelo usando o CV de janela expansiva
- [ ] **Modelos neurais**: N-BEATS, Temporal Fusion Transformer (via PyTorch/Darts)
- [ ] **Features temporais avançadas**: termos de Fourier, decomposição wavelet
- [ ] **Otimização de memória**: quantizar features pra int8/float16 onde possível
- [ ] **Deploy via API**: servir predições com FastAPI + Docker

## Licença

MIT
