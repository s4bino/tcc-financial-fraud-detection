"""Fixtures compartilhadas pela suíte de testes.

As fixtures reproduzem, em escala reduzida, as características estruturais da
base utilizada no trabalho (creditcard.csv, ULB/Kaggle):

  - 31 colunas na ordem Time, V1..V28, Amount, Class;
  - Time contínua e crescente, em segundos desde a primeira transação;
  - Amount contínua, fortemente assimétrica e em escala distinta das demais;
  - V1..V28 com média zero e variâncias decrescentes, como componentes
    principais ordenados;
  - Class binária e desbalanceada.

Reproduzir a forma da base — e não seus valores — permite que os testes
exercitem as mesmas condições de contorno enfrentadas pelo pipeline real
(estratificação sob desbalanceamento, escalas heterogêneas) sem depender do
arquivo de 150 MB, que não é versionado.
"""

import numpy as np
import pandas as pd
import pytest

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
DATASET_COLUMNS = ["Time", *PCA_COLUMNS, "Amount", "Class"]

# Duração total da base original, em segundos (48 horas).
TIME_SPAN_SECONDS = 172_792


def build_dataset(n_samples: int = 1_000, n_frauds: int = 20, seed: int = 42) -> pd.DataFrame:
    """Constrói um DataFrame com a mesma estrutura da base do trabalho.

    Parâmetros:
        n_samples (int): número total de transações.
        n_frauds (int): número de transações rotuladas como fraude (Class = 1).
        seed (int): semente do gerador, para reprodutibilidade.
    """
    if n_frauds > n_samples:
        raise ValueError("n_frauds não pode exceder n_samples")

    rng = np.random.default_rng(seed)

    data = {"Time": np.sort(rng.uniform(0, TIME_SPAN_SECONDS, n_samples))}

    # Variâncias decrescentes, como em componentes principais ordenados.
    for position, column in enumerate(PCA_COLUMNS):
        data[column] = rng.normal(0.0, 1.0 / np.sqrt(position + 1), n_samples)

    # Distribuição log-normal: assimétrica e com cauda pesada, como Amount.
    data["Amount"] = rng.lognormal(mean=3.0, sigma=1.5, size=n_samples)

    labels = np.zeros(n_samples, dtype=int)
    labels[rng.choice(n_samples, size=n_frauds, replace=False)] = 1
    data["Class"] = labels

    return pd.DataFrame(data, columns=DATASET_COLUMNS)


@pytest.fixture
def dataset_factory():
    """Expõe build_dataset aos testes que precisam de outras proporções."""
    return build_dataset


@pytest.fixture
def dataset() -> pd.DataFrame:
    """Base sintética de 1.000 transações com 2% de fraudes."""
    return build_dataset(n_samples=1_000, n_frauds=20)


@pytest.fixture
def small_dataset() -> pd.DataFrame:
    """Base reduzida, para testes que treinam modelos."""
    return build_dataset(n_samples=200, n_frauds=20)


@pytest.fixture
def dataset_csv(tmp_path, dataset) -> str:
    """Caminho de um CSV contendo a base sintética."""
    path = tmp_path / "creditcard.csv"
    dataset.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def inner_folds_dir(tmp_path, small_dataset) -> str:
    """Diretório no formato consumido pelo grid search.

    Reproduz o layout produzido por internal_k_folds: para cada fold, um par
    fold_<n>_train.csv / fold_<n>_val.csv. Ambas as partições preservam a
    presença das duas classes, condição necessária para que as métricas de
    avaliação sejam definidas.
    """
    folds_dir = tmp_path / "inner_folds"
    folds_dir.mkdir()

    frauds = small_dataset[small_dataset["Class"] == 1]
    legitimate = small_dataset[small_dataset["Class"] == 0]

    for fold in (1, 2):
        offset = fold - 1

        fraud_val = frauds.iloc[offset::2]
        fraud_train = frauds.drop(index=fraud_val.index)
        legit_val = legitimate.iloc[offset::4]
        legit_train = legitimate.drop(index=legit_val.index)

        train = pd.concat([fraud_train, legit_train]).sample(frac=1, random_state=fold)
        val = pd.concat([fraud_val, legit_val]).sample(frac=1, random_state=fold)

        train.to_csv(folds_dir / f"fold_{fold}_train.csv", index=False)
        val.to_csv(folds_dir / f"fold_{fold}_val.csv", index=False)

    return str(folds_dir)


@pytest.fixture
def outer_folds_dir(tmp_path, small_dataset) -> str:
    """Diretório no formato consumido pela avaliação externa.

    Reproduz o layout produzido por external_k_folds: para cada fold, um par
    fold_<n>_train.csv / fold_<n>_test.csv. As duas partições preservam a
    presença das duas classes, e os números dos folds coincidem com os de
    inner_folds_dir, como na base real.
    """
    folds_dir = tmp_path / "outer_folds"
    folds_dir.mkdir()

    frauds = small_dataset[small_dataset["Class"] == 1]
    legitimate = small_dataset[small_dataset["Class"] == 0]

    for fold in (1, 2):
        offset = fold - 1

        test = pd.concat([frauds.iloc[offset::2], legitimate.iloc[offset::2]])
        train = small_dataset.drop(index=test.index)

        train.sample(frac=1, random_state=fold).to_csv(
            folds_dir / f"fold_{fold}_train.csv", index=False
        )
        test.sample(frac=1, random_state=fold).to_csv(
            folds_dir / f"fold_{fold}_test.csv", index=False
        )

    return str(folds_dir)
