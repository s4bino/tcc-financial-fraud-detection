"""Testes de features.build_preprocessor.

O transformador padroniza as variáveis derivadas antes do HDBSCAN* aprimorado
e dos modelos supervisionados. Se a padronização usasse estatísticas da
validação ou do teste, haveria vazamento de informação; se alcançasse V1 a V28,
igualaria o peso de componentes de variância muito diferente na distância.
"""

import numpy as np
import pytest

from features import V_COLUMNS, build_preprocessor, engineer_features

SCALED_COLUMNS = ["log_amount", "hour_sin", "hour_cos"]


@pytest.fixture
def train_and_test(dataset):
    features = engineer_features(dataset.drop(columns=["Class"]))
    return features.iloc[:800], features.iloc[800:]


def test_returns_the_31_columns_with_the_scaled_ones_first(train_and_test):
    train, _ = train_and_test
    scaled = build_preprocessor().fit_transform(train)

    assert scaled.shape == (len(train), 31)
    np.testing.assert_allclose(scaled[:, 3:], train[V_COLUMNS].to_numpy())


def test_standardizes_the_derived_variables_on_the_training_set(train_and_test):
    train, _ = train_and_test
    scaled = build_preprocessor().fit_transform(train)

    np.testing.assert_allclose(scaled[:, :3].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled[:, :3].std(axis=0), 1.0)


def test_transforms_new_data_with_the_training_statistics(train_and_test):
    """A validação e o teste recebem a média e o desvio do treino, não os próprios."""
    train, test = train_and_test
    preprocessor = build_preprocessor()
    preprocessor.fit(train)

    expected = (test[SCALED_COLUMNS] - train[SCALED_COLUMNS].mean()) / train[SCALED_COLUMNS].std(
        ddof=0
    )
    np.testing.assert_allclose(preprocessor.transform(test)[:, :3], expected.to_numpy())


def test_leaves_the_pca_components_of_new_data_unchanged(train_and_test):
    train, test = train_and_test
    preprocessor = build_preprocessor()
    preprocessor.fit(train)

    np.testing.assert_allclose(preprocessor.transform(test)[:, 3:], test[V_COLUMNS].to_numpy())
