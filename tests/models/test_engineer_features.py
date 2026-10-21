"""Testes de features.engineer_features.

A função define as variáveis que o HDBSCAN* aprimorado e os modelos
supervisionados recebem. Se o valor bruto de Time escapasse para a saída, os
modelos voltariam a usar a ordem das transações como se fosse informação; se a
hora não fosse cíclica, 23h e 0h ficariam nos extremos opostos da escala.
"""

import numpy as np
import pandas as pd
import pytest

from features import V_COLUMNS, engineer_features

OUTPUT_COLUMNS = [*V_COLUMNS, "log_amount", "hour_sin", "hour_cos"]


def transaction(time, amount=0.0):
    """Uma transação com V1 a V28 nulos, para isolar o efeito de Time e Amount."""
    row = {"Time": time, **dict.fromkeys(V_COLUMNS, 0.0), "Amount": amount}
    return pd.DataFrame([row])


def hour_point(time):
    features = engineer_features(transaction(time))
    return np.array([features["hour_sin"].iloc[0], features["hour_cos"].iloc[0]])


def test_returns_the_31_columns_in_order(dataset):
    features = engineer_features(dataset.drop(columns=["Class"]))
    assert list(features.columns) == OUTPUT_COLUMNS


def test_drops_raw_time_and_amount(dataset):
    features = engineer_features(dataset.drop(columns=["Class"]))
    assert "Time" not in features.columns
    assert "Amount" not in features.columns


def test_keeps_the_pca_components_unchanged(dataset):
    X = dataset.drop(columns=["Class"])
    pd.testing.assert_frame_equal(engineer_features(X)[V_COLUMNS], X[V_COLUMNS])


def test_preserves_the_index(dataset):
    X = dataset.drop(columns=["Class"]).iloc[::7]
    assert engineer_features(X).index.equals(X.index)


def test_does_not_modify_the_input(dataset):
    X = dataset.drop(columns=["Class"])
    before = X.copy()
    engineer_features(X)
    pd.testing.assert_frame_equal(X, before)


@pytest.mark.parametrize("amount, expected", [(0.0, 0.0), (99.0, np.log(100.0))])
def test_log_amount_is_log1p(amount, expected):
    features = engineer_features(transaction(0.0, amount))
    assert features["log_amount"].iloc[0] == pytest.approx(expected)


def test_hour_one_of_the_second_day():
    # 90000 s = 25 h: a mesma posição do ciclo que 1 h.
    assert hour_point(90_000) == pytest.approx([np.sin(2 * np.pi / 24), np.cos(2 * np.pi / 24)])


def test_same_hour_on_both_days_gives_the_same_point():
    assert hour_point(3 * 3600 + 120) == pytest.approx(hour_point(86_400 + 3 * 3600 + 120))


def test_hour_lies_on_the_unit_circle(dataset):
    features = engineer_features(dataset.drop(columns=["Class"]))
    radius = features["hour_sin"] ** 2 + features["hour_cos"] ** 2
    assert radius.to_numpy() == pytest.approx(1.0)


def test_23h_and_0h_are_neighbours():
    # Numa escala linear, 23h e 0h estariam a 23 horas de distância.
    late = hour_point(23 * 3600)
    midnight = hour_point(0)
    one_am = hour_point(3600)
    noon = hour_point(12 * 3600)
    assert np.linalg.norm(late - midnight) < np.linalg.norm(late - noon)
    assert np.linalg.norm(late - midnight) == pytest.approx(np.linalg.norm(one_am - midnight))
