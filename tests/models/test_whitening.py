"""Testes do branqueamento: a euclidiana nos dados transformados é a distância de Mahalanobis."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.covariance import MinCovDet

from whitening import MahalanobisWhitener, check_whitening_params

TOLERANCE = 1e-8


@pytest.fixture
def correlated():
    """Três variáveis correlacionadas e de escalas diferentes, com algumas anomalias."""
    rng = np.random.RandomState(0)
    mixing = np.array([[2.0, 0.0, 0.0], [1.5, 0.5, 0.0], [0.3, -0.8, 0.1]])
    X = rng.normal(size=(400, 3)) @ mixing.T
    X[:8] += rng.normal(0, 6, size=(8, 3))
    return X


def test_empirical_whitening_turns_euclidean_into_mahalanobis(correlated):
    X_train, X_eval = correlated[:300], correlated[300:]
    whitener = MahalanobisWhitener("empirical").fit(X_train)

    inverse = np.linalg.inv(np.cov(X_train, rowvar=False))
    expected = cdist(X_eval, X_train, "mahalanobis", VI=inverse)
    observed = cdist(whitener.transform(X_eval), whitener.transform(X_train))

    np.testing.assert_allclose(observed, expected, atol=TOLERANCE)


def test_empirical_whitening_leaves_the_train_centred_with_unit_covariance(correlated):
    Z = MahalanobisWhitener("empirical").fit_transform(correlated)

    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=TOLERANCE)
    np.testing.assert_allclose(np.cov(Z, rowvar=False), np.eye(3), atol=TOLERANCE)


def test_mcd_norm_is_the_robust_distance_to_the_centre(correlated):
    """Sem amostragem, a norma transformada é a distância do MinCovDet com a mesma semente."""
    whitener = MahalanobisWhitener("mcd", support_fraction=0.9, random_state=1).fit(correlated)
    reference = MinCovDet(support_fraction=0.9, random_state=1).fit(correlated)

    observed = np.linalg.norm(whitener.transform(correlated), axis=1)

    np.testing.assert_allclose(observed, np.sqrt(reference.mahalanobis(correlated)), rtol=1e-6)


def test_mcd_resists_the_anomalies_that_inflate_the_empirical_covariance(correlated):
    """As oito anomalias ficam mais distantes do centro robusto que do centro empírico."""
    robust = MahalanobisWhitener("mcd", random_state=0).fit(correlated)
    empirical = MahalanobisWhitener("empirical").fit(correlated)

    robust_rank = np.argsort(-np.linalg.norm(robust.transform(correlated), axis=1))[:8]
    empirical_distance = np.linalg.norm(empirical.transform(correlated[:8]), axis=1)

    assert set(robust_rank) == set(range(8))
    assert np.all(np.linalg.norm(robust.transform(correlated[:8]), axis=1) > empirical_distance)


def test_transform_uses_only_what_was_fitted_on_the_train(correlated):
    whitener = MahalanobisWhitener("empirical").fit(correlated[:300])
    rows = correlated[300:]

    batch = whitener.transform(rows)
    one_by_one = np.vstack([whitener.transform(row[None, :]) for row in rows])

    np.testing.assert_allclose(batch, one_by_one, atol=TOLERANCE)
    np.testing.assert_allclose(whitener.location_, correlated[:300].mean(axis=0), atol=TOLERANCE)


def test_mcd_sample_is_reproducible_by_the_seed(correlated):
    first = MahalanobisWhitener("mcd", sample_size=200, random_state=7).fit(correlated)
    second = MahalanobisWhitener("mcd", sample_size=200, random_state=7).fit(correlated)

    np.testing.assert_array_equal(first.components_, second.components_)
    np.testing.assert_array_equal(first.location_, second.location_)


def test_singular_covariance_is_floored_instead_of_dividing_by_zero(correlated):
    X = np.column_stack([correlated, correlated[:, 0] + correlated[:, 1]])

    whitener = MahalanobisWhitener("empirical").fit(X)

    assert whitener.n_floored_ == 1
    assert np.isfinite(whitener.transform(X)).all()


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"covariance": "robust"}, "covariance"),
        ({"support_fraction": 0}, "support_fraction"),
        ({"support_fraction": 1.5}, "support_fraction"),
        ({"sample_size": 1}, "sample_size"),
        ({"sample_size": 10.5}, "sample_size"),
        ({"eigen_floor": 0}, "eigen_floor"),
        ({"eigen_floor": 1}, "eigen_floor"),
    ],
)
def test_invalid_parameters_are_refused_before_fitting(correlated, params, message):
    with pytest.raises(ValueError, match=message):
        MahalanobisWhitener(**params).fit(correlated)


def test_valid_parameters_pass_the_check():
    check_whitening_params("mcd", 1.0, 2, 1e-9)
    check_whitening_params("empirical", 0.5, 30_000, 0.5)


def test_covariance_needs_more_points_than_variables():
    with pytest.raises(ValueError, match="mais pontos que variáveis"):
        MahalanobisWhitener("empirical").fit(np.ones((3, 3)))
