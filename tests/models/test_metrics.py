"""Testes de metrics: o grau de anomalia, o GLOSH e o limiar de alerta.

O grau de anomalia combina escores de naturezas diferentes pelos postos em
relação ao treino. Sem o cuML, as distâncias aos vizinhos vêm do scikit-learn,
com a mesma interface. O GLOSH (Campello et al., 2015) compara a densidade em que
cada ponto deixa o seu último agrupamento com a maior densidade desse
agrupamento. As árvores condensadas dos testes são montadas à mão, com os
valores esperados calculados pela Equação (8) do artigo; o último teste confere
o resultado com a biblioteca hdbscan, quando instalada
(uv run --with hdbscan python -m pytest tests/models/test_metrics.py). O limiar
de alerta da base subamostrada pondera cada legítima pelo inverso da fração
mantida, e os testes conferem que ele devolve a fração de alertas do treino
completo, que o quantil simples não devolve.
"""

import json

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from metrics import (
    alert_threshold,
    cluster_max_lambdas,
    ensemble_score,
    glosh_anomaly_score,
    glosh_neighbors,
    glosh_new_points,
    glosh_scores,
    knn_distances,
    manual_anomaly_score,
    percentile_rank,
    read_legitimate_weight,
)

# Pontos 0 a 6 e raiz 7. A raiz se divide em 8 e 9 (λ = 0.2), e 8 se divide em 10 e 11
# (λ = 0.5). O ponto 6 sai da raiz em λ = 0.1, como ruído.
TREE = {
    "parent": [7, 7, 8, 8, 10, 10, 11, 11, 9, 9, 7],
    "child": [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6],
    "lambda_val": [0.2, 0.2, 0.5, 0.5, 8.0, 6.0, 3.0, 3.0, 1.0, 1.0, 0.1],
}
# λ_max por agrupamento, na ordem 7 a 11: o 8 e a raiz herdam o 8.0 do neto 10.
MAX_LAMBDAS = [8.0, 8.0, 1.0, 8.0, 3.0]
# 1 - λ/λ_max de cada ponto: o ponto 1 sai de 10 em 6 (de 8), o ponto 6 sai da raiz em 0.1 (de 8).
GLOSH = [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.9875]
# Distâncias núcleo do treino, com λ ≤ 1/d_core em todo ponto, como numa hierarquia real.
TRAIN_CORE_DISTANCES = np.array([0.125, 0.15, 0.3, 0.3, 1.0, 1.0, 5.0])


def reordered(tree, order):
    return {field: [values[i] for i in order] for field, values in tree.items()}


@pytest.fixture
def scaled_sets():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 3))
    X_eval = rng.normal(size=(50, 3))
    train_probabilities = rng.uniform(size=200)
    probabilities = rng.uniform(size=50)
    return X_train, X_eval, train_probabilities, probabilities


def test_percentile_rank_is_the_empirical_cdf_of_the_reference():
    ranks = percentile_rank(np.array([1, 2, 3, 4]), np.array([0.5, 2, 3.5, 10]))

    np.testing.assert_allclose(ranks, [0.0, 0.5, 0.75, 1.0])


def test_ensemble_score_averages_the_ranks_of_each_score():
    train_scores = [np.array([0.1, 0.2, 0.4, 0.9]), np.array([1.0, 2.0, 3.0, 4.0])]

    combined = ensemble_score(train_scores, [np.array([0.95]), np.array([2.5])])

    np.testing.assert_allclose(combined, [0.75])


def test_knn_distances_do_not_count_the_training_point_itself(scaled_sets):
    """A coluna j é o (j+1)-ésimo vizinho no treino, no treino e no conjunto avaliado."""
    X_train, X_eval, _, _ = scaled_sets

    train_distances, distances = knn_distances(X_train, X_eval, 5)

    np.testing.assert_allclose(train_distances, np.sort(cdist(X_train, X_train))[:, 1:6])
    np.testing.assert_allclose(distances, np.sort(cdist(X_eval, X_train))[:, :5])


def test_manual_anomaly_score_ranks_membership_and_density_against_the_training_set(scaled_sets):
    X_train, X_eval, train_probabilities, probabilities = scaled_sets
    min_samples = 4
    train_density = np.sort(cdist(X_train, X_train))[:, min_samples]
    density = np.sort(cdist(X_eval, X_train))[:, min_samples - 1]
    train_components = [1.0 - train_probabilities, train_density]

    train_degree, degree, components = manual_anomaly_score(
        X_train, X_eval, train_probabilities, probabilities, min_samples
    )

    np.testing.assert_allclose(train_degree, ensemble_score(train_components, train_components))
    np.testing.assert_allclose(degree, ensemble_score(train_components, components))
    np.testing.assert_allclose(components[0], 1.0 - probabilities)
    np.testing.assert_allclose(components[1], density)


def test_manual_anomaly_score_reads_precomputed_neighbors_for_any_smaller_min_samples(
    scaled_sets,
):
    """Na grade, o kNN do maior min_samples serve a todas as configurações."""
    X_train, X_eval, train_probabilities, probabilities = scaled_sets
    neighbors = knn_distances(X_train, X_eval, 10)

    for min_samples in (2, 6, 10):
        computed = manual_anomaly_score(
            X_train, X_eval, train_probabilities, probabilities, min_samples
        )
        reused = manual_anomaly_score(
            X_train, X_eval, train_probabilities, probabilities, min_samples, neighbors=neighbors
        )
        for expected, actual in zip(computed[:2], reused[:2], strict=True):
            np.testing.assert_allclose(actual, expected)


@pytest.fixture
def tree_sets():
    """Treino de 7 pontos, um por folha de TREE, e 5 pontos avaliados."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(7, 2)), rng.normal(size=(5, 2))


def test_manual_anomaly_score_with_glosh_replaces_membership_and_keeps_the_density(tree_sets):
    X_train, X_eval = tree_sets
    min_samples = 3
    train_glosh, glosh = glosh_anomaly_score(TREE, X_train, X_eval, min_samples)
    train_distances, distances = knn_distances(X_train, X_eval, min_samples)
    train_components = [train_glosh, train_distances[:, min_samples - 1]]
    expected_components = [glosh, distances[:, min_samples - 1]]

    train_degree, degree, components = manual_anomaly_score(
        X_train, X_eval, None, None, min_samples, component="glosh", tree=TREE
    )

    np.testing.assert_allclose(train_degree, ensemble_score(train_components, train_components))
    np.testing.assert_allclose(degree, ensemble_score(train_components, expected_components))
    for expected, actual in zip(expected_components, components, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_manual_anomaly_score_with_glosh_reads_both_precomputed_neighbor_sets(tree_sets):
    """Na grade, o kNN e os vizinhos do GLOSH do maior min_samples servem a todas."""
    X_train, X_eval = tree_sets
    neighbors = knn_distances(X_train, X_eval, 3)
    tree_neighbors = glosh_neighbors(X_train, X_eval, 3)

    for min_samples in (2, 3):
        computed = manual_anomaly_score(
            X_train, X_eval, None, None, min_samples, component="glosh", tree=TREE
        )
        reused = manual_anomaly_score(
            X_train,
            X_eval,
            None,
            None,
            min_samples,
            neighbors=neighbors,
            component="glosh",
            tree=TREE,
            tree_neighbors=tree_neighbors,
        )
        for expected, actual in zip(computed[:2], reused[:2], strict=True):
            np.testing.assert_allclose(actual, expected)


def test_manual_anomaly_score_rejects_missing_inputs_and_unknown_components(scaled_sets):
    X_train, X_eval, train_probabilities, probabilities = scaled_sets

    with pytest.raises(ValueError, match="exige train_probabilities"):
        manual_anomaly_score(X_train, X_eval, None, None, 4)
    with pytest.raises(ValueError, match="exige a árvore condensada"):
        manual_anomaly_score(X_train, X_eval, None, None, 4, component="glosh")
    with pytest.raises(ValueError, match="'membership' ou 'glosh'"):
        manual_anomaly_score(
            X_train, X_eval, train_probabilities, probabilities, 4, component="pertinencia"
        )


def test_cluster_max_lambdas_propagates_from_the_deepest_cluster_to_the_root():
    np.testing.assert_allclose(cluster_max_lambdas(TREE), MAX_LAMBDAS)


def test_cluster_max_lambdas_does_not_depend_on_row_order_or_cluster_numbering():
    """O agrupamento 8 passa a ser filho do 11, com rótulo menor que o do pai."""
    relabel = {8: 11, 11: 8}
    tree = {
        "parent": [relabel.get(label, label) for label in TREE["parent"]],
        "child": [relabel.get(label, label) for label in TREE["child"]],
        "lambda_val": TREE["lambda_val"],
    }
    tree = reordered(tree, np.random.default_rng(0).permutation(len(TREE["child"])))

    np.testing.assert_allclose(cluster_max_lambdas(tree), [8.0, 3.0, 1.0, 8.0, 8.0])


def test_glosh_scores_follow_equation_8():
    np.testing.assert_allclose(glosh_scores(TREE), GLOSH)


def test_glosh_scores_are_zero_for_points_at_infinite_density():
    """Pontos repetidos têm distância núcleo zero e saem do agrupamento em λ infinito."""
    tree = {
        "parent": [3, 4, 4, 3],
        "child": [4, 0, 1, 2],
        "lambda_val": [0.5, np.inf, np.inf, 0.2],
    }

    np.testing.assert_allclose(glosh_scores(tree), [0.0, 0.0, 1.0])


def test_glosh_scores_reject_a_tree_without_one_row_per_point():
    tree = {"parent": [3, 3], "child": [0, 1], "lambda_val": [1.0, 1.0]}

    with pytest.raises(ValueError, match="uma linha por ponto"):
        glosh_scores(tree)


def test_a_new_point_identical_to_a_training_point_gets_the_same_glosh():
    n = len(TRAIN_CORE_DISTANCES)

    scores = glosh_new_points(
        TREE,
        TRAIN_CORE_DISTANCES,
        core_distances=TRAIN_CORE_DISTANCES,
        neighbor_indices=np.arange(n)[:, np.newaxis],
        neighbor_distances=np.zeros((n, 1)),
    )

    np.testing.assert_allclose(scores, GLOSH)


def test_a_new_point_joining_before_its_neighbor_cluster_is_born_uses_the_ancestor():
    """Ligado ao ponto 2 em λ = 0.25, antes de o agrupamento 11 nascer (λ = 0.5), o ponto
    novo pertence ao 8, cujo λ_max é 8: 1 - 0.25/8, e não 1 - 0.25/3."""
    scores = glosh_new_points(
        TREE,
        TRAIN_CORE_DISTANCES,
        core_distances=np.array([4.0]),
        neighbor_indices=np.array([[2]]),
        neighbor_distances=np.array([[4.0]]),
    )

    np.testing.assert_allclose(scores, [1 - 0.25 / 8])


def test_a_new_point_follows_the_neighbor_that_keeps_it_in_a_cluster_the_longest():
    """O ponto 6 é o vizinho de menor distância de alcançabilidade mútua (0.9 contra 1.0),
    mas sai da raiz em λ = 0.1. Pelo ponto 4, o ponto novo fica no agrupamento 9 até λ = 1,
    a densidade máxima do agrupamento."""
    train_core_distances = TRAIN_CORE_DISTANCES.copy()
    train_core_distances[6] = 0.2

    scores = glosh_new_points(
        TREE,
        train_core_distances,
        core_distances=np.array([0.9]),
        neighbor_indices=np.array([[6, 4]]),
        neighbor_distances=np.array([[0.3, 0.9]]),
    )

    np.testing.assert_allclose(scores, [0.0])


def test_a_far_new_point_is_compared_with_the_whole_dataset():
    scores = glosh_new_points(
        TREE,
        TRAIN_CORE_DISTANCES,
        core_distances=np.array([100.0]),
        neighbor_indices=np.array([[6]]),
        neighbor_distances=np.array([[100.0]]),
    )

    np.testing.assert_allclose(scores, [1 - 0.01 / 8])


def test_glosh_anomaly_score_reads_precomputed_neighbors_for_any_smaller_min_samples():
    """Na grade, os vizinhos do maior min_samples servem a todas as configurações."""
    rng = np.random.default_rng(0)
    X_train, X_eval = rng.normal(size=(7, 2)), rng.normal(size=(5, 2))
    neighbors = glosh_neighbors(X_train, X_eval, 3)

    for min_samples in (2, 3):
        computed = glosh_anomaly_score(TREE, X_train, X_eval, min_samples)
        reused = glosh_anomaly_score(TREE, X_train, X_eval, min_samples, neighbors=neighbors)
        for expected, actual in zip(computed, reused, strict=True):
            np.testing.assert_allclose(actual, expected)
        assert np.all((computed[1] >= 0) & (computed[1] <= 1))


def test_glosh_anomaly_score_rejects_min_samples_below_two():
    X = np.zeros((7, 2))

    with pytest.raises(ValueError, match="pelo menos 2"):
        glosh_anomaly_score(TREE, X, X, 1)


@pytest.fixture
def hdbscan_clusterer():
    """HDBSCAN* da biblioteca hdbscan, que conta o próprio ponto na distância núcleo."""
    hdbscan = pytest.importorskip("hdbscan")
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(0, 0.5, (300, 2)),
            rng.normal(5, 1.5, (300, 2)),
            rng.uniform(-6, 12, (40, 2)),
        ]
    )
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10, prediction_data=True)
    return clusterer.fit(X), X


def test_matches_the_hdbscan_library_on_the_training_points(hdbscan_clusterer):
    clusterer, _ = hdbscan_clusterer

    np.testing.assert_allclose(
        glosh_scores(clusterer.condensed_tree_.to_numpy()), clusterer.outlier_scores_
    )


def test_glosh_neighbors_reproduce_the_core_distances_of_hdbscan(hdbscan_clusterer):
    clusterer, X = hdbscan_clusterer

    train_distances, _, _ = glosh_neighbors(X, X[:5], 10)

    np.testing.assert_allclose(train_distances[:, 9], clusterer.prediction_data_.core_distances)


def write_info(tmp_path, info):
    path = tmp_path / "fold_1_internal_info.json"
    path.write_text(json.dumps(info), encoding="utf-8")
    return path


def test_the_full_base_gives_the_legitimates_weight_one(tmp_path):
    assert read_legitimate_weight(write_info(tmp_path, {"fold": 1})) == 1.0


def test_each_kept_legitimate_weighs_the_legitimates_it_represents(tmp_path):
    info = {"undersampling": {"num_legitimate_before": 181_962, "num_legitimate_after": 64_086}}

    assert read_legitimate_weight(write_info(tmp_path, info)) == pytest.approx(2.8393, abs=1e-4)


def test_without_weights_the_threshold_is_the_linear_quantile_of_the_training_scores():
    scores = np.random.default_rng(0).normal(size=1_000)

    assert alert_threshold(scores, 0.01) == np.quantile(scores, 0.99)


def test_the_weighted_threshold_counts_each_legitimate_as_many_times_as_its_weight():
    rng = np.random.default_rng(1)
    legitimate, frauds = rng.normal(0, 1, 300), rng.normal(3, 1, 20)
    scores = np.r_[legitimate, frauds]
    labels = np.r_[np.zeros(300, int), np.ones(20, int)]
    expanded = np.r_[np.repeat(legitimate, 3), frauds]

    assert alert_threshold(scores, 0.05, labels, 3.0) == np.quantile(
        expanded, 0.95, method="inverted_cdf"
    )


def test_the_weighted_threshold_recovers_the_alert_rate_of_the_full_training_set():
    # proporções do treino interno: 315 fraudes, 35,2% das legítimas mantidas
    rng = np.random.default_rng(2)
    legitimate, frauds = rng.normal(0, 1, 180_000), rng.normal(3, 1, 315)
    full = np.r_[legitimate, frauds]
    kept = rng.choice(legitimate, round(0.352 * len(legitimate)), replace=False)
    scores = np.r_[kept, frauds]
    labels = np.r_[np.zeros(len(kept), int), np.ones(len(frauds), int)]
    weight = len(legitimate) / len(kept)

    weighted = alert_threshold(scores, 0.001, labels, weight)
    plain = alert_threshold(scores, 0.001)

    assert np.mean(full >= weighted) == pytest.approx(0.001, rel=0.15)
    assert np.mean(full >= plain) < 0.0006


def test_the_weighted_threshold_requires_the_training_labels():
    with pytest.raises(ValueError, match="train_labels"):
        alert_threshold(np.arange(10.0), 0.1, legitimate_weight=2.0)
