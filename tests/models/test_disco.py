"""Testes do DISCO (Beer et al., 2025): as Equações 1 a 8 e os casos especiais do artigo.

Os exemplos unidimensionais têm os valores esperados calculados à mão: com min_points = 2, a
distância de núcleo é a distância ao vizinho mais próximo, já que o ponto conta como o seu primeiro
vizinho. Os valores de REFERENCE_VALUES vieram do código oficial dos autores
(https://github.com/pasiweber/DISCO, commit e685d5b) sobre os dados de reference_data, o que mantém
a conferência com o original sem depender daquele repositório.
"""

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

from disco import DEFAULT_MIN_POINTS, NOISE_LABEL, DCTree, disco_samples, disco_score

TOLERANCE = 1e-10

# Dois grupos separados: distância 1 dentro de cada um e 8 entre eles.
TWO_GROUPS = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
TWO_GROUPS_LABELS = np.array([0, 0, 0, 1, 1, 1])
TWO_GROUPS_SCORE = (8 - 1) / 8

# (min_points, média, escores dos pontos de POINTS_CHECKED)
POINTS_CHECKED = (0, 50, 90, -1)
REFERENCE_VALUES = {
    "clusters_and_noise": (
        5,
        0.8419854045841002,
        [0.8995505313196491, 0.7830108139690559, 0.9252319506000871, 0.5652427841946567],
    ),
    "clusters_only": (
        5,
        0.9174002791001966,
        [0.9260426003128243, 0.8661998081444514, 0.9538964151285085, 0.9532309487959028],
    ),
    "one_cluster_and_noise": (
        5,
        0.2182792603295736,
        [0.8037877915982715, 0.09456028355207641, -0.6712542130760478, 0.6858154320433661],
    ),
    "larger_min_points": (
        10,
        0.7969424502640674,
        [0.8686164456420082, 0.7413659130452663, 0.9081577678093292, 0.41495870222108133],
    ),
    "singleton_clusters": (
        5,
        0.4984513379877023,
        [0.0, 0.7830108139690559, 0.9252319506000871, 0.5652427841946567],
    ),
}


def reference_data(name):
    """Monta os pontos e os rótulos de cada caso de REFERENCE_VALUES."""
    rng = np.random.RandomState(7)
    X = np.r_[
        rng.normal(0, 0.3, (40, 3)),
        rng.normal(4, 0.5, (30, 3)),
        rng.normal((8, 0, 0), 0.2, (30, 3)),
        rng.uniform(-2, 10, (20, 3)),
    ]
    labels = np.r_[np.zeros(40, int), np.ones(30, int), np.full(30, 2), np.full(20, NOISE_LABEL)]
    if name == "clusters_only":
        return X[:100], labels[:100]
    if name == "one_cluster_and_noise":
        return X, np.where(labels == 0, 0, NOISE_LABEL)
    if name == "singleton_clusters":
        labels[:3] = [10, 11, 12]
    return X, labels


def dc_distance_matrix(X, min_points):
    """Calcula a distância dc pela definição: a maior aresta do caminho na árvore mínima."""
    euclidean = cdist(X, X)
    core = np.sort(euclidean, axis=1)[:, min_points - 1]
    reachability = np.maximum(np.maximum(euclidean, core[None, :]), core[:, None])
    np.fill_diagonal(reachability, 0)
    # A soma de 1 evita que uma aresta de peso zero desapareça da matriz esparsa.
    mst = minimum_spanning_tree(reachability + 1).toarray() - 1
    i, j = np.nonzero(mst > -1)
    distances = np.full_like(reachability, np.inf)
    np.fill_diagonal(distances, 0)
    for a, b, length in zip(i, j, mst[i, j], strict=True):
        distances[a, b] = distances[b, a] = length
    for k in range(len(X)):
        distances = np.minimum(distances, np.maximum(distances[:, [k]], distances[[k], :]))
    return distances


@pytest.mark.parametrize("name", REFERENCE_VALUES)
def test_matches_the_official_implementation(name):
    X, labels = reference_data(name)
    min_points, mean, points = REFERENCE_VALUES[name]

    scores = disco_samples(X, labels, min_points)

    assert scores.mean() == pytest.approx(mean, abs=TOLERANCE)
    assert scores[list(POINTS_CHECKED)] == pytest.approx(points, abs=TOLERANCE)


def test_cluster_points_follow_the_silhouette_over_the_dc_distance():
    scores = disco_samples(TWO_GROUPS, TWO_GROUPS_LABELS, min_points=2)

    assert scores == pytest.approx(np.full(6, TWO_GROUPS_SCORE))


def test_nearest_cluster_is_the_one_with_the_smallest_mean_distance():
    # Equação 4: o mínimo é tomado entre os outros agrupamentos, e não sobre todos os pontos deles.
    X = np.array([[0.0], [1.0], [10.0], [11.0], [100.0], [101.0]])

    scores = disco_samples(X, np.array([0, 0, 1, 1, 2, 2]), min_points=2)

    assert scores[0] == pytest.approx((9 - 1) / 9)
    assert scores[4] == pytest.approx((89 - 1) / 89)


def test_well_placed_noise_scores_high():
    X = np.r_[TWO_GROUPS, [[50.0]]]
    labels = np.r_[TWO_GROUPS_LABELS, [NOISE_LABEL]]

    scores = disco_samples(X, labels, min_points=2)

    # Núcleo do ruído = 38, núcleo dos agrupamentos = 1, distância dc ao mais próximo = 38.
    assert scores[-1] == pytest.approx((38 - 1) / 38)
    assert scores[:-1] == pytest.approx(np.full(6, TWO_GROUPS_SCORE))


def test_noise_denser_than_a_cluster_is_penalized_even_when_far():
    X = np.array([[0.0], [10.0], [20.0], [100.0], [101.0], [102.0]])
    labels = np.array([0, 0, 0, NOISE_LABEL, NOISE_LABEL, NOISE_LABEL])

    scores = disco_samples(X, labels, min_points=2)

    # Equações 5 a 7: longe do agrupamento (0,875), mas mais denso que ele (-0,9). Vale o mínimo.
    assert scores[3:] == pytest.approx(np.full(3, (1 - 10) / 10))


def test_noise_inside_a_cluster_is_not_rewarded():
    X = np.arange(5.0)[:, None]

    scores = disco_samples(X, np.array([0, 0, NOISE_LABEL, 0, 0]), min_points=2)

    assert scores == pytest.approx(np.zeros(5))


def test_only_noise_scores_minus_one():
    assert disco_score(TWO_GROUPS, np.full(6, NOISE_LABEL)) == -1.0


def test_single_cluster_without_noise_scores_zero():
    assert disco_score(TWO_GROUPS, np.zeros(6, int)) == 0.0


def test_single_cluster_with_noise_compares_with_the_nearest_noise():
    X = np.array([[0.0], [1.0], [2.0], [50.0]])

    scores = disco_samples(X, np.array([0, 0, 0, NOISE_LABEL]), min_points=2)

    # Equação 8: distância média interna 1 e distância dc 48 até o ruído.
    assert scores == pytest.approx(np.full(4, (48 - 1) / 48))


def test_singleton_clusters_score_zero():
    scores = disco_samples(TWO_GROUPS, np.array([0, 0, 0, 1, 1, 2]), min_points=2)

    assert scores[5] == 0.0
    assert scores[0] > 0


def test_every_point_a_singleton_scores_zero():
    assert disco_score(TWO_GROUPS, np.arange(6), min_points=2) == 0.0


def test_duplicates_use_zero_for_an_undefined_ratio():
    # Pontos repetidos ao menos min_points vezes têm núcleo 0 (Seção 3.1 do artigo).
    X = np.array([[0.0]] * 4 + [[5.0]] * 4)
    labels = np.array([0, 0, 0, 0, NOISE_LABEL, NOISE_LABEL, NOISE_LABEL, NOISE_LABEL])

    scores = disco_samples(X, labels, min_points=3)

    assert scores[4:] == pytest.approx(np.zeros(4))


def test_all_points_identical_are_finite():
    scores = disco_samples(np.zeros((10, 2)), np.r_[np.zeros(5, int), np.ones(5, int)])

    assert scores == pytest.approx(np.zeros(10))


def test_tree_distances_match_the_definition():
    X = np.random.RandomState(0).normal(size=(60, 3))
    min_points = 4
    distances = dc_distance_matrix(X, min_points)
    tree = DCTree(X, min_points)
    mask = np.zeros(60, dtype=bool)
    mask[[1, 7, 30, 44]] = True

    expected_sum = distances[:, mask].sum(axis=1)
    masked_distances = distances[:, mask].copy()
    masked_distances[np.arange(60)[mask], np.arange(4)] = np.inf

    assert tree.sum_distances_to(mask) == pytest.approx(expected_sum)
    assert tree.min_distance_to(mask) == pytest.approx(masked_distances.min(axis=1))


def test_core_distance_counts_the_point_as_its_own_first_neighbor():
    tree = DCTree(np.array([[0.0], [1.0], [3.0], [7.0]]), min_points=2)

    assert tree.core_distances == pytest.approx([1, 1, 2, 4])


def test_score_is_the_mean_of_the_samples():
    X, labels = reference_data("clusters_and_noise")

    assert disco_score(X, labels) == pytest.approx(disco_samples(X, labels).mean())


def test_reusing_the_tree_gives_the_same_result():
    X, labels = reference_data("clusters_and_noise")
    tree = DCTree(X, DEFAULT_MIN_POINTS)

    assert disco_samples(X, labels, tree=tree) == pytest.approx(disco_samples(X, labels))


def test_one_tree_serves_different_partitions():
    X, labels = reference_data("clusters_and_noise")
    other = np.where(labels == 2, NOISE_LABEL, labels)
    tree = DCTree(X, DEFAULT_MIN_POINTS)

    assert disco_score(X, other, tree=tree) == pytest.approx(disco_score(X, other))


def test_result_does_not_depend_on_the_order_of_the_points():
    X, labels = reference_data("clusters_and_noise")
    order = np.random.RandomState(1).permutation(len(X))

    assert disco_samples(X[order], labels[order]) == pytest.approx(disco_samples(X, labels)[order])


def test_result_does_not_depend_on_the_cluster_ids():
    X, labels = reference_data("clusters_and_noise")
    renamed = np.where(labels == NOISE_LABEL, NOISE_LABEL, labels * 7 + 3)

    assert disco_score(X, renamed) == pytest.approx(disco_score(X, labels))


def test_scores_stay_between_minus_one_and_one():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(200, 4))

    scores = disco_samples(X, rng.randint(-1, 4, 200))

    assert scores.min() >= -1 and scores.max() <= 1


def test_min_points_may_be_as_large_as_the_dataset():
    assert np.isfinite(disco_score(TWO_GROUPS, TWO_GROUPS_LABELS, min_points=6))


@pytest.mark.parametrize("min_points", [0, 7])
def test_invalid_min_points_is_rejected(min_points):
    with pytest.raises(ValueError, match="min_points"):
        DCTree(TWO_GROUPS, min_points)


def test_mismatched_sizes_are_rejected():
    with pytest.raises(ValueError, match="rótulos"):
        disco_samples(TWO_GROUPS, np.zeros(5, int))


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="vazio"):
        disco_samples(np.empty((0, 2)), np.empty(0, int))


@pytest.mark.parametrize(
    "points, min_points",
    [(TWO_GROUPS[:5], 2), (TWO_GROUPS, 3)],
    ids=["outro número de pontos", "outro min_points"],
)
def test_tree_from_other_data_is_rejected(points, min_points):
    tree = DCTree(points, min_points)

    with pytest.raises(ValueError, match="árvore"):
        disco_samples(TWO_GROUPS, TWO_GROUPS_LABELS, min_points=2, tree=tree)
