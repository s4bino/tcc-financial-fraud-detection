"""Testes de cross_validation.stratified_folds.

Esta função implementa o ciclo externo da validação cruzada aninhada. As
propriedades verificadas aqui são exatamente as afirmadas na metodologia do
trabalho: partições mutuamente exclusivas, cobertura integral da base e
preservação da proporção de fraudes em cada fold.

Sob desbalanceamento de 0,17%, a amostragem estratificada não é uma
preferência estilística — é o que impede que um fold receba fraudes demais ou
de menos e distorça a estimativa de desempenho.
"""

import numpy as np
import pytest

from cross_validation import stratified_folds


@pytest.fixture
def folds(dataset):
    return stratified_folds(dataset, target="Class", n_splits=5, random_state=42)


def test_produces_the_requested_number_of_folds(folds):
    assert len(folds) == 5


def test_folds_are_numbered_from_one(folds):
    assert [fold["fold"] for fold in folds] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("n_splits", [2, 3, 5, 10])
def test_honours_different_split_counts(dataset, n_splits):
    folds = stratified_folds(dataset, "Class", n_splits, random_state=42)
    assert len(folds) == n_splits


def test_train_and_test_are_disjoint_within_each_fold(folds):
    for fold in folds:
        overlap = np.intersect1d(fold["train_idx"], fold["test_idx"])
        assert overlap.size == 0, f"fold {fold['fold']} sobrepõe treino e teste"


def test_train_and_test_together_cover_the_whole_base(folds, dataset):
    for fold in folds:
        union = np.union1d(fold["train_idx"], fold["test_idx"])
        assert union.size == len(dataset)


def test_every_row_is_tested_exactly_once_across_the_folds(folds, dataset):
    """Cobertura integral: a base inteira é avaliada, sem repetição.

    É a propriedade que sustenta a afirmação de que o método 'assegura a
    cobertura da base, analisando todas as transações de forma independente'.
    """
    all_test_indices = np.concatenate([fold["test_idx"] for fold in folds])

    assert all_test_indices.size == len(dataset)
    assert np.array_equal(np.sort(all_test_indices), np.arange(len(dataset)))


def test_preserves_the_fraud_ratio_in_every_test_partition(folds, dataset):
    population_ratio = dataset["Class"].mean()

    for fold in folds:
        test_ratio = dataset.iloc[fold["test_idx"]]["Class"].mean()
        # Tolerância de um ponto percentual: com 20 fraudes em 5 folds, o
        # arredondamento inteiro por fold já responde por parte do desvio.
        assert test_ratio == pytest.approx(population_ratio, abs=0.01)


def test_preserves_the_fraud_ratio_in_every_train_partition(folds, dataset):
    population_ratio = dataset["Class"].mean()

    for fold in folds:
        train_ratio = dataset.iloc[fold["train_idx"]]["Class"].mean()
        assert train_ratio == pytest.approx(population_ratio, abs=0.01)


def test_every_fold_contains_both_classes(folds, dataset):
    """Sem as duas classes, métricas como AUC e recall ficam indefinidas."""
    for fold in folds:
        for partition in ("train_idx", "test_idx"):
            classes = dataset.iloc[fold[partition]]["Class"].unique()
            assert set(classes) == {0, 1}, f"fold {fold['fold']}, {partition}"


def test_test_partitions_have_approximately_equal_sizes(folds, dataset):
    sizes = [fold["test_idx"].size for fold in folds]
    assert max(sizes) - min(sizes) <= 1


def test_the_split_is_eighty_twenty_for_five_folds(folds, dataset):
    for fold in folds:
        assert fold["train_idx"].size == pytest.approx(0.8 * len(dataset), rel=0.01)
        assert fold["test_idx"].size == pytest.approx(0.2 * len(dataset), rel=0.01)


def test_is_reproducible_for_the_same_seed(dataset):
    first = stratified_folds(dataset, "Class", 5, random_state=42)
    second = stratified_folds(dataset, "Class", 5, random_state=42)

    for fold_a, fold_b in zip(first, second):
        assert np.array_equal(fold_a["train_idx"], fold_b["train_idx"])
        assert np.array_equal(fold_a["test_idx"], fold_b["test_idx"])


def test_different_seeds_produce_different_partitions(dataset):
    first = stratified_folds(dataset, "Class", 5, random_state=42)
    second = stratified_folds(dataset, "Class", 5, random_state=7)

    assert any(not np.array_equal(a["test_idx"], b["test_idx"]) for a, b in zip(first, second))


def test_shuffles_instead_of_slicing_the_base_in_order(folds):
    """A base original é ordenada por Time; folds contíguos introduziriam viés
    temporal, já que a taxa de fraude varia fortemente com a hora do dia."""
    first_test = folds[0]["test_idx"]
    assert not np.array_equal(first_test, np.arange(first_test.size))


def test_indices_are_valid_positions_in_the_base(folds, dataset):
    for fold in folds:
        for partition in ("train_idx", "test_idx"):
            indices = fold[partition]
            assert indices.min() >= 0
            assert indices.max() < len(dataset)


def test_survives_the_real_imbalance_ratio(dataset_factory):
    """Com 0,17% de fraudes, cada fold recebe pouquíssimos positivos."""
    base = dataset_factory(n_samples=10_000, n_frauds=17)
    folds = stratified_folds(base, "Class", 5, random_state=42)

    for fold in folds:
        assert base.iloc[fold["test_idx"]]["Class"].sum() >= 1


def test_fewer_positives_than_folds_degrades_silently(dataset_factory):
    """Com menos fraudes que folds, a estratificação não falha — ela degrada.

    StratifiedKFold apenas emite um aviso e prossegue, e alguns folds ficam
    sem nenhuma fraude na partição de teste. Nessa condição recall fica
    indefinido e AUC não pode ser calculado. É uma restrição concreta da base
    do trabalho: as 492 fraudes são o fator que limita o número de folds, não
    as 284.807 transações.
    """
    base = dataset_factory(n_samples=100, n_frauds=3)

    with pytest.warns(UserWarning, match="least populated class"):
        folds = stratified_folds(base, "Class", n_splits=10, random_state=42)

    frauds_per_fold = [base.iloc[fold["test_idx"]]["Class"].sum() for fold in folds]
    assert min(frauds_per_fold) == 0


def test_rejects_more_splits_than_samples(dataset_factory):
    """Limite absoluto: não há como formar mais folds que amostras."""
    base = dataset_factory(n_samples=20, n_frauds=4)

    with pytest.raises(ValueError):
        stratified_folds(base, "Class", n_splits=25, random_state=42)
