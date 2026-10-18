"""Testes de cross_validation.external_k_folds.

A função materializa em disco o ciclo externo da validação aninhada: para cada
fold, um par de CSVs de treino e teste mais um JSON de metadados. Todo o
restante do pipeline consome esses arquivos, de modo que qualquer inconsistência
aqui se propaga silenciosamente até os resultados finais.
"""

import json
import os

import pandas as pd
import pytest

from cross_validation import external_k_folds

N_SPLITS = 5


@pytest.fixture
def output_dir(tmp_path, dataset_csv):
    directory = tmp_path / "outer_folds"
    external_k_folds(
        csv_path=dataset_csv,
        output_dir=str(directory),
        target="Class",
        n_splits=N_SPLITS,
        random_state=42,
    )
    return directory


def read_fold(directory, fold, partition):
    return pd.read_csv(directory / f"fold_{fold}_{partition}.csv")


def read_metadata(directory, fold):
    with open(directory / f"fold_{fold}_info.json", encoding="utf-8") as file:
        return json.load(file)


def test_creates_three_files_per_fold(output_dir):
    for fold in range(1, N_SPLITS + 1):
        assert (output_dir / f"fold_{fold}_train.csv").exists()
        assert (output_dir / f"fold_{fold}_test.csv").exists()
        assert (output_dir / f"fold_{fold}_info.json").exists()


def test_does_not_create_extra_folds(output_dir):
    train_files = list(output_dir.glob("fold_*_train.csv"))
    assert len(train_files) == N_SPLITS


def test_creates_the_output_directory_when_absent(tmp_path, dataset_csv):
    directory = tmp_path / "nao" / "existe" / "ainda"
    external_k_folds(dataset_csv, str(directory), "Class", n_splits=2, random_state=42)
    assert directory.is_dir()


def test_rejects_a_missing_input_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        external_k_folds(
            csv_path=str(tmp_path / "inexistente.csv"),
            output_dir=str(tmp_path / "saida"),
            target="Class",
        )


def test_preserves_every_column_of_the_original_base(output_dir, dataset):
    for fold in range(1, N_SPLITS + 1):
        for partition in ("train", "test"):
            frame = read_fold(output_dir, fold, partition)
            assert list(frame.columns) == list(dataset.columns)


def test_train_and_test_sum_to_the_original_size(output_dir, dataset):
    for fold in range(1, N_SPLITS + 1):
        train = read_fold(output_dir, fold, "train")
        test = read_fold(output_dir, fold, "test")
        assert len(train) + len(test) == len(dataset)


def test_respects_the_eighty_twenty_proportion(output_dir, dataset):
    for fold in range(1, N_SPLITS + 1):
        test = read_fold(output_dir, fold, "test")
        assert len(test) == pytest.approx(0.2 * len(dataset), rel=0.02)


def test_no_transaction_appears_in_both_partitions_of_a_fold(output_dir):
    """Vazamento entre treino e teste inflaria artificialmente o desempenho."""
    for fold in range(1, N_SPLITS + 1):
        train = read_fold(output_dir, fold, "train")
        test = read_fold(output_dir, fold, "test")

        shared = pd.merge(train, test, how="inner")
        assert shared.empty, f"fold {fold} repete transações entre treino e teste"


def test_metadata_declares_the_partition_sizes(output_dir):
    for fold in range(1, N_SPLITS + 1):
        metadata = read_metadata(output_dir, fold)
        train = read_fold(output_dir, fold, "train")
        test = read_fold(output_dir, fold, "test")

        assert metadata["num_train_samples"] == len(train)
        assert metadata["num_test_samples"] == len(test)


def test_metadata_carries_the_reproducibility_parameters(output_dir):
    metadata = read_metadata(output_dir, 1)
    assert metadata["random_state"] == 42
    assert metadata["target_column"] == "Class"
    assert metadata["fold"] == 1


def test_metadata_records_the_class_distribution_of_both_partitions(output_dir):
    for fold in range(1, N_SPLITS + 1):
        metadata = read_metadata(output_dir, fold)

        for key in ("class_distribution_train", "class_distribution_test"):
            distribution = metadata[key]
            assert set(distribution) <= {"0", "1"}
            assert sum(distribution.values()) == pytest.approx(1.0)


def test_metadata_indices_match_the_written_files(output_dir):
    for fold in range(1, N_SPLITS + 1):
        metadata = read_metadata(output_dir, fold)
        assert len(metadata["train_indices"]) == metadata["num_train_samples"]
        assert len(metadata["test_indices"]) == metadata["num_test_samples"]


def test_metadata_paths_point_to_existing_files(output_dir):
    metadata = read_metadata(output_dir, 1)
    assert os.path.exists(metadata["train_file"])
    assert os.path.exists(metadata["test_file"])


def test_test_partitions_together_reconstruct_the_whole_base(output_dir, dataset):
    """Cobertura integral verificada sobre os arquivos efetivamente gravados."""
    partitions = [read_fold(output_dir, fold, "test") for fold in range(1, N_SPLITS + 1)]
    reunited = pd.concat(partitions, ignore_index=True)

    assert len(reunited) == len(dataset)

    expected = dataset.sort_values(list(dataset.columns)).reset_index(drop=True)
    obtained = reunited.sort_values(list(dataset.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, obtained)


def test_preserves_the_fraud_ratio_across_folds(output_dir, dataset):
    population_ratio = dataset["Class"].mean()

    for fold in range(1, N_SPLITS + 1):
        test = read_fold(output_dir, fold, "test")
        assert test["Class"].mean() == pytest.approx(population_ratio, abs=0.01)


def test_accepts_the_target_given_by_index(tmp_path, dataset_csv, dataset):
    directory = tmp_path / "por_indice"
    external_k_folds(dataset_csv, str(directory), target=len(dataset.columns) - 1, n_splits=2)
    metadata = read_metadata(directory, 1)
    assert metadata["target_column"] == "Class"


def test_is_reproducible_for_the_same_seed(tmp_path, dataset_csv):
    first = tmp_path / "primeira"
    second = tmp_path / "segunda"

    for directory in (first, second):
        external_k_folds(dataset_csv, str(directory), "Class", n_splits=3, random_state=42)

    for fold in range(1, 4):
        pd.testing.assert_frame_equal(
            read_fold(first, fold, "test"), read_fold(second, fold, "test")
        )
