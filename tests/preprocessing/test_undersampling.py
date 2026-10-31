"""Testes de undersampling: os folds com as legítimas subamostradas no treino.

A subamostragem só pode tocar as partições de treino. As fraudes, poucas e
caras de perder, ficam todas, a validação interna e o teste externo seguem
iguais aos originais, na proporção real, e o treino interno continua contido no
treino externo. Sem essas propriedades, as métricas deixariam de ser
comparáveis às dos folds originais ou passariam a medir uma prevalência
artificial.
"""

import json

import numpy as np
import pandas as pd
import pytest

from cross_validation import external_k_folds, internal_k_folds
from undersampling import (
    internal_split,
    legitimate_fraction,
    undersample_legitimate,
    undersampled_k_folds,
)

N_SPLITS = 3
VAL_SIZE = 0.2
TARGET_LEGITIMATE = 490  # metade das 980 legítimas da base sintética


@pytest.fixture
def outer_dir(tmp_path, dataset_csv):
    directory = tmp_path / "outer_folds"
    external_k_folds(dataset_csv, str(directory), "Class", n_splits=N_SPLITS, random_state=42)
    return directory


@pytest.fixture
def inner_dir(tmp_path, outer_dir):
    directory = tmp_path / "inner_folds"
    internal_k_folds(str(outer_dir), str(directory), "Class", val_size=VAL_SIZE, random_state=42)
    return directory


@pytest.fixture
def undersampled_dir(tmp_path, dataset_csv, outer_dir):
    directory = tmp_path / "undersampled"
    undersampled_k_folds(
        dataset_csv,
        str(outer_dir),
        str(directory),
        "Class",
        target_legitimate=TARGET_LEGITIMATE,
        val_size=VAL_SIZE,
        random_state=42,
    )
    return directory


def read_csv(directory, name):
    return pd.read_csv(directory / name)


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def outer_indices(directory, fold):
    return set(read_json(directory / f"fold_{fold}_info.json")["train_indices"])


def inner_indices(undersampled_dir, fold):
    path = undersampled_dir / "inner_folds" / f"fold_{fold}_internal_info.json"
    return set(read_json(path)["internal_validation"]["train_final_indices"])


def test_legitimate_fraction_scales_the_whole_dataset_to_the_target():
    labels = np.array([0] * 8 + [1] * 2)

    assert legitimate_fraction(labels, 4) == pytest.approx(0.5)


@pytest.mark.parametrize("target", [0, 9])
def test_legitimate_fraction_rejects_targets_outside_the_legitimate_count(target):
    with pytest.raises(ValueError, match="entre 1 e 8"):
        legitimate_fraction(np.array([0] * 8 + [1] * 2), target)


def test_undersample_legitimate_keeps_every_fraud_and_the_original_order():
    indices = np.arange(100, 120)
    labels = np.array([1, 0, 0, 0, 0] * 4)

    kept = undersample_legitimate(indices, labels, 0.5, np.random.default_rng(0))

    assert set(indices[labels == 1]) <= set(kept)
    assert np.sum(np.isin(kept, indices[labels == 0])) == 8
    assert np.all(np.diff(kept) > 0)


def test_internal_split_reproduces_internal_k_folds(dataset, outer_dir, inner_dir):
    for fold in range(1, N_SPLITS + 1):
        train_idx = np.array(read_json(outer_dir / f"fold_{fold}_info.json")["train_indices"])
        labels = dataset["Class"].to_numpy()[train_idx]

        inner_idx, val_idx = internal_split(train_idx, labels, VAL_SIZE, 42)

        pd.testing.assert_frame_equal(
            dataset.iloc[inner_idx].reset_index(drop=True),
            read_csv(inner_dir, f"fold_{fold}_train.csv"),
        )
        pd.testing.assert_frame_equal(
            dataset.iloc[val_idx].reset_index(drop=True),
            read_csv(inner_dir, f"fold_{fold}_val.csv"),
        )


def test_creates_the_same_files_as_the_original_folds(undersampled_dir):
    for fold in range(1, N_SPLITS + 1):
        for name in ("train.csv", "test.csv", "info.json"):
            assert (undersampled_dir / "outer_folds" / f"fold_{fold}_{name}").exists()
        for name in ("train.csv", "val.csv", "internal_info.json"):
            assert (undersampled_dir / "inner_folds" / f"fold_{fold}_{name}").exists()


def test_keeps_every_fraud_of_the_training_partitions(undersampled_dir, outer_dir, inner_dir):
    for original_dir, subdir in ((outer_dir, "outer_folds"), (inner_dir, "inner_folds")):
        for fold in range(1, N_SPLITS + 1):
            original = read_csv(original_dir, f"fold_{fold}_train.csv")
            reduced = read_csv(undersampled_dir / subdir, f"fold_{fold}_train.csv")

            pd.testing.assert_frame_equal(
                reduced[reduced["Class"] == 1].reset_index(drop=True),
                original[original["Class"] == 1].reset_index(drop=True),
            )


def test_keeps_the_requested_fraction_of_the_legitimate_transactions(undersampled_dir, outer_dir):
    for fold in range(1, N_SPLITS + 1):
        original = read_csv(outer_dir, f"fold_{fold}_train.csv")
        reduced = read_csv(undersampled_dir / "outer_folds", f"fold_{fold}_train.csv")

        assert (reduced["Class"] == 0).sum() == round(0.5 * (original["Class"] == 0).sum())
        assert outer_indices(undersampled_dir / "outer_folds", fold) <= outer_indices(
            outer_dir, fold
        )


def test_evaluation_partitions_are_identical_to_the_original_ones(
    undersampled_dir, outer_dir, inner_dir
):
    """Validação e teste na proporção real, comparáveis aos folds originais."""
    for fold in range(1, N_SPLITS + 1):
        pd.testing.assert_frame_equal(
            read_csv(undersampled_dir / "outer_folds", f"fold_{fold}_test.csv"),
            read_csv(outer_dir, f"fold_{fold}_test.csv"),
        )
        pd.testing.assert_frame_equal(
            read_csv(undersampled_dir / "inner_folds", f"fold_{fold}_val.csv"),
            read_csv(inner_dir, f"fold_{fold}_val.csv"),
        )


def test_the_inner_train_stays_inside_the_outer_train(dataset, undersampled_dir, outer_dir):
    for fold in range(1, N_SPLITS + 1):
        train_idx = np.array(read_json(outer_dir / f"fold_{fold}_info.json")["train_indices"])
        original_inner, _ = internal_split(
            train_idx, dataset["Class"].to_numpy()[train_idx], VAL_SIZE, 42
        )
        inner = inner_indices(undersampled_dir, fold)

        assert inner <= outer_indices(undersampled_dir / "outer_folds", fold)
        assert inner <= set(original_inner)


def test_the_outer_test_never_reaches_the_training_partitions(undersampled_dir):
    for fold in range(1, N_SPLITS + 1):
        test = read_csv(undersampled_dir / "outer_folds", f"fold_{fold}_test.csv")
        for subdir in ("outer_folds", "inner_folds"):
            train = read_csv(undersampled_dir / subdir, f"fold_{fold}_train.csv")
            assert pd.merge(test, train, how="inner").empty


def test_metadata_indices_point_to_the_written_rows(dataset, undersampled_dir):
    for fold in range(1, N_SPLITS + 1):
        outer = read_json(undersampled_dir / "outer_folds" / f"fold_{fold}_info.json")
        inner = read_json(undersampled_dir / "inner_folds" / f"fold_{fold}_internal_info.json")
        rows = {
            ("outer_folds", "train"): outer["train_indices"],
            ("outer_folds", "test"): outer["test_indices"],
            ("inner_folds", "train"): inner["internal_validation"]["train_final_indices"],
            ("inner_folds", "val"): inner["internal_validation"]["val_indices"],
        }
        for (subdir, name), indices in rows.items():
            pd.testing.assert_frame_equal(
                dataset.iloc[indices].reset_index(drop=True),
                read_csv(undersampled_dir / subdir, f"fold_{fold}_{name}.csv"),
            )


def test_metadata_reports_the_undersampling_of_each_training_partition(undersampled_dir):
    summary = read_json(undersampled_dir / "outer_folds" / "fold_1_info.json")["undersampling"]
    train = read_csv(undersampled_dir / "outer_folds", "fold_1_train.csv")

    assert summary["legitimate_fraction"] == pytest.approx(0.5)
    assert summary["seed"] == [42, 1]
    assert summary["num_legitimate_after"] == (train["Class"] == 0).sum()
    assert summary["num_frauds"] == (train["Class"] == 1).sum()
    assert set(summary["ks_by_column"]) == set(train.columns.drop("Class"))
    assert 0 <= summary["ks_max_statistic"] <= 1


def test_is_reproducible_for_the_same_seed(tmp_path, dataset_csv, outer_dir):
    outputs = []
    for name in ("primeira", "segunda"):
        directory = tmp_path / name
        undersampled_k_folds(
            dataset_csv, str(outer_dir), str(directory), "Class", target_legitimate=490
        )
        outputs.append(read_csv(directory / "outer_folds", "fold_1_train.csv"))

    pd.testing.assert_frame_equal(*outputs)
