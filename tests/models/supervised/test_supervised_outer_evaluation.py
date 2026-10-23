"""Testes da avaliação externa dos modelos supervisionados.

best_params_by_fold indica a melhor configuração de cada fold interno, com a
métrica obtida, e supervised_outer_evaluation treina o modelo com uma única
configuração no treino de cada fold externo e o avalia no teste externo, como
no HDBSCAN* aprimorado. Se uma rodada for avaliada em outro teste, ou com outra
configuração, o desempenho reportado no trabalho não corresponde à validação
cruzada aninhada descrita.

As funções auxiliares clean_params e build_model, compartilhadas com a busca
em grade, também são testadas aqui.
"""

import numpy as np
import pandas as pd
import pytest

from grid_search import (
    best_params_by_fold,
    build_model,
    clean_params,
    manual_grid_search,
    supervised_outer_evaluation,
)

METRIC_COLUMNS = ["auprc", "precision", "recall", "f1"]

RF_PARAMS = {"n_estimators": 5, "min_samples_leaf": 2}


def test_clean_params_turns_nan_into_none():
    """None vira NaN ao passar por uma coluna numérica do pandas e precisa voltar."""
    assert clean_params({"max_samples": np.nan}) == {"max_samples": None}


def test_clean_params_restores_integer_hyperparameters():
    params = clean_params({"n_estimators": 100.0, "max_depth": 6.0, "learning_rate": 0.1})

    assert params == {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
    assert isinstance(params["n_estimators"], int)
    assert isinstance(params["max_depth"], int)


def test_clean_params_keeps_categorical_values():
    assert clean_params({"max_features": "sqrt", "class_weight": None}) == {
        "max_features": "sqrt",
        "class_weight": None,
    }


def test_xgboost_weights_the_frauds_by_the_class_ratio():
    y_train = pd.Series([0] * 90 + [1] * 10)
    model = build_model("xgb", {"n_estimators": 5}, y_train)

    assert model.get_params()["scale_pos_weight"] == pytest.approx(9.0)


def test_build_model_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="não reconhecido"):
        build_model("svm", {}, pd.Series([0, 1]))


def test_best_params_by_fold_picks_the_best_combination_of_each_fold():
    df_folds = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "n_estimators": [100, 200, 100, 200],
            "auprc": [0.8, 0.6, 0.5, 0.7],
        }
    )

    best = best_params_by_fold(df_folds, {"n_estimators": [100, 200]})

    assert best == {
        1: {"params": {"n_estimators": 100}, "best_score": 0.8},
        2: {"params": {"n_estimators": 200}, "best_score": 0.7},
    }


def test_best_params_by_fold_uses_the_requested_metric():
    df_folds = pd.DataFrame(
        {
            "fold": [1, 1],
            "n_estimators": [100, 200],
            "auprc": [0.8, 0.6],
            "f1": [0.3, 0.5],
        }
    )

    best = best_params_by_fold(df_folds, {"n_estimators": [100, 200]}, metric="f1")

    assert best == {1: {"params": {"n_estimators": 200}, "best_score": 0.5}}


def test_best_params_by_fold_restores_the_hyperparameter_types():
    df_folds = pd.DataFrame(
        {"fold": [1], "n_estimators": [300.0], "max_samples": [np.nan], "auprc": [0.5]}
    )

    best = best_params_by_fold(df_folds, {"n_estimators": [300], "max_samples": [None]})

    assert best[1]["params"] == {"n_estimators": 300, "max_samples": None}


def test_best_params_by_fold_keeps_integer_columns_as_integers():
    """Lida como Series, a linha viraria float, e min_samples_leaf=1.0 é uma fração no RF."""
    df_folds = pd.DataFrame({"fold": [1], "min_samples_leaf": [1], "auprc": [0.5]})

    best = best_params_by_fold(df_folds, {"min_samples_leaf": [1]})

    assert isinstance(best[1]["params"]["min_samples_leaf"], int)


@pytest.fixture
def report(tmp_path, outer_folds_dir):
    return supervised_outer_evaluation(
        outer_folds_dir=outer_folds_dir,
        best_parameter_achieved=RF_PARAMS,
        target="Class",
        model_select="rf",
        output_dir=str(tmp_path / "results"),
        n_jobs=1,
    )


def fold_rows(report):
    return report[report["fold"] != "Média"]


@pytest.mark.slow
def test_reports_one_row_per_outer_fold_and_the_mean(report):
    assert list(report["fold"]) == [1, 2, "Média"]


@pytest.mark.slow
def test_every_round_uses_the_same_configuration(report):
    for name, value in RF_PARAMS.items():
        assert (fold_rows(report)[name] == value).all()


@pytest.mark.slow
def test_evaluates_every_fraud_of_its_own_outer_test(report, outer_folds_dir):
    """TP + FN é o total de fraudes do teste externo da rodada."""
    for _, row in fold_rows(report).iterrows():
        test = pd.read_csv(f"{outer_folds_dir}/fold_{row['fold']}_test.csv")
        assert row["TP"] + row["FN"] == test["Class"].sum()


@pytest.mark.slow
def test_confusion_counts_agree_with_precision_and_recall(report):
    for _, row in fold_rows(report).iterrows():
        if row["TP"] + row["FP"] > 0:
            assert row["precision"] == pytest.approx(row["TP"] / (row["TP"] + row["FP"]))
        assert row["recall"] == pytest.approx(row["TP"] / (row["TP"] + row["FN"]))


@pytest.mark.slow
def test_metrics_stay_within_their_valid_range(report):
    for metric in METRIC_COLUMNS:
        assert fold_rows(report)[metric].between(0.0, 1.0).all()


@pytest.mark.slow
def test_mean_row_is_the_average_of_the_folds(report):
    """A linha "Média" é o número levado ao trabalho."""
    mean = report[report["fold"] == "Média"].iloc[0]

    for metric in METRIC_COLUMNS:
        assert mean[metric] == pytest.approx(fold_rows(report)[metric].mean())


@pytest.mark.slow
def test_writes_the_report_named_after_the_model(tmp_path, outer_folds_dir):
    output_dir = tmp_path / "resultados"

    supervised_outer_evaluation(
        outer_folds_dir=outer_folds_dir,
        best_parameter_achieved={"n_estimators": 5, "max_depth": 2},
        model_select="xgb",
        output_dir=str(output_dir),
        n_jobs=1,
    )

    written = pd.read_csv(output_dir / "xgb_outer_final_report.csv")
    assert list(written["fold"].astype(str)) == ["1", "2", "Média"]


@pytest.mark.slow
def test_runs_on_the_configuration_chosen_by_the_grid_search(
    tmp_path, inner_folds_dir, outer_folds_dir
):
    """O fluxo do script principal: busca em grade, melhor de cada fold e a maior entre elas."""
    grid = {"n_estimators": [5, 10], "min_samples_leaf": [1, 3]}
    df_folds, _, _ = manual_grid_search(
        folds_dir=inner_folds_dir,
        param_grid=grid,
        output_dir=str(tmp_path / "grid"),
        parallel=False,
    )

    best_params_per_fold = best_params_by_fold(df_folds, grid)
    best_fold = max(best_params_per_fold, key=lambda fold: best_params_per_fold[fold]["best_score"])
    best_parameter_achieved = best_params_per_fold[best_fold]["params"]

    report = supervised_outer_evaluation(
        outer_folds_dir=outer_folds_dir,
        best_parameter_achieved=best_parameter_achieved,
        output_dir=str(tmp_path / "outer"),
        n_jobs=1,
    )

    assert best_params_per_fold[best_fold]["best_score"] == pytest.approx(df_folds["auprc"].max())
    for name, value in best_parameter_achieved.items():
        assert (fold_rows(report)[name] == value).all()
