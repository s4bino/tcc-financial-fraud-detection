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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import grid_search
from grid_search import (
    best_params_by_fold,
    build_model,
    clean_params,
    manual_grid_search,
    supervised_outer_evaluation,
    training_weights,
)
from metrics import MEAN_ROW_LABEL, STD_ROW_LABEL

METRIC_COLUMNS = ["auprc", "precision", "recall", "f1"]
SUMMARY_ROWS = [MEAN_ROW_LABEL, STD_ROW_LABEL]

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


def test_xgboost_restores_the_class_proportion_of_the_full_base():
    """Com legítimas subamostradas, cada uma pesa pelas que representa, como no HDBSCAN*."""
    model = build_model("xgb", {"n_estimators": 5}, legitimate_weight=2.84)

    assert model.get_params()["scale_pos_weight"] == pytest.approx(1 / 2.84)


def test_xgboost_is_not_reweighted_on_the_full_base():
    model = build_model("xgb", {"n_estimators": 5})

    assert model.get_params()["scale_pos_weight"] == pytest.approx(1.0)


def test_random_forest_without_class_weight_restores_the_class_proportion():
    """Cada legítima pesa pelas que representa, e cada fraude pesa 1, como no HDBSCAN*."""
    weights = training_weights("rf", {"class_weight": None}, pd.Series([0, 1, 0]), 2.84)

    np.testing.assert_allclose(weights, [2.84, 1.0, 2.84])


@pytest.mark.parametrize(
    "model_select, params",
    [("rf", {"class_weight": "balanced_subsample"}), ("xgb", {"n_estimators": 5})],
    ids=["classes já igualadas", "XGBoost, reponderado pelo scale_pos_weight"],
)
def test_other_models_are_not_reweighted_in_the_fit(model_select, params):
    assert training_weights(model_select, params, pd.Series([0, 1, 0]), 2.84) is None


def test_build_model_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="não reconhecido"):
        build_model("svm", {})


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
    return report[~report["fold"].isin(SUMMARY_ROWS)]


@pytest.mark.slow
def test_reports_one_row_per_outer_fold_the_mean_and_the_std(report):
    assert list(report["fold"]) == [1, 2, *SUMMARY_ROWS]


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
def test_std_row_is_the_sample_deviation_of_the_folds(report):
    std = report[report["fold"] == STD_ROW_LABEL].iloc[0]

    for metric in METRIC_COLUMNS:
        assert std[metric] == pytest.approx(fold_rows(report)[metric].std(ddof=1))


@pytest.mark.slow
def test_the_hour_of_the_day_does_not_reach_the_model(tmp_path, outer_folds_dir):
    """Com a hora fora das variáveis, mudar Time nos folds não muda nenhuma previsão."""
    shifted_dir = tmp_path / "shifted"
    shifted_dir.mkdir()
    rng = np.random.RandomState(0)
    for path in sorted(Path(outer_folds_dir).iterdir()):
        target = shifted_dir / path.name
        if path.suffix == ".csv":
            fold = pd.read_csv(path)
            fold["Time"] = rng.uniform(0, 172_800, len(fold))
            fold.to_csv(target, index=False)
        else:
            target.write_bytes(path.read_bytes())
    common = dict(best_parameter_achieved=RF_PARAMS, model_select="rf", n_jobs=1)

    original = supervised_outer_evaluation(
        outer_folds_dir=outer_folds_dir, output_dir=str(tmp_path / "a"), **common
    )
    shifted = supervised_outer_evaluation(
        outer_folds_dir=str(shifted_dir), output_dir=str(tmp_path / "b"), **common
    )

    for metric in [*METRIC_COLUMNS, "TP", "FP", "FN"]:
        np.testing.assert_allclose(fold_rows(shifted)[metric], fold_rows(original)[metric])


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
    assert list(written["fold"].astype(str)) == ["1", "2", *SUMMARY_ROWS]


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


@pytest.mark.slow
def test_outer_resume_skips_the_folds_already_saved(tmp_path, outer_folds_dir, monkeypatch):
    output_dir = tmp_path / "results"
    common = dict(
        outer_folds_dir=outer_folds_dir,
        best_parameter_achieved=RF_PARAMS,
        model_select="rf",
        output_dir=str(output_dir),
        n_jobs=1,
    )
    full = supervised_outer_evaluation(**common)
    full[full["fold"] == 1].to_csv(output_dir / "rf_outer_final_report.csv", index=False)
    fitted = []
    original_build = grid_search.build_model
    monkeypatch.setattr(
        grid_search,
        "build_model",
        lambda *args, **kwargs: fitted.append(args) or original_build(*args, **kwargs),
    )

    resumed = supervised_outer_evaluation(**common, resume=True)

    assert len(fitted) == 1
    assert list(resumed["fold"]) == [1, 2, *SUMMARY_ROWS]
    for metric in METRIC_COLUMNS:
        np.testing.assert_allclose(resumed[metric].astype(float), full[metric].astype(float))


@pytest.mark.slow
def test_outer_resume_refuses_another_configuration(tmp_path, outer_folds_dir):
    common = dict(
        outer_folds_dir=outer_folds_dir, model_select="rf", output_dir=str(tmp_path), n_jobs=1
    )
    supervised_outer_evaluation(**common, best_parameter_achieved=RF_PARAMS)

    with pytest.raises(ValueError, match="outra grade ou configuração"):
        supervised_outer_evaluation(
            **common, best_parameter_achieved={**RF_PARAMS, "n_estimators": 7}, resume=True
        )
