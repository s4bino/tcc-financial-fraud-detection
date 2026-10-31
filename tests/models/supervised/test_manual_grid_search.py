"""Testes de grid_search.manual_grid_search.

A função percorre a grade de hiperparâmetros sobre os folds internos e elege a
melhor combinação segundo uma métrica. É onde a calibração acontece: se o
resumo por combinação não corresponder às métricas por fold, ou se a seleção
não maximizar a métrica pedida, o hiperparâmetro reportado no trabalho não é o
que os dados indicam.

Os testes usam grades mínimas e modelos com poucas árvores — o objetivo é
verificar a mecânica da busca, não o desempenho preditivo.
"""

import numpy as np
import pandas as pd
import pytest

from grid_search import manual_grid_search

pytestmark = pytest.mark.slow

RF_GRID = {"n_estimators": [5, 10], "min_samples_leaf": [1]}
XGB_GRID = {"n_estimators": [5], "max_depth": [2]}

METRIC_COLUMNS = ["auprc", "precision", "recall", "f1"]


@pytest.fixture
def result(tmp_path, inner_folds_dir):
    return manual_grid_search(
        folds_dir=inner_folds_dir,
        param_grid=RF_GRID,
        target="Class",
        metric="auprc",
        output_dir=str(tmp_path / "results"),
        parallel=False,
        model_select="rf",
    )


def test_returns_folds_summary_and_best_params(result):
    df_folds, df_summary, best_params = result

    assert isinstance(df_folds, pd.DataFrame)
    assert isinstance(df_summary, pd.DataFrame)
    assert isinstance(best_params, dict)


def test_evaluates_every_combination_on_every_fold(result):
    df_folds, _, _ = result
    n_combinations = len(RF_GRID["n_estimators"]) * len(RF_GRID["min_samples_leaf"])
    n_folds = 2

    assert len(df_folds) == n_combinations * n_folds


def test_summary_has_one_row_per_combination(result):
    _, df_summary, _ = result
    assert len(df_summary) == 2


def test_reports_the_four_evaluation_metrics(result):
    df_folds, df_summary, _ = result

    for column in METRIC_COLUMNS:
        assert column in df_folds.columns
        assert column in df_summary.columns


def test_records_the_hyperparameters_alongside_the_metrics(result):
    df_folds, df_summary, _ = result

    for column in RF_GRID:
        assert column in df_folds.columns
        assert column in df_summary.columns


def test_summary_metrics_are_the_mean_across_folds(result):
    """O resumo é a média por combinação — é o número levado ao trabalho."""
    df_folds, df_summary, _ = result

    for combo_id, summary in df_summary.set_index("combo_id").iterrows():
        folds = df_folds[df_folds["combo_id"] == combo_id]

        for metric in METRIC_COLUMNS:
            assert summary[metric] == pytest.approx(folds[metric].mean())


def test_best_params_maximise_the_requested_metric(result):
    df_folds, df_summary, best_params = result

    best_row = df_summary.loc[df_summary["auprc"].idxmax()]
    for name, value in best_params.items():
        assert best_row[name] == value


def test_confusion_counts_agree_with_precision_and_recall(result):
    """TP, FP e FN são as contagens da mesma predição que gera a precisão e o recall."""
    df_folds, _, _ = result

    for _, row in df_folds.iterrows():
        if row["TP"] + row["FP"] > 0:
            assert row["precision"] == pytest.approx(row["TP"] / (row["TP"] + row["FP"]))
        assert row["recall"] == pytest.approx(row["TP"] / (row["TP"] + row["FN"]))


def test_best_params_contain_exactly_the_grid_keys(result):
    _, _, best_params = result
    assert set(best_params) == set(RF_GRID)


def test_metrics_stay_within_their_valid_range(result):
    df_folds, _, _ = result

    for metric in METRIC_COLUMNS:
        assert df_folds[metric].between(0.0, 1.0).all()


def test_records_training_and_inference_times(result):
    df_folds, _, _ = result

    assert (df_folds["train_time_min"] > 0).all()
    assert (df_folds["test_time_min"] >= 0).all()


def test_writes_both_result_files(tmp_path, inner_folds_dir):
    output_dir = tmp_path / "resultados"
    manual_grid_search(
        folds_dir=inner_folds_dir,
        param_grid=RF_GRID,
        output_dir=str(output_dir),
        parallel=False,
    )

    assert (output_dir / "results_by_fold.csv").exists()
    assert (output_dir / "results_summary.csv").exists()


def test_written_files_match_the_returned_frames(tmp_path, inner_folds_dir):
    output_dir = tmp_path / "resultados"
    df_folds, df_summary, _ = manual_grid_search(
        folds_dir=inner_folds_dir,
        param_grid=RF_GRID,
        output_dir=str(output_dir),
        parallel=False,
    )

    assert len(pd.read_csv(output_dir / "results_by_fold.csv")) == len(df_folds)
    assert len(pd.read_csv(output_dir / "results_summary.csv")) == len(df_summary)


def test_creates_the_output_directory_when_absent(tmp_path, inner_folds_dir):
    output_dir = tmp_path / "nao" / "existe"
    manual_grid_search(
        folds_dir=inner_folds_dir,
        param_grid=RF_GRID,
        output_dir=str(output_dir),
        parallel=False,
    )
    assert output_dir.is_dir()


def test_rejects_a_metric_that_is_not_reported(tmp_path, inner_folds_dir):
    with pytest.raises(ValueError, match="não encontrada"):
        manual_grid_search(
            folds_dir=inner_folds_dir,
            param_grid=RF_GRID,
            metric="auc",
            output_dir=str(tmp_path / "resultados"),
            parallel=False,
        )


@pytest.mark.parametrize("metric", ["auprc", "precision", "recall", "f1"])
def test_accepts_every_reported_metric_as_selection_criterion(tmp_path, inner_folds_dir, metric):
    _, df_summary, best_params = manual_grid_search(
        folds_dir=inner_folds_dir,
        param_grid=RF_GRID,
        metric=metric,
        output_dir=str(tmp_path / metric),
        parallel=False,
    )

    best_row = df_summary.loc[df_summary[metric].idxmax()]
    assert best_row["n_estimators"] == best_params["n_estimators"]


def test_parallel_execution_matches_the_sequential_one(tmp_path, inner_folds_dir):
    """A paralelização não pode alterar o resultado da busca."""
    common = dict(folds_dir=inner_folds_dir, param_grid=RF_GRID, target="Class")

    _, sequential, _ = manual_grid_search(
        **common, output_dir=str(tmp_path / "serial"), parallel=False
    )
    _, concurrent, _ = manual_grid_search(
        **common, output_dir=str(tmp_path / "paralelo"), parallel=True, n_jobs=2
    )

    ordered = concurrent.sort_values("combo_id").reset_index(drop=True)
    expected = sequential.sort_values("combo_id").reset_index(drop=True)

    for metric in METRIC_COLUMNS:
        np.testing.assert_allclose(expected[metric], ordered[metric])


def test_supports_the_xgboost_estimator(tmp_path, inner_folds_dir):
    df_folds, df_summary, best_params = manual_grid_search(
        folds_dir=inner_folds_dir,
        param_grid=XGB_GRID,
        output_dir=str(tmp_path / "xgb"),
        parallel=False,
        model_select="xgb",
    )

    assert len(df_summary) == 1
    assert set(best_params) == set(XGB_GRID)
    assert df_folds["auprc"].between(0.0, 1.0).all()


def test_is_reproducible_across_runs(tmp_path, inner_folds_dir):
    """random_state fixo em 42: duas execuções devem coincidir."""
    common = dict(folds_dir=inner_folds_dir, param_grid=RF_GRID, parallel=False)

    _, first, _ = manual_grid_search(**common, output_dir=str(tmp_path / "a"))
    _, second, _ = manual_grid_search(**common, output_dir=str(tmp_path / "b"))

    for metric in METRIC_COLUMNS:
        np.testing.assert_allclose(first[metric], second[metric])


def test_an_empty_folds_directory_fails_instead_of_returning_empty_results(tmp_path):
    """Caracteriza a limitação atual.

    Sem nenhum fold, a função não emite um erro descritivo: o dicionário de
    parâmetros só é criado dentro do laço de folds, e o resumo o referencia em
    seguida. O teste registra o comportamento observado para que uma futura
    mudança nessa área seja percebida.
    """
    empty = tmp_path / "vazio"
    empty.mkdir()

    with pytest.raises(NameError):
        manual_grid_search(
            folds_dir=str(empty),
            param_grid=RF_GRID,
            output_dir=str(tmp_path / "resultados"),
            parallel=False,
        )
