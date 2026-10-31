"""Testes da integração do DISCO em run_hdbscan_enhanced_pipeline.py.

O pipeline importa o cuML e o dbcv, que só existem na VM com GPU. Os testes os substituem por
módulos vazios: nenhum deles agrupa os dados, só conferem o caminho da métrica até a seleção.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import paths
from disco import disco_score

PIPELINE_FILE = (
    Path(__file__).resolve().parents[2] / "src/models/unsupervised/run_hdbscan_enhanced_pipeline.py"
)
FAKE_DBCV_SCORE = 0.25


@pytest.fixture(scope="module")
def pipeline():
    cluster = types.ModuleType("cuml.cluster")
    cluster.HDBSCAN = object
    hdbscan_module = types.ModuleType("cuml.cluster.hdbscan")
    hdbscan_module.approximate_predict = None
    fake_dbcv = types.ModuleType("dbcv")
    fake_dbcv.dbcv = lambda *args, **kwargs: FAKE_DBCV_SCORE
    fakes = {
        "cuml": types.ModuleType("cuml"),
        "cuml.cluster": cluster,
        "cuml.cluster.hdbscan": hdbscan_module,
        "dbcv": fake_dbcv,
    }
    with pytest.MonkeyPatch.context() as patch:
        for name, module in fakes.items():
            patch.setitem(sys.modules, name, module)
        spec = importlib.util.spec_from_file_location("disco_pipeline_under_test", PIPELINE_FILE)
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


@pytest.fixture
def partition():
    rng = np.random.RandomState(3)
    X = np.r_[rng.normal(0, 0.3, (40, 3)), rng.normal(5, 0.3, (40, 3))]
    return X, np.r_[np.zeros(40, int), np.ones(40, int)]


def test_config_runs_on_the_undersampled_base(pipeline):
    assert pipeline.CONFIG.internal_folds_dir == paths.UNDERSAMPLED_PATH.internal_folds_dir
    assert pipeline.CONFIG.outer_folds_dir == paths.UNDERSAMPLED_PATH.outer_folds_dir
    assert pipeline.CONFIG.grid_output_dir.name.endswith("_undersampled")
    assert pipeline.CONFIG.report_output_dir.name.endswith("_undersampled")


def test_config_selects_by_disco_and_keeps_dbcv_in_the_history(pipeline):
    assert pipeline.validation_metric_names(pipeline.CONFIG) == ("disco", "dbcv")
    assert pipeline.selection_column(pipeline.CONFIG) == "disco_score"
    pipeline.validate_config(pipeline.CONFIG)


def test_validation_scores_have_both_metrics(pipeline, partition):
    X, labels = partition

    scores = pipeline.compute_validation_scores(X, labels, pipeline.CONFIG)

    assert scores["dbcv_score"] == FAKE_DBCV_SCORE
    assert scores["disco_score"] == pytest.approx(disco_score(X, labels))


def test_disco_runs_on_the_whole_train_set_even_above_the_dbcv_sample(pipeline, partition):
    X, labels = partition
    config = pipeline.CONFIG.__class__(**{**pipeline.CONFIG.__dict__, "dbcv_max_sample_size": 10})

    assert pipeline.disco_validation_score(X, labels, config) == pytest.approx(
        disco_score(X, labels)
    )


def test_tree_is_built_once_per_train_set(pipeline, partition, monkeypatch):
    X, labels = partition
    built = []
    real_tree = pipeline.DCTree

    def counting_tree(*args):
        built.append(args)
        return real_tree(*args)

    monkeypatch.setattr(pipeline, "DCTree", counting_tree)
    monkeypatch.setattr(pipeline, "FOLD_TREE_CACHE", pipeline.FoldTreeCache())

    for partition_labels in (labels, np.where(labels == 1, -1, labels), np.zeros(80, int) - 1):
        pipeline.disco_validation_score(X, partition_labels, pipeline.CONFIG)
    assert len(built) == 1

    pipeline.disco_validation_score(X.copy(), labels, pipeline.CONFIG)
    assert len(built) == 2


def test_only_noise_gets_the_invalid_score(pipeline, partition):
    X, _ = partition

    score = pipeline.disco_validation_score(X, np.full(80, -1), pipeline.CONFIG)

    assert score == pipeline.INVALID_VALIDATION_SCORE


def test_disco_error_stops_the_run_instead_of_becoming_a_score(pipeline, partition):
    X, _ = partition

    with pytest.raises(ValueError):
        pipeline.disco_validation_score(X, np.zeros(10, int), pipeline.CONFIG)


def test_selection_follows_disco_when_dbcv_disagrees(pipeline):
    grid = {name: 50 for name in pipeline.CONFIG.param_grid}
    history = pd.DataFrame(
        [
            {"fold": 1, "combo_id": 0, **grid, "disco_score": 0.1, "dbcv_score": 0.9},
            {"fold": 1, "combo_id": 1, **grid, "disco_score": 0.3, "dbcv_score": 0.2},
        ]
    )

    summary, best = pipeline.select_best_trials(history, pipeline.CONFIG)

    assert summary.iloc[0]["combo_id"] == 1
    assert best[1]["best_disco_score"] == 0.3


def test_resume_refuses_a_history_without_disco(pipeline):
    old_history = [{"fold": 1, "combo_id": 0, "dbcv_score": 0.2}]

    with pytest.raises(ValueError, match="disco_score"):
        pipeline.check_resumable_history(old_history, pipeline.CONFIG)


def test_resume_accepts_a_complete_or_empty_history(pipeline):
    history = [{"fold": 1, "combo_id": 0, "dbcv_score": 0.2, "disco_score": 0.3}]

    pipeline.check_resumable_history(history, pipeline.CONFIG)
    pipeline.check_resumable_history([], pipeline.CONFIG)
