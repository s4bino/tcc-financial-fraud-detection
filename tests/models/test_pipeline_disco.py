"""Testes de run_hdbscan_enhanced_pipeline.py: integração do DISCO e variáveis da distância.

O pipeline importa o cuML e o dbcv, que só existem na VM com GPU. Os testes os substituem por
módulos vazios: nenhum deles agrupa os dados, só conferem o caminho da métrica até a seleção e
as variáveis que chegam ao HDBSCAN*.
"""

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import paths
from disco import disco_score
from features import V_COLUMNS
from whitening import MahalanobisWhitener

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


def default_config(pipeline):
    """A execução da rodada final que já rodou: base completa com o MCD."""
    return pipeline.build_config("full", "mahalanobis_mcd")


@pytest.mark.parametrize("base", ["full", "undersampled"])
@pytest.mark.parametrize("distance", ["euclidean", "mahalanobis", "mahalanobis_mcd"])
def test_config_reads_and_writes_the_same_base(pipeline, base, distance):
    """Os resultados de uma base nunca vão para as pastas da outra."""
    folds = paths.FOLD_PATHS[base]
    config = pipeline.build_config(base, distance)

    assert config.internal_folds_dir == folds.internal_folds_dir
    assert config.outer_folds_dir == folds.outer_folds_dir
    assert config.distance == distance
    assert config.grid_output_dir == config.report_output_dir
    assert config.grid_output_dir.parent == paths.RESULTS_DIR / paths.FINAL_RUNS_DIR_NAME
    assert config.grid_output_dir.name == f"hdbscan_{distance}{folds.results_suffix}"


def test_every_run_of_the_final_round_has_its_own_folder(pipeline):
    runs = [
        pipeline.build_config(base, distance).grid_output_dir
        for base in paths.FOLD_PATHS
        for distance in pipeline.DISTANCES
    ]

    assert len(set(runs)) == len(runs) == 6


def test_runs_differ_only_in_base_and_distance(pipeline):
    reference = pipeline.run_config_record(default_config(pipeline))
    other = pipeline.run_config_record(pipeline.build_config("undersampled", "euclidean"))
    differing = {key for key in reference if reference[key] != other[key]}

    assert differing == {
        "internal_folds_dir",
        "outer_folds_dir",
        "grid_output_dir",
        "report_output_dir",
        "distance",
    }
    assert "n_processes" not in reference


@pytest.mark.parametrize("distance", ["euclidean", "mahalanobis", "mahalanobis_mcd"])
def test_every_distance_reports_the_center_distance(pipeline, distance):
    config = pipeline.build_config("full", distance)

    pipeline.validate_config(config)
    assert "center_distance" in pipeline.anomaly_score_names(config)


def test_config_selects_only_by_disco(pipeline):
    assert pipeline.validation_metric_names(default_config(pipeline)) == ("disco",)
    assert pipeline.selection_column(default_config(pipeline)) == "disco_score"
    pipeline.validate_config(default_config(pipeline))


def test_dbcv_is_plugged_in_by_the_config(pipeline, partition):
    X, labels = partition
    config = dataclasses.replace(default_config(pipeline), extra_validation_metrics=("dbcv",))

    scores = pipeline.compute_validation_scores(X, labels, config)

    assert scores["dbcv_score"] == FAKE_DBCV_SCORE
    assert scores["disco_score"] == pytest.approx(disco_score(X, labels))


def test_disco_runs_on_the_whole_train_set_even_above_the_dbcv_sample(pipeline, partition):
    X, labels = partition
    config = dataclasses.replace(default_config(pipeline), dbcv_max_sample_size=10)

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
        pipeline.disco_validation_score(X, partition_labels, default_config(pipeline))
    assert len(built) == 1

    pipeline.disco_validation_score(X.copy(), labels, default_config(pipeline))
    assert len(built) == 2


def test_only_noise_gets_the_invalid_score(pipeline, partition):
    X, _ = partition

    score = pipeline.disco_validation_score(X, np.full(80, -1), default_config(pipeline))

    assert score == pipeline.INVALID_VALIDATION_SCORE


def test_disco_error_stops_the_run_instead_of_becoming_a_score(pipeline, partition):
    X, _ = partition

    with pytest.raises(ValueError):
        pipeline.disco_validation_score(X, np.zeros(10, int), default_config(pipeline))


def test_selection_follows_disco_when_dbcv_disagrees(pipeline):
    grid = {name: 50 for name in default_config(pipeline).param_grid}
    history = pd.DataFrame(
        [
            {"fold": 1, "combo_id": 0, **grid, "disco_score": 0.1, "dbcv_score": 0.9},
            {"fold": 1, "combo_id": 1, **grid, "disco_score": 0.3, "dbcv_score": 0.2},
        ]
    )

    summary, best = pipeline.select_best_trials(history, default_config(pipeline))

    assert summary.iloc[0]["combo_id"] == 1
    assert best[1]["best_disco_score"] == 0.3


def test_resume_refuses_a_history_without_disco(pipeline):
    old_history = [{"fold": 1, "combo_id": 0, "dbcv_score": 0.2}]

    with pytest.raises(ValueError, match="disco_score"):
        pipeline.check_resumable_history(old_history, default_config(pipeline))


def test_resume_accepts_a_complete_or_empty_history(pipeline):
    rate_columns = {
        pipeline.alert_rate_column("f1", rate): 0.4
        for rate in pipeline.tracked_alert_rates(default_config(pipeline))
    }
    history = [{"fold": 1, "combo_id": 0, "dbcv_score": 0.2, "disco_score": 0.3, **rate_columns}]

    pipeline.check_resumable_history(history, default_config(pipeline))
    pipeline.check_resumable_history([], default_config(pipeline))


def test_resume_refuses_a_history_without_an_extra_alert_rate(pipeline):
    config = dataclasses.replace(default_config(pipeline), extra_alert_rates=(0.0017,))
    old_history = [{"fold": 1, "combo_id": 0, "disco_score": 0.3}]

    with pytest.raises(ValueError, match="f1_at_0.0017"):
        pipeline.check_resumable_history(old_history, config)


def test_extra_alert_rates_only_move_the_threshold(pipeline):
    rng = np.random.RandomState(0)
    train_scores, evaluated = rng.rand(2000), rng.rand(500)
    y_eval = pd.Series((evaluated > 0.97).astype(int))
    fold_data = pipeline.FoldData(None, None, y_eval, pd.Series(np.zeros(2000, int)), 1.0)
    config = dataclasses.replace(
        default_config(pipeline), alert_rate=0.01, extra_alert_rates=(0.05, 0.01, 0.05)
    )

    row = pipeline.tracked_alert_metrics(
        fold_data, pipeline.ScorePair(train_scores, evaluated), config
    )

    assert list(row) == [f"{name}_at_0.05" for name in pipeline.ALERT_RATE_COLUMNS]
    expected = pipeline.alert_metrics(y_eval, evaluated, np.quantile(train_scores, 0.95))
    for name in pipeline.ALERT_RATE_COLUMNS:
        assert row[f"{name}_at_0.05"] == expected[name]
    assert "F1@0.05=" in pipeline.format_metrics(
        {"min_cluster_size": 50, "min_samples": 300, "f1": 0.4, **row}, config
    )


def test_config_rejects_an_alert_rate_outside_zero_and_one(pipeline):
    config = dataclasses.replace(default_config(pipeline), extra_alert_rates=(0.002, 1.5))

    with pytest.raises(ValueError, match="1.5"):
        pipeline.validate_config(config)


def load_inner_fold(pipeline, folds_dir, excluded_features, standardize_v_columns=False):
    config = dataclasses.replace(
        default_config(pipeline),
        excluded_features=excluded_features,
        standardize_v_columns=standardize_v_columns,
    )
    return pipeline.standardize(
        pipeline.load_fold(
            Path(folds_dir),
            1,
            pipeline.INNER_EVAL_FILE,
            pipeline.INNER_INFO_FILE,
            "Class",
            excluded_features,
        ),
        pipeline.standardized_columns(config),
    )


def test_excluded_features_leave_the_distance_but_not_the_fold_files(pipeline, inner_folds_dir):
    files = sorted(Path(inner_folds_dir).iterdir())
    contents_before = [path.read_bytes() for path in files]

    fold_data = load_inner_fold(pipeline, inner_folds_dir, ("hour_sin", "hour_cos"))

    raw_train = pd.read_csv(Path(inner_folds_dir) / "fold_1_train.csv")
    assert fold_data.X_train.shape == (len(raw_train), 29)
    assert fold_data.X_eval.shape[1] == 29
    np.testing.assert_allclose(fold_data.X_train[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(fold_data.X_train[:, 1:], raw_train[V_COLUMNS].to_numpy())
    assert sorted(Path(inner_folds_dir).iterdir()) == files
    assert [path.read_bytes() for path in files] == contents_before


def test_no_exclusion_keeps_the_31_variables(pipeline, inner_folds_dir):
    fold_data = load_inner_fold(pipeline, inner_folds_dir, ())

    assert fold_data.X_train.shape[1] == 31
    assert fold_data.X_eval.shape[1] == 31


def test_standardized_columns_follow_the_v_option_and_the_exclusions(pipeline):
    without_hour = ("hour_sin", "hour_cos")
    v_off = dataclasses.replace(
        default_config(pipeline), excluded_features=without_hour, standardize_v_columns=False
    )
    v_on = dataclasses.replace(v_off, standardize_v_columns=True)

    assert pipeline.standardized_columns(v_off) == ["log_amount"]
    assert pipeline.standardized_columns(v_on) == ["log_amount", *V_COLUMNS]


def test_standardized_v_columns_get_unit_variance_only_from_the_train(pipeline, inner_folds_dir):
    files = sorted(Path(inner_folds_dir).iterdir())
    contents_before = [path.read_bytes() for path in files]

    fold_data = load_inner_fold(
        pipeline, inner_folds_dir, ("hour_sin", "hour_cos"), standardize_v_columns=True
    )

    assert fold_data.X_train.shape[1] == 29
    np.testing.assert_allclose(fold_data.X_train.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(fold_data.X_train.std(axis=0), 1.0, rtol=1e-10)
    assert [path.read_bytes() for path in files] == contents_before


def test_glosh_degree_passes_the_tree_and_both_neighbor_sets(pipeline, monkeypatch):
    calls = []

    def fake_degree(X_train, X_eval, train_probabilities, probabilities, min_samples, **kwargs):
        calls.append((train_probabilities, probabilities, min_samples, kwargs))
        return np.zeros(2), np.ones(3), []

    tree = np.arange(4)
    model = types.SimpleNamespace(condensed_tree_=types.SimpleNamespace(to_numpy=lambda: tree))
    tree_neighbors, distance_neighbors = ("tree",), ("knn",)
    monkeypatch.setattr(pipeline, "manual_anomaly_score", fake_degree)

    scores = pipeline.ANOMALY_SCORERS["manual_anomaly_score_glosh"].score(
        model, np.zeros((2, 1)), np.zeros((3, 1)), 7, (tree_neighbors, distance_neighbors)
    )

    _, _, min_samples, kwargs = calls[0]
    assert min_samples == 7
    assert kwargs["component"] == "glosh"
    assert kwargs["tree"] is tree
    assert kwargs["tree_neighbors"] is tree_neighbors
    assert kwargs["neighbors"] is distance_neighbors
    np.testing.assert_array_equal(scores.evaluated, np.ones(3))
    assert not pipeline.ANOMALY_SCORERS["manual_anomaly_score_glosh"].needs_prediction_data


def test_config_rejects_an_unknown_excluded_feature(pipeline):
    config = dataclasses.replace(default_config(pipeline), excluded_features=("hour",))

    with pytest.raises(ValueError, match="hour"):
        pipeline.validate_config(config)


def test_config_rejects_an_unknown_distance(pipeline):
    config = dataclasses.replace(default_config(pipeline), distance="manhattan")

    with pytest.raises(ValueError, match="distance"):
        pipeline.validate_config(config)


def correlated_fold(pipeline):
    rng = np.random.RandomState(5)
    mixing = np.array([[2.0, 0.0, 0.0], [1.5, 0.5, 0.0], [0.3, -0.8, 0.1]])
    X_train = rng.normal(size=(500, 3)) @ mixing.T
    X_eval = rng.normal(size=(60, 3)) @ mixing.T + 1.0
    y_eval, y_train = pd.Series(np.zeros(60)), pd.Series(np.zeros(500))
    return pipeline.FoldData(X_train, X_eval, y_eval, y_train, 1.0)


def test_euclidean_distance_only_moves_the_train_mean_to_the_origin(pipeline):
    fold_data = correlated_fold(pipeline)
    config = dataclasses.replace(default_config(pipeline), distance="euclidean")

    centered = pipeline.whiten(fold_data, config)

    center = fold_data.X_train.mean(axis=0)
    np.testing.assert_allclose(centered.X_train.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(centered.X_train, fold_data.X_train - center)
    np.testing.assert_allclose(centered.X_eval, fold_data.X_eval - center)
    pairwise = np.linalg.norm(fold_data.X_eval[:, None] - fold_data.X_train[None], axis=2)
    np.testing.assert_allclose(
        np.linalg.norm(centered.X_eval[:, None] - centered.X_train[None], axis=2), pairwise
    )
    assert centered.y_eval is fold_data.y_eval


@pytest.mark.parametrize(
    ("distance", "covariance"), [("mahalanobis", "empirical"), ("mahalanobis_mcd", "mcd")]
)
def test_mahalanobis_distances_whiten_with_their_covariance_fitted_on_the_train(
    pipeline, distance, covariance
):
    fold_data = correlated_fold(pipeline)
    config = dataclasses.replace(default_config(pipeline), distance=distance)

    whitened = pipeline.whiten(fold_data, config)

    reference = MahalanobisWhitener(covariance, random_state=config.random_state)
    reference.fit(fold_data.X_train)
    np.testing.assert_array_equal(whitened.X_train, reference.transform(fold_data.X_train))
    np.testing.assert_array_equal(whitened.X_eval, reference.transform(fold_data.X_eval))
    assert whitened.y_eval is fold_data.y_eval


def test_empirical_mahalanobis_leaves_the_train_with_unit_covariance(pipeline):
    config = dataclasses.replace(default_config(pipeline), distance="mahalanobis")

    whitened = pipeline.whiten(correlated_fold(pipeline), config)

    np.testing.assert_allclose(np.cov(whitened.X_train, rowvar=False), np.eye(3), atol=1e-8)


def test_center_distance_is_the_norm_of_the_whitened_rows(pipeline):
    X_train, X_eval = np.array([[3.0, 4.0], [0.0, 1.0]]), np.array([[6.0, 8.0]])
    scorer = pipeline.ANOMALY_SCORERS["center_distance"]

    scores = scorer.score(None, X_train, X_eval, 50, ())

    np.testing.assert_allclose(scores.train, [5.0, 1.0])
    np.testing.assert_allclose(scores.evaluated, [10.0])
    assert scorer.neighbors(X_train, X_eval, 50) == ()


def test_prepare_fold_whitens_with_the_mcd_without_touching_the_fold_files(
    pipeline, inner_folds_dir
):
    """Com os V gaussianos da fixture, o treino branqueado tem covariância perto da identidade."""
    files = sorted(Path(inner_folds_dir).iterdir())
    contents_before = [path.read_bytes() for path in files]
    config = dataclasses.replace(default_config(pipeline), distance="mahalanobis_mcd")

    fold_data = pipeline.prepare_fold(
        pipeline.load_fold(
            Path(inner_folds_dir),
            1,
            pipeline.INNER_EVAL_FILE,
            pipeline.INNER_INFO_FILE,
            "Class",
            config.excluded_features,
        ),
        config,
    )

    raw_train = pd.read_csv(Path(inner_folds_dir) / "fold_1_train.csv")
    assert fold_data.X_train.shape == (len(raw_train), 29)
    assert np.isfinite(fold_data.X_eval).all()
    np.testing.assert_allclose(np.cov(fold_data.X_train, rowvar=False), np.eye(29), atol=0.25)
    assert [path.read_bytes() for path in files] == contents_before
