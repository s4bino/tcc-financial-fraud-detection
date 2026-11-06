"""Validação cruzada aninhada do HDBSCAN* aprimorado, em versão modular.

Segue a lógica de run_hdbscan_enhanced_grid_search.py, na base subamostrada e com o DISCO como
métrica de seleção. O ciclo interno escolhe a configuração de cada fold pela métrica de validação,
e o ciclo externo mede a detecção de fraudes com o escore de anomalia configurado. O DBCV entra
no histórico, sem papel na seleção, quando incluído em extra_validation_metrics. A origem de cada
trecho nos notebooks do Kaggle está registrada nos comentários de
run_hdbscan_enhanced_grid_search.py.
"""

import argparse
import itertools
import json
import os
import re
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import dbcv
import numpy as np
import pandas as pd
from cuml.cluster import HDBSCAN
from cuml.cluster.hdbscan import approximate_predict
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

import paths
from disco import DCTree, disco_score
from features import DERIVED_COLUMNS, V_COLUMNS, build_preprocessor, engineer_features
from metrics import (
    alert_threshold,
    glosh_anomaly_score,
    glosh_neighbors,
    knn_distances,
    manual_anomaly_score,
    read_legitimate_weight,
    to_numpy,
)
from whitening import (
    DEFAULT_EIGEN_FLOOR,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SUPPORT_FRACTION,
    MahalanobisWhitener,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuração do pipeline: caminhos, grade, sementes, folds e métricas.

    Parameters
    ----------
    internal_folds_dir : Path
        Pasta com fold_k_train.csv e fold_k_val.csv do ciclo interno.
    outer_folds_dir : Path
        Pasta com fold_k_train.csv e fold_k_test.csv do ciclo externo.
    grid_output_dir : Path
        Pasta dos resultados da busca em grade.
    report_output_dir : Path
        Pasta do relatório do ciclo externo.
    target : str
        Coluna alvo.
    excluded_features : tuple[str, ...]
        Variáveis de engineer_features retiradas antes da padronização, e por isso fora da
        distância do HDBSCAN*, das métricas de validação e dos escores. A exclusão acontece na
        memória, ao ler cada fold: os arquivos dos folds não mudam.
    standardize_v_columns : bool
        Se True, V1 a V28 também são padronizados com as estatísticas do treino, e cada
        componente passa a ter o mesmo peso na distância. Se False, os V entram sem alteração,
        como em build_preprocessor.
    distance : str
        Distância entre as transações no HDBSCAN*, nas métricas de validação e nos escores.
        "euclidean" usa os dados padronizados. "mahalanobis" e "mahalanobis_mcd" branqueiam o
        fold depois da padronização, com a covariância do treino ou com a covariância robusta
        do MCD, de modo que a euclidiana sobre os dados branqueados é a distância de
        Mahalanobis. O branqueamento é ajustado só no treino de cada fold, e os parâmetros do
        MCD são os fixos de whitening.py. O estudo da escolha está em
        docs/scratch/notes/mahalanobis-distance-study.md.
    param_grid : dict[str, list[int]]
        Valores inteiros de cada hiperparâmetro do HDBSCAN*, combinados em produto cartesiano.
    n_folds : int
        Número de folds de cada ciclo, o mesmo da geração dos folds.
    alert_rate : float
        Fração de alertas, que fixa o limiar no quantil 1 - alert_rate do escore do treino,
        ponderado na base subamostrada por alert_threshold.
    extra_alert_rates : tuple[float, ...]
        Outras frações de alerta, registradas no histórico e no relatório pela precisão, pelo
        recall, pelo F1, por TP e por FP. Só o limiar muda: o modelo e o escore são os mesmos
        de alert_rate, e a seleção não depende delas.
    random_state : int
        Semente da amostragem do DBCV e da amostra e do ajuste do MCD.
    dbcv_max_sample_size : int
        Número máximo de pontos usados no DBCV.
    n_processes : int
        Número de processos do DBCV.
    disco_min_points : int
        Parâmetro μ da distância de núcleo do DISCO, fixo em toda a grade.
    selection_metric : str
        Métrica de validação que escolhe a configuração de cada fold.
    extra_validation_metrics : tuple[str, ...]
        Outras métricas de validação registradas no histórico, sem papel na seleção.
    anomaly_score : str
        Escore de anomalia que define os alertas.
    extra_anomaly_scores : tuple[str, ...]
        Outros escores de anomalia, registrados no histórico pela AUPRC.
    """

    internal_folds_dir: Path
    outer_folds_dir: Path
    grid_output_dir: Path
    report_output_dir: Path
    target: str
    excluded_features: tuple[str, ...]
    standardize_v_columns: bool
    distance: str
    param_grid: dict[str, list[int]]
    n_folds: int
    alert_rate: float
    extra_alert_rates: tuple[float, ...]
    random_state: int
    dbcv_max_sample_size: int
    n_processes: int
    disco_min_points: int
    selection_metric: str
    extra_validation_metrics: tuple[str, ...]
    anomaly_score: str
    extra_anomaly_scores: tuple[str, ...]


FOLDS = paths.FULL_PATH

CONFIG = PipelineConfig(
    internal_folds_dir=FOLDS.internal_folds_dir,
    outer_folds_dir=FOLDS.outer_folds_dir,
    grid_output_dir=FOLDS.results_dir(
        paths.RESULTS_DIR, "grid_results_hdbscan_degree_glosh_disco_no_hour_mcd"
    ),
    report_output_dir=FOLDS.results_dir(
        paths.RESULTS_DIR, "final_reports_hdbscan_degree_glosh_disco_no_hour_mcd"
    ),
    target="Class",
    excluded_features=("hour_sin", "hour_cos"),
    standardize_v_columns=False,
    distance="mahalanobis_mcd",
    param_grid={"min_cluster_size": [50], "min_samples": [50, 100, 200, 300]},
    n_folds=5,
    alert_rate=0.0008,
    extra_alert_rates=(0.0005, 0.0012, 0.0017, 0.0025, 0.0035),
    random_state=42,
    dbcv_max_sample_size=28_000,
    n_processes=os.cpu_count() or 1,
    disco_min_points=5,
    selection_metric="disco",
    extra_validation_metrics=(),
    anomaly_score="manual_anomaly_score_glosh",
    extra_anomaly_scores=("glosh", "manual_anomaly_score", "center_distance"),
)

EUCLIDEAN = "euclidean"
# Covariância do branqueamento de cada distância de Mahalanobis, em MahalanobisWhitener
WHITENING_BY_DISTANCE = {"mahalanobis": "empirical", "mahalanobis_mcd": "mcd"}
DISTANCES = (EUCLIDEAN, *WHITENING_BY_DISTANCE)
NOISE_LABEL = -1
INVALID_VALIDATION_SCORE = -1.0
CLASS_LABELS = (0, 1)
FOLD_TRAIN_PATTERN = re.compile(r"fold_(?P<fold>\d+)_train\.csv")
TRAIN_FILE = "fold_{fold}_train.csv"
INNER_EVAL_FILE = "fold_{fold}_val.csv"
OUTER_EVAL_FILE = "fold_{fold}_test.csv"
INNER_INFO_FILE = "fold_{fold}_internal_info.json"
OUTER_INFO_FILE = "fold_{fold}_info.json"
GRID_RESULTS_FILE = "hdbscan_enhanced_grid_results_by_fold.csv"
GRID_SUMMARY_FILE = "hdbscan_enhanced_grid_best_summary.csv"
BEST_PARAMS_FILE = "best_params_hdbscan_enhanced.json"
OUTER_REPORT_FILE = "hdbscan_enhanced_outer_final_report.csv"
INNER_DETECTION_COLUMNS = ("f1", "precision", "recall", "auprc")
OUTER_DETECTION_COLUMNS = ("precision", "recall", "f1", "auprc")
CONFUSION_COLUMNS = ("TP", "FP", "FN")
ALERT_RATE_COLUMNS = ("precision", "recall", "f1", "TP", "FP")
MEAN_ROW_LABEL = "Média"
INNER_STAGE = "Fold interno"
OUTER_STAGE = "Fold externo"
JSON_INDENT = 4
DISPLAY_DECIMALS = 3


class FoldData(NamedTuple):
    """Treino e conjunto avaliado de um fold.

    Parameters
    ----------
    X_train : pd.DataFrame | np.ndarray
        Variáveis do treino, sem transações duplicadas.
    X_eval : pd.DataFrame | np.ndarray
        Variáveis da validação ou do teste.
    y_eval : pd.Series
        Rótulos do conjunto avaliado.
    y_train : pd.Series
        Rótulos do treino, usados só nos pesos do limiar de alerta da base subamostrada.
    legitimate_weight : float
        Peso de cada legítima do treino no limiar de alerta, de read_legitimate_weight.
    """

    X_train: pd.DataFrame | np.ndarray
    X_eval: pd.DataFrame | np.ndarray
    y_eval: pd.Series
    y_train: pd.Series
    legitimate_weight: float


class ScorePair(NamedTuple):
    """Escores de anomalia do treino e do conjunto avaliado.

    Parameters
    ----------
    train : np.ndarray
        Escore de cada ponto do treino.
    evaluated : np.ndarray
        Escore de cada ponto da validação ou do teste.
    """

    train: np.ndarray
    evaluated: np.ndarray


Neighbors = tuple[np.ndarray | tuple[np.ndarray, ...], ...]
ValidationMetric = Callable[[np.ndarray, np.ndarray, PipelineConfig], float]


@dataclass(frozen=True)
class AnomalyScorer:
    """Escore de anomalia do registry de detecção.

    Parameters
    ----------
    neighbors : Callable[[np.ndarray, np.ndarray, int], Neighbors]
        Busca, uma vez por fold, os vizinhos de que o escore precisa.
    score : Callable[[HDBSCAN, np.ndarray, np.ndarray, int, Neighbors], ScorePair]
        Calcula o escore do treino e do conjunto avaliado.
    needs_prediction_data : bool
        Se o HDBSCAN* precisa ser ajustado com prediction_data=True.
    needs_mahalanobis : bool
        Se o escore só tem sentido com uma distância de Mahalanobis.
    """

    neighbors: Callable[[np.ndarray, np.ndarray, int], Neighbors]
    score: Callable[[HDBSCAN, np.ndarray, np.ndarray, int, Neighbors], ScorePair]
    needs_prediction_data: bool = False
    needs_mahalanobis: bool = False


def discover_folds(folds_dir: Path, n_folds: int) -> list[int]:
    """Lista os folds de uma pasta a partir dos arquivos fold_k_train.csv.

    Parameters
    ----------
    folds_dir : Path
        Pasta dos folds.
    n_folds : int
        Número esperado de folds.

    Returns
    -------
    list[int]
        Números dos folds, em ordem crescente.
    """
    matches = (FOLD_TRAIN_PATTERN.fullmatch(path.name) for path in folds_dir.iterdir())
    folds = sorted(int(match["fold"]) for match in matches if match)
    if len(folds) != n_folds:
        raise ValueError(f"{folds_dir} tem {len(folds)} folds, e a configuração espera {n_folds}")
    return folds


def model_features(X: pd.DataFrame, excluded_features: tuple[str, ...]) -> pd.DataFrame:
    """Deriva as variáveis de engineer_features e retira as excluídas da distância.

    Parameters
    ----------
    X : pd.DataFrame
        Transações com as colunas originais, sem a coluna alvo.
    excluded_features : tuple[str, ...]
        Variáveis de engineer_features que não entram no modelo.

    Returns
    -------
    pd.DataFrame
        Variáveis de engineer_features, sem as excluídas e com o índice de X.
    """
    return engineer_features(X).drop(columns=list(excluded_features))


def load_fold(
    folds_dir: Path,
    fold: int,
    eval_file: str,
    info_file: str,
    target: str,
    excluded_features: tuple[str, ...],
) -> FoldData:
    """Lê um fold e deriva as variáveis, sem as duplicatas do treino.

    Os arquivos do fold são só lidos: a exclusão de variáveis acontece nas cópias em memória.

    Parameters
    ----------
    folds_dir : Path
        Pasta dos folds.
    fold : int
        Número do fold.
    eval_file : str
        Modelo do nome do arquivo avaliado, de validação ou de teste.
    info_file : str
        Modelo do nome do JSON do fold, de onde vem o peso das legítimas no limiar de alerta.
    target : str
        Coluna alvo.
    excluded_features : tuple[str, ...]
        Variáveis de engineer_features que não entram no modelo.

    Returns
    -------
    FoldData
        Variáveis de model_features no treino e no conjunto avaliado, os rótulos dos dois e o
        peso das legítimas.
    """
    train = pd.read_csv(folds_dir / TRAIN_FILE.format(fold=fold))
    evaluated = pd.read_csv(folds_dir / eval_file.format(fold=fold))
    X_train = model_features(train.drop(columns=[target]).drop_duplicates(), excluded_features)
    return FoldData(
        X_train=X_train,
        X_eval=model_features(evaluated.drop(columns=[target]), excluded_features),
        y_eval=evaluated[target],
        y_train=train.loc[X_train.index, target],
        legitimate_weight=read_legitimate_weight(folds_dir / info_file.format(fold=fold)),
    )


def standardized_columns(config: PipelineConfig) -> list[str]:
    """Lista as variáveis que a padronização do fold recebe.

    Parameters
    ----------
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    list[str]
        Variáveis derivadas e, com standardize_v_columns, V1 a V28, sem as retiradas por
        model_features, que build_preprocessor não deve procurar.
    """
    candidates = [*DERIVED_COLUMNS, *(V_COLUMNS if config.standardize_v_columns else [])]
    return [column for column in candidates if column not in config.excluded_features]


def standardize(fold_data: FoldData, scaled_columns: Iterable[str]) -> FoldData:
    """Padroniza o fold com estatísticas calculadas só no treino.

    Parameters
    ----------
    fold_data : FoldData
        Fold com as variáveis de model_features.
    scaled_columns : Iterable[str]
        Variáveis padronizadas, de standardized_columns. As demais entram sem alteração.

    Returns
    -------
    FoldData
        O mesmo fold, com scaled_columns padronizadas por build_preprocessor.
    """
    preprocessor = build_preprocessor(scaled_columns)
    return fold_data._replace(
        X_train=preprocessor.fit_transform(fold_data.X_train),
        X_eval=preprocessor.transform(fold_data.X_eval),
    )


def whiten(fold_data: FoldData, config: PipelineConfig) -> FoldData:
    """Branqueia o fold para a distância de Mahalanobis, com a covariância só do treino.

    Depois do branqueamento, a distância euclidiana entre as transações é a distância de
    Mahalanobis, e a norma de cada linha é a distância dela ao centro estimado no treino.

    Parameters
    ----------
    fold_data : FoldData
        Fold padronizado.
    config : PipelineConfig
        Configuração do pipeline, com a distância e a semente do MCD.

    Returns
    -------
    FoldData
        O mesmo fold, sem alteração com distance="euclidean" e branqueado nos demais casos.
    """
    if config.distance == EUCLIDEAN:
        return fold_data
    whitener = MahalanobisWhitener(
        covariance=WHITENING_BY_DISTANCE[config.distance], random_state=config.random_state
    ).fit(fold_data.X_train)
    eigenvalues = whitener.eigenvalues_
    print(
        f"Distância {config.distance}: autovalores da covariância de {eigenvalues[0]:.2e} a "
        f"{eigenvalues[-1]:.2e}, {whitener.n_floored_} no piso"
    )
    return fold_data._replace(
        X_train=whitener.transform(fold_data.X_train),
        X_eval=whitener.transform(fold_data.X_eval),
    )


def prepare_fold(fold_data: FoldData, config: PipelineConfig) -> FoldData:
    """Padroniza e, com uma distância de Mahalanobis, branqueia o fold, com estatísticas do treino.

    Parameters
    ----------
    fold_data : FoldData
        Fold de load_fold.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    FoldData
        O fold pronto para o HDBSCAN*, as métricas de validação e os escores.
    """
    return whiten(standardize(fold_data, standardized_columns(config)), config)


def build_grid(param_grid: Mapping[str, Iterable[int]]) -> pd.DataFrame:
    """Monta a grade de hiperparâmetros como produto cartesiano.

    Parameters
    ----------
    param_grid : Mapping[str, Iterable[int]]
        Valores de cada hiperparâmetro.

    Returns
    -------
    pd.DataFrame
        Uma linha por combinação; o índice é o combo_id.
    """
    return pd.DataFrame(list(itertools.product(*param_grid.values())), columns=list(param_grid))


def load_history(path: Path) -> list[dict]:
    """Lê o histórico gravado por uma execução anterior.

    Parameters
    ----------
    path : Path
        CSV do histórico por fold e combinação.

    Returns
    -------
    list[dict]
        Uma entrada por combinação já avaliada, ou lista vazia se o arquivo não existe.
    """
    return pd.read_csv(path).to_dict("records") if path.is_file() else []


def check_resumable_history(history: list[dict], config: PipelineConfig) -> None:
    """Recusa retomar um histórico sem alguma métrica de validação ou taxa de alerta configurada.

    Parameters
    ----------
    history : list[dict]
        Histórico lido por load_history.
    config : PipelineConfig
        Configuração do pipeline.
    """
    expected = {f"{name}_score" for name in validation_metric_names(config)} | {
        alert_rate_column("f1", rate) for rate in tracked_alert_rates(config)
    }
    missing = sorted(expected - set().union(*history)) if history else []
    if missing:
        raise ValueError(
            f"O histórico em {config.grid_output_dir} não tem {missing}. Arquive a pasta ou rode "
            "sem --resume para não misturar execuções diferentes."
        )


def save_table(rows: list[dict], path: Path) -> None:
    """Grava uma lista de linhas como CSV, sem o índice.

    Parameters
    ----------
    rows : list[dict]
        Linhas da tabela.
    path : Path
        Arquivo de destino.
    """
    pd.DataFrame(rows).to_csv(path, index=False)


def has_clusters(labels: np.ndarray) -> bool:
    """Indica se a partição tem ao menos um agrupamento além do ruído.

    Parameters
    ----------
    labels : np.ndarray
        Rótulos do HDBSCAN*.

    Returns
    -------
    bool
        True se algum rótulo for diferente do ruído.
    """
    return bool(set(labels) - {NOISE_LABEL})


def cluster_counts(labels: np.ndarray) -> dict[str, int]:
    """Conta os agrupamentos e os pontos de ruído de uma partição.

    Parameters
    ----------
    labels : np.ndarray
        Rótulos do HDBSCAN*.

    Returns
    -------
    dict[str, int]
        n_clusters e n_noise_points.
    """
    return {
        "n_clusters": len(set(labels) - {NOISE_LABEL}),
        "n_noise_points": int(np.count_nonzero(labels == NOISE_LABEL)),
    }


def distinct_points(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mantém uma ocorrência de cada ponto, em float64.

    Parameters
    ----------
    X : np.ndarray
        Pontos padronizados.
    labels : np.ndarray
        Rótulo de cada ponto.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pontos distintos, em ordem lexicográfica, e os seus rótulos.
    """
    points = X.astype(np.float64)
    _, first_indices = np.unique(points, axis=0, return_index=True)
    return points[first_indices], labels[first_indices]


def subsample_points(
    X: np.ndarray, labels: np.ndarray, max_size: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Limita os pontos a max_size, com amostra estratificada pelos rótulos.

    Parameters
    ----------
    X : np.ndarray
        Pontos distintos.
    labels : np.ndarray
        Rótulo de cada ponto.
    max_size : int
        Número máximo de pontos.
    random_state : int
        Semente da amostragem.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pontos e rótulos da amostra, aleatória simples quando a estratificação não é possível.
    """
    if len(X) <= max_size:
        return X, labels
    try:
        X_sample, _, labels_sample, _ = train_test_split(
            X, labels, train_size=max_size, random_state=random_state, stratify=labels
        )
    except ValueError:
        indices = np.random.RandomState(random_state).choice(len(X), size=max_size, replace=False)
        return X[indices], labels[indices]
    return X_sample, labels_sample


def dbcv_score(X_train: np.ndarray, labels: np.ndarray, config: PipelineConfig) -> float:
    """Calcula o DBCV da partição do treino.

    Parameters
    ----------
    X_train : np.ndarray
        Treino padronizado.
    labels : np.ndarray
        Rótulos do HDBSCAN*.
    config : PipelineConfig
        Tamanho máximo da amostra, semente e número de processos do DBCV.

    Returns
    -------
    float
        DBCV em [-1, 1], ou -1 se não houver agrupamento ou se o cálculo falhar.
    """
    if not has_clusters(labels):
        return INVALID_VALIDATION_SCORE
    try:
        points, point_labels = subsample_points(
            *distinct_points(X_train, labels), config.dbcv_max_sample_size, config.random_state
        )
        if not has_clusters(point_labels):
            return INVALID_VALIDATION_SCORE
        return dbcv.dbcv(points, point_labels, noise_id=NOISE_LABEL, n_processes=config.n_processes)
    except Exception as error:
        warnings.warn(f"Erro no DBCV: {error}", RuntimeWarning, stacklevel=2)
        return INVALID_VALIDATION_SCORE


class FoldTreeCache:
    """Guarda a árvore do DISCO do último treino, que não muda entre as combinações de um fold."""

    def __init__(self) -> None:
        self._X_train: np.ndarray | None = None
        self._tree: DCTree | None = None

    def get(self, X_train: np.ndarray, min_points: int) -> DCTree:
        """Devolve a árvore de X_train, construída só quando o treino ou min_points mudam.

        Parameters
        ----------
        X_train : np.ndarray
            Treino padronizado. O mesmo objeto é reaproveitado ao longo do fold.
        min_points : int
            Parâmetro μ da distância de núcleo.

        Returns
        -------
        DCTree
            Árvore de X_train.
        """
        if X_train is not self._X_train or self._tree.min_points != min_points:
            self._X_train = X_train
            self._tree = DCTree(X_train, min_points)
        return self._tree


FOLD_TREE_CACHE = FoldTreeCache()


def disco_validation_score(
    X_train: np.ndarray, labels: np.ndarray, config: PipelineConfig
) -> float:
    """Calcula o DISCO da partição sobre o treino inteiro, sem amostragem.

    O DISCO está definido para qualquer partição, e uma partição só de ruído vale -1. Por isso, ao
    contrário do DBCV, um erro aqui interrompe a execução em vez de virar um escore.

    Parameters
    ----------
    X_train : np.ndarray
        Treino padronizado.
    labels : np.ndarray
        Rótulos do HDBSCAN*.
    config : PipelineConfig
        Valor de μ do DISCO.

    Returns
    -------
    float
        DISCO em [-1, 1].
    """
    tree = FOLD_TREE_CACHE.get(X_train, config.disco_min_points)
    return disco_score(X_train, labels, config.disco_min_points, tree)


VALIDATION_METRICS: dict[str, ValidationMetric] = {
    "dbcv": dbcv_score,
    "disco": disco_validation_score,
}


def validation_metric_names(config: PipelineConfig) -> tuple[str, ...]:
    """Lista as métricas de validação calculadas, a de seleção primeiro.

    Parameters
    ----------
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    tuple[str, ...]
        Nomes do registry VALIDATION_METRICS, sem repetição.
    """
    return tuple(dict.fromkeys((config.selection_metric, *config.extra_validation_metrics)))


def selection_column(config: PipelineConfig) -> str:
    """Devolve a coluna do histórico que escolhe a melhor configuração.

    Parameters
    ----------
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    str
        Nome da coluna da métrica de seleção, como dbcv_score.
    """
    return f"{config.selection_metric}_score"


def compute_validation_scores(
    X_train: np.ndarray, labels: np.ndarray, config: PipelineConfig
) -> dict[str, float]:
    """Calcula as métricas de validação configuradas para uma partição do treino.

    Parameters
    ----------
    X_train : np.ndarray
        Treino padronizado.
    labels : np.ndarray
        Rótulos do HDBSCAN*.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    dict[str, float]
        Valor de cada métrica, na coluna nome_score.
    """
    return {
        f"{name}_score": VALIDATION_METRICS[name](X_train, labels, config)
        for name in validation_metric_names(config)
    }


def score_glosh(
    model: HDBSCAN, X_train: np.ndarray, X_eval: np.ndarray, min_samples: int, neighbors: Neighbors
) -> ScorePair:
    """Calcula o GLOSH do treino e do conjunto avaliado.

    Parameters
    ----------
    model : HDBSCAN
        HDBSCAN* ajustado no treino.
    X_train : np.ndarray
        Treino padronizado.
    X_eval : np.ndarray
        Conjunto avaliado padronizado.
    min_samples : int
        min_samples do modelo.
    neighbors : Neighbors
        Vizinhos de glosh_neighbors.

    Returns
    -------
    ScorePair
        GLOSH do treino e do conjunto avaliado.
    """
    return ScorePair(
        *glosh_anomaly_score(
            model.condensed_tree_.to_numpy(), X_train, X_eval, min_samples, neighbors=neighbors
        )
    )


def score_manual_anomaly(
    model: HDBSCAN, X_train: np.ndarray, X_eval: np.ndarray, min_samples: int, neighbors: Neighbors
) -> ScorePair:
    """Calcula o grau de anomalia do treino e do conjunto avaliado.

    Parameters
    ----------
    model : HDBSCAN
        HDBSCAN* ajustado no treino com prediction_data=True.
    X_train : np.ndarray
        Treino padronizado.
    X_eval : np.ndarray
        Conjunto avaliado padronizado.
    min_samples : int
        min_samples do modelo.
    neighbors : Neighbors
        Distâncias de knn_distances.

    Returns
    -------
    ScorePair
        Grau de anomalia do treino e do conjunto avaliado.
    """
    _, probabilities = approximate_predict(model, X_eval)
    train_degree, degree, _ = manual_anomaly_score(
        X_train, X_eval, model.probabilities_, probabilities, min_samples, neighbors=neighbors
    )
    return ScorePair(train_degree, degree)


def glosh_degree_neighbors(X_train: np.ndarray, X_eval: np.ndarray, n_neighbors: int) -> Neighbors:
    """Busca os vizinhos do grau de anomalia com o GLOSH: os do GLOSH e os da distância kNN.

    Parameters
    ----------
    X_train : np.ndarray
        Treino padronizado.
    X_eval : np.ndarray
        Conjunto avaliado padronizado.
    n_neighbors : int
        min_samples da busca, em geral o maior da grade.

    Returns
    -------
    Neighbors
        Par com o trio de glosh_neighbors e o par de knn_distances.
    """
    return (
        glosh_neighbors(X_train, X_eval, n_neighbors),
        knn_distances(X_train, X_eval, n_neighbors),
    )


def score_manual_anomaly_glosh(
    model: HDBSCAN, X_train: np.ndarray, X_eval: np.ndarray, min_samples: int, neighbors: Neighbors
) -> ScorePair:
    """Calcula o grau de anomalia com o GLOSH no lugar de 1 - p(x).

    O grau é a média dos postos percentis do GLOSH e da distância ao vizinho de ordem
    min_samples, como em manual_anomaly_score com component="glosh".

    Parameters
    ----------
    model : HDBSCAN
        HDBSCAN* ajustado no treino.
    X_train : np.ndarray
        Treino padronizado.
    X_eval : np.ndarray
        Conjunto avaliado padronizado.
    min_samples : int
        min_samples do modelo.
    neighbors : Neighbors
        Vizinhos de glosh_degree_neighbors.

    Returns
    -------
    ScorePair
        Grau de anomalia do treino e do conjunto avaliado.
    """
    tree_neighbors, distance_neighbors = neighbors
    train_degree, degree, _ = manual_anomaly_score(
        X_train,
        X_eval,
        None,
        None,
        min_samples,
        neighbors=distance_neighbors,
        component="glosh",
        tree=model.condensed_tree_.to_numpy(),
        tree_neighbors=tree_neighbors,
    )
    return ScorePair(train_degree, degree)


def no_neighbors(X_train: np.ndarray, X_eval: np.ndarray, n_neighbors: int) -> Neighbors:
    """Atende à interface do registry para escores que não usam vizinhos.

    Parameters
    ----------
    X_train : np.ndarray
        Treino preparado.
    X_eval : np.ndarray
        Conjunto avaliado preparado.
    n_neighbors : int
        min_samples da busca, sem uso aqui.

    Returns
    -------
    Neighbors
        Tupla vazia.
    """
    return ()


def score_center_distance(
    model: HDBSCAN, X_train: np.ndarray, X_eval: np.ndarray, min_samples: int, neighbors: Neighbors
) -> ScorePair:
    """Calcula a distância de cada transação ao centro estimado no treino.

    Com os dados branqueados por whiten, a distância é a de Mahalanobis, e o centro é a média do
    treino (distance="mahalanobis") ou o centro robusto do MCD (distance="mahalanobis_mcd"). É
    a norma euclidiana de cada linha. O escore não depende do HDBSCAN* nem do min_samples e
    serve de referência para o GLOSH: mede o afastamento global do núcleo dos dados, sem
    vizinhança.

    Parameters
    ----------
    model : HDBSCAN
        HDBSCAN* ajustado no treino, sem uso aqui.
    X_train : np.ndarray
        Treino branqueado.
    X_eval : np.ndarray
        Conjunto avaliado branqueado.
    min_samples : int
        min_samples do modelo, sem uso aqui.
    neighbors : Neighbors
        Tupla vazia de no_neighbors.

    Returns
    -------
    ScorePair
        Distância ao centro no treino e no conjunto avaliado.
    """
    return ScorePair(np.linalg.norm(X_train, axis=1), np.linalg.norm(X_eval, axis=1))


ANOMALY_SCORERS: dict[str, AnomalyScorer] = {
    "glosh": AnomalyScorer(neighbors=glosh_neighbors, score=score_glosh),
    "manual_anomaly_score": AnomalyScorer(
        neighbors=knn_distances, score=score_manual_anomaly, needs_prediction_data=True
    ),
    "manual_anomaly_score_glosh": AnomalyScorer(
        neighbors=glosh_degree_neighbors, score=score_manual_anomaly_glosh
    ),
    "center_distance": AnomalyScorer(
        neighbors=no_neighbors, score=score_center_distance, needs_mahalanobis=True
    ),
}


def anomaly_score_names(config: PipelineConfig) -> tuple[str, ...]:
    """Lista os escores de anomalia calculados, o dos alertas primeiro.

    Parameters
    ----------
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    tuple[str, ...]
        Nomes do registry ANOMALY_SCORERS, sem repetição.
    """
    return tuple(dict.fromkeys((config.anomaly_score, *config.extra_anomaly_scores)))


def extra_anomaly_names(config: PipelineConfig) -> tuple[str, ...]:
    """Lista os escores de anomalia acompanhados só pela AUPRC.

    Parameters
    ----------
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    tuple[str, ...]
        Escores calculados além do que define os alertas.
    """
    return tuple(name for name in anomaly_score_names(config) if name != config.anomaly_score)


def needs_prediction_data(config: PipelineConfig) -> bool:
    """Indica se algum escore configurado exige prediction_data=True no ajuste.

    Parameters
    ----------
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    bool
        True se algum escore usar approximate_predict.
    """
    return any(ANOMALY_SCORERS[name].needs_prediction_data for name in anomaly_score_names(config))


def compute_neighbors(
    fold_data: FoldData, n_neighbors: int, config: PipelineConfig
) -> dict[str, Neighbors]:
    """Busca os vizinhos de cada escore configurado, uma vez por fold.

    Parameters
    ----------
    fold_data : FoldData
        Fold padronizado.
    n_neighbors : int
        min_samples da busca, em geral o maior da grade.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    dict[str, Neighbors]
        Vizinhos de cada escore.
    """
    return {
        name: ANOMALY_SCORERS[name].neighbors(fold_data.X_train, fold_data.X_eval, n_neighbors)
        for name in anomaly_score_names(config)
    }


def compute_anomaly_scores(
    model: HDBSCAN,
    fold_data: FoldData,
    min_samples: int,
    neighbors: Mapping[str, Neighbors],
    config: PipelineConfig,
) -> dict[str, ScorePair]:
    """Calcula os escores de anomalia configurados.

    Parameters
    ----------
    model : HDBSCAN
        HDBSCAN* ajustado no treino.
    fold_data : FoldData
        Fold padronizado.
    min_samples : int
        min_samples do modelo.
    neighbors : Mapping[str, Neighbors]
        Vizinhos de cada escore, de compute_neighbors.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    dict[str, ScorePair]
        Escores do treino e do conjunto avaliado, por nome.
    """
    return {
        name: ANOMALY_SCORERS[name].score(
            model, fold_data.X_train, fold_data.X_eval, min_samples, neighbors[name]
        )
        for name in anomaly_score_names(config)
    }


def fold_threshold(train_scores: np.ndarray, fold_data: FoldData, alert_rate: float) -> float:
    """Calcula o limiar de alerta no escore do treino, ponderado na base subamostrada.

    Parameters
    ----------
    train_scores : np.ndarray
        Escore de anomalia do treino.
    fold_data : FoldData
        Fold com os rótulos do treino e o peso das legítimas.
    alert_rate : float
        Fração de alertas.

    Returns
    -------
    float
        Limiar de alert_threshold para a fração alert_rate.
    """
    return alert_threshold(train_scores, alert_rate, fold_data.y_train, fold_data.legitimate_weight)


def alert_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Mede os alertas de um limiar.

    Parameters
    ----------
    y_true : pd.Series
        Rótulos do conjunto avaliado.
    scores : np.ndarray
        Escore de anomalia do conjunto avaliado.
    threshold : float
        Limiar de alerta, a partir do qual a transação é sinalizada.

    Returns
    -------
    dict[str, float]
        precision, recall, f1, TP, FP e FN.
    """
    predictions = (scores >= threshold).astype(int)
    _, fp, fn, tp = confusion_matrix(y_true, predictions, labels=CLASS_LABELS).ravel()
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def detection_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Mede a detecção de fraudes com o limiar de alerta e a AUPRC do escore.

    Parameters
    ----------
    y_true : pd.Series
        Rótulos do conjunto avaliado.
    scores : np.ndarray
        Escore de anomalia do conjunto avaliado.
    threshold : float
        Limiar de alerta, a partir do qual a transação é sinalizada.

    Returns
    -------
    dict[str, float]
        As métricas de alert_metrics e a auprc.
    """
    return {
        **alert_metrics(y_true, scores, threshold),
        "auprc": average_precision_score(y_true, scores),
    }


def tracked_alert_rates(config: PipelineConfig) -> tuple[float, ...]:
    """Lista as frações de alerta acompanhadas além de alert_rate.

    Parameters
    ----------
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    tuple[float, ...]
        As frações de extra_alert_rates, sem repetição e sem alert_rate.
    """
    return tuple(
        rate for rate in dict.fromkeys(config.extra_alert_rates) if rate != config.alert_rate
    )


def alert_rate_column(metric: str, rate: float) -> str:
    """Nomeia a coluna de uma métrica medida com outra fração de alertas.

    Parameters
    ----------
    metric : str
        Métrica de ALERT_RATE_COLUMNS.
    rate : float
        Fração de alertas.

    Returns
    -------
    str
        Nome da coluna, como f1_at_0.0012.
    """
    return f"{metric}_at_{rate:g}"


def tracked_alert_metrics(
    fold_data: FoldData, scores: ScorePair, config: PipelineConfig
) -> dict[str, float]:
    """Mede os alertas do escore de config.anomaly_score em cada fração de extra_alert_rates.

    Parameters
    ----------
    fold_data : FoldData
        Fold com os rótulos e o peso das legítimas.
    scores : ScorePair
        Escore dos alertas no treino e no conjunto avaliado.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    dict[str, float]
        As métricas de ALERT_RATE_COLUMNS de cada fração, nas colunas de alert_rate_column.
    """
    row = {}
    for rate in tracked_alert_rates(config):
        threshold = fold_threshold(scores.train, fold_data, rate)
        metrics = alert_metrics(fold_data.y_eval, scores.evaluated, threshold)
        row |= {alert_rate_column(name, rate): metrics[name] for name in ALERT_RATE_COLUMNS}
    return row


def tracked_auprc(
    y_true: pd.Series, scores: Mapping[str, ScorePair], config: PipelineConfig
) -> dict[str, float]:
    """Calcula a AUPRC dos escores acompanhados além do que define os alertas.

    Parameters
    ----------
    y_true : pd.Series
        Rótulos do conjunto avaliado.
    scores : Mapping[str, ScorePair]
        Escores calculados, por nome.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    dict[str, float]
        AUPRC de cada escore extra, na coluna auprc_nome.
    """
    return {
        f"auprc_{name}": average_precision_score(y_true, scores[name].evaluated)
        for name in extra_anomaly_names(config)
    }


def select_columns(values: Mapping[str, float], columns: Iterable[str]) -> dict[str, float]:
    """Seleciona e ordena entradas de um dicionário.

    Parameters
    ----------
    values : Mapping[str, float]
        Valores por nome.
    columns : Iterable[str]
        Nomes desejados, na ordem de saída.

    Returns
    -------
    dict[str, float]
        As entradas pedidas, nessa ordem.
    """
    return {column: values[column] for column in columns}


def fit_hdbscan(
    X_train: np.ndarray, params: Mapping[str, int], prediction_data: bool
) -> tuple[HDBSCAN, float]:
    """Ajusta o HDBSCAN* do cuML no treino.

    Parameters
    ----------
    X_train : np.ndarray
        Treino padronizado.
    params : Mapping[str, int]
        Hiperparâmetros da combinação.
    prediction_data : bool
        Se o ajuste guarda os dados de approximate_predict.

    Returns
    -------
    tuple[HDBSCAN, float]
        Modelo ajustado e tempo de treino, em minutos.
    """
    start = time.time()
    model = HDBSCAN(**params, prediction_data=prediction_data)
    model.fit(X_train)
    return model, (time.time() - start) / 60


def evaluate_combination(
    fold: int,
    combo_id: int,
    params: Mapping[str, int],
    fold_data: FoldData,
    neighbors: Mapping[str, Neighbors],
    config: PipelineConfig,
) -> dict[str, float]:
    """Avalia uma combinação da grade em um fold interno.

    Parameters
    ----------
    fold : int
        Número do fold interno.
    combo_id : int
        Posição da combinação na grade.
    params : Mapping[str, int]
        Hiperparâmetros da combinação.
    fold_data : FoldData
        Fold padronizado.
    neighbors : Mapping[str, Neighbors]
        Vizinhos do fold para o maior min_samples da grade.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    dict[str, float]
        Linha do histórico, com as métricas de validação e de detecção.
    """
    model, train_time_min = fit_hdbscan(fold_data.X_train, params, needs_prediction_data(config))
    labels = to_numpy(model.labels_)
    validation = compute_validation_scores(fold_data.X_train, labels, config)
    start = time.time()
    scores = compute_anomaly_scores(model, fold_data, params["min_samples"], neighbors, config)
    threshold = fold_threshold(scores[config.anomaly_score].train, fold_data, config.alert_rate)
    val_time_min = (time.time() - start) / 60
    detection = detection_metrics(
        fold_data.y_eval, scores[config.anomaly_score].evaluated, threshold
    )
    return {
        "fold": fold,
        "combo_id": combo_id,
        **params,
        **validation,
        **cluster_counts(labels),
        **select_columns(detection, INNER_DETECTION_COLUMNS),
        **tracked_auprc(fold_data.y_eval, scores, config),
        **select_columns(detection, CONFUSION_COLUMNS),
        **tracked_alert_metrics(fold_data, scores[config.anomaly_score], config),
        "train_time_min": train_time_min,
        "val_time_min": val_time_min,
    }


def run_inner_fold(
    fold: int,
    grid: pd.DataFrame,
    completed: set[tuple[int, int]],
    history: list[dict],
    results_path: Path,
    config: PipelineConfig,
) -> None:
    """Avalia as combinações pendentes de um fold interno e grava o histórico a cada uma.

    Parameters
    ----------
    fold : int
        Número do fold interno.
    grid : pd.DataFrame
        Grade de build_grid.
    completed : set[tuple[int, int]]
        Pares (fold, combo_id) já avaliados.
    history : list[dict]
        Histórico, que recebe uma linha por combinação avaliada.
    results_path : Path
        CSV do histórico, regravado após cada combinação.
    config : PipelineConfig
        Configuração do pipeline.
    """
    fold_data = prepare_fold(
        load_fold(
            config.internal_folds_dir,
            fold,
            INNER_EVAL_FILE,
            INNER_INFO_FILE,
            config.target,
            config.excluded_features,
        ),
        config,
    )
    neighbors = compute_neighbors(fold_data, int(grid["min_samples"].max()), config)
    for combo_id, combo in grid.iterrows():
        if (fold, combo_id) in completed:
            continue
        params = {name: int(value) for name, value in combo.items()}
        history.append(evaluate_combination(fold, combo_id, params, fold_data, neighbors, config))
        save_table(history, results_path)
        print(f"Combo {combo_id} | {format_metrics(history[-1], config)}")


def best_trial(trials: pd.DataFrame, column: str) -> pd.Series:
    """Escolhe a combinação de maior valor na métrica de seleção.

    Parameters
    ----------
    trials : pd.DataFrame
        Combinações de um fold.
    column : str
        Coluna da métrica de seleção.

    Returns
    -------
    pd.Series
        Linha da melhor combinação.
    """
    return trials.sort_values(column, ascending=False).iloc[0]


def run_inner_cycle(
    grid: pd.DataFrame, config: PipelineConfig, resume: bool, only_folds: list[int] | None
) -> pd.DataFrame:
    """Executa a busca em grade nos folds internos.

    Parameters
    ----------
    grid : pd.DataFrame
        Grade de build_grid.
    config : PipelineConfig
        Configuração do pipeline.
    resume : bool
        Se True, retoma o histórico gravado e pula as combinações já avaliadas.
    only_folds : list[int] | None
        Folds internos a executar; None ou lista vazia executam todos.

    Returns
    -------
    pd.DataFrame
        Histórico com uma linha por fold e combinação.
    """
    config.grid_output_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.grid_output_dir / GRID_RESULTS_FILE
    history = load_history(results_path) if resume else []
    check_resumable_history(history, config)
    completed = {(int(row["fold"]), int(row["combo_id"])) for row in history}
    folds = discover_folds(config.internal_folds_dir, config.n_folds)
    if only_folds:
        folds = [fold for fold in folds if fold in set(only_folds)]
    for fold in folds:
        if all((fold, combo_id) in completed for combo_id in grid.index):
            continue
        run_inner_fold(fold, grid, completed, history, results_path, config)
        fold_trials = pd.DataFrame([row for row in history if row["fold"] == fold])
        best_row = best_trial(fold_trials, selection_column(config)).to_dict()
        print_fold_line(INNER_STAGE, fold, best_row, config)
    return pd.DataFrame(history)


def select_best_trials(
    history: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    """Escolhe, em cada fold interno, a combinação de maior métrica de seleção.

    Parameters
    ----------
    history : pd.DataFrame
        Histórico de run_inner_cycle.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    tuple[pd.DataFrame, dict[int, dict[str, float]]]
        Linha da melhor combinação de cada fold e, por fold, os hiperparâmetros e o valor da
        métrica de seleção.
    """
    column = selection_column(config)
    summary_rows = []
    best_per_fold = {}
    for fold in sorted(int(fold) for fold in history["fold"].unique()):
        best_row = best_trial(history[history["fold"] == fold], column)
        best_per_fold[fold] = {
            **{name: int(best_row[name]) for name in config.param_grid},
            f"best_{column}": float(best_row[column]),
        }
        summary_rows.append(best_row.to_dict())
    return pd.DataFrame(summary_rows), best_per_fold


def save_inner_results(
    history: pd.DataFrame,
    summary: pd.DataFrame,
    best_per_fold: Mapping[int, Mapping[str, float]],
    output_dir: Path,
) -> None:
    """Grava o histórico, o resumo por fold e os melhores hiperparâmetros.

    Parameters
    ----------
    history : pd.DataFrame
        Histórico de run_inner_cycle.
    summary : pd.DataFrame
        Melhor combinação de cada fold.
    best_per_fold : Mapping[int, Mapping[str, float]]
        Hiperparâmetros e métrica de seleção da melhor combinação de cada fold.
    output_dir : Path
        Pasta dos resultados da busca em grade.
    """
    history.to_csv(output_dir / GRID_RESULTS_FILE, index=False)
    summary.to_csv(output_dir / GRID_SUMMARY_FILE, index=False)
    (output_dir / BEST_PARAMS_FILE).write_text(
        json.dumps(best_per_fold, indent=JSON_INDENT), encoding="utf-8"
    )


def best_configuration(
    best_per_fold: Mapping[int, Mapping[str, float]], config: PipelineConfig
) -> Mapping[str, float]:
    """Escolhe, entre os folds internos, a configuração de maior métrica de seleção.

    Parameters
    ----------
    best_per_fold : Mapping[int, Mapping[str, float]]
        Saída de select_best_trials.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    Mapping[str, float]
        Hiperparâmetros e valor da métrica de seleção da configuração escolhida.
    """
    key = f"best_{selection_column(config)}"
    return max(best_per_fold.values(), key=lambda fold_best: fold_best[key])


def evaluate_outer_fold(fold: int, params: Mapping[str, int], config: PipelineConfig) -> dict:
    """Avalia a configuração escolhida em um fold externo.

    Parameters
    ----------
    fold : int
        Número do fold externo.
    params : Mapping[str, int]
        Hiperparâmetros escolhidos no ciclo interno.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    dict
        Linha do relatório, com as métricas de detecção no teste.
    """
    fold_data = prepare_fold(
        load_fold(
            config.outer_folds_dir,
            fold,
            OUTER_EVAL_FILE,
            OUTER_INFO_FILE,
            config.target,
            config.excluded_features,
        ),
        config,
    )
    model, train_time_min = fit_hdbscan(fold_data.X_train, params, needs_prediction_data(config))
    start = time.time()
    neighbors = compute_neighbors(fold_data, params["min_samples"], config)
    scores = compute_anomaly_scores(model, fold_data, params["min_samples"], neighbors, config)
    threshold = fold_threshold(scores[config.anomaly_score].train, fold_data, config.alert_rate)
    predict_time_min = (time.time() - start) / 60
    detection = detection_metrics(
        fold_data.y_eval, scores[config.anomaly_score].evaluated, threshold
    )
    return {
        "fold": fold,
        **params,
        **select_columns(detection, OUTER_DETECTION_COLUMNS),
        **tracked_auprc(fold_data.y_eval, scores, config),
        **select_columns(detection, CONFUSION_COLUMNS),
        **tracked_alert_metrics(fold_data, scores[config.anomaly_score], config),
        "train_time_min": train_time_min,
        "predict_time_min": predict_time_min,
    }


def run_outer_cycle(params: Mapping[str, int], config: PipelineConfig) -> pd.DataFrame:
    """Avalia a configuração escolhida em todos os folds externos.

    Parameters
    ----------
    params : Mapping[str, int]
        Hiperparâmetros escolhidos no ciclo interno.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    pd.DataFrame
        Uma linha por fold externo.
    """
    rows = []
    for fold in discover_folds(config.outer_folds_dir, config.n_folds):
        rows.append(evaluate_outer_fold(fold, params, config))
        print_fold_line(OUTER_STAGE, fold, rows[-1], config)
    return pd.DataFrame(rows)


def append_mean_row(report: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta ao relatório a média de cada coluna entre os folds.

    Parameters
    ----------
    report : pd.DataFrame
        Uma linha por fold externo.

    Returns
    -------
    pd.DataFrame
        O relatório com a linha final da média.
    """
    mean = report.mean().to_dict()
    mean["fold"] = MEAN_ROW_LABEL
    return pd.concat([report, pd.DataFrame([mean])], ignore_index=True)


def save_outer_report(report: pd.DataFrame, output_dir: Path) -> Path:
    """Grava o relatório do ciclo externo.

    Parameters
    ----------
    report : pd.DataFrame
        Relatório com a linha da média.
    output_dir : Path
        Pasta do relatório.

    Returns
    -------
    Path
        Caminho do CSV gravado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / OUTER_REPORT_FILE
    report.to_csv(path, index=False)
    return path


def metric_labels(config: PipelineConfig) -> dict[str, str]:
    """Associa cada coluna de métrica ao rótulo das linhas de acompanhamento.

    Parameters
    ----------
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    dict[str, str]
        Rótulo de cada coluna, na ordem de exibição.
    """
    validation = {f"{name}_score": name.upper() for name in validation_metric_names(config)}
    detection = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "auprc": f"AUPRC {config.anomaly_score}",
    }
    extras = {f"auprc_{name}": f"AUPRC {name}" for name in extra_anomaly_names(config)}
    rates = {alert_rate_column("f1", rate): f"F1@{rate:g}" for rate in tracked_alert_rates(config)}
    return validation | detection | extras | rates


def format_metrics(row: Mapping[str, float], config: PipelineConfig) -> str:
    """Formata os hiperparâmetros e as métricas de uma linha de resultado.

    Parameters
    ----------
    row : Mapping[str, float]
        Linha do histórico ou do relatório.
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    str
        Pares nome=valor separados por barras.
    """
    params = [f"{name}={int(row[name])}" for name in config.param_grid]
    metrics = [
        f"{label}={row[column]:.{DISPLAY_DECIMALS}f}"
        for column, label in metric_labels(config).items()
        if column in row
    ]
    return " | ".join(params + metrics)


def print_fold_line(
    stage: str, fold: int, row: Mapping[str, float], config: PipelineConfig
) -> None:
    """Imprime, em uma linha, a configuração e as métricas de um fold.

    Parameters
    ----------
    stage : str
        Ciclo do fold, interno ou externo.
    fold : int
        Número do fold.
    row : Mapping[str, float]
        Linha do fold, a melhor combinação no ciclo interno.
    config : PipelineConfig
        Configuração do pipeline.
    """
    print(f"[{stage} {fold}/{config.n_folds}] {format_metrics(row, config)}")


def validate_config(config: PipelineConfig) -> None:
    """Confere as métricas, as variáveis excluídas, as frações de alerta e a distância.

    Parameters
    ----------
    config : PipelineConfig
        Configuração do pipeline.
    """
    unknown_features = sorted(set(config.excluded_features) - {*V_COLUMNS, *DERIVED_COLUMNS})
    if unknown_features:
        raise ValueError(f"Variáveis desconhecidas em excluded_features: {unknown_features}")
    invalid_rates = [
        rate for rate in (config.alert_rate, *config.extra_alert_rates) if not 0 < rate < 1
    ]
    if invalid_rates:
        raise ValueError(f"Frações de alerta fora do intervalo (0, 1): {invalid_rates}")
    for names, registry in (
        (validation_metric_names(config), VALIDATION_METRICS),
        (anomaly_score_names(config), ANOMALY_SCORERS),
    ):
        unknown = sorted(set(names) - set(registry))
        if unknown:
            raise ValueError(f"Métricas desconhecidas: {unknown}. Disponíveis: {sorted(registry)}")
    if config.distance not in DISTANCES:
        raise ValueError(f"distance={config.distance!r}: use um de {DISTANCES}.")
    mahalanobis_only = [
        name for name in anomaly_score_names(config) if ANOMALY_SCORERS[name].needs_mahalanobis
    ]
    if mahalanobis_only and config.distance == EUCLIDEAN:
        raise ValueError(
            f"{mahalanobis_only} exigem uma distância de Mahalanobis, e não {EUCLIDEAN!r}."
        )


def distance_label(config: PipelineConfig) -> str:
    """Descreve a distância configurada, para o cabeçalho do log.

    Parameters
    ----------
    config : PipelineConfig
        Configuração do pipeline.

    Returns
    -------
    str
        A distância e, nas de Mahalanobis, a covariância e os parâmetros fixos do branqueamento.
    """
    if config.distance == EUCLIDEAN:
        return EUCLIDEAN
    details = [f"piso dos autovalores {DEFAULT_EIGEN_FLOOR:g}"]
    if config.distance == "mahalanobis_mcd":
        details = [
            f"MCD com support_fraction={DEFAULT_SUPPORT_FRACTION:g}",
            f"amostra de até {DEFAULT_SAMPLE_SIZE} pontos",
            *details,
        ]
    else:
        details = ["covariância empírica do treino", *details]
    return f"{config.distance} ({', '.join(details)})"


def parse_args() -> argparse.Namespace:
    """Lê as opções de execução da linha de comando.

    Returns
    -------
    argparse.Namespace
        resume e folds.
    """
    parser = argparse.ArgumentParser(
        description="Grid search do HDBSCAN aprimorado com retomada por fold."
    )
    parser.add_argument(
        "--resume", action="store_true", help="lê o CSV parcial e pula os folds já concluídos"
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="executa apenas estes folds internos, por exemplo: --folds 2 3",
    )
    return parser.parse_args()


def main() -> None:
    """Executa o ciclo interno, escolhe a melhor configuração e a avalia no ciclo externo."""
    args = parse_args()
    validate_config(CONFIG)

    print("HDBSCAN* aprimorado: validação cruzada aninhada")
    print(
        f"Grade {CONFIG.param_grid} | seleção por {CONFIG.selection_metric} | alertas por "
        f"{CONFIG.anomaly_score} com alert_rate={CONFIG.alert_rate} | folds internos: "
        f"{args.folds or 'todos'} | retomada: {'sim' if args.resume else 'não'}"
    )
    print(f"Variáveis fora da distância: {list(CONFIG.excluded_features) or 'nenhuma'}")
    print(f"Variáveis padronizadas: {standardized_columns(CONFIG)}")
    print(f"Distância: {distance_label(CONFIG)}")
    print(f"Taxas de alerta acompanhadas: {list(tracked_alert_rates(CONFIG)) or 'nenhuma'}")

    history = run_inner_cycle(build_grid(CONFIG.param_grid), CONFIG, args.resume, args.folds)
    summary, best_per_fold = select_best_trials(history, CONFIG)
    save_inner_results(history, summary, best_per_fold, CONFIG.grid_output_dir)
    best = best_configuration(best_per_fold, CONFIG)
    params = {name: int(best[name]) for name in CONFIG.param_grid}
    column = selection_column(CONFIG)
    best_row = {**params, column: best[f"best_{column}"]}

    print(f"Melhor configuração do ciclo interno: {format_metrics(best_row, CONFIG)}")
    report = append_mean_row(run_outer_cycle(params, CONFIG))
    report_path = save_outer_report(report, CONFIG.report_output_dir)

    print(f"Média nos folds externos: {format_metrics(report.iloc[-1].to_dict(), CONFIG)}")
    print(f"Resultados salvos em {CONFIG.grid_output_dir} e {report_path}")


if __name__ == "__main__":
    main()
