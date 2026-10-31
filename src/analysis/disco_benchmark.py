"""Mede o custo do DISCO e o compara com o DBCV na base subamostrada, para docs/metrics/disco.md.

Precisa do dbcv, que quebra com o SciPy 1.18, e por isso roda em um ambiente à parte:

    uv run --no-project --python 3.13 --with numpy --with "scipy<1.17" --with scikit-learn
        --with pandas --with psutil --with "git+https://github.com/FelSiq/DBCV"
        python src/analysis/disco_benchmark.py {scaling,partitions,grid}

A etapa scaling compara o tempo e a memória nas mesmas amostras, cada medição em um processo
próprio. Com --original-dir, inclui o código oficial do DISCO
(https://github.com/pasiweber/DISCO), carregado sem o __init__.py do pacote, que exige outras
métricas. A etapa partitions mede o custo de uma partição sobre uma árvore pronta. A etapa grid
compara as duas métricas na grade do pipeline, com o HDBSCAN do scikit-learn em CPU no lugar do
cuML. As tabelas vão para results/unsupervised/disco/.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src/models"))

from disco import DCTree, disco_score  # noqa: E402
from features import build_preprocessor, engineer_features  # noqa: E402
from paths import LOCAL_UNDERSAMPLED_PATH  # noqa: E402

OUTPUT_DIR = ROOT / "results/unsupervised/disco"
FOLD = 1
TARGET = "Class"
RANDOM_STATE = 42
# Os mesmos valores de CONFIG em run_hdbscan_enhanced_pipeline.py, que importa o cuML.
MIN_CLUSTER_SIZE = 50
MIN_SAMPLES_GRID = (100, 150, 200, 250, 300, 350, 400)
DBCV_SAMPLE_SIZE = 28_000

SCALING_MIN_SAMPLES = 100
SCALING_RUNS = (
    *(
        (size, name)
        for size in (2_000, 5_000, 10_000, 15_000)
        for name in ("dbcv", "official", "disco")
    ),
    (28_000, "dbcv"),
    (28_000, "disco"),
    (None, "disco"),
)
# Com 28 mil pontos, as matrizes do código oficial não caberiam na memória desta máquina (16 GB).
PARTITION_CLUSTER_COUNTS = (3, 10, 30)
PARTITION_NOISE_FRACTION = 0.3
MEMORY_POLL_SECONDS = 0.1
RUN_TIMEOUT_SECONDS = 1800


def load_train():
    """Lê o treino do fold interno com o mesmo tratamento de load_fold e standardize do pipeline."""
    train = pd.read_csv(LOCAL_UNDERSAMPLED_PATH.internal_folds_dir / f"fold_{FOLD}_train.csv")
    features = engineer_features(train.drop(columns=[TARGET]).drop_duplicates())
    return np.asarray(build_preprocessor().fit_transform(features), dtype=np.float64)


def fit_labels(X, min_samples):
    """Agrupa o treino com o HDBSCAN do scikit-learn e devolve os rótulos e o tempo."""
    start = time.perf_counter()
    model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE, min_samples=min_samples, n_jobs=-1, copy=True
    )
    return model.fit_predict(X), time.perf_counter() - start


def sample_like_dbcv(X, labels, size):
    """Mantém os pontos distintos e amostra por rótulo, como dbcv_score no pipeline."""
    _, first_indices = np.unique(X, axis=0, return_index=True)
    X, labels = X[first_indices], labels[first_indices]
    if size is None or size >= len(X):
        return X, labels
    sample, _, sample_labels, _ = train_test_split(
        X, labels, train_size=size, random_state=RANDOM_STATE, stratify=labels
    )
    return sample, sample_labels


def run_dbcv(X, labels):
    import dbcv

    return dbcv.dbcv(X, labels, noise_id=-1, n_processes=os.cpu_count() or 1)


def load_official(original_dir):
    """Carrega só o disco.py e o dctree.py do código oficial."""
    evaluation = Path(original_dir) / "src/Evaluation"
    for package in ("src", "src.Evaluation", "src.Evaluation.dcdistances"):
        sys.modules[package] = types.ModuleType(package)
    modules = {
        "src.Evaluation.dcdistances.dctree": evaluation / "dcdistances/dctree.py",
        "src.Evaluation.disco": evaluation / "disco.py",
    }
    for name, path in modules.items():
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules["src.Evaluation.disco"].disco_score


def measure(name, X, labels, original_dir):
    """Executa uma métrica e devolve a nota, o tempo e o pico de memória, com os processos."""
    process = psutil.Process()
    peak = [process.memory_info().rss]
    finished = threading.Event()

    def watch_memory():
        while not finished.wait(MEMORY_POLL_SECONDS):
            try:
                children = process.children(recursive=True)
                used = process.memory_info().rss + sum(c.memory_info().rss for c in children)
            except psutil.Error:
                continue
            peak[0] = max(peak[0], used)

    threading.Thread(target=watch_memory, daemon=True).start()
    metrics = {"dbcv": run_dbcv, "disco": disco_score}
    if name == "official":
        metrics["official"] = load_official(original_dir)
    start = time.perf_counter()
    score = metrics[name](X, labels)
    seconds = time.perf_counter() - start
    finished.set()
    return {"score": float(score), "seconds": seconds, "peak_gb": peak[0] / 1e9}


def run_scaling(original_dir):
    """Mede cada par de tamanho e métrica em um processo próprio, para o pico não se misturar."""
    X = load_train()
    labels, _ = fit_labels(X, SCALING_MIN_SAMPLES)
    labels_path = OUTPUT_DIR / "scaling_labels.npy"
    np.save(labels_path, labels)
    rows = []
    for size, name in SCALING_RUNS:
        if name == "official" and original_dir is None:
            continue
        command = [sys.executable, __file__, "one", name, str(size or 0), str(labels_path)]
        if original_dir:
            command += ["--original-dir", original_dir]
        output = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=RUN_TIMEOUT_SECONDS
        )
        rows.append(json.loads(output.stdout.strip().splitlines()[-1]))
        print(rows[-1], flush=True)
    labels_path.unlink()
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "scaling.csv", index=False)


def run_one(name, size, labels_path, original_dir):
    X, labels = sample_like_dbcv(load_train(), np.load(labels_path), size or None)
    row = {"metric": name, "n_points": len(X), **measure(name, X, labels, original_dir)}
    print(json.dumps(row))


def run_partitions():
    """Mede a construção da árvore e o custo de cada partição com vários agrupamentos."""
    X = load_train()
    start = time.perf_counter()
    tree = DCTree(X)
    build_seconds = time.perf_counter() - start
    rng = np.random.RandomState(RANDOM_STATE)
    rows = []
    for n_clusters in PARTITION_CLUSTER_COUNTS:
        labels = KMeans(n_clusters, n_init=1, random_state=RANDOM_STATE).fit_predict(X)
        labels[rng.rand(len(X)) < PARTITION_NOISE_FRACTION] = -1
        start = time.perf_counter()
        score = disco_score(X, labels, tree=tree)
        rows.append(
            {
                "n_points": len(X),
                "n_clusters": n_clusters,
                "tree_seconds": build_seconds,
                "partition_seconds": time.perf_counter() - start,
                "disco": score,
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "partitions.csv", index=False)


def run_grid():
    """Calcula o DISCO no treino inteiro e o DBCV na amostra, em cada combinação da grade."""
    X = load_train()
    start = time.perf_counter()
    tree = DCTree(X)
    print(f"Árvore do DISCO: {time.perf_counter() - start:.1f} s", flush=True)
    rows = []
    for min_samples in MIN_SAMPLES_GRID:
        labels, hdbscan_seconds = fit_labels(X, min_samples)
        start = time.perf_counter()
        disco_value = disco_score(X, labels, tree=tree)
        disco_seconds = time.perf_counter() - start
        start = time.perf_counter()
        dbcv_value = run_dbcv(*sample_like_dbcv(X, labels, DBCV_SAMPLE_SIZE))
        rows.append(
            {
                "min_samples": min_samples,
                "n_clusters": len(set(labels.tolist()) - {-1}),
                "noise_fraction": float(np.mean(labels == -1)),
                "disco": disco_value,
                "dbcv": float(dbcv_value),
                "disco_seconds": disco_seconds,
                "dbcv_seconds": time.perf_counter() - start,
                "hdbscan_seconds": hdbscan_seconds,
            }
        )
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "grid.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["scaling", "partitions", "grid", "one"])
    parser.add_argument("one_args", nargs="*", help="uso interno da etapa scaling")
    parser.add_argument("--original-dir", default=None, help="pasta do código oficial do DISCO")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "scaling":
        run_scaling(args.original_dir)
    elif args.stage == "partitions":
        run_partitions()
    elif args.stage == "grid":
        run_grid()
    else:
        name, size, labels_path = args.one_args
        run_one(name, int(size), labels_path, args.original_dir)


if __name__ == "__main__":
    main()
