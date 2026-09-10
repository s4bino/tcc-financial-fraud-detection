"""
Execução do grid search do HDBSCAN na infraestrutura da Modal.

Este arquivo é uma adaptação de `run_hdbscan_grid_search.py` para execução
remota. Os blocos de cálculo — pré-processamento, ajuste do HDBSCAN, cálculo do
DBCV, métricas e agregação — foram copiados literalmente do original; o que
muda aqui é apenas a orquestração: em qual máquina cada bloco roda e em que
ordem. Nenhum número produzido por este arquivo difere do que o original
produziria, desde que os mesmos folds sejam usados.

MOTIVO DA SEPARAÇÃO EM ETAPAS
-----------------------------
No original, um único laço alterna entre a GPU (o `fit` do cuML, ~5 s) e a CPU
(o DBCV, minutos). A GPU fica ociosa durante o DBCV, sendo cobrada o tempo
todo. Como a Modal permite declarar hardware por função, o trabalho é dividido:

    1. `gpu_fold_stage`   — GPU. Ajusta o HDBSCAN e aplica `approximate_predict`
                            em todas as combinações de um fold. Grava os rótulos
                            e a matriz escalonada no Volume. Um contêiner por
                            fold, em paralelo.

    2. `dbcv_stage`       — SEM GPU. Calcula o DBCV de uma combinação a partir do
                            que a etapa 1 gravou. Um contêiner por combinação,
                            em paralelo.

    3. `aggregate_stage`  — CPU. Junta as duas metades, monta os relatórios da
                            Etapa 1 e elege os melhores parâmetros por fold.

    4. `outer_stage`      — GPU. Avaliação nos folds externos com o melhor
                            parâmetro do ciclo interno (Etapa 2).

As imagens também são separadas: a etapa do DBCV não carrega o RAPIDS (~3 GB),
então seus contêineres sobem em segundos.

MEMÓRIA
-------
O DBCV monta matrizes de distância de `n²` para n = MAX_DBCV_SAMPLE_SIZE. Com
28.000 pontos, cada matriz float64 ocupa ~6,3 GB, e o `n_processes` multiplica
esse consumo. Daí o teto alto em DBCV_MEMORY_MIB.

Atenção ao `DBCV_N_PROCESSES`: o original usa `os.cpu_count()`, que dentro de um
contêiner reporta os núcleos da máquina hospedeira, não o limite da função.
Deixar assim abriria processos demais e estouraria a memória — por isso o valor
é fixado explicitamente aqui.

USO
---
    pip install modal
    modal setup

    # envio dos folds (uma vez)
    modal volume create tcc-hdbscan-dados
    modal volume put tcc-hdbscan-dados data/processed/inner_folds  /data/processed/inner_folds
    modal volume put tcc-hdbscan-dados data/processed/outer_folds  /data/processed/outer_folds

    # teste rápido das imagens, antes de gastar tempo de GPU
    modal run src/models/run_hdbscan_grid_search_modal.py::smoke

    # execução; --detach mantém o job vivo com o terminal fechado
    modal run --detach src/models/run_hdbscan_grid_search_modal.py

    # resultados de volta
    modal volume get tcc-hdbscan-dados /results/unsupervised results/
"""

import modal

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

APP_NAME = "tcc-hdbscan"
VOLUME_NAME = "tcc-hdbscan-dados"
VOLUME_MOUNT = "/vol"

INTERNAL_FOLDS_DIR = f"{VOLUME_MOUNT}/data/processed/inner_folds"
OUTER_FOLDS_DIR = f"{VOLUME_MOUNT}/data/processed/outer_folds"
RESULTS_DIR = f"{VOLUME_MOUNT}/results/unsupervised"
CACHE_DIR = f"{VOLUME_MOUNT}/cache"  # matrizes e rótulos trocados entre etapas

TARGET = "Class"
MAX_DBCV_SAMPLE_SIZE = 28000

PARAM_GRID = {
    "min_cluster_size": [5, 25, 50],
    "min_samples": [280, 290, 300, 315, 325, 350, 400],
}

# --- Hardware -----------------------------------------------------------------

GPU_TYPE = "T4"  # o fit leva ~5 s; uma GPU maior não compensa o custo
GPU_CPU = 4.0
GPU_MEMORY_MIB = (8192, 32768)
GPU_TIMEOUT_S = 2 * 60 * 60

# Matrizes de distância n² dominam o consumo: piso de 32 GiB, teto de 128 GiB.
DBCV_CPU = 8.0
DBCV_MEMORY_MIB = (32768, 131072)
DBCV_TIMEOUT_S = 6 * 60 * 60

# Substitui o `os.cpu_count()` do original, que não respeita o limite do
# contêiner. Mantenha <= DBCV_CPU: cada processo replica parte das matrizes.
DBCV_N_PROCESSES = 8

# Teto de contêineres simultâneos na etapa do DBCV. Reduza se esbarrar em
# limites da conta. Em SDKs da Modal anteriores a 1.0 o parâmetro do decorador
# chama-se `concurrency_limit` em vez de `max_containers`.
DBCV_MAX_CONTAINERS = 25

# ==============================================================================
# IMAGENS
# ==============================================================================

_BASE_PACKAGES = ("numpy", "pandas", "scikit-learn")

# Etapa de GPU: RAPIDS, sem o DBCV.
cuml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_BASE_PACKAGES)
    .pip_install("cuml-cu12", extra_index_url="https://pypi.nvidia.com")
)

# Etapa de CPU: DBCV, sem o RAPIDS.
dbcv_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*_BASE_PACKAGES)
    .pip_install("git+https://github.com/FelSiq/DBCV")
)

# Agregação: só pandas.
cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(*_BASE_PACKAGES)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ==============================================================================
# AUXILIARES DE ORQUESTRAÇÃO
# ==============================================================================


def _combinations():
    """As combinações da grade, na mesma ordem e com os mesmos `combo_id` do
    original — que os obtém de `df_param_combinations.iterrows()`."""
    import itertools

    import pandas as pd

    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(itertools.product(*param_values))
    return pd.DataFrame(all_combinations, columns=param_names)


def _cache_paths(fold, combo_id=None):
    matrix = f"{CACHE_DIR}/fold_{fold}_X_train_scaled.npy"
    if combo_id is None:
        return matrix
    return matrix, f"{CACHE_DIR}/fold_{fold}_combo_{combo_id}_labels.npy"


# ==============================================================================
# ETAPA 0 — DESCOBERTA DOS FOLDS
# ==============================================================================


@app.function(image=cpu_image, volumes={VOLUME_MOUNT: volume}, timeout=600)
def discover_folds():
    """Mesma descoberta do original: os folds vêm dos arquivos `*_train.csv`."""
    import os

    volume.reload()

    folds = sorted({
        int(f.split("_")[1])
        for f in os.listdir(INTERNAL_FOLDS_DIR)
        if f.endswith("_train.csv")
    })
    print(f"Folds encontrados para Avaliação (HDBSCAN): {folds}")

    df_param_combinations = _combinations()
    print(f"Total de combinações HDBSCAN por fold: {len(df_param_combinations)}")

    return folds


# ==============================================================================
# ETAPA 1a — GPU: AJUSTE E AVALIAÇÃO NA VALIDAÇÃO
# ==============================================================================


@app.function(
    image=cuml_image,
    gpu=GPU_TYPE,
    cpu=GPU_CPU,
    memory=GPU_MEMORY_MIB,
    timeout=GPU_TIMEOUT_S,
    volumes={VOLUME_MOUNT: volume},
)
def gpu_fold_stage(fold: int):
    """Roda todas as combinações de um fold interno na GPU.

    Devolve um registro por combinação, com `dbcv_score` ainda vazio, e grava no
    Volume o que a etapa do DBCV precisa: a matriz escalonada de treino (uma por
    fold) e os rótulos (um por combinação).
    """
    import os
    import time

    import numpy as np
    from cuml.cluster import HDBSCAN
    from cuml.cluster.hdbscan import approximate_predict
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    from sklearn.preprocessing import RobustScaler, StandardScaler
    import pandas as pd

    volume.reload()
    os.makedirs(CACHE_DIR, exist_ok=True)

    df_param_combinations = _combinations()
    target = TARGET
    internal_folds_dir = INTERNAL_FOLDS_DIR

    print(f"\n--- Iniciando Grid Search para o Fold Interno {fold} ---")

    train_path = os.path.join(internal_folds_dir, f"fold_{fold}_train.csv")
    val_path   = os.path.join(internal_folds_dir, f"fold_{fold}_val.csv")

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)

    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    X_val   = df_val.drop(columns=[target])
    y_val   = df_val[target]
    del df_train, df_val

    # ==============================================================
    # 2. PRÉ-PROCESSAMENTO
    # ==============================================================
    preprocessor = ColumnTransformer(
        transformers=[
            ("rob_scaler", RobustScaler(), ['Amount']),
            ("std_scaler", StandardScaler(), ['Time'])
        ],
        remainder="passthrough"
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)

    # A matriz de treino é a entrada do DBCV; grava-se uma vez por fold.
    np.save(_cache_paths(fold), np.asarray(X_train_scaled))

    fold_records = []

    # ==============================================================
    # GRID
    # ==============================================================
    for combo_id, combo in df_param_combinations.iterrows():
        params_dict = combo.to_dict()

        min_cluster_size = int(params_dict["min_cluster_size"])
        min_samples = int(params_dict["min_samples"])

        # ==============================================================
        # 3. TREINAMENTO
        # ==============================================================
        t0 = time.time()
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True
        )
        model.fit(X_train_scaled)
        train_time = time.time() - t0

        labels = model.labels_.to_numpy() if hasattr(model.labels_, 'to_numpy') else model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # ==============================================================
        # 4. A MÉTRICA DE OTIMIZAÇÃO: DBCV
        #
        # Calculada em `dbcv_stage`, sem GPU. Aqui apenas persistimos os
        # rótulos de que ela depende.
        # ==============================================================
        _, labels_path = _cache_paths(fold, combo_id)
        np.save(labels_path, labels)

        # ==============================================================
        # 5. AVALIAÇÃO PREDITIVA NA VALIDAÇÃO (Tracking oculto)
        # ==============================================================
        t1 = time.time()
        val_labels, val_probs = approximate_predict(model, X_val_scaled)
        val_time = time.time() - t1

        if hasattr(val_labels, 'get'): val_labels = val_labels.get()
        if hasattr(val_probs, 'get'):  val_probs = val_probs.get()

        preds = (val_labels == -1).astype(int)
        preds_proba = 1.0 - val_probs

        f1    = f1_score(y_val, preds, zero_division=0)
        prec  = precision_score(y_val, preds, zero_division=0)
        rec   = recall_score(y_val, preds, zero_division=0)
        auc   = roc_auc_score(y_val, preds_proba)
        auprc = average_precision_score(y_val, preds_proba)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()

        # A ordem das chaves reproduz a do original: é ela que determina a
        # ordem das colunas do CSV por fold.
        fold_records.append({
            "fold": fold,
            "combo_id": combo_id,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "dbcv_score": None,  # preenchido em `aggregate_stage`
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "auc": auc,
            "auprc": auprc,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "train_time": train_time,
            "val_time": val_time
        })

        print(f"Combo {combo_id} | min_cluster={min_cluster_size} | min_samples={min_samples} | F1 Val={f1:.4f}")

    del X_train, y_train, X_val, y_val

    volume.commit()
    return fold_records


# ==============================================================================
# ETAPA 1b — CPU: DBCV
# ==============================================================================


@app.function(
    image=dbcv_image,
    cpu=DBCV_CPU,
    memory=DBCV_MEMORY_MIB,
    timeout=DBCV_TIMEOUT_S,
    max_containers=DBCV_MAX_CONTAINERS,
    volumes={VOLUME_MOUNT: volume},
)
def dbcv_stage(task: dict):
    """DBCV de uma combinação, sem GPU.

    O bloco de cálculo é o do original, palavra por palavra. Como a amostragem
    usa `random_state=42` sobre a mesma matriz e os mesmos rótulos, o escore é
    idêntico ao que sairia no laço único.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    import dbcv

    volume.reload()

    fold = task["fold"]
    combo_id = task["combo_id"]
    max_dbcv_sample_size = MAX_DBCV_SAMPLE_SIZE
    n_processes = DBCV_N_PROCESSES

    matrix_path, labels_path = _cache_paths(fold, combo_id)
    X_train_scaled = np.load(matrix_path)
    labels = np.load(labels_path)

    valid_clusters = set(labels) - {-1}
    if len(valid_clusters) >= 1:
        try:
            X_mat = X_train_scaled.astype(np.float64)

            _, unique_indices = np.unique(X_mat, axis=0, return_index=True)

            X_mat_unique = X_mat[unique_indices]
            labels_unique = labels[unique_indices]

            if len(X_mat_unique) > max_dbcv_sample_size:
                try:
                    X_mat_unique, _, labels_unique, _ = train_test_split(
                        X_mat_unique,
                        labels_unique,
                        train_size=max_dbcv_sample_size,
                        random_state=42,
                        stratify=labels_unique
                    )
                except ValueError:
                    rng = np.random.RandomState(42)
                    sample_idx = rng.choice(len(X_mat_unique), size=max_dbcv_sample_size, replace=False)
                    X_mat_unique = X_mat_unique[sample_idx]
                    labels_unique = labels_unique[sample_idx]

            if len(set(labels_unique) - {-1}) >= 1:
                dbcv_score = dbcv.dbcv(X_mat_unique, labels_unique, noise_id=-1, n_processes=n_processes)
            else:
                dbcv_score = -1.0

        except Exception as e:
            print(f"Erro no DBCV: {e}")
            dbcv_score = -1.0
    else:
        dbcv_score = -1.0

    print(f"Fold {fold} | Combo {combo_id} | DBCV={dbcv_score:.4f}")
    return {"fold": fold, "combo_id": combo_id, "dbcv_score": float(dbcv_score)}


# ==============================================================================
# ETAPA 1c — AGREGAÇÃO E SELEÇÃO
# ==============================================================================


@app.function(
    image=cpu_image,
    cpu=2.0,
    memory=(4096, 16384),
    timeout=900,
    volumes={VOLUME_MOUNT: volume},
)
def aggregate_stage(records: list, dbcv_results: list, folds: list):
    """Junta GPU e DBCV e reproduz o bloco de relatórios do original."""
    import json
    import os

    import pandas as pd

    output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    scores = {(r["fold"], r["combo_id"]): r["dbcv_score"] for r in dbcv_results}

    # Reordena por (fold, combo_id) para reproduzir a ordem em que o laço único
    # empilharia os registros — de que depende o desempate do `sort_values`.
    all_trials_history = sorted(records, key=lambda r: (r["fold"], r["combo_id"]))
    for record in all_trials_history:
        record["dbcv_score"] = scores[(record["fold"], record["combo_id"])]

    best_params_per_fold = {}

    # ==============================================================
    # 6. GERAÇÃO DE RELATÓRIOS E SELEÇÃO DOS MELHORES
    # ==============================================================
    df_folds = pd.DataFrame(all_trials_history)
    df_folds.to_csv(os.path.join(output_dir, "hdbscan_grid_results_by_fold.csv"), index=False)

    summary_data = []
    for fold in folds:
        fold_data = df_folds[df_folds['fold'] == fold]

        best_row = fold_data.sort_values("dbcv_score", ascending=False).iloc[0]

        best_params_per_fold[fold] = {
            "min_cluster_size": int(best_row["min_cluster_size"]),
            "min_samples": int(best_row["min_samples"]),
            "best_dbcv_score": float(best_row["dbcv_score"])
        }
        summary_data.append(best_row.to_dict())

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, "hdbscan_grid_best_summary.csv"), index=False)

    with open(os.path.join(output_dir, "best_params_hdbscan.json"), "w") as f:
        json.dump(best_params_per_fold, f, indent=4)

    volume.commit()
    return best_params_per_fold


# ==============================================================================
# ETAPA 2 — GPU: AVALIAÇÃO EXTERNA
# ==============================================================================


@app.function(
    image=cuml_image,
    gpu=GPU_TYPE,
    cpu=GPU_CPU,
    memory=GPU_MEMORY_MIB,
    timeout=GPU_TIMEOUT_S,
    volumes={VOLUME_MOUNT: volume},
)
def outer_stage(best_parameter_achieved: dict):
    """Avaliação nos folds externos — o `hdbscan_outer_evaluation` do original."""
    import os
    import time

    from cuml.cluster import HDBSCAN
    from cuml.cluster.hdbscan import approximate_predict
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    from sklearn.preprocessing import RobustScaler, StandardScaler
    import pandas as pd

    volume.reload()

    outer_folds_dir = OUTER_FOLDS_DIR
    target = TARGET
    output_dir = RESULTS_DIR

    os.makedirs(output_dir, exist_ok=True)
    final_metrics = []

    print("\n--- Iniciando Avaliação Externa (Outer Folds) com Predição ---")

    folds = sorted({
        int(f.split("_")[1])
        for f in os.listdir(outer_folds_dir)
        if f.endswith("_train.csv")
    })

    min_cluster_size = int(best_parameter_achieved["min_cluster_size"])
    min_samples = int(best_parameter_achieved["min_samples"])

    for fold in folds:
        print(f"Avaliando Fold Externo {fold} | min_cluster_size={min_cluster_size} | min_samples={min_samples}")

        train_path = os.path.join(outer_folds_dir, f"fold_{fold}_train.csv")
        test_path = os.path.join(outer_folds_dir, f"fold_{fold}_test.csv")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        X_train = df_train.drop(columns=[target])
        y_train = df_train[target]
        X_test = df_test.drop(columns=[target])
        y_test = df_test[target]

        preprocessor = ColumnTransformer(
            transformers=[
                ("rob_scaler", RobustScaler(), ['Amount']),
                ("std_scaler", StandardScaler(), ['Time'])
            ],
            remainder="passthrough"
        )

        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)

        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True
        )

        t0 = time.time()
        model.fit(X_train_scaled) # Treina com a versão escalonada
        train_time = time.time() - t0

        t1 = time.time()
        test_labels, test_probabilities = approximate_predict(model, X_test_scaled) # Avalia na versão escalonada
        predict_time = time.time() - t1

        if hasattr(test_labels, 'get'): test_labels = test_labels.get()
        if hasattr(test_probabilities, 'get'): test_probabilities = test_probabilities.get()

        preds = (test_labels == -1).astype(int)
        preds_proba = 1.0 - test_probabilities

        f1      = f1_score(y_test, preds, zero_division=0)
        prec    = precision_score(y_test, preds, zero_division=0)
        rec     = recall_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, preds_proba)
        auprc   = average_precision_score(y_test, preds_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

        final_metrics.append({
            "fold": fold,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "auprc": auprc,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "train_time": train_time,
            "predict_time": predict_time
        })

    df_final = pd.DataFrame(final_metrics)
    mean_metrics = df_final.mean().to_dict()
    mean_metrics["fold"] = "Média"
    df_final = pd.concat([df_final, pd.DataFrame([mean_metrics])], ignore_index=True)

    report_path = os.path.join(output_dir, "hdbscan_outer_final_report.csv")
    df_final.to_csv(report_path, index=False)
    print(f"\nResultados preditivos finais reportados em: {report_path}")

    volume.commit()
    return df_final.to_csv(index=False)


# ==============================================================================
# VERIFICAÇÃO DAS IMAGENS
# ==============================================================================


@app.function(image=cuml_image, gpu=GPU_TYPE, timeout=600)
def _check_cuml():
    import subprocess

    import cuml
    print("cuML:", cuml.__version__)
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    return "cuml ok"


@app.function(image=dbcv_image, cpu=DBCV_CPU, memory=DBCV_MEMORY_MIB, timeout=600)
def _check_dbcv():
    import numpy as np
    import dbcv

    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(50, 3), rng.randn(50, 3) + 8.0])
    y = np.array([0] * 50 + [1] * 50)
    print("DBCV de referência:", dbcv.dbcv(X, y, noise_id=-1, n_processes=2))
    return "dbcv ok"


@app.local_entrypoint()
def smoke():
    """Constrói as duas imagens e confere GPU e DBCV antes da execução real."""
    print(_check_cuml.remote())
    print(_check_dbcv.remote())


# ==============================================================================
# PIPELINE
# ==============================================================================


@app.local_entrypoint()
def main():
    print("Iniciando pipeline...")
    print(f"Processos alocados para o DBCV (por contêiner): {DBCV_N_PROCESSES}")

    folds = discover_folds.remote()

    print("=== ETAPA 1: Grid Search Interno (Treino + Tracking na Validação) ===")

    # Um contêiner com GPU por fold, em paralelo.
    records = []
    for fold_records in gpu_fold_stage.map(folds):
        records.extend(fold_records)

    # Um contêiner sem GPU por combinação, em paralelo.
    tasks = [{"fold": r["fold"], "combo_id": r["combo_id"]} for r in records]
    print(f"Cálculos de DBCV a distribuir: {len(tasks)}")
    dbcv_results = list(dbcv_stage.map(tasks))

    best_params_per_fold = aggregate_stage.remote(records, dbcv_results, folds)

    print("\nMelhores parâmetros encontrados por fold:")
    import json
    print(json.dumps(best_params_per_fold, indent=4))

    best_parameter_achieved = max(
        best_params_per_fold.values(),
        key=lambda fold_params: fold_params["best_dbcv_score"]
    )

    print("\nMelhor parâmetro obtido no ciclo interno:")
    print(json.dumps(best_parameter_achieved, indent=4))

    print("\n=== ETAPA 2: Avaliação Externa (Verificando Fraudes com Labels Reais) ===")
    final_report_csv = outer_stage.remote(best_parameter_achieved)

    print("\nRelatório Final (Outer Folds):")
    print(final_report_csv)

    print(
        "\nArtefatos gravados no Volume. Para baixá-los:\n"
        f"    modal volume get {VOLUME_NAME} /results/unsupervised results/"
    )
