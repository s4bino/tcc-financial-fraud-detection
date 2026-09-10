!pip install "git+https://github.com/FelSiq/DBCV"
!pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12

import os
import time
import json
import itertools

import numpy as np
import pandas as pd
from cuml.cluster import HDBSCAN
from cuml.cluster.hdbscan import approximate_predict
import dbcv

from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.model_selection import train_test_split

def hdbscan_internal_grid_with_tracking(
    internal_folds_dir,
    param_grid,
    target="Class",
    output_dir="results/unsupervised",
    n_processes=None,
    max_dbcv_sample_size=20000
):
    """
    Treina o modelo na base de treino usando Grid Search clássico e otimiza pelo DBCV.
    Testa os clusters na base de validação usando approximate_predict para rastrear métricas reais.
    A grid deve sempre conter 'min_cluster_size' e 'min_samples'.
    """
    os.makedirs(output_dir, exist_ok=True)

    folds = sorted({
        int(f.split("_")[1])
        for f in os.listdir(internal_folds_dir)
        if f.endswith("_train.csv")
    })
    print(f"Folds encontrados para Avaliação (HDBSCAN): {folds}")


    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    df_param_combinations = pd.DataFrame(all_combinations, columns=param_names)

    # df_param_combinations = pd.DataFrame(param_grid)
    print(f"Total de combinações HDBSCAN por fold: {len(df_param_combinations)}")

    best_params_per_fold = {}
    all_trials_history = []

    for fold in folds:
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
            # ==============================================================
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

            all_trials_history.append({
                "fold": fold,
                "combo_id": combo_id,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "dbcv_score": dbcv_score,
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

            print(f"Combo {combo_id} | min_cluster={min_cluster_size} | min_samples={min_samples} | DBCV={dbcv_score:.4f} | F1 Val={f1:.4f}")

        del X_train, y_train, X_val, y_val

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

    return df_folds, df_summary, best_params_per_fold

def hdbscan_outer_evaluation(
    outer_folds_dir,
    best_parameter_achieved,
    target="Class",
    output_dir="results/unsupervised"
):
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

    return df_final


if __name__ == "__main__":
    INTERNAL_FOLDS_DIR = "/content/drive/MyDrive/TCC - DADOS CV/internal_folds"
    OUTER_FOLDS_DIR = "/content/drive/MyDrive/TCC - DADOS CV/folds_output"

    RESULTS_DIR = "/content/drive/MyDrive/RESULTS"

    param_grid_hdbscan = {
        'min_cluster_size': [5, 25, 50],
        'min_samples': [280, 290, 300, 315, 325, 350, 400]
    }

    n_cores_disponiveis = os.cpu_count() or 1
    print(f"Iniciando pipeline... Processos alocados para o DBCV: {n_cores_disponiveis}")

    print("=== ETAPA 1: Grid Search Interno (Treino + Tracking na Validação) ===")
    df_folds, df_summary, best_params_per_fold = hdbscan_internal_grid_with_tracking(
        internal_folds_dir=INTERNAL_FOLDS_DIR,
        param_grid=param_grid_hdbscan,
        target="Class",
        output_dir=f"{RESULTS_DIR}/grid_results_hdbscan",
        n_processes=n_cores_disponiveis,
        max_dbcv_sample_size=28000
    )

    print("\nMelhores parâmetros encontrados por fold:")
    print(json.dumps(best_params_per_fold, indent=4))

    best_parameter_achieved = max(
        best_params_per_fold.values(),
        key=lambda fold_params: fold_params["best_dbcv_score"]
    )

    print("\nMelhor parâmetro obtido no ciclo interno:")
    print(json.dumps(best_parameter_achieved, indent=4))

    print("\n=== ETAPA 2: Avaliação Externa (Verificando Fraudes com Labels Reais) ===")
    df_final_report = hdbscan_outer_evaluation(
        outer_folds_dir=OUTER_FOLDS_DIR,
        best_parameter_achieved=best_parameter_achieved,
        target="Class",
        output_dir=f"{RESULTS_DIR}/final_reports_hdbscan"
    )

    print("\nRelatório Final (Outer Folds):")
    print(df_final_report)