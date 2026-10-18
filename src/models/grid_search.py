import os
import time
import itertools

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from joblib import Parallel, delayed


def manual_grid_search(
    folds_dir,
    param_grid,
    target="Class",
    metric="f1",
    output_dir="results/supervised",
    parallel=False,
    n_jobs=-1,
    model_select: str = "rf",
):
    """
    Realiza um grid search manual usando folds internos e retorna métricas por fold e médias por combinação.

    Objetivo:
        Avaliar todas as combinações de hiperparâmetros do RandomForestClassifier
        usando folds internos (train/val), calcular métricas de avaliação e
        identificar os melhores parâmetros segundo uma métrica escolhida.

    Parâmetros:
        folds_dir (str): Pasta contendo os arquivos de fold: fold_1_train.csv, fold_1_val.csv, etc.
        param_grid (dict): Dicionário com listas de valores de hiperparâmetros.
        target (str): Nome da coluna alvo.
        metric (str): Métrica usada para escolher o melhor modelo ("f1", "auc", "precision", "recall").
        output_dir (str): Diretório onde os CSVs de resultados serão salvos.
        parallel (bool): Se True, executa combinações em paralelo.
        n_jobs (int): Número de jobs para paralelização (-1 = todos).
        model_select (str): Modelo a ser usado: "rf" para RandomForest, "xgb" para XGBoost.

    Retorno:
        df_folds (pd.DataFrame): Métricas detalhadas por fold e combinação.
        df_summary (pd.DataFrame): Métricas médias por combinação de hiperparâmetros.
        best_params (dict): Dicionário com os melhores parâmetros segundo a métrica escolhida.

    Exemplo de uso:
        param_grid = {
            'n_estimators': [100, 200],
            'max_features': ['sqrt', 0.5],
            'min_samples_leaf': [1, 2]
        }

        df_folds, df_summary, best_params = manual_grid_search(
            folds_dir="data/processed/inner_folds",
            param_grid=param_grid,
            target="target",
            metric="f1",
            output_dir="results/supervised",
            parallel=True,
            n_jobs=-1
        )

        print("Melhores parâmetros:", best_params)
    """

    os.makedirs(output_dir, exist_ok=True)

    # --- 1) encontrar todos os folds ---
    folds = sorted(
        {int(f.split("_")[1]) for f in os.listdir(folds_dir) if f.endswith("_train.csv")}
    )
    print(f"Folds encontrados: {folds}")

    # --- 2) criar todas as combinações de hiperparâmetros ---
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    df_param_combinations = pd.DataFrame(
        all_combinations, columns=param_names
    )  # primeiro montou a combinação de grid depois caminhou sobre ela
    print(f"Total de combinações: {len(df_param_combinations)}")

    # --- função que avalia uma combinação ---
    def evaluate_combo(combo_id, combo, model_select):
        metrics_fold = []
        model = None

        for fold in folds:
            # 1. Carregamento de dados
            train_path = os.path.join(folds_dir, f"fold_{fold}_train.csv")
            val_path = os.path.join(folds_dir, f"fold_{fold}_val.csv")

            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)

            X_train = df_train.drop(columns=[target])
            y_train = df_train[target]
            X_val = df_val.drop(columns=[target])
            y_val = df_val[target]

            # Libera memória do dataframe bruto, já que separamos X e y
            del df_train
            del df_val

            params_dict = combo.to_dict()

            if "n_estimators" in params_dict:
                params_dict["n_estimators"] = int(params_dict["n_estimators"])

            if "max_depth" in params_dict:
                params_dict["max_depth"] = int(params_dict["max_depth"])

            if "max_samples" in params_dict and pd.isna(params_dict["max_samples"]):
                params_dict["max_samples"] = None

            # ---------------------------------------------------------------------------------

            if model_select == "rf":
                model = RandomForestClassifier(**params_dict, random_state=42, n_jobs=1)
            elif model_select == "xgb":
                scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()
                model = XGBClassifier(
                    **params_dict,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    scale_pos_weight=scale_weight,
                    n_jobs=1,
                )

            print(f"Evaluando combo_id {combo_id} no fold {fold}...")

            # ---------------------------------------------------------------------------------

            # treino
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            # teste
            t1 = time.time()
            preds = model.predict(X_val)
            test_time = time.time() - t1
            preds_proba = model.predict_proba(X_val)[:, 1]

            # métricas
            auc = roc_auc_score(y_val, preds_proba)
            f1 = f1_score(y_val, preds)
            prec = precision_score(y_val, preds)
            rec = recall_score(y_val, preds)

            metrics_fold.append(
                {
                    "combo_id": combo_id,
                    "fold": fold,
                    **params_dict,
                    "auc": auc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "train_time": train_time,
                    "test_time": test_time,
                }
            )

            # Limpeza final de memória do loop
            del X_train, y_train, X_val, y_val, model

        # médias por combinação
        metrics_summary = {
            "combo_id": combo_id,
            **params_dict,
            "auc": np.mean([m["auc"] for m in metrics_fold]),
            "f1": np.mean([m["f1"] for m in metrics_fold]),
            "precision": np.mean([m["precision"] for m in metrics_fold]),
            "recall": np.mean([m["recall"] for m in metrics_fold]),
            "train_time": np.mean([m["train_time"] for m in metrics_fold]),
            "test_time": np.mean([m["test_time"] for m in metrics_fold]),
        }

        return metrics_fold, metrics_summary

    # --- 3) executar combinações ---
    if parallel:
        results = Parallel(n_jobs=n_jobs, verbose=10, backend="threading")(
            delayed(evaluate_combo)(combo_id, combo, model_select)
            for combo_id, combo in df_param_combinations.iterrows()
        )
    else:
        results = [
            evaluate_combo(combo_id, combo, model_select)
            for combo_id, combo in df_param_combinations.iterrows()
        ]

    print("Grid search concluído.")

    # separar fold-wise e summary
    rows_folds = [m for combo_metrics, _ in results for m in combo_metrics]
    rows_summary = [summary for _, summary in results]

    df_folds = pd.DataFrame(rows_folds)
    df_summary = pd.DataFrame(rows_summary)

    # --- salvar CSVs ---
    df_folds.to_csv(os.path.join(output_dir, "results_by_fold.csv"), index=False)
    df_summary.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    print("\nResultados salvos em:")
    print(os.path.join(output_dir, "results_by_fold.csv"))
    print(os.path.join(output_dir, "results_summary.csv"))

    # --- 4) melhor combinação segundo a métrica ---
    if metric not in df_summary.columns:
        raise ValueError(
            f"Métrica '{metric}' não encontrada! Escolha entre: {list(df_summary.columns)}"
        )

    best_row = df_summary.sort_values(metric, ascending=False).iloc[0]
    best_params = {k: best_row[k] for k in param_names}

    return df_folds, df_summary, best_params
