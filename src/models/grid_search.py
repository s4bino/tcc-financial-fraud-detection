import itertools
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from features import build_preprocessor, engineer_features


def clean_params(params):
    """
    Converte hiperparâmetros vindos de um DataFrame para os tipos esperados pelos modelos.

    As colunas do pandas transformam None em NaN e inteiros em float quando misturam tipos.
    NaN volta a None, e n_estimators e max_depth voltam a int.
    """
    params = {
        name: None if pd.api.types.is_scalar(value) and pd.isna(value) else value
        for name, value in params.items()
    }

    for name in ("n_estimators", "max_depth"):
        if params.get(name) is not None:
            params[name] = int(params[name])

    return params


def build_model(model_select, params, y_train, n_jobs=1):
    """
    Instancia o RandomForestClassifier ("rf") ou o XGBClassifier ("xgb") com os hiperparâmetros
    dados e random_state fixo em 42.

    O XGBoost recebe scale_pos_weight igual à razão entre transações legítimas e fraudes de
    y_train, que compensa o desbalanceamento das classes.
    """
    if model_select == "rf":
        return RandomForestClassifier(**params, random_state=42, n_jobs=n_jobs)

    if model_select == "xgb":
        scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_weight,
            n_jobs=n_jobs,
        )

    raise ValueError(f"Modelo '{model_select}' não reconhecido! Escolha entre 'rf' e 'xgb'.")


def classification_metrics(y_true, preds, preds_proba):
    """
    Métricas registradas para os modelos, as mesmas do HDBSCAN* aprimorado: precisão, recall,
    F1, TP, FP e FN das classes preditas e AUPRC (precisão média) da probabilidade de fraude.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    return {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "auprc": average_precision_score(y_true, preds_proba),
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def manual_grid_search(
    folds_dir,
    param_grid,
    target="Class",
    metric="auprc",
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

        As variáveis de entrada são as de engineer_features, padronizadas por build_preprocessor
        ajustado só no treino de cada fold, como no HDBSCAN* aprimorado: V1 a V28, log_amount e
        a hora do dia codificada em seno e cosseno. O valor bruto de Time não entra nos modelos.

        As métricas são as mesmas registradas pelo HDBSCAN* aprimorado: a AUPRC (precisão
        média) da probabilidade de fraude, que compara os modelos, e precisão, recall, F1, TP,
        FP e FN das classes preditas por model.predict.

    Parâmetros:
        folds_dir (str): Pasta contendo os arquivos de fold: fold_1_train.csv, fold_1_val.csv, etc.
        param_grid (dict): Dicionário com listas de valores de hiperparâmetros.
        target (str): Nome da coluna alvo.
        metric (str): Métrica usada para escolher o melhor modelo ("auprc", "f1", "precision", "recall").
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
            metric="auprc",
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

            preprocessor = build_preprocessor()

            X_train = preprocessor.fit_transform(engineer_features(df_train.drop(columns=[target])))
            y_train = df_train[target]
            X_val = preprocessor.transform(engineer_features(df_val.drop(columns=[target])))
            y_val = df_val[target]

            # Libera memória do dataframe bruto, já que separamos X e y
            del df_train
            del df_val

            params_dict = clean_params(combo.to_dict())

            # ---------------------------------------------------------------------------------

            model = build_model(model_select, params_dict, y_train)

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

            metrics_fold.append(
                {
                    "combo_id": combo_id,
                    "fold": fold,
                    **params_dict,
                    **classification_metrics(y_val, preds, preds_proba),
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
            "f1": np.mean([m["f1"] for m in metrics_fold]),
            "precision": np.mean([m["precision"] for m in metrics_fold]),
            "recall": np.mean([m["recall"] for m in metrics_fold]),
            "auprc": np.mean([m["auprc"] for m in metrics_fold]),
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


def best_params_by_fold(df_folds, param_grid, metric="auprc"):
    """
    Escolhe, em cada fold interno, a combinação de hiperparâmetros de maior métrica.

    Objetivo:
        É a primeira etapa da seleção, a mesma do HDBSCAN* aprimorado com o DBCV: cada ciclo
        interno indica a sua melhor combinação, e a de maior métrica entre elas é aplicada a
        todas as rodadas externas por supervised_outer_evaluation, conforme o critério descrito
        na Seção 3.1.3.4 da monografia.

    Parâmetros:
        df_folds (pd.DataFrame): métricas por fold e combinação, retornadas por manual_grid_search.
        param_grid (dict): grade usada na busca, da qual vêm os nomes dos hiperparâmetros.
        metric (str): métrica usada na escolha.

    Retorno:
        dict {fold: {"params": hiperparâmetros, "best_score": valor da métrica}}, com os tipos
        dos hiperparâmetros ajustados por clean_params.

    Exemplo de uso:
        best_params_per_fold = best_params_by_fold(df_folds, param_grid, metric="auprc")
        best_parameter_achieved = max(
            best_params_per_fold.values(), key=lambda fold_best: fold_best["best_score"]
        )["params"]
    """
    best_params_per_fold = {}

    for fold, fold_rows in df_folds.groupby("fold"):
        best_index = fold_rows[metric].idxmax()
        best_row = fold_rows.loc[[best_index], list(param_grid)]
        best_params_per_fold[int(fold)] = {
            "params": clean_params(best_row.to_dict("records")[0]),
            "best_score": float(fold_rows.loc[best_index, metric]),
        }

    return best_params_per_fold


def supervised_outer_evaluation(
    outer_folds_dir,
    best_parameter_achieved,
    target="Class",
    model_select="rf",
    output_dir="results/supervised",
    n_jobs=-1,
):
    """
    Executa o ciclo externo da validação cruzada aninhada para o Random Forest ou o XGBoost.

    Objetivo:
        Para cada fold externo (fold_k_train.csv e fold_k_test.csv em outer_folds_dir), treina o
        modelo sobre todo o treino externo com a configuração best_parameter_achieved e o avalia
        no teste externo. Todas as rodadas externas usam a mesma configuração, a de maior
        métrica entre as melhores de cada ciclo interno, conforme o critério de seleção descrito
        na Seção 3.1.3.4 da monografia e o mesmo do HDBSCAN* aprimorado. As variáveis de entrada
        vêm de engineer_features, padronizadas por build_preprocessor ajustado só no treino
        externo, e as métricas de classification_metrics, as mesmas do ciclo interno e do
        HDBSCAN* aprimorado.

        O relatório traz uma linha por fold e uma linha final, "Média", com a média aritmética
        das métricas entre os folds, e é gravado em <model_select>_outer_final_report.csv.

    Parâmetros:
        outer_folds_dir (str): pasta com os arquivos fold_k_train.csv e fold_k_test.csv.
        best_parameter_achieved (dict): hiperparâmetros avaliados, como o "params" de uma das
            entradas de best_params_by_fold.
        target (str): nome da coluna alvo.
        model_select (str): "rf" para RandomForest, "xgb" para XGBoost.
        output_dir (str): pasta do relatório final.
        n_jobs (int): número de núcleos usados no treino de cada modelo (-1 = todos).

    Retorno:
        pd.DataFrame com os hiperparâmetros, precisão, recall, F1, AUPRC, TP, FP, FN e tempos de
        treino e de predição de cada fold, seguido da linha "Média".

    Exemplo de uso:
        df_folds, _, _ = manual_grid_search(
            folds_dir="data/processed/inner_folds", param_grid=param_grid, model_select="xgb"
        )

        best_params_per_fold = best_params_by_fold(df_folds, param_grid)
        best_parameter_achieved = max(
            best_params_per_fold.values(), key=lambda fold_best: fold_best["best_score"]
        )["params"]

        report = supervised_outer_evaluation(
            outer_folds_dir="data/processed/outer_folds",
            best_parameter_achieved=best_parameter_achieved,
            model_select="xgb",
        )

        print(report[report["fold"] == "Média"][["precision", "recall", "f1", "auprc"]])
    """
    os.makedirs(output_dir, exist_ok=True)
    final_metrics = []

    print(f"\n--- Iniciando Avaliação Externa (Outer Folds) do modelo {model_select} ---")

    folds = sorted(
        {int(f.split("_")[1]) for f in os.listdir(outer_folds_dir) if f.endswith("_train.csv")}
    )

    params = best_parameter_achieved

    for fold in folds:
        print(f"Avaliando Fold Externo {fold} | {params}")

        df_train = pd.read_csv(os.path.join(outer_folds_dir, f"fold_{fold}_train.csv"))
        df_test = pd.read_csv(os.path.join(outer_folds_dir, f"fold_{fold}_test.csv"))

        preprocessor = build_preprocessor()

        X_train = preprocessor.fit_transform(engineer_features(df_train.drop(columns=[target])))
        y_train = df_train[target]
        X_test = preprocessor.transform(engineer_features(df_test.drop(columns=[target])))
        y_test = df_test[target]

        del df_train, df_test

        model = build_model(model_select, params, y_train, n_jobs=n_jobs)

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t1 = time.time()
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]
        predict_time = time.time() - t1

        final_metrics.append(
            {
                "fold": fold,
                **params,
                **classification_metrics(y_test, preds, preds_proba),
                "train_time": train_time,
                "predict_time": predict_time,
            }
        )

        del X_train, y_train, X_test, y_test, model

    df_final = pd.DataFrame(final_metrics)
    # A média cobre só as métricas e os tempos, já que os hiperparâmetros podem variar por fold
    mean_columns = [
        "precision",
        "recall",
        "f1",
        "auprc",
        "TP",
        "FP",
        "FN",
        "train_time",
        "predict_time",
    ]
    mean_metrics = df_final[mean_columns].mean().to_dict()
    mean_metrics["fold"] = "Média"
    df_final = pd.concat([df_final, pd.DataFrame([mean_metrics])], ignore_index=True)

    report_path = os.path.join(output_dir, f"{model_select}_outer_final_report.csv")
    df_final.to_csv(report_path, index=False)
    print(f"\nResultados preditivos finais reportados em: {report_path}")

    return df_final
