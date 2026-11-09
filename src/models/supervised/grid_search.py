import itertools
import os
import threading
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

from features import EXCLUDED_FEATURES, build_preprocessor, model_features, scaled_columns
from metrics import append_fold_summary, read_legitimate_weight


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


def comparable_value(value):
    """
    Normaliza um hiperparâmetro para comparar o lido de um CSV com o da grade.

    O CSV devolve None como NaN, inteiros como float e números de colunas mistas como texto, de
    modo que 0, 0.0 e "0" precisam virar o mesmo valor.
    """
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def saved_rows_of_this_grid(path, expected_params, key):
    """
    Lê as linhas gravadas por uma execução anterior, para retomá-la.

    Objetivo:
        Confere que cada linha gravada tem os hiperparâmetros esperados para a sua chave, a
        posição na grade (combo_id) na busca em grade ou o fold no ciclo externo. Uma grade ou
        uma configuração diferente faria a retomada misturar execuções, e por isso levanta erro.
        Os hiperparâmetros das linhas devolvidas vêm de expected_params, com os tipos certos, e
        só as métricas vêm do CSV, que perde os tipos das colunas mistas.

    Parâmetros:
        path (str): CSV gravado pela execução anterior.
        expected_params (dict): {chave: hiperparâmetros esperados}.
        key (str): coluna que identifica a linha, "combo_id" ou "fold".

    Retorno:
        list[dict] com as linhas gravadas, ou lista vazia se o arquivo não existe.
    """
    if not os.path.exists(path):
        return []

    saved = pd.read_csv(path)
    saved = saved[saved[key].astype(str) != "Média"]
    rows = []
    for _, row in saved.iterrows():
        identifier = int(row[key])
        params = expected_params.get(identifier)
        if params is None or any(
            comparable_value(row[name]) != comparable_value(value) for name, value in params.items()
        ):
            raise ValueError(
                f"{path} tem resultados de outra grade ou configuração em {key}={identifier}. "
                "Arquive o arquivo ou rode sem resume."
            )
        rows.append(
            {
                name: identifier if name == key else params.get(name, row[name])
                for name in saved.columns
            }
        )
    return rows


def build_model(model_select, params, legitimate_weight=1.0, n_jobs=1):
    """
    Instancia o RandomForestClassifier ("rf") ou o XGBClassifier ("xgb") com os hiperparâmetros
    dados e random_state fixo em 42.

    O XGBoost recebe scale_pos_weight igual a 1 / legitimate_weight, o mesmo peso das legítimas
    que o HDBSCAN* aprimorado usa no limiar de alerta (read_legitimate_weight). Pesar as fraudes
    por esse inverso equivale a pesar cada legítima pelo número de legítimas do treino original
    que ela representa, de modo que o treino volta à proporção de classes de antes da
    subamostragem, a mesma da validação e do teste. Na base completa, o peso é 1 e o treino não
    é reponderado. Nos folds internos subamostrados, essa escolha deu F1 médio de 0,881 na
    validação, contra 0,836 com scale_pos_weight igual à razão entre legítimas e fraudes do
    treino subamostrado, que superestima a probabilidade de fraude no limiar de 0,5.
    """
    if model_select == "rf":
        return RandomForestClassifier(**params, random_state=42, n_jobs=n_jobs)

    if model_select == "xgb":
        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=1.0 / legitimate_weight,
            n_jobs=n_jobs,
        )

    raise ValueError(f"Modelo '{model_select}' não reconhecido! Escolha entre 'rf' e 'xgb'.")


def training_weights(model_select, params, y_train, legitimate_weight):
    """
    Peso de cada transação do treino no fit do modelo.

    Objetivo:
        Só o Random Forest sem class_weight é reponderado: cada legítima pesa legitimate_weight,
        o número de legítimas do treino original que ela representa (read_legitimate_weight),
        e cada fraude pesa 1. O treino volta à proporção de classes de antes da subamostragem,
        a mesma da validação e do teste, como o scale_pos_weight faz no XGBoost (build_model) e
        o peso das legítimas faz no limiar de alerta do HDBSCAN* aprimorado. Com
        class_weight="balanced_subsample", as classes já são igualadas em cada amostra, e o
        peso das legítimas desfaria esse equilíbrio, já que o scikit-learn multiplica os dois.
        Na base completa, o peso é 1 e nada muda.

    Parâmetros:
        model_select (str): "rf" ou "xgb".
        params (dict): hiperparâmetros do modelo, já tratados por clean_params.
        y_train (pd.Series): rótulos do treino, 0 para legítima e 1 para fraude.
        legitimate_weight (float): peso de cada legítima, de read_legitimate_weight.

    Retorno:
        np.ndarray com o peso de cada transação, na ordem de y_train, ou None quando o modelo
        não é reponderado.

    Exemplo de uso:
        training_weights("rf", {"class_weight": None}, pd.Series([0, 0, 1]), 2.84)
        # array([2.84, 2.84, 1.  ])
    """
    if model_select == "rf" and params.get("class_weight") is None:
        return np.where(y_train == 0, legitimate_weight, 1.0)
    return None


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
    resume=False,
    excluded_features=EXCLUDED_FEATURES,
):
    """
    Realiza um grid search manual usando folds internos e retorna métricas por fold e médias por combinação.

    Objetivo:
        Avaliar todas as combinações de hiperparâmetros do RandomForestClassifier
        usando folds internos (train/val), calcular métricas de avaliação e
        identificar os melhores parâmetros segundo uma métrica escolhida.

        As variáveis de entrada são as de model_features, padronizadas por build_preprocessor
        ajustado só no treino de cada fold, como no HDBSCAN* aprimorado: V1 a V28 e log_amount,
        sem as variáveis de excluded_features, que por padrão retira a hora do dia. O valor
        bruto de Time não entra nos modelos.

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
        resume (bool): Se True, retoma o results_by_fold.csv gravado em output_dir e pula as
            combinações já avaliadas. Se False, começa do zero. O arquivo é regravado ao fim de
            cada combinação, de modo que uma execução interrompida perde só as combinações em
            andamento.
        excluded_features (Sequence[str]): variáveis de engineer_features que não entram no
            modelo. O padrão é EXCLUDED_FEATURES, o mesmo do HDBSCAN* aprimorado.

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
    if not folds:
        raise ValueError(f"Nenhum fold_k_train.csv em {folds_dir}")
    print(f"Folds encontrados: {folds}")

    # --- 2) criar todas as combinações de hiperparâmetros ---
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    df_param_combinations = pd.DataFrame(
        all_combinations, columns=param_names
    )  # primeiro montou a combinação de grid depois caminhou sobre ela
    print(f"Total de combinações: {len(df_param_combinations)}")

    # --- 2.1) retomada: combinações gravadas por uma execução anterior ---
    folds_path = os.path.join(output_dir, "results_by_fold.csv")
    grid_params = {
        combo_id: clean_params(combo.to_dict())
        for combo_id, combo in df_param_combinations.iterrows()
    }
    if resume:
        saved_rows = saved_rows_of_this_grid(folds_path, grid_params, "combo_id")
    else:
        saved_rows = []
        if os.path.exists(folds_path):
            os.remove(folds_path)
    completed = {row["combo_id"] for row in saved_rows}
    pending = df_param_combinations.drop(index=list(completed))
    print(f"Combinações já gravadas: {len(completed)} | pendentes: {len(pending)}")
    save_lock = threading.Lock()

    def save_combo(metrics_fold):
        with save_lock:
            saved_rows.extend(metrics_fold)
            pd.DataFrame(saved_rows).sort_values(["combo_id", "fold"]).to_csv(
                folds_path, index=False
            )

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

            preprocessor = build_preprocessor(scaled_columns(excluded_features))

            X_train = preprocessor.fit_transform(
                model_features(df_train.drop(columns=[target]), excluded_features)
            )
            y_train = df_train[target]
            X_val = preprocessor.transform(
                model_features(df_val.drop(columns=[target]), excluded_features)
            )
            y_val = df_val[target]

            # Libera memória do dataframe bruto, já que separamos X e y
            del df_train
            del df_val

            params_dict = clean_params(combo.to_dict())

            # ---------------------------------------------------------------------------------

            weight = read_legitimate_weight(
                os.path.join(folds_dir, f"fold_{fold}_internal_info.json")
            )
            model = build_model(model_select, params_dict, weight)

            print(f"Evaluando combo_id {combo_id} no fold {fold}...")

            # ---------------------------------------------------------------------------------

            # treino
            t0 = time.time()
            model.fit(
                X_train,
                y_train,
                sample_weight=training_weights(model_select, params_dict, y_train, weight),
            )
            train_time_min = (time.time() - t0) / 60

            # teste
            t1 = time.time()
            preds = model.predict(X_val)
            test_time_min = (time.time() - t1) / 60
            preds_proba = model.predict_proba(X_val)[:, 1]

            metrics_fold.append(
                {
                    "combo_id": combo_id,
                    "fold": fold,
                    **params_dict,
                    **classification_metrics(y_val, preds, preds_proba),
                    "train_time_min": train_time_min,
                    "test_time_min": test_time_min,
                }
            )

            # Limpeza final de memória do loop
            del X_train, y_train, X_val, y_val, model

        save_combo(metrics_fold)
        mean_auprc = np.mean([m["auprc"] for m in metrics_fold])
        mean_f1 = np.mean([m["f1"] for m in metrics_fold])
        print(f"Combo {combo_id} | AUPRC={mean_auprc:.4f} | F1={mean_f1:.4f}", flush=True)

    # --- 3) executar combinações ---
    if parallel:
        Parallel(n_jobs=n_jobs, verbose=10, backend="threading")(
            delayed(evaluate_combo)(combo_id, combo, model_select)
            for combo_id, combo in pending.iterrows()
        )
    else:
        for combo_id, combo in pending.iterrows():
            evaluate_combo(combo_id, combo, model_select)

    print("Grid search concluído.")

    # métricas por fold, as gravadas antes e as desta execução, e as médias por combinação
    df_folds = pd.DataFrame(saved_rows).sort_values(["combo_id", "fold"]).reset_index(drop=True)
    summary_columns = ["f1", "precision", "recall", "auprc", "train_time_min", "test_time_min"]
    df_summary = pd.DataFrame(
        [
            {
                "combo_id": combo_id,
                **grid_params[combo_id],
                **combo_rows[summary_columns].mean().to_dict(),
            }
            for combo_id, combo_rows in df_folds.groupby("combo_id")
        ]
    )

    # --- salvar CSVs ---
    df_folds.to_csv(folds_path, index=False)
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
        na Seção 4.3.2 da monografia.

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
    resume=False,
    excluded_features=EXCLUDED_FEATURES,
):
    """
    Executa o ciclo externo da validação cruzada aninhada para o Random Forest ou o XGBoost.

    Objetivo:
        Para cada fold externo (fold_k_train.csv e fold_k_test.csv em outer_folds_dir), treina o
        modelo sobre todo o treino externo com a configuração best_parameter_achieved e o avalia
        no teste externo. Todas as rodadas externas usam a mesma configuração, a de maior
        métrica entre as melhores de cada ciclo interno, conforme o critério de seleção descrito
        na Seção 4.3.2 da monografia e o mesmo do HDBSCAN* aprimorado. As variáveis de entrada
        vêm de model_features, sem as de excluded_features, padronizadas por build_preprocessor
        ajustado só no treino externo, e as métricas de classification_metrics, as mesmas do
        ciclo interno e do HDBSCAN* aprimorado.

        O relatório traz uma linha por fold e duas linhas finais, "Média" e "Desvio padrão",
        de append_fold_summary, e é gravado em <model_select>_outer_final_report.csv.

    Parâmetros:
        outer_folds_dir (str): pasta com os arquivos fold_k_train.csv e fold_k_test.csv.
        best_parameter_achieved (dict): hiperparâmetros avaliados, como o "params" de uma das
            entradas de best_params_by_fold.
        target (str): nome da coluna alvo.
        model_select (str): "rf" para RandomForest, "xgb" para XGBoost.
        output_dir (str): pasta do relatório final.
        n_jobs (int): número de núcleos usados no treino de cada modelo (-1 = todos).
        resume (bool): se True, retoma o relatório gravado em output_dir e pula os folds já
            avaliados com a mesma configuração. O relatório é regravado ao fim de cada fold.
        excluded_features (Sequence[str]): variáveis de engineer_features que não entram no
            modelo, as mesmas do ciclo interno.

    Retorno:
        pd.DataFrame com os hiperparâmetros, precisão, recall, F1, AUPRC, TP, FP, FN e tempos de
        treino e de predição, em minutos, de cada fold, seguido das linhas "Média" e
        "Desvio padrão".

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
    report_path = os.path.join(output_dir, f"{model_select}_outer_final_report.csv")

    print(f"\n--- Iniciando Avaliação Externa (Outer Folds) do modelo {model_select} ---")

    folds = sorted(
        {int(f.split("_")[1]) for f in os.listdir(outer_folds_dir) if f.endswith("_train.csv")}
    )

    params = best_parameter_achieved

    final_metrics = (
        saved_rows_of_this_grid(report_path, {fold: params for fold in folds}, "fold")
        if resume
        else []
    )
    completed = {row["fold"] for row in final_metrics}

    for fold in folds:
        if fold in completed:
            print(f"Fold Externo {fold} já gravado, pulado")
            continue
        print(f"Avaliando Fold Externo {fold} | {params}")

        df_train = pd.read_csv(os.path.join(outer_folds_dir, f"fold_{fold}_train.csv"))
        df_test = pd.read_csv(os.path.join(outer_folds_dir, f"fold_{fold}_test.csv"))

        preprocessor = build_preprocessor(scaled_columns(excluded_features))

        X_train = preprocessor.fit_transform(
            model_features(df_train.drop(columns=[target]), excluded_features)
        )
        y_train = df_train[target]
        X_test = preprocessor.transform(
            model_features(df_test.drop(columns=[target]), excluded_features)
        )
        y_test = df_test[target]

        del df_train, df_test

        weight = read_legitimate_weight(os.path.join(outer_folds_dir, f"fold_{fold}_info.json"))
        model = build_model(model_select, params, weight, n_jobs=n_jobs)

        t0 = time.time()
        model.fit(
            X_train,
            y_train,
            sample_weight=training_weights(model_select, params, y_train, weight),
        )
        train_time_min = (time.time() - t0) / 60

        t1 = time.time()
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]
        predict_time_min = (time.time() - t1) / 60

        final_metrics.append(
            {
                "fold": fold,
                **params,
                **classification_metrics(y_test, preds, preds_proba),
                "train_time_min": train_time_min,
                "predict_time_min": predict_time_min,
            }
        )

        del X_train, y_train, X_test, y_test, model
        pd.DataFrame(final_metrics).sort_values("fold").to_csv(report_path, index=False)

    df_final = pd.DataFrame(final_metrics).sort_values("fold").reset_index(drop=True)
    # O resumo cobre só as métricas e os tempos, já que os hiperparâmetros podem variar por fold
    summary_columns = [
        "precision",
        "recall",
        "f1",
        "auprc",
        "TP",
        "FP",
        "FN",
        "train_time_min",
        "predict_time_min",
    ]
    df_final = append_fold_summary(df_final, summary_columns)

    df_final.to_csv(report_path, index=False)
    print(f"\nResultados preditivos finais reportados em: {report_path}")

    return df_final
