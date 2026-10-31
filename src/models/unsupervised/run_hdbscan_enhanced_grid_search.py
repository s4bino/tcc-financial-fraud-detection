import itertools
import json
import os
import sys
import time
from pathlib import Path

import dbcv
import numpy as np
import pandas as pd
from cuml.cluster import HDBSCAN
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# features.py, metrics.py e paths.py ficam em src/models/, um nível acima deste diretório. No
# Colab, os quatro arquivos sobem juntos para /content, e o diretório do próprio script já os
# contém.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import paths  # noqa: E402
from features import build_preprocessor, engineer_features  # noqa: E402
from metrics import (  # noqa: E402
    alert_threshold,
    glosh_anomaly_score,
    glosh_neighbors,
    read_legitimate_weight,
    to_numpy,
)


def hdbscan_enhanced_internal_grid_with_tracking(
    internal_folds_dir,
    param_grid,
    target="Class",
    output_dir="results/unsupervised",
    n_processes=None,
    max_dbcv_sample_size=20000,
    alert_rate=0.01,
    resume=False,
    only_folds=None,
):
    """
    Executa o ciclo interno da validação cruzada aninhada do HDBSCAN* aprimorado: grid search
    selecionado pelo DBCV, com acompanhamento das métricas de detecção na validação.

    Objetivo:
        Para cada fold interno (fold_k_train.csv e fold_k_val.csv em internal_folds_dir), avalia
        todas as combinações do produto cartesiano de param_grid, que deve conter
        min_cluster_size e min_samples. Em cada fold:

        1. Remove a coluna alvo, elimina as transações duplicadas apenas do treino, deriva as
           variáveis com engineer_features e as padroniza com build_preprocessor ajustado só no
           treino.
        2. Busca uma única vez, com glosh_neighbors, os vizinhos mais próximos no treino para o
           maior min_samples da grade, reaproveitados por todas as combinações.
        3. Para cada combinação pendente, treina o HDBSCAN* do cuML (CAMPELLO; MOULAVI; SANDER,
           2013) e calcula sobre o treino o DBCV (MOULAVI et al., 2014),
           índice interno de validação de agrupamentos por densidade, em [-1, 1], que não usa
           rótulos. Por ter custo quadrático, o DBCV é calculado sobre os pontos distintos e,
           acima de max_dbcv_sample_size, sobre uma amostra estratificada pelos rótulos dos
           clusters (random_state=42), ou aleatória simples quando a estratificação não é
           possível. Sem cluster válido ou em caso de erro, o DBCV recebe -1.
        4. Calcula com glosh_anomaly_score o GLOSH (CAMPELLO et al., 2015) do treino, a partir
           da árvore condensada do modelo, e o da validação, pela inserção de cada transação na
           hierarquia do treino. O limiar é o quantil 1 - alert_rate do GLOSH do treino,
           calculado por alert_threshold, que na base subamostrada pondera as legítimas pelo
           peso de read_legitimate_weight, lido do fold_k_internal_info.json. As transações da
           validação com escore igual ou superior a ele, os outliers de maior escore, são
           sinalizadas como fraude. Registram-se precisão, recall, F1, AUPRC do GLOSH, TP, FP,
           FN e os tempos de treino e de validação, em minutos.

        Os rótulos da validação servem apenas ao acompanhamento: a escolha da configuração de cada
        fold usa somente o DBCV, o que preserva o caráter não supervisionado da seleção. Após cada
        combinação, o histórico é regravado em hdbscan_enhanced_grid_results_by_fold.csv, o que
        permite retomar uma execução interrompida. Ao final, a combinação de maior DBCV de cada
        fold é gravada em hdbscan_enhanced_grid_best_summary.csv e best_params_hdbscan_enhanced.json.

    Parâmetros:
        internal_folds_dir (str): pasta com os arquivos fold_k_train.csv e fold_k_val.csv.
        param_grid (dict): listas de valores para min_cluster_size e min_samples.
        target (str): nome da coluna alvo.
        output_dir (str): pasta dos CSVs e do JSON de resultados.
        n_processes (int | None): número de processos usados no cálculo do DBCV.
        max_dbcv_sample_size (int): número máximo de pontos usados no DBCV.
        alert_rate (float): fração esperada de alertas, que define o limiar no escore do treino.
        resume (bool): se True, lê o CSV parcial e pula os pares (fold, combo_id) já avaliados.
            Como combo_id é a posição da combinação no produto cartesiano, a grade precisa ser a
            mesma da execução anterior.
        only_folds (list[int] | None): restringe a execução a estes folds internos.

    Retorno:
        df_folds (pd.DataFrame): uma linha por fold e combinação, com o DBCV e as métricas de validação.
        df_summary (pd.DataFrame): a linha de maior DBCV de cada fold.
        best_params_per_fold (dict): por fold, min_cluster_size, min_samples e best_dbcv_score.
            As chaves são inteiros no retorno e texto no JSON gravado.

    Exemplo de uso:
        param_grid = {"min_cluster_size": [50], "min_samples": [150, 300]}

        df_folds, df_summary, best_params_per_fold = hdbscan_enhanced_internal_grid_with_tracking(
            internal_folds_dir="data/processed/inner_folds",
            param_grid=param_grid,
            output_dir="results/unsupervised",
            n_processes=8,
            max_dbcv_sample_size=28000,
            alert_rate=0.001,
            resume=True
        )

        print(best_params_per_fold[1])   # configuração de maior DBCV no fold interno 1
    """
    os.makedirs(output_dir, exist_ok=True)

    folds = sorted(
        {int(f.split("_")[1]) for f in os.listdir(internal_folds_dir) if f.endswith("_train.csv")}
    )
    print(f"Folds encontrados para Avaliação (HDBSCAN aprimorado): {folds}")

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    df_param_combinations = pd.DataFrame(all_combinations, columns=param_names)

    print(f"Total de combinações HDBSCAN por fold: {len(df_param_combinations)}")

    max_min_samples = int(df_param_combinations["min_samples"].max())

    best_params_per_fold = {}
    all_trials_history = []
    completed_trials = set()

    partial_path = os.path.join(output_dir, "hdbscan_enhanced_grid_results_by_fold.csv")

    if resume and os.path.isfile(partial_path):
        all_trials_history = pd.read_csv(partial_path).to_dict("records")
        completed_trials = {(int(row["fold"]), int(row["combo_id"])) for row in all_trials_history}
        print(f"Retomando de {partial_path}")
        print(f"Combinações já concluídas, que serão puladas: {len(completed_trials)}")

    if only_folds:
        folds = [f for f in folds if f in set(only_folds)]
        print(f"Restrito aos folds: {folds}")

    for fold in folds:
        pending_combos = [
            combo_id
            for combo_id in df_param_combinations.index
            if (fold, combo_id) not in completed_trials
        ]
        if not pending_combos:
            print(f"\n--- Fold Interno {fold} já concluído, pulando ---")
            continue

        print(
            f"\n--- Iniciando Grid Search para o Fold Interno {fold} ({len(pending_combos)} combinações pendentes) ---"
        )

        train_path = os.path.join(internal_folds_dir, f"fold_{fold}_train.csv")
        val_path = os.path.join(internal_folds_dir, f"fold_{fold}_val.csv")

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)

        # credit-card-fraud-detection-ds, célula 9, linha 2; credit-card-fraud-detection, célula 10, linha 3
        X_train = engineer_features(df_train.drop(columns=[target]).drop_duplicates())
        X_val = engineer_features(df_val.drop(columns=[target]))
        y_val = df_val[target]
        # os rótulos do treino entram só nos pesos do limiar de alerta da base subamostrada
        y_train = df_train.loc[X_train.index, target]
        legitimate_weight = read_legitimate_weight(
            os.path.join(internal_folds_dir, f"fold_{fold}_internal_info.json")
        )
        print(f"Peso de cada legítima no limiar de alerta: {legitimate_weight:.4f}")
        del df_train, df_val

        # ==============================================================
        # 2. PRÉ-PROCESSAMENTO
        # ==============================================================
        preprocessor = build_preprocessor()

        X_train_scaled = preprocessor.fit_transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)

        t_knn = time.time()
        neighbors = glosh_neighbors(X_train_scaled, X_val_scaled, max_min_samples)
        print(
            f"Vizinhos do treino ({max_min_samples}) e da validação ({2 * max_min_samples}) "
            f"calculados em {(time.time() - t_knn) / 60:.2f} min"
        )

        # ==============================================================
        # GRID
        # ==============================================================
        for combo_id, combo in df_param_combinations.iterrows():
            if (fold, combo_id) in completed_trials:
                continue

            params_dict = combo.to_dict()

            min_cluster_size = int(params_dict["min_cluster_size"])
            min_samples = int(params_dict["min_samples"])

            # ==============================================================
            # 3. TREINAMENTO
            # ==============================================================
            t0 = time.time()
            model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            model.fit(X_train_scaled)
            train_time_min = (time.time() - t0) / 60

            labels = to_numpy(model.labels_)
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
                                stratify=labels_unique,
                            )
                        except ValueError:
                            rng = np.random.RandomState(42)
                            sample_idx = rng.choice(
                                len(X_mat_unique), size=max_dbcv_sample_size, replace=False
                            )
                            X_mat_unique = X_mat_unique[sample_idx]
                            labels_unique = labels_unique[sample_idx]

                    if len(set(labels_unique) - {-1}) >= 1:
                        dbcv_score = dbcv.dbcv(
                            X_mat_unique, labels_unique, noise_id=-1, n_processes=n_processes
                        )
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
            train_glosh, val_glosh = glosh_anomaly_score(
                model.condensed_tree_.to_numpy(),
                X_train_scaled,
                X_val_scaled,
                min_samples,
                neighbors=neighbors,
            )
            # credit-card-fraud-detection-ds, célula 36, linha 2; credit-card-fraud-detection, célula 55, linhas 2-5
            threshold = alert_threshold(train_glosh, alert_rate, y_train, legitimate_weight)
            val_time_min = (time.time() - t1) / 60

            # fds-et-19-anomaly-detection-models, célula 17, linha 3
            preds = (val_glosh >= threshold).astype(int)

            f1 = f1_score(y_val, preds, zero_division=0)
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            auprc = average_precision_score(y_val, val_glosh)
            tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()

            all_trials_history.append(
                {
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
                    "auprc": auprc,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "train_time_min": train_time_min,
                    "val_time_min": val_time_min,
                }
            )

            pd.DataFrame(all_trials_history).to_csv(partial_path, index=False)

            print(
                f"Combo {combo_id} [{combo_id + 1}/{len(df_param_combinations)}] | min_cluster={min_cluster_size} | min_samples={min_samples} | "
                f"clusters={n_clusters} | ruído={n_noise} ({n_noise / len(labels):.1%}) | DBCV={dbcv_score:.4f} | {(time.time() - t0) / 60:.1f} min\n"
                f"    AUPRC Val (GLOSH)={auprc:.4f} | "
                f"Precisão={prec:.3f} | Recall={rec:.3f} | F1 Val={f1:.4f} | TP={tp} FP={fp} FN={fn}"
            )

        del X_train, X_val, y_train, y_val, neighbors

        print(f"Fold interno {fold} concluído e gravado.")

    # ==============================================================
    # 6. GERAÇÃO DE RELATÓRIOS E SELEÇÃO DOS MELHORES
    # ==============================================================
    df_folds = pd.DataFrame(all_trials_history)
    df_folds.to_csv(
        os.path.join(output_dir, "hdbscan_enhanced_grid_results_by_fold.csv"), index=False
    )

    summary_data = []
    for fold in sorted(int(f) for f in df_folds["fold"].unique()):
        fold_data = df_folds[df_folds["fold"] == fold]

        best_row = fold_data.sort_values("dbcv_score", ascending=False).iloc[0]

        best_params_per_fold[fold] = {
            "min_cluster_size": int(best_row["min_cluster_size"]),
            "min_samples": int(best_row["min_samples"]),
            "best_dbcv_score": float(best_row["dbcv_score"]),
        }
        summary_data.append(best_row.to_dict())

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(
        os.path.join(output_dir, "hdbscan_enhanced_grid_best_summary.csv"), index=False
    )

    with open(os.path.join(output_dir, "best_params_hdbscan_enhanced.json"), "w") as f:
        json.dump(best_params_per_fold, f, indent=4)

    return df_folds, df_summary, best_params_per_fold


def hdbscan_enhanced_outer_evaluation(
    outer_folds_dir,
    best_parameter_achieved,
    target="Class",
    output_dir="results/unsupervised",
    alert_rate=0.01,
):
    """
    Executa o ciclo externo da validação cruzada aninhada: estima, nos folds externos de teste, o
    desempenho de detecção do HDBSCAN* aprimorado com uma configuração fixa.

    Objetivo:
        Para cada fold externo (fold_k_train.csv e fold_k_test.csv em outer_folds_dir), treina o
        HDBSCAN* com o min_cluster_size e o min_samples de best_parameter_achieved sobre todo o
        treino externo, depois de remover as duplicatas, derivar as variáveis e padronizá-las com
        estatísticas do treino. O teste segue a mesma regra de decisão do ciclo interno: o GLOSH
        de cada transação do teste, calculado por glosh_anomaly_score, é comparado ao quantil
        1 - alert_rate do GLOSH do treino, ponderado na base subamostrada com o peso lido do
        fold_k_info.json. Os rótulos do teste entram apenas no cálculo das métricas.

        Todas as rodadas externas usam a mesma configuração, conforme o critério de seleção
        adotado no trabalho e descrito na Seção 3.1.3.4 da monografia. O relatório traz uma linha
        por fold e uma linha final, "Média", com a média aritmética das métricas entre os folds,
        e é gravado em hdbscan_enhanced_outer_final_report.csv.

    Parâmetros:
        outer_folds_dir (str): pasta com os arquivos fold_k_train.csv e fold_k_test.csv.
        best_parameter_achieved (dict): configuração avaliada, com min_cluster_size e min_samples.
        target (str): nome da coluna alvo.
        output_dir (str): pasta do relatório final.
        alert_rate (float): fração esperada de alertas, que define o limiar no escore do treino.

    Retorno:
        pd.DataFrame com precisão, recall, F1, AUPRC do GLOSH, TP, FP, FN e tempos de treino e
        de predição, em minutos, de cada fold, seguido da linha "Média".

    Exemplo de uso:
        best = {"min_cluster_size": 50, "min_samples": 300}

        report = hdbscan_enhanced_outer_evaluation(
            outer_folds_dir="data/processed/outer_folds",
            best_parameter_achieved=best,
            output_dir="results/unsupervised",
            alert_rate=0.001
        )

        print(report[report["fold"] == "Média"][["precision", "recall", "f1", "auprc"]])
    """
    os.makedirs(output_dir, exist_ok=True)
    final_metrics = []

    print("\n--- Iniciando Avaliação Externa (Outer Folds) com Predição ---")

    folds = sorted(
        {int(f.split("_")[1]) for f in os.listdir(outer_folds_dir) if f.endswith("_train.csv")}
    )

    min_cluster_size = int(best_parameter_achieved["min_cluster_size"])
    min_samples = int(best_parameter_achieved["min_samples"])

    for fold in folds:
        print(
            f"Avaliando Fold Externo {fold} | min_cluster_size={min_cluster_size} | min_samples={min_samples}"
        )

        train_path = os.path.join(outer_folds_dir, f"fold_{fold}_train.csv")
        test_path = os.path.join(outer_folds_dir, f"fold_{fold}_test.csv")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # credit-card-fraud-detection-ds, célula 9, linha 2; credit-card-fraud-detection, célula 10, linha 3
        X_train = engineer_features(df_train.drop(columns=[target]).drop_duplicates())
        X_test = engineer_features(df_test.drop(columns=[target]))
        y_test = df_test[target]
        # os rótulos do treino entram só nos pesos do limiar de alerta da base subamostrada
        y_train = df_train.loc[X_train.index, target]
        legitimate_weight = read_legitimate_weight(
            os.path.join(outer_folds_dir, f"fold_{fold}_info.json")
        )
        print(f"Peso de cada legítima no limiar de alerta: {legitimate_weight:.4f}")

        preprocessor = build_preprocessor()

        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)

        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

        t0 = time.time()
        model.fit(X_train_scaled)
        train_time_min = (time.time() - t0) / 60

        t1 = time.time()
        train_glosh, test_glosh = glosh_anomaly_score(
            model.condensed_tree_.to_numpy(), X_train_scaled, X_test_scaled, min_samples
        )
        # credit-card-fraud-detection-ds, célula 36, linha 2; credit-card-fraud-detection, célula 55, linhas 2-5
        threshold = alert_threshold(train_glosh, alert_rate, y_train, legitimate_weight)
        predict_time_min = (time.time() - t1) / 60

        # fds-et-19-anomaly-detection-models, célula 17, linha 3
        preds = (test_glosh >= threshold).astype(int)

        f1 = f1_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        auprc = average_precision_score(y_test, test_glosh)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

        final_metrics.append(
            {
                "fold": fold,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auprc": auprc,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "train_time_min": train_time_min,
                "predict_time_min": predict_time_min,
            }
        )

    df_final = pd.DataFrame(final_metrics)
    mean_metrics = df_final.mean().to_dict()
    mean_metrics["fold"] = "Média"
    df_final = pd.concat([df_final, pd.DataFrame([mean_metrics])], ignore_index=True)

    report_path = os.path.join(output_dir, "hdbscan_enhanced_outer_final_report.csv")
    df_final.to_csv(report_path, index=False)
    print(f"\nResultados preditivos finais reportados em: {report_path}")

    return df_final


if __name__ == "__main__":
    import argparse

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
    args = parser.parse_args()

    # base desta execução: paths.FULL_PATH (todas as transações) ou paths.UNDERSAMPLED_PATH
    FOLDS = paths.UNDERSAMPLED_PATH

    # fds-et-19-anomaly-detection-models, célula 5, linha 4 (top-1%)
    ALERT_RATE = 0.001

    param_grid_hdbscan = {
        "min_cluster_size": [50],
        "min_samples": [225, 250, 275, 300, 325, 350],
    }

    n_cores_disponiveis = os.cpu_count() or 1
    print(f"Iniciando pipeline... Processos alocados para o DBCV: {n_cores_disponiveis}")

    print("=== ETAPA 1: Grid Search Interno (Treino + Tracking na Validação) ===")
    df_folds, df_summary, best_params_per_fold = hdbscan_enhanced_internal_grid_with_tracking(
        internal_folds_dir=FOLDS.internal_folds_dir,
        param_grid=param_grid_hdbscan,
        target="Class",
        # pasta própria do GLOSH: com --resume, a do grau de anomalia seria lida como concluída
        output_dir=FOLDS.results_dir(paths.RESULTS_DIR, "grid_results_hdbscan_glosh"),
        n_processes=n_cores_disponiveis,
        max_dbcv_sample_size=28000,
        alert_rate=ALERT_RATE,
        resume=args.resume,
        only_folds=args.folds,
    )

    print("\nMelhores parâmetros encontrados por fold:")
    print(json.dumps(best_params_per_fold, indent=4))

    best_parameter_achieved = max(
        best_params_per_fold.values(), key=lambda fold_params: fold_params["best_dbcv_score"]
    )

    print("\nMelhor parâmetro obtido no ciclo interno:")
    print(json.dumps(best_parameter_achieved, indent=4))

    print("\n=== ETAPA 2: Avaliação Externa (Verificando Fraudes com Labels Reais) ===")
    df_final_report = hdbscan_enhanced_outer_evaluation(
        outer_folds_dir=FOLDS.outer_folds_dir,
        best_parameter_achieved=best_parameter_achieved,
        target="Class",
        output_dir=FOLDS.results_dir(paths.RESULTS_DIR, "final_reports_hdbscan_glosh"),
        alert_rate=ALERT_RATE,
    )

    print("\nRelatório Final (Outer Folds):")
    print(df_final_report)
