import os
import sys
from pathlib import Path

import psutil

# grid_search importa o features.py de src/models/, um nível acima deste diretório
sys.path.append(str(Path(__file__).resolve().parents[1]))

from grid_search import *  # noqa: E402

if __name__ == "__main__":
    # Detectando núcleos da máquina
    num_cores = os.cpu_count()
    num_physical = psutil.cpu_count(logical=False)

    print(f"Cores lógicos (hyperthreading): {num_cores}")
    print(f"Cores físicos: {num_physical}\n")

    # Grid de hiperparâmetros pra o random forest
    param_grid_rf = {
        "n_estimators": [100, 200, 300],
        "max_features": ["sqrt", 0.5, 0.7],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "max_samples": [0.1, 0.2, None],
        "min_samples_leaf": [1, 2, 3],
    }

    # ------------------------------------------------------
    # Grid específico para XGBoost
    param_grid_xgb = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "gamma": [0, 0.5, 1],
    }

    # ------------------------------------------------------

    model_select = "xgb"  # modelo: "rf" ou "xgb"
    param_grid = {"rf": param_grid_rf, "xgb": param_grid_xgb}[model_select]
    output_dir = f"results/supervised/{model_select}"  # onde salvar CSVs, um por modelo

    df_folds, df_summary, _ = manual_grid_search(
        folds_dir="data/processed/inner_folds",  # pasta com CSVs
        param_grid=param_grid,  # grid de hiperparâmetros
        target="Class",  # nome da coluna alvo
        metric="auprc",  # métrica para escolher o melhor modelo
        output_dir=output_dir,
        parallel=True,  # True para paralelizar
        n_jobs=6,  # -1 = usar todos os núcleos
        model_select=model_select,
    )

    print(df_summary.head())  # resumo por combinação
    print(df_folds.head())  # métricas por fold

    # Como no HDBSCAN* aprimorado: cada fold interno indica a sua melhor combinação, e a de
    # maior AUPRC entre elas é aplicada a todas as rodadas externas
    best_params_per_fold = best_params_by_fold(df_folds, param_grid, metric="auprc")
    print("Melhores parâmetros por fold segundo a AUPRC:", best_params_per_fold)

    best_parameter_achieved = max(
        best_params_per_fold.values(), key=lambda fold_best: fold_best["best_score"]
    )["params"]
    print("Melhor parâmetro obtido no ciclo interno:", best_parameter_achieved)

    df_final_report = supervised_outer_evaluation(
        outer_folds_dir="data/processed/outer_folds",
        best_parameter_achieved=best_parameter_achieved,
        target="Class",
        model_select=model_select,
        output_dir=output_dir,
    )

    print("\nRelatório Final (Outer Folds):")
    print(df_final_report)
