import os
import psutil
import numpy as np
import pandas as pd
from grid_search import *

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

    df_folds, df_summary, best_params = manual_grid_search(
        folds_dir="data/processed/inner_folds",  # pasta com CSVs
        param_grid=param_grid_xgb,  # grid de hiperparâmetros
        target="Class",  # nome da coluna alvo
        metric="f1",  # métrica para escolher o melhor modelo
        output_dir="results/supervised",  # onde salvar CSVs
        parallel=True,  # True para paralelizar
        n_jobs=6,  # -1 = usar todos os núcleos
        model_select="xgb",  # modelo: "rf" ou "xgb"
    )

    print("Melhores parâmetros segundo F1:", best_params)
    print(df_summary.head())  # resumo por combinação
    print(df_folds.head())  # métricas por fold
