import argparse
import os
import sys
from pathlib import Path

import psutil

# grid_search importa o features.py e o metrics.py de src/models/, um nível acima deste
# diretório, onde também fica o paths.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

import paths  # noqa: E402
from grid_search import *  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Busca em grade dos modelos supervisionados.")
    parser.add_argument("--model", choices=["rf", "xgb"], required=True, help="modelo a executar")
    parser.add_argument(
        "--resume", action="store_true", help="retoma os CSVs gravados e pula o que já foi feito"
    )
    parser.add_argument(
        "--drive", action="store_true", help="lê os folds e grava os resultados no Drive (Colab)"
    )
    args = parser.parse_args()

    # Detectando núcleos da máquina
    num_cores = os.cpu_count()
    num_physical = psutil.cpu_count(logical=False)

    print(f"Cores lógicos (hyperthreading): {num_cores}")
    print(f"Cores físicos: {num_physical}\n")

    # Grid de hiperparâmetros pra o random forest
    param_grid_rf = {
        "n_estimators": [100, 200, 300],
        "max_features": ["sqrt", 0.5, 0.7],
        "class_weight": [None, "balanced_subsample"],
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

    model_select = args.model
    param_grid = {"rf": param_grid_rf, "xgb": param_grid_xgb}[model_select]

    # base desta execução: a completa (FULL_PATH) ou a subamostrada (UNDERSAMPLED_PATH), no
    # repositório local ou no Drive montado no Colab
    folds = paths.UNDERSAMPLED_PATH if args.drive else paths.LOCAL_UNDERSAMPLED_PATH
    results_root = paths.RESULTS_DIR if args.drive else paths.LOCAL_RESULTS_DIR
    # onde salvar CSVs, um por modelo, numa pasta própria de cada base
    output_dir = folds.results_dir(results_root, "supervised") / model_select

    df_folds, df_summary, _ = manual_grid_search(
        folds_dir=folds.internal_folds_dir,  # pasta com CSVs
        param_grid=param_grid,  # grid de hiperparâmetros
        target="Class",  # nome da coluna alvo
        metric="auprc",  # métrica para escolher o melhor modelo
        output_dir=output_dir,
        parallel=True,  # True para paralelizar
        n_jobs=num_physical,  # uma combinação por núcleo físico
        model_select=model_select,
        resume=args.resume,
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
        outer_folds_dir=folds.outer_folds_dir,
        best_parameter_achieved=best_parameter_achieved,
        target="Class",
        model_select=model_select,
        output_dir=output_dir,
        resume=args.resume,
    )

    print("\nRelatório Final (Outer Folds):")
    print(df_final_report)
