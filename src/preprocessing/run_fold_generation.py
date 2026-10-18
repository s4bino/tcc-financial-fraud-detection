import cross_validation as cv
import json
import pandas as pd

if __name__ == "__main__":
    # Caminho do dataset original (CSV)
    csv_path = "data/raw/creditcard.csv"

    # Pasta onde vão ficar os folds
    output_dir = "data/processed/outer_folds"  # será criada se não existir

    # Índice ou nome da coluna alvo
    target = "Class"  # ou o número da coluna, ex: 30

    credito = pd.read_csv(csv_path)
    credito.shape
    credito.columns

    # 284807 linhas e 31 colunas sendo a última coluna o atributo alvo

    cv.external_k_folds(
        csv_path=csv_path, output_dir=output_dir, target=target, n_splits=5, random_state=42
    )

    with open("data/processed/outer_folds/fold_2_info.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # print(data)
    print(data.keys())

    data["fold"]
    data["train_indices"]
    data["test_indices"]
    data["train_file"]
    data["test_file"]

    # cada fold mantém aproximadamente a mesma proporção das
    # classes do dataset original

    # 99.8275 % das amostras de treino são normais
    # 0.1724 % das amostras de treino são fraudes
    data["class_distribution_train"]

    # 99.8262 % das amostras de teste são normais
    # 0.1738 % das amostras de teste são fraudes
    data["class_distribution_test"]

    data["class_distribution_train"]["1"]

    data["target_column"]
    data["random_state"]

    # 284807 * 0.8 = 227845.6
    data["num_train_samples"]

    # 284807 * 0.2 = 56961.4
    data["num_test_samples"]

    data["num_train_samples"] + data["num_test_samples"]
    data["num_train_samples"] + data["num_test_samples"] == 284807

    cv.internal_k_folds(
        folds_dir="data/processed/outer_folds",  # onde está o treino/teste externo
        internal_output_dir="data/processed/inner_folds",  # nova pasta para salvar treino_final/val
        target="Class",
        val_size=0.2,
        random_state=42,
    )
