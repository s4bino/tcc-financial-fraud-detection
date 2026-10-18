import os
import json
import pandas as pd
from tqdm import tqdm
from typing import Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Resolve nome da coluna alvo (pode ser índice ou string)
# ---------------------------------------------------------
def resolve_target_column(df: pd.DataFrame, target) -> str:
    if isinstance(target, int):
        if target < 0 or target >= len(df.columns):
            raise IndexError(
                f"Índice de coluna inválido: {target}. O DataFrame tem {len(df.columns)} colunas."
            )
        return df.columns[target]

    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(
                f"A coluna '{target}' não existe. Colunas disponíveis: {list(df.columns)}"
            )
        return target

    raise TypeError("target deve ser str (nome) ou int (índice da coluna).")


# ---------------------------------------------------------
# Cria os índices dos folds estratificados
# ---------------------------------------------------------
def stratified_folds(df: pd.DataFrame, target: str, n_splits: int, random_state: Optional[int]):
    X = df.drop(columns=[target])
    y = df[target]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for fold_number, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        folds.append({"fold": fold_number, "train_idx": train_idx, "test_idx": test_idx})
    return folds


# ---------------------------------------------------------
# Criação dos CSVs + Metadados JSON
# ---------------------------------------------------------
def external_k_folds(
    csv_path: str, output_dir: str, target, n_splits: int = 5, random_state: Optional[int] = 42
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    # cria pasta de saída
    os.makedirs(output_dir, exist_ok=True)

    print(f"Lendo dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    target_name = resolve_target_column(df, target)
    print(f"Coluna alvo usada: '{target_name}'\n")

    folds = stratified_folds(df, target_name, n_splits, random_state)

    print(f"Gerando {n_splits} folds estratificados...\n")

    # Barra de progresso ENQUANTO cria arquivos
    for fold_data in tqdm(folds, desc="Processando folds"):
        fold = fold_data["fold"]
        train_idx = fold_data["train_idx"]
        test_idx = fold_data["test_idx"]

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        train_path = os.path.join(output_dir, f"fold_{fold}_train.csv")
        test_path = os.path.join(output_dir, f"fold_{fold}_test.csv")

        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        # -------------------------------------------------
        # METADADOS DO FOLD
        # -------------------------------------------------
        metadata = {
            "fold": fold,
            "random_state": random_state,
            "target_column": target_name,
            "num_train_samples": len(df_train),
            "num_test_samples": len(df_test),
            "class_distribution_train": df_train[target_name]
            .value_counts(normalize=True)
            .to_dict(),
            "class_distribution_test": df_test[target_name].value_counts(normalize=True).to_dict(),
            "train_file": train_path,
            "test_file": test_path,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
        }

        meta_path = os.path.join(output_dir, f"fold_{fold}_info.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\n Processo concluído!")
    print(f"Todos os CSVs e metadados foram salvos em: {output_dir}\n")


def internal_k_folds(
    folds_dir: str, internal_output_dir: str, target, val_size: float = 0.2, random_state: int = 42
):
    """
    Cria validação interna estratificada usando train_test_split,
    salvando os arquivos em outra pasta.
    """

    # Criar pasta para os folds internos
    os.makedirs(internal_output_dir, exist_ok=True)

    print(f"Gerando validação interna estratificada (val_size={val_size})...\n")
    print(f"Salvando resultados em: {internal_output_dir}\n")

    # lista apenas os train.csv criados pelo external
    fold_train_files = sorted(
        [f for f in os.listdir(folds_dir) if f.startswith("fold_") and f.endswith("_train.csv")]
    )

    for train_file in tqdm(fold_train_files, desc="Processando folds internos"):
        # ----------------------------
        # Identificar o número do fold
        # ----------------------------
        fold_id = int(train_file.split("_")[1])

        train_path_ext = os.path.join(folds_dir, train_file)
        meta_path_ext = os.path.join(folds_dir, f"fold_{fold_id}_info.json")

        df_train = pd.read_csv(train_path_ext)

        # Resolve nome da coluna alvo
        target_name = resolve_target_column(df_train, target)

        # ----------------------------
        # Criar split interno
        # ----------------------------
        df_train_final, df_val = train_test_split(
            df_train,
            test_size=val_size,
            shuffle=True,
            stratify=df_train[target_name],
            random_state=random_state,
        )

        df_train_final = df_train_final.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        # ----------------------------
        # Salvar arquivos internos em outra pasta
        # ----------------------------
        train_final_path = os.path.join(internal_output_dir, f"fold_{fold_id}_train.csv")
        val_path = os.path.join(internal_output_dir, f"fold_{fold_id}_val.csv")

        df_train_final.to_csv(train_final_path, index=False)
        df_val.to_csv(val_path, index=False)

        # ----------------------------
        # Criar metadados internos (não altera o externo!)
        # ----------------------------
        with open(meta_path_ext, "r", encoding="utf-8") as f:
            metadata_ext = json.load(f)

        # copia info externa mas sem alterar o original
        metadata_int = {
            "fold": fold_id,
            "external_metadata": metadata_ext,
            "internal_validation": {
                "method": "train_test_split",
                "val_size": val_size,
                "random_state": random_state,
                "num_train_final_samples": len(df_train_final),
                "num_val_samples": len(df_val),
                "class_distribution_train_final": (
                    df_train_final[target_name]
                    .value_counts(normalize=True)
                    .round(6)
                    .astype(float)
                    .to_dict()
                ),
                "class_distribution_val": (
                    df_val[target_name]
                    .value_counts(normalize=True)
                    .round(6)
                    .astype(float)
                    .to_dict()
                ),
                "train_final_file": train_final_path,
                "val_file": val_path,
            },
        }

        # Salvar metadata interna no novo diretório
        meta_path_int = os.path.join(internal_output_dir, f"fold_{fold_id}_internal_info.json")

        with open(meta_path_int, "w", encoding="utf-8") as f:
            json.dump(metadata_int, f, indent=4, ensure_ascii=False)

    print("\nProcesso concluído!")
    print(f"Todos os arquivos foram salvos em: {internal_output_dir}\n")
