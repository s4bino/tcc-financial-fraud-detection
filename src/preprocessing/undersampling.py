"""Subamostragem aleatória das transações legítimas nas partições de treino dos folds.

Cada treino, externo e interno, mantém todas as fraudes e uma fração sorteada das legítimas. A
validação interna e o teste externo são copiados sem alteração, na proporção real de fraudes,
para que as métricas continuem estimando o desempenho na distribuição em que o modelo seria
usado e comparáveis às dos folds originais. Os folds de partida são os de
data/processed/outer_folds, lidos pelos índices gravados nos metadados, de modo que cada
transação continua no mesmo fold e na mesma partição de antes.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cross_validation import resolve_target_column


def legitimate_fraction(labels, target_legitimate):
    """
    Calcula a fração das legítimas que leva a base inteira a target_legitimate transações.

    Objetivo:
        Converte o tamanho desejado da classe majoritária, dado para a base inteira, na taxa de
        amostragem aplicada a cada partição de treino. Com 284.315 legítimas na base e
        target_legitimate = 100.000, a fração é 0,3517, e um treino externo com 227.452
        legítimas passa a ter cerca de 80.000.

    Parâmetros:
        labels (np.ndarray): rótulos da base inteira, com 0 para as legítimas.
        target_legitimate (int): número de legítimas que a base teria depois da subamostragem.

    Retorno:
        float em (0, 1].

    Exemplo de uso:
        legitimate_fraction(df["Class"].to_numpy(), 100_000)   # 0.3517...
    """
    n_legitimate = int(np.sum(np.asarray(labels) == 0))
    if not 0 < target_legitimate <= n_legitimate:
        raise ValueError(
            f"target_legitimate deve estar entre 1 e {n_legitimate}, não {target_legitimate}"
        )
    return target_legitimate / n_legitimate


def undersample_legitimate(indices, labels, fraction, rng):
    """
    Sorteia uma fração das legítimas de uma partição e mantém todas as fraudes.

    Objetivo:
        Aplica a subamostragem aleatória simples, sem reposição, à classe majoritária. As
        legítimas mantidas formam uma amostra aleatória das legítimas da partição, o que
        preserva, em esperança, a distribuição conjunta das suas variáveis. As fraudes, poucas
        e caras de perder, ficam todas. A ordem original da partição é preservada.

    Parâmetros:
        indices (np.ndarray): índices, na base original, das transações da partição.
        labels (np.ndarray): rótulos dessas transações, na mesma ordem.
        fraction (float): fração das legítimas mantida.
        rng (np.random.Generator): gerador do sorteio.

    Retorno:
        np.ndarray com os índices mantidos, na ordem de indices.

    Exemplo de uso:
        rng = np.random.default_rng([42, fold])
        kept = undersample_legitimate(train_idx, y[train_idx], 0.35, rng)
    """
    indices, labels = np.asarray(indices), np.asarray(labels)
    legitimate = indices[labels == 0]
    sampled = rng.choice(legitimate, size=round(fraction * len(legitimate)), replace=False)
    return indices[(labels != 0) | np.isin(indices, sampled)]


def internal_split(train_indices, labels, val_size, random_state):
    """
    Reproduz, em índices da base original, a divisão de internal_k_folds.

    Objetivo:
        internal_k_folds divide o treino externo com train_test_split estratificado e grava só
        os CSVs. A divisão depende apenas do número de linhas, dos rótulos e da semente, então
        repeti-la sobre as posições do treino externo devolve as mesmas partições, agora como
        índices da base, que permitem subamostrar o treino interno sem perder o vínculo com o
        treino externo.

    Parâmetros:
        train_indices (np.ndarray): índices do treino externo, na ordem do CSV do fold.
        labels (np.ndarray): rótulos do treino externo, na mesma ordem.
        val_size (float): o mesmo val_size usado em internal_k_folds.
        random_state (int): a mesma semente usada em internal_k_folds.

    Retorno:
        train_final_indices (np.ndarray): índices do treino interno, na ordem do CSV.
        val_indices (np.ndarray): índices da validação interna, na ordem do CSV.

    Exemplo de uso:
        inner_idx, val_idx = internal_split(train_idx, y[train_idx], 0.2, 42)
    """
    train_indices = np.asarray(train_indices)
    train_positions, val_positions = train_test_split(
        np.arange(len(train_indices)),
        test_size=val_size,
        shuffle=True,
        stratify=labels,
        random_state=random_state,
    )
    return train_indices[train_positions], train_indices[val_positions]


def representativeness(kept, removed):
    """
    Compara, variável a variável, as legítimas mantidas com as removidas.

    Objetivo:
        Aplica o teste de Kolmogorov-Smirnov de duas amostras a cada coluna. Como as duas
        partes vêm de um mesmo sorteio aleatório, a estatística D deve ficar próxima de zero, e
        os p-valores, distribuídos de modo aproximadamente uniforme. Valores de D altos
        indicariam que a amostra mudou o perfil das transações legítimas.

    Parâmetros:
        kept (pd.DataFrame): legítimas mantidas, só com as variáveis de entrada.
        removed (pd.DataFrame): legítimas removidas, com as mesmas colunas.

    Retorno:
        dict com, para cada coluna, a estatística D e o p-valor.

    Exemplo de uso:
        ks = representativeness(df.loc[kept, features], df.loc[removed, features])
    """
    results = {}
    for column in kept.columns:
        test = ks_2samp(kept[column], removed[column])
        results[column] = {"statistic": float(test.statistic), "p_value": float(test.pvalue)}
    return results


def undersampling_summary(df, target_name, partition, kept, fraction, seed):
    """
    Reúne os metadados da subamostragem de uma partição de treino.

    Parâmetros:
        df (pd.DataFrame): base original.
        target_name (str): coluna alvo.
        partition (np.ndarray): índices da partição antes da subamostragem.
        kept (np.ndarray): índices mantidos.
        fraction (float): fração das legítimas mantida.
        seed (list[int]): semente do sorteio, [random_state, fold].

    Retorno:
        dict com contagens, fração, semente e o teste de representatividade.
    """
    labels = df[target_name].to_numpy()
    legitimate = partition[labels[partition] == 0]
    kept_legitimate = kept[labels[kept] == 0]
    removed_legitimate = np.setdiff1d(legitimate, kept_legitimate)
    features = df.columns.drop(target_name)
    ks = representativeness(df.loc[kept_legitimate, features], df.loc[removed_legitimate, features])
    return {
        "method": "subamostragem aleatória simples das legítimas, sem reposição",
        "legitimate_fraction": fraction,
        "seed": seed,
        "num_legitimate_before": len(legitimate),
        "num_legitimate_after": len(kept_legitimate),
        "num_frauds": int(np.sum(labels[kept] != 0)),
        "ks_max_statistic": max(result["statistic"] for result in ks.values()),
        "ks_by_column": ks,
    }


def class_distribution(frame, target_name):
    return frame[target_name].value_counts(normalize=True).round(6).astype(float).to_dict()


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def undersampled_k_folds(
    csv_path: str,
    folds_dir: str,
    output_dir: str,
    target,
    target_legitimate: int = 100_000,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Gera os folds externos e internos com as legítimas subamostradas nas partições de treino.

    Objetivo:
        Para cada fold de folds_dir, sorteia uma vez as legítimas do treino externo e grava, em
        output_dir/outer_folds e output_dir/inner_folds, os mesmos arquivos de external_k_folds
        e internal_k_folds:
          - treino externo: todas as fraudes e a fração sorteada das legítimas.
          - teste externo: igual ao original.
          - treino interno: as transações do treino interno original que ficaram no treino
            externo subamostrado, de modo que continua contido nele.
          - validação interna: igual à original.
        Os metadados repetem os campos dos folds originais e acrescentam o bloco
        "undersampling", com contagens, semente e o teste de representatividade.

    Parâmetros:
        csv_path (str): base original (creditcard.csv).
        folds_dir (str): pasta dos folds externos originais, com os fold_<n>_info.json.
        output_dir (str): pasta de saída, que recebe outer_folds/ e inner_folds/.
        target (str | int): coluna alvo, pelo nome ou pelo índice.
        target_legitimate (int): legítimas que a base inteira teria (legitimate_fraction).
        val_size (float): o val_size usado na geração dos folds internos originais.
        random_state (int): semente da subamostragem e da divisão interna original.
    """
    df = pd.read_csv(csv_path)
    target_name = resolve_target_column(df, target)
    labels = df[target_name].to_numpy()
    fraction = legitimate_fraction(labels, target_legitimate)

    outer_dir = os.path.join(output_dir, "outer_folds")
    inner_dir = os.path.join(output_dir, "inner_folds")
    os.makedirs(outer_dir, exist_ok=True)
    os.makedirs(inner_dir, exist_ok=True)

    print(f"Fração das legítimas mantida no treino: {fraction:.4f} ({target_legitimate} na base)")

    metadata_files = sorted(
        f for f in os.listdir(folds_dir) if f.startswith("fold_") and f.endswith("_info.json")
    )
    for metadata_file in tqdm(metadata_files, desc="Subamostrando folds"):
        with open(os.path.join(folds_dir, metadata_file), encoding="utf-8") as file:
            original = json.load(file)
        fold = original["fold"]
        train_idx = np.array(original["train_indices"])
        test_idx = np.array(original["test_indices"])
        seed = [random_state, fold]

        kept = undersample_legitimate(
            train_idx, labels[train_idx], fraction, np.random.default_rng(seed)
        )
        inner_idx, val_idx = internal_split(train_idx, labels[train_idx], val_size, random_state)
        inner_kept = inner_idx[np.isin(inner_idx, kept)]

        partitions = {
            (outer_dir, "train"): kept,
            (outer_dir, "test"): test_idx,
            (inner_dir, "train"): inner_kept,
            (inner_dir, "val"): val_idx,
        }
        frames = {}
        for (directory, name), indices in partitions.items():
            frame = df.iloc[indices].reset_index(drop=True)
            path = os.path.join(directory, f"fold_{fold}_{name}.csv")
            frame.to_csv(path, index=False)
            frames[directory, name] = (frame, path)

        outer_train, outer_train_path = frames[outer_dir, "train"]
        outer_test, outer_test_path = frames[outer_dir, "test"]
        outer_metadata = {
            "fold": fold,
            "random_state": random_state,
            "target_column": target_name,
            "num_train_samples": len(outer_train),
            "num_test_samples": len(outer_test),
            "class_distribution_train": class_distribution(outer_train, target_name),
            "class_distribution_test": class_distribution(outer_test, target_name),
            "train_file": outer_train_path,
            "test_file": outer_test_path,
            "train_indices": kept.tolist(),
            "test_indices": test_idx.tolist(),
            "source_folds": os.path.join(folds_dir, metadata_file),
            "undersampling": undersampling_summary(
                df, target_name, train_idx, kept, fraction, seed
            ),
        }
        write_json(outer_metadata, os.path.join(outer_dir, f"fold_{fold}_info.json"))

        inner_train, inner_train_path = frames[inner_dir, "train"]
        validation, validation_path = frames[inner_dir, "val"]
        inner_metadata = {
            "fold": fold,
            "external_metadata": outer_metadata,
            "internal_validation": {
                "method": "train_test_split",
                "val_size": val_size,
                "random_state": random_state,
                "num_train_final_samples": len(inner_train),
                "num_val_samples": len(validation),
                "class_distribution_train_final": class_distribution(inner_train, target_name),
                "class_distribution_val": class_distribution(validation, target_name),
                "train_final_file": inner_train_path,
                "val_file": validation_path,
                "train_final_indices": inner_kept.tolist(),
                "val_indices": val_idx.tolist(),
            },
            "undersampling": undersampling_summary(
                df, target_name, inner_idx, inner_kept, fraction, seed
            ),
        }
        write_json(inner_metadata, os.path.join(inner_dir, f"fold_{fold}_internal_info.json"))

    print("\nProcesso concluído!")
    print(f"Folds subamostrados salvos em: {output_dir}\n")
