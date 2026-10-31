"""Caminhos dos folds e das pastas de resultados, num só lugar, para os scripts de entrada.

Cada base de folds existe em duas versões: a completa, com todas as transações, e a
subamostrada, com as legítimas reduzidas só nas partições de treino
(src/preprocessing/run_undersampled_fold_generation.py). Os scripts escolhem a base passando um
destes conjuntos de caminhos às funções que iniciam o pipeline. Os resultados de cada base vão
para pastas próprias, para que o --resume de uma não leia os da outra como concluídos.

No Colab, este arquivo sobe para /content junto com o script, o features.py, o metrics.py e o
disco.py.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FoldPaths:
    """
    Pastas dos folds de uma base e o sufixo das suas pastas de resultados.

    Parâmetros:
        internal_folds_dir (Path): fold_k_train.csv e fold_k_val.csv do ciclo interno.
        outer_folds_dir (Path): fold_k_train.csv e fold_k_test.csv do ciclo externo.
        results_suffix (str): sufixo acrescentado ao nome das pastas de resultados.
    """

    internal_folds_dir: Path
    outer_folds_dir: Path
    results_suffix: str

    def results_dir(self, root, name):
        """
        Monta a pasta de resultados de um experimento nesta base.

        Parâmetros:
            root (Path): pasta que reúne os resultados, como RESULTS_DIR.
            name (str): nome da pasta do experimento na base completa.

        Retorno:
            Path de root/name, com o sufixo da base.

        Exemplo de uso:
            UNDERSAMPLED_PATH.results_dir(RESULTS_DIR, "grid_results_hdbscan_glosh")
            # /content/drive/MyDrive/RESULTS/grid_results_hdbscan_glosh_undersampled
        """
        return Path(root) / f"{name}{self.results_suffix}"


# Google Drive montado no Colab, onde rodam os modelos não supervisionados
DRIVE_DIR = Path("/content/drive/MyDrive")
DRIVE_DATA_DIR = DRIVE_DIR / "TCC - DADOS CV"
RESULTS_DIR = DRIVE_DIR / "RESULTS"

FULL_PATH = FoldPaths(
    internal_folds_dir=DRIVE_DATA_DIR / "internal_folds",
    outer_folds_dir=DRIVE_DATA_DIR / "folds_output",
    results_suffix="",
)
UNDERSAMPLED_PATH = FoldPaths(
    internal_folds_dir=DRIVE_DATA_DIR / "undersampled" / "internal_folds",
    outer_folds_dir=DRIVE_DATA_DIR / "undersampled" / "folds_output",
    results_suffix="_undersampled",
)

# Repositório local, relativo à raiz, onde rodam os modelos supervisionados
LOCAL_RESULTS_DIR = Path("results")

LOCAL_FULL_PATH = FoldPaths(
    internal_folds_dir=Path("data/processed/inner_folds"),
    outer_folds_dir=Path("data/processed/outer_folds"),
    results_suffix="",
)
LOCAL_UNDERSAMPLED_PATH = FoldPaths(
    internal_folds_dir=Path("data/undersampled/inner_folds"),
    outer_folds_dir=Path("data/undersampled/outer_folds"),
    results_suffix="_undersampled",
)
