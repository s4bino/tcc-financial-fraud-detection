"""Caminhos dos folds e das pastas de resultados, num só lugar, para os scripts de entrada.

Cada base de folds existe em duas versões: a completa, com todas as transações, e a
subamostrada, com as legítimas reduzidas só nas partições de treino
(src/preprocessing/run_undersampled_fold_generation.py). Os scripts escolhem a base pelo --base,
que seleciona um destes conjuntos de caminhos em FOLD_PATHS. Os resultados de cada base vão para
pastas próprias, para que o --resume de uma não leia os da outra como concluídos, e
guard_run_config impede que uma pasta receba execuções com configurações diferentes.

No Colab, este arquivo sobe para /content junto com o script, o features.py e o metrics.py, mais o
disco.py no HDBSCAN* e o grid_search.py nos supervisionados.
"""

import json
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


# Google Drive montado no Colab, onde rodam os modelos não supervisionados e, com --drive, os
# supervisionados
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

# Repositório local, relativo à raiz, onde rodam os modelos supervisionados sem --drive
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

# Bases escolhidas pelo --base dos scripts de entrada
FOLD_PATHS = {"full": FULL_PATH, "undersampled": UNDERSAMPLED_PATH}
LOCAL_FOLD_PATHS = {"full": LOCAL_FULL_PATH, "undersampled": LOCAL_UNDERSAMPLED_PATH}

# Rodada final: todos os modelos, nas duas bases, com a mesma configuração. Cada execução grava
# numa pasta própria dentro de uma destas, nomeada pelo modelo e pela base. No Drive, as duas
# ficam em RESULTS, longe das pastas das rodadas exploratórias, e espelham as de results/ no
# repositório, para onde os resultados são copiados sem mudar de nome.
SUPERVISED_DIR_NAME = "supervised"
UNSUPERVISED_DIR_NAME = "unsupervised"
RUN_CONFIG_FILE = "run_config.json"


def guard_run_config(output_dir, config, resume):
    """
    Grava a configuração de uma execução na sua pasta de resultados e recusa retomar outra.

    Objetivo:
        A retomada dos scripts identifica o que já foi feito pelos arquivos da pasta. Se a pasta
        guarda uma execução com outra configuração, como outra distância ou outra grade, o
        --resume a leria como concluída. A função compara a configuração pedida com a gravada
        em run_config.json e interrompe a execução quando alguma chave difere ou quando a pasta
        tem resultados sem esse arquivo. Sem --resume, a execução começa do zero e regrava o
        arquivo.

    Parâmetros:
        output_dir (str | Path): pasta de resultados da execução, criada se não existe.
        config (dict): configuração serializável em JSON. Caminhos viram texto, e tuplas,
            listas, como no arquivo gravado.
        resume (bool): se a execução retoma os resultados da pasta.

    Retorno:
        Path do run_config.json gravado.

    Exemplo de uso:
        guard_run_config("results/supervised/rf", {"model": "rf"}, resume=True)
        # na pasta com {"model": "xgb"}: ValueError, a execução não começa
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RUN_CONFIG_FILE
    record = json.loads(json.dumps(config, default=str))
    if resume and path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        changed = sorted(
            key for key in record.keys() | saved.keys() if record.get(key) != saved.get(key)
        )
        if changed:
            raise ValueError(
                f"{path} registra outra configuração, com {changed} diferentes. Use outra pasta "
                "ou rode sem --resume para começar do zero."
            )
    elif resume and any(output_dir.glob("*.csv")):
        raise ValueError(
            f"{output_dir} tem resultados sem {RUN_CONFIG_FILE}, e não há como saber se são desta "
            "configuração. Use outra pasta ou rode sem --resume."
        )
    path.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
    return path
