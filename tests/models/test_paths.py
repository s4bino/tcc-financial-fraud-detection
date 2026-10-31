"""Testes de paths: os caminhos dos folds e das pastas de resultados.

Os testes de lógica rodam sempre. Os que leem os folds de data/ e o espelho do
Drive (Google Drive para Desktop, em G:) são pulados quando os arquivos não
estão na máquina, porque os dados não são versionados. A pasta do espelho pode
ser trocada pela variável de ambiente DRIVE_MIRROR.
"""

import filecmp
import os
from pathlib import Path

import pandas as pd
import pytest

import paths
from metrics import read_legitimate_weight

N_FOLDS = 5
REPOSITORY = Path(__file__).resolve().parents[2]
COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
DRIVE_MIRROR = Path(os.environ.get("DRIVE_MIRROR", "G:/Meu Drive"))
BASES = {
    "completa": (paths.FULL_PATH, paths.LOCAL_FULL_PATH),
    "subamostrada": (paths.UNDERSAMPLED_PATH, paths.LOCAL_UNDERSAMPLED_PATH),
}
FILES = {"internal_folds_dir": ("train", "val"), "outer_folds_dir": ("train", "test")}
INFO_FILES = {"internal_folds_dir": "internal_info", "outer_folds_dir": "info"}
# 100 mil das 284.315 legítimas da base: cada legítima mantida representa cerca de 2,84
LEGITIMATE_WEIGHTS = {"completa": 1.0, "subamostrada": 284_315 / 100_000}


def info_files(folds):
    """Os 10 JSONs de uma base, um por fold de cada ciclo, com o peso das legítimas."""
    return [
        Path(getattr(folds, cycle)) / f"fold_{fold}_{name}.json"
        for cycle, name in INFO_FILES.items()
        for fold in range(1, N_FOLDS + 1)
    ]


def assert_legitimate_weight(base, info_paths):
    for path in info_paths:
        weight = read_legitimate_weight(path)
        assert weight == pytest.approx(LEGITIMATE_WEIGHTS[base], rel=0.005), f"{base}: {path}"


def fold_files(folds):
    """Os 20 CSVs de uma base: treino e validação internos, treino e teste externos."""
    return [
        (cycle, fold, name, Path(getattr(folds, cycle)) / f"fold_{fold}_{name}.csv")
        for cycle, names in FILES.items()
        for fold in range(1, N_FOLDS + 1)
        for name in names
    ]


def local(folds):
    """A mesma base com os caminhos resolvidos a partir da raiz do repositório."""
    return paths.FoldPaths(
        REPOSITORY / folds.internal_folds_dir,
        REPOSITORY / folds.outer_folds_dir,
        folds.results_suffix,
    )


def require_local(folds):
    resolved = local(folds)
    if not Path(resolved.internal_folds_dir).is_dir():
        pytest.skip(f"{folds.internal_folds_dir} não existe nesta máquina")
    return resolved


@pytest.mark.parametrize("folds", [paths.FULL_PATH, paths.LOCAL_FULL_PATH])
def test_the_full_base_keeps_the_original_results_folders(folds):
    assert folds.results_dir("RESULTS", "grid_results_hdbscan_glosh") == Path(
        "RESULTS/grid_results_hdbscan_glosh"
    )


@pytest.mark.parametrize("folds", [paths.UNDERSAMPLED_PATH, paths.LOCAL_UNDERSAMPLED_PATH])
def test_the_undersampled_base_writes_to_its_own_results_folders(folds):
    assert folds.results_dir("RESULTS", "grid_results_hdbscan_glosh") == Path(
        "RESULTS/grid_results_hdbscan_glosh_undersampled"
    )


@pytest.mark.parametrize(
    ("full", "undersampled"),
    [
        (paths.FULL_PATH, paths.UNDERSAMPLED_PATH),
        (paths.LOCAL_FULL_PATH, paths.LOCAL_UNDERSAMPLED_PATH),
    ],
)
def test_the_two_bases_never_share_a_folder(full, undersampled):
    assert full.internal_folds_dir != undersampled.internal_folds_dir
    assert full.outer_folds_dir != undersampled.outer_folds_dir
    assert full.results_suffix != undersampled.results_suffix


def test_the_drive_undersampled_folds_live_inside_the_drive_data_folder():
    for cycle in FILES:
        full = Path(getattr(paths.FULL_PATH, cycle))
        undersampled = Path(getattr(paths.UNDERSAMPLED_PATH, cycle))

        assert undersampled == paths.DRIVE_DATA_DIR / "undersampled" / full.name


@pytest.mark.parametrize("base", BASES)
def test_every_local_fold_file_loads_with_the_dataset_columns(base):
    folds = require_local(BASES[base][1])

    for cycle, fold, name, path in fold_files(folds):
        assert path.is_file(), f"{base}: falta {path}"
        frame = pd.read_csv(path, nrows=1_000)
        assert list(frame.columns) == COLUMNS, f"{base}: colunas de {cycle} {fold} {name}"


def test_the_local_undersampled_training_sets_keep_every_fraud_with_fewer_legitimates():
    full = require_local(paths.LOCAL_FULL_PATH)
    undersampled = require_local(paths.LOCAL_UNDERSAMPLED_PATH)

    for cycle in FILES:
        original = pd.read_csv(Path(getattr(full, cycle)) / "fold_1_train.csv")
        reduced = pd.read_csv(Path(getattr(undersampled, cycle)) / "fold_1_train.csv")

        assert reduced["Class"].sum() == original["Class"].sum()
        assert len(reduced) < len(original)
        assert reduced["Class"].mean() == pytest.approx(0.0049, abs=0.0005)


def test_the_local_evaluation_files_are_the_same_in_both_bases():
    full = require_local(paths.LOCAL_FULL_PATH)
    undersampled = require_local(paths.LOCAL_UNDERSAMPLED_PATH)

    for cycle, fold, name, path in fold_files(full):
        if name == "train":
            continue
        twin = Path(getattr(undersampled, cycle)) / path.name
        assert filecmp.cmp(path, twin, shallow=False), f"{cycle} {fold} {name} difere"


@pytest.mark.parametrize("base", BASES)
def test_the_drive_mirror_has_every_fold_file_the_colab_reads(base):
    if not DRIVE_MIRROR.is_dir():
        pytest.skip(f"espelho do Drive ausente em {DRIVE_MIRROR}")
    drive = BASES[base][0]

    for *_, path in fold_files(drive):
        mirrored = DRIVE_MIRROR / path.relative_to(paths.DRIVE_DIR)
        assert mirrored.is_file(), f"{base}: falta {mirrored}"


def test_no_script_writes_the_drive_folders_outside_paths():
    """Os caminhos do Drive ficam só em paths.py, para mudarem num lugar só."""
    sources = (REPOSITORY / "src").rglob("*.py")
    hardcoded = [
        source.relative_to(REPOSITORY)
        for source in sources
        if source.name != "paths.py" and "TCC - DADOS CV" in source.read_text(encoding="utf-8")
    ]

    assert hardcoded == []


@pytest.mark.parametrize("base", BASES)
def test_every_local_fold_info_gives_the_legitimate_weight_of_its_base(base):
    folds = require_local(BASES[base][1])

    assert_legitimate_weight(base, info_files(folds))


@pytest.mark.parametrize("base", BASES)
def test_every_drive_fold_info_gives_the_legitimate_weight_of_its_base(base):
    if not DRIVE_MIRROR.is_dir():
        pytest.skip(f"espelho do Drive ausente em {DRIVE_MIRROR}")
    drive = BASES[base][0]
    mirrored = [DRIVE_MIRROR / path.relative_to(paths.DRIVE_DIR) for path in info_files(drive)]

    assert_legitimate_weight(base, mirrored)
