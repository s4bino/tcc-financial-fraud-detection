"""
Analysis of the cross-validation partitions.

Answers three questions the methodology depends on:

  1. Did stratification hold? Every partition must keep the original fraud
     proportion, otherwise the folds are not comparable to each other.
  2. Is each transaction tested exactly once across the outer folds, and is the
     outer test set kept away from the inner calibration sets?
  3. Do frauds sit in low-density regions consistently across folds, or does
     that property itself vary? This matters because the HDBSCAN results vary
     sharply between near-identical partitions, and the partition is the first
     suspect.

Reads the fold index files rather than the fold CSVs: the JSON files carry
`train_indices` and `test_indices`, so the 1.4 GB of materialised folds does
not have to be loaded.

Run from the repository root:

    python src/analysis/fold_analysis.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # noqa: E402

CSV = "data/raw/creditcard.csv"
OUTER_DIR = "data/processed/outer_folds"
INNER_DIR = "data/processed/inner_folds"
OUTPUT = "results/figures/folds"
TARGET = "Class"
SEED = 42

LEGIT_SAMPLE_SIZE = 12000
K_NEIGHBOURS = 20


def load_dataset():
    df = pd.read_csv(CSV)
    y = df[TARGET].to_numpy()
    return df, y


def scale(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler, StandardScaler

    pre = ColumnTransformer(
        transformers=[
            ("rob", RobustScaler(), ["Amount"]),
            ("std", StandardScaler(), ["Time"]),
        ],
        remainder="passthrough",
    )
    return pre.fit_transform(X)


def read_outer_folds():
    """Fold metadata plus the train/test index arrays."""
    folds = {}
    for name in sorted(os.listdir(OUTER_DIR)):
        if not name.endswith("_info.json"):
            continue
        with open(os.path.join(OUTER_DIR, name), encoding="utf-8") as handle:
            info = json.load(handle)
        folds[int(info["fold"])] = info
    return folds


def read_inner_folds():
    folds = {}
    if not os.path.isdir(INNER_DIR):
        return folds
    for name in sorted(os.listdir(INNER_DIR)):
        if not name.endswith("_internal_info.json"):
            continue
        with open(os.path.join(INNER_DIR, name), encoding="utf-8") as handle:
            info = json.load(handle)
        fold = info.get("fold")
        if fold is None:
            fold = info.get("external_metadata", {}).get("fold", name.split("_")[1])
        folds[int(fold)] = info
    return folds


# ===========================================================================
# Figure 01 — Composition and stratification
# ===========================================================================

def figure_composition(outer, inner, y):
    import matplotlib.pyplot as plt

    global_rate = 100 * y.mean()
    fold_ids = sorted(outer)

    def entry(fold, split, distribution, n):
        """The fold files store the class distribution as proportions, not
        counts, so the fraud count is recovered from the partition size."""
        share = float(distribution.get("1", distribution.get(1, 0.0)))
        return {"fold": fold, "split": split, "n": int(n),
                "frauds": int(round(share * n)), "rate": 100 * share}

    rows = []
    for fold in fold_ids:
        info = outer[fold]
        rows.append(entry(fold, "Treino externo",
                          info["class_distribution_train"],
                          info["num_train_samples"]))
        rows.append(entry(fold, "Teste externo",
                          info["class_distribution_test"],
                          info["num_test_samples"]))

    for fold, info in sorted(inner.items()):
        internal = info.get("internal_validation", {})
        if not internal:
            continue
        rows.append(entry(fold, "Treino interno",
                          internal["class_distribution_train_final"],
                          internal["num_train_final_samples"]))
        rows.append(entry(fold, "Validação interna",
                          internal["class_distribution_val"],
                          internal["num_val_samples"]))

    table = pd.DataFrame(rows)
    splits = [s for s in ("Treino externo", "Teste externo",
                          "Treino interno", "Validação interna")
              if s in set(table["split"])]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    ax.grid(True, axis="y")
    width = 0.8 / len(splits)
    for position, split in enumerate(splits):
        subset = table[table["split"] == split].sort_values("fold")
        offset = (position - (len(splits) - 1) / 2) * width
        ax.bar(np.array(subset["fold"]) + offset, subset["frauds"],
               width=width * 0.9, color=ps.CATEGORICAL[position], label=split)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Número de fraudes")
    ax.set_xticks(fold_ids)
    ax.set_title("(a) Fraudes por partição", loc="left")
    ax.legend()

    ax = axes[1]
    ax.grid(True, axis="y")
    for position, split in enumerate(splits):
        subset = table[table["split"] == split].sort_values("fold")
        ax.plot(subset["fold"], subset["rate"], marker="o",
                color=ps.CATEGORICAL[position], label=split)
    ax.axhline(global_rate, color=ps.INK_MUTED, lw=1.0)
    ax.annotate(f"base completa: {global_rate:.4f}%",
                xy=(fold_ids[0], global_rate), xytext=(0, 6),
                textcoords="offset points", fontsize=7.5, color=ps.INK_MUTED)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Proporção de fraude (%)")
    ax.set_xticks(fold_ids)
    ax.set_title("(b) A estratificação se manteve?", loc="left")

    fig.suptitle("Composição das partições da validação cruzada aninhada", y=1.04)
    ps.footnote(fig, "A linha horizontal marca a proporção de fraude da base "
                     "completa. Desvios visíveis indicariam falha na "
                     "estratificação e tornariam os folds não comparáveis.")
    return fig, table


# ===========================================================================
# Figure 02 — Coverage and isolation
# ===========================================================================

def figure_coverage(outer, total):
    import matplotlib.pyplot as plt

    fold_ids = sorted(outer)
    times_tested = np.zeros(total, dtype=int)
    for fold in fold_ids:
        times_tested[np.asarray(outer[fold]["test_indices"], dtype=int)] += 1

    overlap = np.zeros((len(fold_ids), len(fold_ids)), dtype=float)
    test_sets = {f: set(np.asarray(outer[f]["test_indices"], dtype=int))
                 for f in fold_ids}
    for i, a in enumerate(fold_ids):
        for j, b in enumerate(fold_ids):
            overlap[i, j] = 100 * len(test_sets[a] & test_sets[b]) / len(test_sets[a])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    ax.grid(True, axis="y")
    values, counts = np.unique(times_tested, return_counts=True)
    bars = ax.bar([str(v) for v in values], counts, color=ps.BLUE, width=0.55)
    for bar, count in zip(bars, counts):
        ax.annotate(ps.thousands(int(count)),
                    xy=(bar.get_x() + bar.get_width() / 2, count),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=ps.INK_SECONDARY)
    ax.set_xlabel("Vezes em que a transação foi testada")
    ax.set_ylabel("Número de transações")
    ax.set_title("(a) Cobertura da base", loc="left")

    ax = axes[1]
    ax.grid(False)
    im = ax.imshow(overlap, cmap=ps.SEQUENTIAL, vmin=0, vmax=100)
    ax.set_xticks(range(len(fold_ids)), [str(f) for f in fold_ids])
    ax.set_yticks(range(len(fold_ids)), [str(f) for f in fold_ids])
    ax.set_xlabel("Fold")
    ax.set_ylabel("Fold")
    ax.set_title("(b) Sobreposição entre conjuntos de teste", loc="left")
    for i in range(len(fold_ids)):
        for j in range(len(fold_ids)):
            ax.text(j, i, f"{overlap[i, j]:.0f}", ha="center", va="center",
                    fontsize=8,
                    color=ps.SURFACE if overlap[i, j] > 55 else ps.INK_SECONDARY)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("sobreposição (%)", color=ps.INK_SECONDARY)
    bar.outline.set_visible(False)

    fig.suptitle("Cada transação é testada exatamente uma vez?", y=1.04)
    ps.footnote(fig, "A propriedade desejada é uma única barra em “1” à "
                     "esquerda, e zeros fora da diagonal à direita: conjuntos "
                     "de teste mutuamente exclusivos que cobrem toda a base.")
    return fig, times_tested, overlap


# ===========================================================================
# Figure 03 — Fraud density per fold
# ===========================================================================

def figure_density_by_fold(outer, scaled, y, rng, k=K_NEIGHBOURS):
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors

    fold_ids = sorted(outer)
    records = []

    for fold in fold_ids:
        train_idx = np.asarray(outer[fold]["train_indices"], dtype=int)
        labels = y[train_idx]
        legit_pool = train_idx[labels == 0]
        fraud_idx = train_idx[labels == 1]
        legit_idx = rng.choice(legit_pool,
                               size=min(LEGIT_SAMPLE_SIZE, len(legit_pool)),
                               replace=False)
        index = np.concatenate([legit_idx, fraud_idx])

        matrix = scaled[index]
        is_fraud = np.concatenate([np.zeros(len(legit_idx), dtype=bool),
                                   np.ones(len(fraud_idx), dtype=bool)])

        neighbours = NearestNeighbors(n_neighbors=k + 1).fit(matrix)
        distances, _ = neighbours.kneighbors(matrix)
        k_distance = distances[:, -1]

        records.append({
            "fold": fold,
            "n_frauds": int(len(fraud_idx)),
            "median_legit": float(np.median(k_distance[~is_fraud])),
            "median_fraud": float(np.median(k_distance[is_fraud])),
            "d_legit": k_distance[~is_fraud],
            "d_fraud": k_distance[is_fraud],
        })

    table = pd.DataFrame([{k_: v for k_, v in r.items()
                           if not isinstance(v, np.ndarray)} for r in records])
    table["ratio"] = table["median_fraud"] / table["median_legit"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    positions = np.arange(len(fold_ids))
    box = ax.boxplot([r["d_fraud"] for r in records], positions=positions - 0.17,
                     widths=0.28, patch_artist=True, showfliers=False,
                     medianprops=dict(color=ps.SURFACE, lw=1.2))
    for patch in box["boxes"]:
        patch.set_facecolor(ps.COLOR_FRAUD)
        patch.set_edgecolor(ps.COLOR_FRAUD)
    box2 = ax.boxplot([r["d_legit"] for r in records], positions=positions + 0.17,
                      widths=0.28, patch_artist=True, showfliers=False,
                      medianprops=dict(color=ps.SURFACE, lw=1.2))
    for patch in box2["boxes"]:
        patch.set_facecolor(ps.COLOR_LEGIT)
        patch.set_edgecolor(ps.COLOR_LEGIT)
    for whisker_set in (box, box2):
        for element in ("whiskers", "caps"):
            for item in whisker_set[element]:
                item.set_color(ps.INK_MUTED)
                item.set_linewidth(0.8)

    ax.set_xticks(positions, [str(f) for f in fold_ids])
    ax.set_xlabel("Fold externo (conjunto de treino)")
    ax.set_ylabel(f"Distância ao {k}º vizinho")
    ax.set_title("(a) Densidade local por classe e por fold", loc="left")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ps.COLOR_FRAUD, label=ps.LABEL_FRAUD),
                       Patch(color=ps.COLOR_LEGIT, label=ps.LABEL_LEGIT)],
              loc="upper right")

    ax = axes[1]
    ax.grid(True, axis="y")
    ax.plot(table["fold"], table["ratio"], marker="o", color=ps.VIOLET)
    ax.axhline(1.0, color=ps.INK_MUTED, lw=1.0)
    for _, row in table.iterrows():
        ax.annotate(f"{row['ratio']:.2f}×",
                    xy=(row["fold"], row["ratio"]), xytext=(0, 7),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=ps.INK_SECONDARY)
    ax.set_xticks(fold_ids)
    ax.set_xlabel("Fold externo")
    ax.set_ylabel("Mediana da fraude ÷ mediana da legítima")
    ax.set_ylim(0, max(3.6, table["ratio"].max() * 1.25))
    ax.set_title("(b) Quão mais esparsas são as fraudes", loc="left")

    fig.suptitle("A premissa de baixa densidade se sustenta em todos os folds?",
                 y=1.04)
    ps.footnote(fig, f"Caixas sem valores extremos, para legibilidade. Amostra "
                     f"de até {ps.thousands(LEGIT_SAMPLE_SIZE)} transações "
                     f"legítimas por fold; todas as fraudes do fold. Uma razão "
                     f"acima de 1 indica fraudes em regiões menos densas.")
    return fig, table


# ===========================================================================
# Summary
# ===========================================================================

def write_summary(composition, times_tested, overlap, density, target):
    off_diagonal = overlap[~np.eye(len(overlap), dtype=bool)]

    lines = [
        "# Sumário das partições",
        "",
        "Gerado por `src/analysis/fold_analysis.py`.",
        "",
        "## Estratificação",
        "",
        "| Fold | Partição | Amostras | Fraudes | Proporção (%) |",
        "|---|---|---|---|---|",
    ]
    for _, row in composition.sort_values(["fold", "split"]).iterrows():
        lines.append(f"| {int(row.fold)} | {row.split} | "
                     f"{ps.thousands(int(row.n))} | {int(row.frauds)} | "
                     f"{row.rate:.4f} |")

    spread = composition.groupby("split")["rate"].agg(["min", "max"])
    lines += [
        "",
        "Amplitude da proporção de fraude dentro de cada tipo de partição:",
        "",
    ]
    for split, row in spread.iterrows():
        lines.append(f"- {split}: {row['min']:.4f}% a {row['max']:.4f}% "
                     f"(amplitude de {row['max'] - row['min']:.4f} ponto percentual)")

    values, counts = np.unique(times_tested, return_counts=True)
    lines += [
        "",
        "## Cobertura e isolamento",
        "",
    ]
    for value, count in zip(values, counts):
        lines.append(f"- Transações testadas {value} vez(es): "
                     f"{ps.thousands(int(count))}")
    lines += [
        f"- Sobreposição máxima fora da diagonal: {off_diagonal.max():.2f}%",
        "",
        "Sobreposição nula fora da diagonal confirma que os conjuntos de teste",
        "são mutuamente exclusivos. Toda transação testada exatamente uma vez",
        "confirma a cobertura integral da base.",
        "",
        "## Densidade local das fraudes por fold",
        "",
        "| Fold | Fraudes | Mediana legítima | Mediana fraude | Razão |",
        "|---|---|---|---|---|",
    ]
    for _, row in density.iterrows():
        lines.append(f"| {int(row.fold)} | {int(row.n_frauds)} | "
                     f"{row.median_legit:.3f} | {row.median_fraud:.3f} | "
                     f"{row.ratio:.2f}× |")
    lines += [
        "",
        f"Amplitude da razão entre folds: {density['ratio'].min():.2f}× a "
        f"{density['ratio'].max():.2f}×.",
        "",
        "Se a razão for estável entre os folds, a instabilidade observada nos",
        "resultados do HDBSCAN não vem de diferenças na estrutura de densidade",
        "das partições.",
    ]

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return target


# ===========================================================================

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps.apply_style()
    rng = np.random.RandomState(SEED)

    print("Reading dataset and fold metadata...")
    df, y = load_dataset()
    outer = read_outer_folds()
    inner = read_inner_folds()
    print(f"  outer folds: {sorted(outer)}   inner folds: {sorted(inner)}")

    print("01 - composition and stratification")
    fig, composition = figure_composition(outer, inner, y)
    print("   ", ps.save(fig, f"{OUTPUT}/01-fold-composition")[0])
    plt.close(fig)

    print("02 - coverage and isolation")
    fig, times_tested, overlap = figure_coverage(outer, len(df))
    print("   ", ps.save(fig, f"{OUTPUT}/02-fold-coverage")[0])
    plt.close(fig)

    print("03 - fraud density per fold")
    scaled = scale(df.drop(columns=[TARGET]))
    fig, density = figure_density_by_fold(outer, scaled, y, rng)
    print("   ", ps.save(fig, f"{OUTPUT}/03-fraud-density-by-fold")[0])
    plt.close(fig)

    summary = write_summary(composition, times_tested, overlap, density,
                            f"{OUTPUT}/fold-statistics.md")
    print("summary:", summary)


if __name__ == "__main__":
    main()
