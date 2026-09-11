"""
Variation of the HDBSCAN grid-search metrics across folds.

Replaces the earlier AUC-only figure. Three questions are addressed:

  1. How much does each metric move between folds for a fixed configuration?
     Instability is the central finding of this work and deserves a figure that
     makes it legible rather than one that hides it in overlapping panels.
  2. Which configurations are stable and which are not?
  3. Does DBCV, the unsupervised selection criterion, actually predict
     supervised performance? If it does not, the whole selection procedure
     rests on a metric that is disconnected from the outcome.

Run from the repository root:

    python src/analysis/hdbscan_metrics.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # noqa: E402

RESULTS = "results/unsupervised/hdbscan_grid_results_by_fold.csv"
OUTPUT = "results/figures/hdbscan"

# Metric key and label.
#
# ROC AUC is deliberately excluded. At 0.172% positives it is dominated by how
# well the negatives are ordered and reads far more favourably than the model
# deserves; AUPRC and precision are the metrics this work reports.
METRICS = [
    ("dbcv_score", "DBCV", True),
    ("auprc", "AUPRC", True),
    ("precision", "Precisão", True),
    ("f1", "F1", True),
    ("recall", "Revocação", True),
]


def load():
    df = pd.read_csv(RESULTS)
    df["config"] = ("mcs " + df["min_cluster_size"].astype(int).astype(str)
                    + " | ms " + df["min_samples"].astype(int).astype(str))
    return df


def pivot(df, metric):
    table = df.pivot_table(index=["min_cluster_size", "min_samples"],
                           columns="fold", values=metric, aggfunc="mean")
    return table.sort_index()


def config_labels(table):
    return [f"mcs {int(a)} | ms {int(b)}" for a, b in table.index]


# ===========================================================================
# Figure 01 — Metric stability heatmaps
# ===========================================================================

def figure_stability(df):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(METRICS),
                             figsize=(3.0 * len(METRICS), 8.4),
                             sharey=True)

    for position, (ax, (key, label, _)) in enumerate(zip(axes, METRICS)):
        table = pivot(df, key)
        matrix = table.to_numpy()
        ax.grid(False)
        im = ax.imshow(matrix, cmap=ps.SEQUENTIAL, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(range(matrix.shape[1]),
                      [str(int(c)) for c in table.columns])
        ax.set_xlabel("Fold")
        ax.set_title(label, loc="left")

        if position == 0:
            ax.set_yticks(range(matrix.shape[0]), config_labels(table),
                          fontsize=7)
            ax.set_ylabel("Configuração")

        bar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.03)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7)

    fig.suptitle("Variação de cada métrica entre os folds internos", y=1.015)
    ps.footnote(fig, "Cada linha é uma combinação de hiperparâmetros; cada "
                     "coluna, um fold. Uma linha de tom uniforme indica "
                     "configuração estável; variação de tom dentro da mesma "
                     "linha indica sensibilidade à partição. As escalas de cor "
                     "são independentes entre métricas.")
    return fig


# ===========================================================================
# Figure 02 — Spread per configuration
# ===========================================================================

def figure_spread(df, key="dbcv_score", label="DBCV"):
    import matplotlib.pyplot as plt

    grouped = df.groupby(["min_cluster_size", "min_samples"])[key]
    summary = grouped.agg(["min", "max", "median", "mean", "std"]).reset_index()
    summary["spread"] = summary["max"] - summary["min"]
    summary = summary.sort_values("spread").reset_index(drop=True)
    labels = [f"mcs {int(a)} | ms {int(b)}"
              for a, b in zip(summary["min_cluster_size"], summary["min_samples"])]

    fig, ax = plt.subplots(figsize=(8.0, 0.30 * len(summary) + 2.0))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    positions = np.arange(len(summary))
    ax.hlines(positions, summary["min"], summary["max"],
              color=ps.GRID, lw=3.2, zorder=1)
    ax.scatter(summary["min"], positions, s=34, color=ps.BLUE, zorder=3,
               edgecolors=ps.SURFACE, linewidths=0.8, label="mínimo entre folds")
    ax.scatter(summary["max"], positions, s=34, color=ps.ORANGE, zorder=3,
               edgecolors=ps.SURFACE, linewidths=0.8, label="máximo entre folds")
    ax.scatter(summary["median"], positions, s=16, color=ps.INK_SECONDARY,
               zorder=4, marker="|", linewidths=1.6, label="mediana")

    ax.set_yticks(positions, labels, fontsize=7.5)
    # Configurations whose spread exceeds twice the median are worth naming:
    # the instability of this pipeline is concentrated, not diffuse.
    threshold = 2 * summary["spread"].median()
    for tick, spread_value in zip(ax.get_yticklabels(), summary["spread"]):
        if spread_value > threshold:
            tick.set_color(ps.ORANGE)
            tick.set_fontweight("bold")
    ax.set_xlabel(label)
    ax.set_title(f"Amplitude do {label} entre os cinco folds, por configuração",
                 pad=10)
    ax.legend(loc="lower right")

    worst = summary.iloc[-1]
    ax.annotate(f"amplitude de {worst['spread']:.3f}",
                xy=(worst["max"], len(summary) - 1), xytext=(-6, 12),
                textcoords="offset points", ha="right", fontsize=8,
                color=ps.INK_SECONDARY)

    ps.footnote(fig, "Ordenado da configuração menos estável, no topo, para a "
                     "mais estável, na base. Uma barra longa significa que os "
                     "mesmos hiperparâmetros produzem resultados muito "
                     "diferentes em partições que diferem em apenas um quinto "
                     "dos dados. Em destaque, as configurações cuja amplitude "
                     "supera o dobro da mediana.")
    return fig, summary


# ===========================================================================
# Figure 03 — Does DBCV predict performance?
# ===========================================================================

def figure_criterion(df):
    import matplotlib.pyplot as plt
    from scipy import stats

    pairs = [("auprc", "AUPRC"), ("precision", "Precisão"), ("f1", "F1")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.0 * len(pairs), 4.4))

    # The DBCV values are bimodal: a dense group below 0.2 and a detached group
    # near 0.96, with nothing in between. A single correlation over both groups
    # mostly measures the gap, so the within-group coefficient is reported too.
    cut = 0.5
    low = df[df["dbcv_score"] < cut]

    correlations = {}
    for ax, (key, label) in zip(axes, pairs):
        ax.grid(True, axis="both")
        ax.scatter(df["dbcv_score"], df[key], s=22, color=ps.BLUE,
                   alpha=0.6, edgecolors=ps.SURFACE, linewidths=0.5)

        rho, p_value = stats.spearmanr(df["dbcv_score"], df[key])
        rho_low, p_low = stats.spearmanr(low["dbcv_score"], low[key])
        correlations[key] = {"rho_all": rho, "p_all": p_value,
                             "rho_low": rho_low, "p_low": p_low}

        ax.set_xlabel("DBCV (critério de seleção, não supervisionado)")
        ax.set_ylabel(label)
        ax.set_title(f"{label}", loc="left")
        ax.annotate(f"ρ global = {rho:+.2f}\nρ no regime inferior = {rho_low:+.2f}",
                    xy=(0.03, 0.97), xycoords="axes fraction",
                    va="top", fontsize=8, color=ps.INK_SECONDARY)

    fig.suptitle("O DBCV prediz o desempenho supervisionado?", y=1.05)
    ps.footnote(fig,
                f"Cada ponto é uma combinação de hiperparâmetros avaliada em um "
                f"fold. O DBCV é calculado sem consultar os rótulos; as métricas "
                f"do eixo vertical os utilizam. A distribuição do DBCV é bimodal "
                f"— {len(low)} das {len(df)} execuções abaixo de {cut}, o "
                f"restante acima de 0,95, sem nada entre os dois — de modo que o "
                f"coeficiente global mede sobretudo a distância entre os dois "
                f"regimes. O coeficiente calculado apenas no regime inferior "
                f"mostra se há ordenação dentro do grupo.")
    return fig, correlations


# ===========================================================================
# Summary
# ===========================================================================

def write_summary(df, spread, correlations, target):
    lines = [
        "# Variação das métricas do HDBSCAN",
        "",
        "Gerado por `src/analysis/hdbscan_metrics.py`.",
        "",
        f"- Combinações avaliadas: "
        f"{df.groupby(['min_cluster_size', 'min_samples']).ngroups}",
        f"- Folds: {sorted(df['fold'].unique().tolist())}",
        f"- Execuções registradas: {len(df)}",
        "",
        "## Amplitude por métrica",
        "",
        "Medida dentro de cada configuração, entre os cinco folds, e depois",
        "resumida pelo maior e pelo menor caso.",
        "",
        "| Métrica | Menor amplitude | Maior amplitude | Amplitude mediana |",
        "|---|---|---|---|",
    ]
    for key, label, _ in METRICS:
        grouped = df.groupby(["min_cluster_size", "min_samples"])[key]
        ranges = grouped.max() - grouped.min()
        lines.append(f"| {label} | {ranges.min():.4f} | {ranges.max():.4f} | "
                     f"{ranges.median():.4f} |")

    worst = spread.iloc[-1]
    best = spread.iloc[0]
    lines += [
        "",
        "## Configurações extremas em DBCV",
        "",
        f"- Mais estável: `min_cluster_size={int(best.min_cluster_size)}`, "
        f"`min_samples={int(best.min_samples)}` — DBCV de {best['min']:.4f} a "
        f"{best['max']:.4f} (amplitude {best['spread']:.4f})",
        f"- Menos estável: `min_cluster_size={int(worst.min_cluster_size)}`, "
        f"`min_samples={int(worst.min_samples)}` — DBCV de {worst['min']:.4f} a "
        f"{worst['max']:.4f} (amplitude {worst['spread']:.4f})",
        "",
        "## O DBCV prediz o desempenho?",
        "",
        "| Métrica | ρ global | p | ρ no regime inferior | p |",
        "|---|---|---|---|---|",
    ]
    for key, values in correlations.items():
        lines.append(f"| {key} | {values['rho_all']:+.4f} | "
                     f"{values['p_all']:.3e} | {values['rho_low']:+.4f} | "
                     f"{values['p_low']:.3e} |")

    lines += [
        "",
        "**Atenção à interpretação.** A distribuição do DBCV é bimodal: um",
        "grupo denso abaixo de 0,2 e um grupo destacado acima de 0,95, sem",
        "nenhuma observação entre os dois. O coeficiente global mede",
        "sobretudo a distância entre esses dois regimes, não uma ordenação",
        "gradual. A leitura defensável é que o DBCV **distingue corretamente",
        "os dois regimes** — e o regime que ele premia é o de melhor",
        "desempenho supervisionado. Afirmar que o DBCV ordena configurações",
        "por qualidade exige olhar o coeficiente calculado dentro do regime",
        "inferior, apresentado na última coluna.",
        "",
        "## Seleção: máximo contra média",
        "",
    ]

    grouped = df.groupby(["min_cluster_size", "min_samples"])["dbcv_score"]
    by_max = grouped.max().idxmax()
    by_mean = grouped.mean().idxmax()
    lines += [
        f"- Vencedora pelo **máximo** entre folds: "
        f"`min_cluster_size={int(by_max[0])}`, `min_samples={int(by_max[1])}`",
        f"- Vencedora pelo **DBCV médio** entre folds: "
        f"`min_cluster_size={int(by_mean[0])}`, `min_samples={int(by_mean[1])}`",
        "",
        ("As duas coincidem." if by_max == by_mean else
         "**As duas divergem.** A escolha do critério muda a configuração "
         "levada à avaliação externa, e essa divergência é em si um resultado "
         "sobre a estabilidade do algoritmo."),
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

    print("Reading HDBSCAN grid results...")
    df = load()
    print(f"  {len(df)} runs, "
          f"{df.groupby(['min_cluster_size', 'min_samples']).ngroups} configurations")

    print("01 - metric stability")
    fig = figure_stability(df)
    print("   ", ps.save(fig, f"{OUTPUT}/01-metric-stability")[0])
    plt.close(fig)

    print("02 - DBCV spread per configuration")
    fig, spread = figure_spread(df)
    print("   ", ps.save(fig, f"{OUTPUT}/02-dbcv-spread-by-config")[0])
    plt.close(fig)

    print("03 - DBCV against supervised performance")
    fig, correlations = figure_criterion(df)
    print("   ", ps.save(fig, f"{OUTPUT}/03-dbcv-vs-performance")[0])
    plt.close(fig)

    summary = write_summary(df, spread, correlations,
                            f"{OUTPUT}/hdbscan-metric-variation.md")
    print("summary:", summary)


if __name__ == "__main__":
    main()
