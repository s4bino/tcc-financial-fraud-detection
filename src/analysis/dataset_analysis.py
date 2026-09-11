"""
Exploratory analysis of the transaction dataset.

Produces the figures and the statistical summary that describe how fraud
behaves: correlation structure, per-feature discriminative power, relevance of
the Time variable, projections onto the most informative components, and the
local density of each class.

The last analysis is the one that tests the premise of this work. If fraud is
an anomaly, fraudulent transactions should sit in lower-density regions than
legitimate ones, and that is measurable through the distance to the nearest
neighbours.

Run from the repository root:

    python src/analysis/dataset_analysis.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # noqa: E402

CSV = "data/raw/creditcard.csv"
OUTPUT = "results/figures/dataset"
TARGET = "Class"
SEED = 42

# Majority-class sample used in the scatter and density figures. All 492 frauds
# are always kept.
LEGIT_SAMPLE_SIZE = 20000
K_NEIGHBOURS = 20


# ===========================================================================
# Preparation
# ===========================================================================

def load():
    df = pd.read_csv(CSV)
    y = df[TARGET].to_numpy()
    X = df.drop(columns=[TARGET])
    return df, X, y


def scale(X):
    """Same preprocessing as the pipeline: RobustScaler on Amount, Z-score on
    Time, PCA components untouched."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler, StandardScaler

    pre = ColumnTransformer(
        transformers=[
            ("rob", RobustScaler(), ["Amount"]),
            ("std", StandardScaler(), ["Time"]),
        ],
        remainder="passthrough",
    )
    scaled = pre.fit_transform(X)
    names = ["Amount", "Time"] + [c for c in X.columns if c not in ("Amount", "Time")]
    return pd.DataFrame(scaled, columns=names)


def discriminative_power(X, y):
    """Average precision (AUPRC) of each feature taken alone as a fraud ranker.

    ROC AUC is not used anywhere in this work. With 0.172% positives it is
    dominated by how well the negatives are ordered, which is not the question:
    what matters is the precision achieved at the top of the ranking, and that
    is what average precision measures.

    Each feature is scored in both directions and the stronger one is kept, so
    the measure reflects the strength of the separation regardless of sign.
    The lift is the AUPRC divided by the prevalence — the AUPRC a random
    ranker would obtain — so a lift of 1 means no information at all.
    """
    from sklearn.metrics import average_precision_score

    prevalence = float(y.mean())
    rows = []
    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        ascending = average_precision_score(y, values)
        descending = average_precision_score(y, -values)
        rows.append({
            "feature": col,
            "auprc": max(ascending, descending),
            "lift": max(ascending, descending) / prevalence,
            "direction": ("maior em fraude" if ascending >= descending
                          else "menor em fraude"),
        })
    table = pd.DataFrame(rows).sort_values("auprc", ascending=False)
    return table.reset_index(drop=True), prevalence


# ===========================================================================
# Figure 01 — Correlation matrix
# ===========================================================================

def figure_correlation(df):
    import matplotlib.pyplot as plt

    corr = df.corr()
    order = list(df.columns)
    matrix = corr.loc[order, order].to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    ax.grid(False)
    im = ax.imshow(matrix, cmap=ps.DIVERGING, vmin=-1, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6.5)
    ax.set_yticklabels(order, fontsize=6.5)
    ax.set_title("Correlação de Pearson entre as variáveis", pad=12)

    # Highlight the three variables that did not go through PCA.
    for label in ("Time", "Amount", "Class"):
        i = order.index(label)
        ax.axhline(i, color=ps.INK, lw=0.5, alpha=0.35)
        ax.axvline(i, color=ps.INK, lw=0.5, alpha=0.35)

    bar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02,
                       ticks=[-1, -0.5, 0, 0.5, 1])
    bar.set_label("coeficiente de correlação", color=ps.INK_SECONDARY)
    bar.outline.set_visible(False)

    ps.footnote(fig, "As linhas destacam Time, Amount e Class, as únicas "
                     "variáveis que não resultam do PCA. V1–V28 são ortogonais "
                     "entre si por construção, o que explica o bloco central "
                     "sem correlação.")
    return fig, corr


# ===========================================================================
# Figure 02 — Discriminative power per feature
# ===========================================================================

def figure_discriminative(table, prevalence):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ordered = table.sort_values("lift")
    colors = [ps.ORANGE if f in ("Time", "Amount") else ps.BLUE
              for f in ordered["feature"]]

    fig, ax = plt.subplots(figsize=(7.4, 8.2))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.barh(ordered["feature"], ordered["lift"], color=colors, height=0.68)

    # Three orders of magnitude separate the weakest feature from the strongest,
    # so the axis is logarithmic and anchored at the random-ranker baseline.
    ax.set_xscale("log")
    ax.set_xlim(1.0, ordered["lift"].max() * 1.6)
    ax.axvline(1.0, color=ps.AXIS, lw=0.9)
    ax.set_xlabel("Lift do AUPRC sobre o acaso (escala logarítmica)")
    ax.set_title("Poder discriminante de cada variável, aplicada sozinha", pad=10)

    for feature in ("Time", "Amount"):
        row = ordered[ordered["feature"] == feature].iloc[0]
        ax.text(row["lift"] * 1.08, list(ordered["feature"]).index(feature),
                f"{row['lift']:.1f}×", va="center", fontsize=8,
                color=ps.INK_SECONDARY)
    best = ordered.iloc[-1]
    ax.text(best["lift"] * 1.08, len(ordered) - 1,
            f"{best['lift']:.0f}×  (AUPRC {best['auprc']:.3f})",
            va="center", fontsize=8, color=ps.INK_SECONDARY)

    ax.legend(handles=[Patch(color=ps.BLUE, label="Componente do PCA (V1–V28)"),
                       Patch(color=ps.ORANGE, label="Variável original")],
              loc="lower right")

    ps.footnote(fig, f"Mede-se o AUPRC, não o AUC ROC: com "
                     f"{100 * prevalence:.3f}% de fraudes, o AUC ROC é dominado "
                     f"pela ordenação dos negativos e superestima a separação. "
                     f"O lift é o AUPRC dividido pela prevalência, isto é, pelo "
                     f"AUPRC de {prevalence:.6f} que um ordenador aleatório "
                     f"obteria. Lift 1 significa nenhuma informação.")
    return fig


# ===========================================================================
# Figure 03 — Is the Time variable relevant?
# ===========================================================================

def figure_time(df, y):
    import matplotlib.pyplot as plt
    from scipy import stats

    hours = df["Time"].to_numpy() / 3600.0
    hour_of_day = hours % 24
    legit, fraud = y == 0, y == 1

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    bins = np.linspace(0, hours.max(), 97)
    for mask, color, label in ((legit, ps.COLOR_LEGIT, ps.LABEL_LEGIT),
                               (fraud, ps.COLOR_FRAUD, ps.LABEL_FRAUD)):
        counts, _ = np.histogram(hours[mask], bins=bins)
        ax.plot(bins[:-1], counts / counts.sum(), color=color, label=label)
    ax.set_xlabel("Horas desde a primeira transação")
    ax.set_ylabel("Proporção da própria classe")
    ax.set_title("(a) Distribuição ao longo da coleta", loc="left")
    ax.legend()

    ax = axes[1]
    bins = np.arange(0, 25, 1)
    for mask, color, label in ((legit, ps.COLOR_LEGIT, ps.LABEL_LEGIT),
                               (fraud, ps.COLOR_FRAUD, ps.LABEL_FRAUD)):
        counts, _ = np.histogram(hour_of_day[mask], bins=bins)
        ax.step(bins[:-1], counts / counts.sum(), where="post",
                color=color, label=label)
    ax.set_xlabel("Hora do dia (Time convertido, módulo 24 h)")
    ax.set_ylabel("Proporção da própria classe")
    ax.set_title("(b) Perfil horário", loc="left")
    ax.set_xticks(range(0, 25, 4))
    ax.legend()

    # The rate per band is what the two curves above imply but do not state:
    # a weak ranker can still multiply the prior odds of fraud several times.
    ax = axes[2]
    ax.grid(True, axis="y")
    band = (hour_of_day // 3).astype(int) * 3
    bands = np.arange(0, 24, 3)
    rates = np.array([10000 * y[band == b].mean() for b in bands])
    colours = [ps.ORANGE if r >= rates.mean() else ps.BLUE for r in rates]
    ax.bar([f"{b:02d}–{b + 3:02d}" for b in bands], rates,
           color=colours, width=0.66)
    ax.axhline(10000 * y.mean(), color=ps.INK_MUTED, lw=1.0)
    ax.annotate(f"média da base: {10000 * y.mean():.1f}",
                xy=(len(bands) - 0.5, 10000 * y.mean()), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=7.5,
                color=ps.INK_MUTED)
    ax.set_xlabel("Faixa horária")
    ax.set_ylabel("Fraudes por 10.000 transações")
    ax.set_title("(c) Risco por faixa horária", loc="left")
    ax.tick_params(axis="x", labelrotation=45)

    ks = stats.ks_2samp(hour_of_day[fraud], hour_of_day[legit])
    ratio = rates.max() / rates.min()
    fig.suptitle("A variável Time separa fraude de transação legítima?", y=1.04)
    ps.footnote(fig, f"Kolmogorov–Smirnov sobre a hora do dia: "
                     f"D = {ks.statistic:.3f}, p = {ks.pvalue:.2e}. Em (a) e "
                     f"(b) as curvas são normalizadas dentro de cada classe, "
                     f"pois as 492 fraudes não seriam visíveis na mesma escala "
                     f"das 284.315 legítimas. O painel (c) mostra o que as "
                     f"curvas implicam mas não declaram: a faixa de maior risco "
                     f"concentra {ratio:.1f} vezes mais fraude por transação "
                     f"que a de menor risco.")
    return fig, ks


# ===========================================================================
# Figure 04 — Class-conditional distributions
# ===========================================================================

def figure_distributions(scaled, y, table, top_n=6):
    import matplotlib.pyplot as plt

    top = list(table.head(top_n)["feature"])
    legit, fraud = y == 0, y == 1

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0))
    for ax, feature in zip(axes.ravel(), top):
        values_legit = scaled[feature][legit]
        values_fraud = scaled[feature][fraud]
        low = min(np.percentile(values_legit, 0.1), values_fraud.min())
        high = max(np.percentile(values_legit, 99.9), values_fraud.max())
        bins = np.linspace(low, high, 70)

        for values, color, label in ((values_legit, ps.COLOR_LEGIT, ps.LABEL_LEGIT),
                                     (values_fraud, ps.COLOR_FRAUD, ps.LABEL_FRAUD)):
            counts, _ = np.histogram(values, bins=bins)
            ax.plot(bins[:-1], counts / max(counts.sum(), 1),
                    color=color, label=label, lw=1.4)

        row = table[table["feature"] == feature].iloc[0]
        ax.set_title(f"{feature}   (AUPRC {row['auprc']:.3f}, "
                     f"lift {row['lift']:.0f}×)", loc="left")
        ax.set_ylabel("densidade")

    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Distribuição das seis variáveis mais discriminantes, por classe",
                 y=1.03)
    ps.footnote(fig, "Cada curva é normalizada dentro da própria classe. Valores "
                     "escalonados pelo mesmo pré-processamento do pipeline.")
    return fig


# ===========================================================================
# Figure 05 — Projections
# ===========================================================================

def figure_projections(scaled, y, table, rng):
    import matplotlib.pyplot as plt

    first, second, third = list(table.head(3)["feature"])
    legit_idx = np.where(y == 0)[0]
    legit_idx = rng.choice(legit_idx,
                           size=min(LEGIT_SAMPLE_SIZE, len(legit_idx)),
                           replace=False)
    fraud_idx = np.where(y == 1)[0]

    fig = plt.figure(figsize=(12.0, 4.4))

    def scatter(ax, x_name, y_name):
        # rasterized=True keeps the point cloud as an image inside the PDF while
        # text and axes stay vector. Without it a single figure with 20.000
        # markers per panel produces a file of tens of megabytes.
        ax.grid(True, axis="both")
        ax.scatter(scaled[x_name].to_numpy()[legit_idx],
                   scaled[y_name].to_numpy()[legit_idx],
                   s=3, color=ps.COLOR_LEGIT, alpha=0.22, linewidths=0,
                   rasterized=True,
                   label=f"{ps.LABEL_LEGIT} (amostra de {ps.thousands(len(legit_idx))})")
        ax.scatter(scaled[x_name].to_numpy()[fraud_idx],
                   scaled[y_name].to_numpy()[fraud_idx],
                   s=16, color=ps.COLOR_FRAUD, alpha=0.95,
                   edgecolors=ps.SURFACE, linewidths=0.6, rasterized=True,
                   label=f"{ps.LABEL_FRAUD} (todas as {len(fraud_idx)})")
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

    ax1 = fig.add_subplot(1, 3, 1)
    scatter(ax1, first, second)
    ax1.set_title(f"(a) {first} × {second}", loc="left")

    ax2 = fig.add_subplot(1, 3, 2)
    scatter(ax2, first, third)
    ax2.set_title(f"(b) {first} × {third}", loc="left")
    ax2.legend(loc="upper left", markerscale=2.2)

    ax3 = fig.add_subplot(1, 3, 3)
    scatter(ax3, second, third)
    ax3.set_title(f"(c) {second} × {third}", loc="left")

    fig.suptitle("Fraudes projetadas nas três componentes mais discriminantes",
                 y=1.04)
    ps.footnote(fig, "As componentes são escolhidas pelo AUPRC individual, não "
                     "pela variância. V1–V28 já resultam de PCA, e a ordem do "
                     "PCA reflete variância total, que não coincide com poder "
                     "de separação entre as classes.")
    return fig, (first, second, third)


# ===========================================================================
# Figure 06 — Three-dimensional view
# ===========================================================================

def figure_projection_3d(scaled, y, table, rng):
    """Four viewpoints of the same three-dimensional cloud.

    A single viewpoint hides structure through occlusion; rotating the same
    cloud shows whether the fraud region is genuinely detached or only appears
    so from one angle.
    """
    import matplotlib.pyplot as plt

    first, second, third = list(table.head(3)["feature"])
    legit_idx = np.where(y == 0)[0]
    legit_idx = rng.choice(legit_idx,
                           size=min(LEGIT_SAMPLE_SIZE, len(legit_idx)),
                           replace=False)
    fraud_idx = np.where(y == 1)[0]

    x_legit = scaled[first].to_numpy()[legit_idx]
    y_legit = scaled[second].to_numpy()[legit_idx]
    z_legit = scaled[third].to_numpy()[legit_idx]
    x_fraud = scaled[first].to_numpy()[fraud_idx]
    y_fraud = scaled[second].to_numpy()[fraud_idx]
    z_fraud = scaled[third].to_numpy()[fraud_idx]

    views = [(18, -60), (18, 30), (62, -60), (6, -95)]
    fig = plt.figure(figsize=(11.5, 10.0))

    for position, (elevation, azimuth) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, position, projection="3d")
        ax.set_facecolor(ps.SURFACE)
        # See the note in figure_projections: four panels of 20.000 vector
        # markers each would make this the heaviest file in the repository.
        ax.scatter(x_legit, y_legit, z_legit, s=2.4,
                   color=ps.COLOR_LEGIT, alpha=0.20, linewidths=0,
                   rasterized=True, label=ps.LABEL_LEGIT)
        ax.scatter(x_fraud, y_fraud, z_fraud, s=13,
                   color=ps.COLOR_FRAUD, alpha=0.9,
                   edgecolors=ps.SURFACE, linewidths=0.4,
                   rasterized=True, label=ps.LABEL_FRAUD)
        ax.set_xlabel(first, labelpad=2)
        ax.set_ylabel(second, labelpad=2)
        ax.set_zlabel(third, labelpad=2)
        ax.tick_params(labelsize=6.5, pad=1)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_title(f"elevação {elevation}°, azimute {azimuth}°",
                     loc="left", fontsize=9, color=ps.INK_SECONDARY)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor(ps.SURFACE)
            pane.pane.set_edgecolor(ps.GRID)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2,
               markerscale=2.4, bbox_to_anchor=(0.5, -0.015))

    fig.suptitle(f"Estrutura tridimensional em {first} × {second} × {third}",
                 y=1.02)
    fig.text(0.0, -0.055, f"O mesmo conjunto de pontos visto de quatro ângulos. "
                          f"Amostra de {ps.thousands(len(legit_idx))} transações "
                          f"legítimas e todas as {len(fraud_idx)} fraudes.",
             ha="left", va="top", fontsize=7.5, color=ps.INK_MUTED)
    return fig


# ===========================================================================
# Figure 07 — Local density: the premise under test
# ===========================================================================

def figure_local_density(scaled, y, rng, k=K_NEIGHBOURS):
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.neighbors import NearestNeighbors

    legit_idx = np.where(y == 0)[0]
    legit_idx = rng.choice(legit_idx,
                           size=min(LEGIT_SAMPLE_SIZE, len(legit_idx)),
                           replace=False)
    fraud_idx = np.where(y == 1)[0]
    index = np.concatenate([legit_idx, fraud_idx])

    matrix = scaled.to_numpy()[index]
    labels = y[index]

    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(matrix)
    distances, _ = neighbours.kneighbors(matrix)
    k_distance = distances[:, -1]

    d_legit, d_fraud = k_distance[labels == 0], k_distance[labels == 1]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Density estimated by kernel rather than histogram: with only 492 frauds a
    # histogram is dominated by binning noise.
    ax = axes[0]
    upper = np.percentile(k_distance, 99.0)
    grid = np.linspace(0, upper, 400)
    for values, color, label in ((d_legit, ps.COLOR_LEGIT, ps.LABEL_LEGIT),
                                 (d_fraud, ps.COLOR_FRAUD, ps.LABEL_FRAUD)):
        kernel = stats.gaussian_kde(values)
        ax.plot(grid, kernel(grid), color=color, label=label)
        ax.fill_between(grid, kernel(grid), color=color, alpha=0.10, linewidth=0)
    ax.axvline(np.median(d_legit), color=ps.COLOR_LEGIT, lw=0.9, alpha=0.6)
    ax.axvline(np.median(d_fraud), color=ps.COLOR_FRAUD, lw=0.9, alpha=0.6)
    ax.set_xlim(0, upper)
    ax.set_xlabel(f"Distância ao {k}º vizinho mais próximo")
    ax.set_ylabel("Densidade estimada por núcleo")
    ax.set_title("(a) Distribuição da densidade local", loc="left")
    ax.legend()

    ax = axes[1]
    ax.grid(True, axis="y")
    quantiles = np.linspace(0, 99, 100)
    ax.plot(quantiles, np.percentile(d_legit, quantiles),
            color=ps.COLOR_LEGIT, label=ps.LABEL_LEGIT)
    ax.plot(quantiles, np.percentile(d_fraud, quantiles),
            color=ps.COLOR_FRAUD, label=ps.LABEL_FRAUD)
    ax.set_xlabel("Percentil (recortado no 99º)")
    ax.set_ylabel(f"Distância ao {k}º vizinho")
    ax.set_title("(b) Curvas de quantis", loc="left")
    ax.legend()

    ks = stats.ks_2samp(d_fraud, d_legit)
    median_legit, median_fraud = np.median(d_legit), np.median(d_fraud)
    ratio = median_fraud / median_legit

    fig.suptitle("Fraudes ocupam regiões de densidade mais baixa?", y=1.04)
    ps.footnote(fig,
                f"Mediana da distância: {median_legit:.2f} para transações "
                f"legítimas e {median_fraud:.2f} para fraudes — razão de "
                f"{ratio:.2f}×. Kolmogorov–Smirnov: D = {ks.statistic:.3f}, "
                f"p = {ks.pvalue:.2e}. Amostra de "
                f"{ps.thousands(len(legit_idx))} legítimas e todas as "
                f"{len(fraud_idx)} fraudes.")
    return fig, {"median_legit": median_legit, "median_fraud": median_fraud,
                 "ratio": ratio, "ks_d": ks.statistic, "ks_p": ks.pvalue}


# ===========================================================================
# Statistical summary
# ===========================================================================

def write_summary(df, y, corr, table, prevalence, ks_time, density, axes, target):
    total = len(df)
    frauds = int(y.sum())
    amount = df["Amount"]
    amount_legit, amount_fraud = amount[y == 0], amount[y == 1]

    class_corr = corr["Class"].drop("Class").abs().sort_values(ascending=False)
    components = [c for c in df.columns if c.startswith("V")]
    corr_components = corr.loc[components, components].to_numpy()
    off_diagonal = corr_components[~np.eye(len(components), dtype=bool)]

    lines = [
        "# Sumário estatístico da base",
        "",
        "Gerado por `src/analysis/dataset_analysis.py`.",
        "",
        "## Composição",
        "",
        f"- Transações: {ps.thousands(total)}",
        f"- Fraudes: {frauds} ({100 * frauds / total:.4f}%)",
        f"- Desbalanceamento: 1 fraude para {(total - frauds) // frauds} legítimas",
        f"- Período coberto: {df['Time'].max() / 3600:.1f} horas",
        "",
        "## Ortogonalidade das componentes do PCA",
        "",
        f"- Correlação absoluta máxima entre V1–V28: {np.abs(off_diagonal).max():.2e}",
        f"- Correlação absoluta média: {np.abs(off_diagonal).mean():.2e}",
        "",
        "Confirma que V1–V28 são mutuamente ortogonais, como se espera de",
        "componentes principais. Nenhum tratamento de multicolinearidade é",
        "necessário entre elas.",
        "",
        "## Métrica adotada",
        "",
        "O poder discriminante é medido por **AUPRC** (precisão média), não por",
        f"AUC ROC. Com {100 * prevalence:.3f}% de fraudes, o AUC ROC é dominado",
        "pela ordenação dos negativos e superestima a separação; o AUPRC mede a",
        "precisão no topo do ranking, que é o que interessa quando apenas uma",
        "fração mínima das transações será acusada.",
        "",
        f"Um ordenador aleatório obtém AUPRC de {prevalence:.6f}. O **lift** é o",
        "AUPRC dividido por esse valor: lift 1 significa nenhuma informação.",
        "",
        "## A variável Time",
        "",
        f"- Correlação com Class: {corr.loc['Time', 'Class']:+.4f}",
        f"- AUPRC isolada: {table[table.feature == 'Time'].auprc.iloc[0]:.4f} "
        f"(lift de {table[table.feature == 'Time'].lift.iloc[0]:.2f}×)",
        f"- Posição no ranking de poder discriminante: "
        f"{1 + list(table.feature).index('Time')} de {len(table)}",
        f"- Kolmogorov–Smirnov sobre a hora do dia: D = {ks_time.statistic:.4f}, "
        f"p = {ks_time.pvalue:.3e}",
        "",
        "## A variável Amount",
        "",
        f"- Correlação com Class: {corr.loc['Amount', 'Class']:+.4f}",
        f"- AUPRC isolada: {table[table.feature == 'Amount'].auprc.iloc[0]:.4f} "
        f"(lift de {table[table.feature == 'Amount'].lift.iloc[0]:.2f}×)",
        f"- Mediana: {amount_legit.median():.2f} (legítimas) contra "
        f"{amount_fraud.median():.2f} (fraudes)",
        f"- Média: {amount_legit.mean():.2f} contra {amount_fraud.mean():.2f}",
        f"- Máximo: {amount_legit.max():.2f} contra {amount_fraud.max():.2f}",
        "",
        "## Variáveis mais discriminantes",
        "",
        "| # | Variável | AUPRC | Lift | Correlação com Class | Sentido |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in table.head(10).iterrows():
        lines.append(f"| {i + 1} | {row.feature} | {row.auprc:.4f} | "
                     f"{row.lift:.1f}× | "
                     f"{corr.loc[row.feature, 'Class']:+.4f} | {row.direction} |")

    lines += [
        "",
        "## Densidade local",
        "",
        f"- Eixos usados nas projeções: {', '.join(axes)}",
        f"- Mediana da distância ao {K_NEIGHBOURS}º vizinho: "
        f"{density['median_legit']:.3f} (legítimas) contra "
        f"{density['median_fraud']:.3f} (fraudes)",
        f"- Razão: {density['ratio']:.2f}×",
        f"- Kolmogorov–Smirnov: D = {density['ks_d']:.4f}, p = {density['ks_p']:.3e}",
        "",
        "Uma razão acima de 1 indica que as fraudes estão, em mediana, mais",
        "distantes de seus vizinhos — ou seja, em regiões menos densas. É a",
        "premissa que sustenta o tratamento de fraude como ruído.",
        "",
        "## Correlação com a variável alvo",
        "",
        "Cinco maiores em valor absoluto:",
        "",
    ]
    for feature in class_corr.head(5).index:
        lines.append(f"- {feature}: {corr.loc[feature, 'Class']:+.4f}")

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

    print("Reading dataset...")
    df, X, y = load()
    print(f"  {ps.thousands(len(df))} transactions, {int(y.sum())} frauds")

    scaled = scale(X)
    table, prevalence = discriminative_power(scaled, y)

    print("01 - correlation matrix")
    fig, corr = figure_correlation(df)
    print("   ", ps.save(fig, f"{OUTPUT}/01-correlation-matrix")[0])
    plt.close(fig)

    print("02 - discriminative power")
    fig = figure_discriminative(table, prevalence)
    print("   ", ps.save(fig, f"{OUTPUT}/02-feature-discriminative-power")[0])
    plt.close(fig)

    print("03 - relevance of Time")
    fig, ks_time = figure_time(df, y)
    print("   ", ps.save(fig, f"{OUTPUT}/03-time-relevance")[0])
    plt.close(fig)

    print("04 - class-conditional distributions")
    fig = figure_distributions(scaled, y, table)
    print("   ", ps.save(fig, f"{OUTPUT}/04-class-distributions")[0])
    plt.close(fig)

    print("05 - projections")
    fig, axes = figure_projections(scaled, y, table, rng)
    print("   ", ps.save(fig, f"{OUTPUT}/05-projections")[0])
    plt.close(fig)

    print("06 - three-dimensional view")
    fig = figure_projection_3d(scaled, y, table, np.random.RandomState(SEED))
    print("   ", ps.save(fig, f"{OUTPUT}/06-projection-3d")[0])
    plt.close(fig)

    print("07 - local density")
    fig, density = figure_local_density(scaled, y, rng)
    print("   ", ps.save(fig, f"{OUTPUT}/07-local-density")[0])
    plt.close(fig)

    summary = write_summary(df, y, corr, table, prevalence, ks_time, density,
                            axes, f"{OUTPUT}/dataset-statistics.md")
    print("summary:", summary)

    table.to_csv(f"{OUTPUT}/feature-discriminative-power.csv", index=False)
    print("table  :", f"{OUTPUT}/feature-discriminative-power.csv")


if __name__ == "__main__":
    main()
