"""
Tables and figures of the Results chapter and of Appendix A, built from the final runs.

Reads the CSVs of the final runs in results/supervised and results/unsupervised and writes what
the monograph shows, so that no value is typed by hand. The precision, recall, F1 and counts of
the HDBSCAN* come from the alert fraction the monograph adopts (ADOPTED_RATE). For Chapter 5:

  1. outer-results.tex: mean and standard deviation of AUPRC, precision, recall and F1 over the
     five outer test folds, for every model and base, with the best mean of each metric in bold.
  2. 01-outer-metrics: boxplots of the same metrics over the five outer test folds.
  3. outer-alerts.tex: alerts, true positives, false positives and false negatives summed over the
     five outer test folds, which together cover every transaction of the dataset.
  4. 02-alert-fraction: precision, recall and F1 of the HDBSCAN* with the MCD distance at each
     alert fraction of the run, against the range of the supervised models.
  5. reference-scores.tex: AUPRC of the anomaly degree of the HDBSCAN* and of the two reference
     scores, GLOSH alone and the distance to the centre.

For Appendix A:

  6. rf-, xgb- and hdbscan-hyperparameters.tex: the grid of each model and the configuration it
     applied to the outer folds in each base. Every inner fold indicates its best combination, and
     the best among them is applied to all outer folds, as in supervised_outer_evaluation and in
     best_configuration of run_hdbscan_enhanced_pipeline.py.
  7. supervised-selection.tex: AUPRC of the chosen supervised configuration in the inner fold that
     indicated it, its mean over the five inner folds, the best mean of the grid and how many
     combinations come within 0.01 of it.
  8. hdbscan-selection.tex and 03-hdbscan-selection: mean clusters, DISCO and AUPRC of the HDBSCAN*
     inner cycle for each m_pts, the highest DISCO among the folds and the chosen configuration.

The tables come out as LaTeX tabulars (the caption, label and source line stay in main.tex), and
the numbers also go to a Markdown summary.

Run from the repository root:

    python src/analysis/results_analysis.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # noqa: E402

SUPERVISED_DIR = "results/supervised"
UNSUPERVISED_DIR = "results/unsupervised"
OUTPUT = "results/figures/evaluation"
TABLES = f"{OUTPUT}/tables"

BASES = (("full", "", "Completa"), ("undersampled", "_undersampled", "Subamostrada"))
MODELS = (("rf", "\\textit{Random Forest}"), ("xgb", "XGBoost"))
# Key, title (figure columns and table cells) and the form used after "HDBSCAN*".
DISTANCES = (
    ("euclidean", "Euclidiana", "euclidiana"),
    ("mahalanobis", "Mahalanobis empírica", "Mahalanobis empírica"),
    ("mahalanobis_mcd", "Mahalanobis com MCD", "Mahalanobis com MCD"),
)
GRID_FILE = "hdbscan_enhanced_grid_results_by_fold.csv"
HDBSCAN_OUTER_FILE = "hdbscan_enhanced_outer_final_report.csv"
SELECTION_METRIC = {"supervised": "auprc", "hdbscan": "disco_score"}
METRICS = (("auprc", "AUPRC"), ("precision", "Precisão"), ("recall", "Revocação"), ("f1", "F1"))
# Reference scores of the HDBSCAN*, as named by the pipeline: the anomaly degree that issues the
# alerts (column auprc), GLOSH alone and the distance to the centre estimated in the training set.
REFERENCE_SCORES = (("auprc", "Grau de anomalia"), ("auprc_glosh", "GLOSH"),
                    ("auprc_center_distance", "Distância ao centro"))
ALERT_DISTANCE = "mahalanobis_mcd"  # distance of the alert fraction figure
# Alert fraction of the HDBSCAN* in the chapter. The runs issue alerts at alert_rate and score the
# same transactions at extra_alert_rates, changing only the threshold. In the inner folds, the F1
# of the MCD distance peaks at 0.0017 in both bases, the evaluated fraction closest to the fraud
# proportion of the dataset (0.172%), and the monograph adopts it.
ADOPTED_RATE = 0.0017
COUNT_METRICS = ("precision", "recall", "f1", "TP", "FP")  # reported at every alert fraction

# Models of the outer evaluation in the order of every table and figure: key, LaTeX label and
# figure label.
OUTER_MODELS = (
    ("rf", "\\textit{Random Forest}", "Random Forest"),
    ("xgb", "XGBoost", "XGBoost"),
) + tuple((f"hdbscan_{key}", f"HDBSCAN*\\newline {inline}", f"HDBSCAN*\n{inline}")
          for key, _, inline in DISTANCES)

# Hyperparameters of the quadros, with the names of the libraries.
DESCRIPTIONS = {
    "rf": {
        "n_estimators": "Número de árvores",
        "max_features": "Variáveis sorteadas em cada divisão, pela raiz quadrada (sqrt) ou por "
                        "uma fração do total",
        "class_weight": "Ponderação das classes, ausente (None) ou balanceada na amostra de "
                        "cada árvore (balanced\\_subsample)",
        "max_samples": "Fração do treino sorteada para cada árvore, ou o treino inteiro (None)",
        "min_samples_leaf": "Número mínimo de transações em cada folha",
    },
    "xgb": {
        "n_estimators": "Número de árvores",
        "max_depth": "Profundidade máxima de cada árvore",
        "learning_rate": "Taxa de aprendizado (\\textit{shrinkage})",
        "gamma": "Penalidade $\\gamma$ sobre o número de folhas",
    },
}

# The two bases keep the same colour in every figure of the chapter. Blue and orange are reserved
# for the classes (plot_style), so the bases take the next hues of the categorical order.
BASE_COLORS = {"full": ps.AQUA, "undersampled": ps.VIOLET}
BASE_MARKERS = {"full": "o", "undersampled": "s"}


def decimal(value, digits=3):
    """Number with the decimal comma, in math mode when negative."""
    text = f"{value:.{digits}f}".replace(".", ",")
    return f"${text.replace(',', '{,}')}$" if value < 0 else text


def plain(value):
    """Hyperparameter value as plain Python, with None for an empty cell."""
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value.item() if hasattr(value, "item") else value


def latex_value(value):
    """Hyperparameter value as written in the quadros: decimal comma, None and library names."""
    value = plain(value)
    if isinstance(value, str):
        try:
            value = plain(float(value))
        except ValueError:  # library name, which may break after the underscore, never hyphenated
            return value.replace("_", "\\_\\allowbreak ")
    if value is None:
        return "None"
    return f"{value:g}".replace(".", ",") if isinstance(value, float) else str(value)


def listing(values):
    """Values of a grid, separated by semicolons when a decimal comma appears."""
    cells = [latex_value(value) for value in values]
    return ("; " if any("," in cell for cell in cells) else ", ").join(cells)


def best_per_fold(history, metric):
    """Row of the best combination in each inner fold, as in the selection code."""
    return history.loc[history.groupby("fold")[metric].idxmax()]


def chosen_row(history, metric):
    """Best combination among the inner-fold winners, the one applied to the outer folds."""
    winners = best_per_fold(history, metric)
    return winners.loc[winners[metric].idxmax()]


def write_table(name, content):
    """Write a LaTeX tabular to the tables folder and return its path."""
    path = f"{TABLES}/{name}.tex"
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(content)
    return path


# --------------------------------------------------------------------------- outer evaluation

def adopt_rate(report, rates):
    """HDBSCAN* report with the standard columns at ADOPTED_RATE and every fraction as _at_ columns.

    The columns without suffix hold the alert_rate of the run. They are copied to _at_ columns, so
    that every fraction is read the same way, and then replaced by the values at ADOPTED_RATE. The
    false negatives follow from the frauds of the fold, which do not depend on the fraction.
    """
    main_rate, extra_rates = rates
    if ADOPTED_RATE not in (main_rate, *extra_rates):
        raise SystemExit(f"A fração {ADOPTED_RATE} não foi avaliada nas rodadas do HDBSCAN*")
    report = report.copy()
    frauds = report["TP"] + report["FN"]
    for column in COUNT_METRICS:
        report[f"{column}_at_{main_rate}"] = report[column]
    for column in COUNT_METRICS:
        report[column] = report[f"{column}_at_{ADOPTED_RATE}"]
    report["FN"] = frauds - report["TP"]
    return report


def outer_reports(rates):
    """Metrics of every model and base in each outer test fold, as one long table."""
    frames = []
    for key, _, _ in OUTER_MODELS:
        for base, suffix, _ in BASES:
            if key.startswith("hdbscan_"):
                path = f"{UNSUPERVISED_DIR}/{key}{suffix}/{HDBSCAN_OUTER_FILE}"
            else:
                path = f"{SUPERVISED_DIR}/{key}{suffix}/{key}_outer_final_report.csv"
            report = pd.read_csv(path)
            # The reports end with two summary rows (mean and standard deviation), left out here.
            folds = pd.to_numeric(report["fold"], errors="coerce")
            report = report[folds.notna()].assign(fold=folds.dropna().astype(int), model=key,
                                                  base=base)
            if sorted(report["fold"]) != [1, 2, 3, 4, 5]:
                raise SystemExit(f"{path} não tem os cinco folds externos")
            frames.append(adopt_rate(report, rates) if key.startswith("hdbscan_") else report)
    return pd.concat(frames, ignore_index=True)


def alert_rates():
    """Alert fraction of the HDBSCAN* runs and the extra fractions evaluated on the same scores."""
    configs = sorted(glob.glob(f"{UNSUPERVISED_DIR}/hdbscan_*/run_config.json"))
    rates = set()
    for path in configs:
        with open(path, encoding="utf-8") as file:
            config = json.load(file)
        rates.add((config["alert_rate"], tuple(config["extra_alert_rates"])))
    if len(rates) != 1:
        raise SystemExit(f"As rodadas do HDBSCAN* usam frações de alerta diferentes: {rates}")
    return rates.pop()


def mean_sd(values, bold=False):
    """Mean and standard deviation over the folds, with the mean in bold when asked."""
    mean = decimal(values.mean())
    if bold:
        mean = f"\\textbf{{{mean}}}"
    return f"{mean} $\\pm$ {decimal(values.std())}"


def base_header(label, columns):
    """Row that opens the block of a base in the tables grouped by base."""
    return f"    \\multicolumn{{{columns}}}{{l}}{{\\textit{{Base {label.lower()}}}}} \\\\"


def results_table(outer):
    """LaTeX tabular with mean and standard deviation of each metric, best mean in bold."""
    lines = [
        "\\begin{tabular}{L{3.8cm} C{2.3cm} C{2.3cm} C{2.3cm} C{2.3cm}}",
        "    \\toprule",
        "    \\textbf{Modelo} & " + " & ".join(f"\\textbf{{{label}}}" for _, label in METRICS)
        + " \\\\",
        "    \\midrule",
    ]
    for b_index, (base, _, base_label) in enumerate(BASES):
        block = outer[outer["base"] == base]
        means = block.groupby("model")[[metric for metric, _ in METRICS]].mean().round(3)
        lines.append(base_header(base_label, len(METRICS) + 1))
        for key, latex_label, _ in OUTER_MODELS:
            folds = block[block["model"] == key]
            cells = [mean_sd(folds[metric], bold=means.loc[key, metric] == means[metric].max())
                     for metric, _ in METRICS]
            lines.append(f"    {latex_label} & " + " & ".join(cells) + " \\\\")
        if b_index < len(BASES) - 1:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def alert_counts(outer):
    """Alerts and confusion counts summed over the outer test folds."""
    sums = outer.groupby(["base", "model"])[["TP", "FP", "FN"]].sum().astype(int)
    sums["alerts"] = sums["TP"] + sums["FP"]
    return sums


def alerts_table(outer):
    """LaTeX tabular of the alerts and confusion counts summed over the outer test folds."""
    sums = alert_counts(outer)
    lines = [
        "\\begin{tabular}{L{3.8cm} C{1.9cm} C{1.9cm} C{1.9cm} C{1.9cm}}",
        "    \\toprule",
        "    \\textbf{Modelo} & \\textbf{Alertas} & \\textbf{VP} & \\textbf{FP} & \\textbf{FN} \\\\",
        "    \\midrule",
    ]
    for b_index, (base, _, base_label) in enumerate(BASES):
        lines.append(base_header(base_label, 5))
        for key, latex_label, _ in OUTER_MODELS:
            row = sums.loc[(base, key)]
            lines.append(f"    {latex_label} & {row['alerts']} & {row['TP']} & {row['FP']} & "
                         f"{row['FN']} \\\\")
        if b_index < len(BASES) - 1:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def reference_table(outer):
    """LaTeX tabular with the AUPRC of the anomaly degree and of the two reference scores."""
    lines = [
        "\\begin{tabular}{L{3.8cm} C{2.6cm} C{2.6cm} C{2.6cm}}",
        "    \\toprule",
        "    \\textbf{Distância} & " + " & ".join(f"\\textbf{{{label}}}"
                                                for _, label in REFERENCE_SCORES) + " \\\\",
        "    \\midrule",
    ]
    for b_index, (base, _, base_label) in enumerate(BASES):
        lines.append(base_header(base_label, len(REFERENCE_SCORES) + 1))
        for key, title, _ in DISTANCES:
            folds = outer[(outer["model"] == f"hdbscan_{key}") & (outer["base"] == base)]
            cells = [mean_sd(folds[column]) for column, _ in REFERENCE_SCORES]
            lines.append(f"    {title} & " + " & ".join(cells) + " \\\\")
        if b_index < len(BASES) - 1:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def alert_curve(outer, rates):
    """Mean precision, recall and F1 of the HDBSCAN* with ALERT_DISTANCE at each alert fraction."""
    main_rate, extra_rates = rates
    rows = []
    for base, _, _ in BASES:
        folds = outer[(outer["model"] == f"hdbscan_{ALERT_DISTANCE}") & (outer["base"] == base)]
        for rate in sorted((main_rate, *extra_rates)):
            rows.append({"base": base, "rate": rate, **{
                metric: float(folds[f"{metric}_at_{rate}"].mean())
                for metric in ("precision", "recall", "f1")}})
    return pd.DataFrame(rows)


def figure_outer_metrics(outer):
    """Boxplots of each metric over the outer test folds, one row per model, one colour per base."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(0)  # fixed jitter, so the figure is reproducible
    centers = np.arange(len(OUTER_MODELS))
    offsets = {"full": -0.19, "undersampled": 0.19}
    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXT_WIDTH_IN, 5.0), sharex=True, sharey=True)
    for ax, (metric, label) in zip(axes.flat, METRICS):
        for base, _, _ in BASES:
            color = BASE_COLORS[base]
            data = [outer[(outer["model"] == key) & (outer["base"] == base)][metric].to_numpy()
                    for key, _, _ in OUTER_MODELS]
            positions = centers + offsets[base]
            line = {"color": color, "linewidth": 1.0}
            ax.boxplot(data, positions=positions, orientation="horizontal", widths=0.3,
                       whis=(0, 100), showfliers=False, patch_artist=True,
                       boxprops={"facecolor": to_rgba(color, 0.12), "edgecolor": color,
                                 "linewidth": 1.0},
                       medianprops={"color": color, "linewidth": 1.8},
                       whiskerprops=line, capprops=line)
            for position, values in zip(positions, data):
                jitter = rng.uniform(-0.07, 0.07, size=len(values))
                ax.scatter(values, position + jitter, s=13, marker=BASE_MARKERS[base],
                           facecolor=color, edgecolor=ps.SURFACE, linewidth=0.5, zorder=3,
                           clip_on=False)
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ps.decimal_comma(ax.xaxis)
    axes[0, 0].set_yticks(centers, [figure_label for _, _, figure_label in OUTER_MODELS])
    axes[0, 0].set_ylim(len(OUTER_MODELS) - 0.5, -0.5)  # first model on top, set once (shared)
    handles = [Line2D([], [], color=BASE_COLORS[base], marker=BASE_MARKERS[base],
                      markerfacecolor=BASE_COLORS[base], markeredgecolor=ps.SURFACE,
                      markersize=6, linewidth=1.8, label=f"Base {label.lower()}")
               for base, _, label in BASES]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    return fig


def figure_alert_fraction(outer, rates):
    """Precision, recall and F1 of the HDBSCAN* with ALERT_DISTANCE against the alert fraction."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator

    curve = alert_curve(outer, rates)
    supervised = (outer[outer["model"].isin([key for key, _ in MODELS])]
                  .groupby(["model", "base"])[["precision", "recall", "f1"]].mean())
    percent = sorted(rate * 100 for rate in curve["rate"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH_IN, 2.7), sharey=True)
    for ax, (metric, label) in zip(axes, METRICS[1:]):
        ax.axhspan(supervised[metric].min(), supervised[metric].max(), color=ps.INK_MUTED,
                   alpha=0.22, linewidth=0, zorder=1)
        ax.axvline(ADOPTED_RATE * 100, color=ps.INK_MUTED, linewidth=0.8, zorder=1)
        for base, _, _ in BASES:
            data = curve[curve["base"] == base]
            color = BASE_COLORS[base]
            ax.plot(data["rate"] * 100, data[metric], color=color, linewidth=1.6, zorder=2)
            ax.scatter(data["rate"] * 100, data[metric], s=22, marker=BASE_MARKERS[base],
                       facecolor=ps.SURFACE, edgecolor=color, linewidth=1.2, zorder=3)
        ax.set_title(label)
        ax.set_xscale("log")
        ax.set_xticks(percent, [f"{value:g}".replace(".", ",") for value in percent])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Fração de alertas (%)")
        ax.set_ylim(0, 1)
        ps.decimal_comma(ax.yaxis)
    handles = [Line2D([], [], color=BASE_COLORS[base], marker=BASE_MARKERS[base],
                      markerfacecolor=ps.SURFACE, markersize=5, linewidth=1.6,
                      label=f"HDBSCAN*, base {label.lower()}") for base, _, label in BASES]
    handles += [Patch(facecolor=ps.INK_MUTED, alpha=0.22, label="Modelos supervisionados"),
                Line2D([], [], color=ps.INK_MUTED, linewidth=0.8, label="Fração adotada")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    return fig


# --------------------------------------------------------------------------- supervised cycle

def supervised_selection():
    """Chosen configuration, its AUPRC in the indicating fold and its mean over the inner folds."""
    rows = []
    for model, model_label in MODELS:
        for base, suffix, base_label in BASES:
            folder = f"{SUPERVISED_DIR}/{model}{suffix}"
            with open(f"{folder}/run_config.json", encoding="utf-8") as file:
                grid = json.load(file)["param_grid"]
            history = pd.read_csv(f"{folder}/results_by_fold.csv")
            chosen = chosen_row(history, SELECTION_METRIC["supervised"])
            same = history["combo_id"] == chosen["combo_id"]
            winners = best_per_fold(history, SELECTION_METRIC["supervised"])
            means = history.groupby("combo_id")["auprc"].mean()
            rows.append({
                "model": model, "model_label": model_label, "base": base, "base_label": base_label,
                "grid": grid,
                "chosen": {name: chosen[name] for name in grid},
                "params": ", ".join(f"{name}={plain(chosen[name])}" for name in grid),
                "indicating_fold": int(chosen["fold"]),
                "auprc_fold": float(chosen["auprc"]),
                "auprc_mean": float(history.loc[same, "auprc"].mean()),
                "distinct_winners": int(winners["combo_id"].nunique()),
                "grid_size": int(history["combo_id"].nunique()),
                "best_mean": float(means.max()),
                "within_001": int((means >= means.max() - 0.01).sum()),
            })
    return pd.DataFrame(rows)


def supervised_table(selection):
    """LaTeX tabular of the AUPRC of the chosen supervised configurations in the inner cycle."""
    lines = [
        "\\begin{tabular}{L{3.0cm} C{2.4cm} C{2.4cm} C{2.4cm} C{2.8cm}}",
        "    \\toprule",
        "    \\textbf{Modelo} & \\textbf{AUPRC no \\textit{fold} que a indicou} & "
        "\\textbf{AUPRC média nos \\textit{folds}} & \\textbf{Maior AUPRC média da grade} & "
        "\\textbf{Combinações a menos de 0,01 da maior média} \\\\",
        "    \\midrule",
    ]
    for b_index, (base, _, base_label) in enumerate(BASES):
        lines.append(base_header(base_label, 5))
        for _, row in selection[selection["base"] == base].iterrows():
            lines.append(f"    {row['model_label']} & {decimal(row['auprc_fold'])} & "
                         f"{decimal(row['auprc_mean'])} & {decimal(row['best_mean'])} & "
                         f"{row['within_001']} de {row['grid_size']} \\\\")
        if b_index < len(BASES) - 1:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------- HDBSCAN* cycle

def hdbscan_inner_cycle():
    """Clusters, DISCO and AUPRC per m_pts and the chosen m_pts, for each distance and base."""
    rows = []
    for distance, distance_label, _ in DISTANCES:
        for base, suffix, base_label in BASES:
            history = pd.read_csv(f"{UNSUPERVISED_DIR}/hdbscan_{distance}{suffix}/{GRID_FILE}")
            chosen = chosen_row(history, SELECTION_METRIC["hdbscan"])
            groups = history.groupby("min_samples")
            means = groups[["n_clusters", "disco_score", "auprc"]].mean()
            maxima = groups["disco_score"].max()
            for m_pts, values in means.iterrows():
                rows.append({
                    "distance": distance, "distance_label": distance_label,
                    "base": base, "base_label": base_label, "m_pts": int(m_pts),
                    "clusters": float(values["n_clusters"]),
                    "disco": float(values["disco_score"]),
                    "disco_max": float(maxima[m_pts]),
                    "auprc": float(values["auprc"]),
                    "chosen": int(m_pts) == int(chosen["min_samples"]),
                    "min_cluster_size": int(chosen["min_cluster_size"]),
                    "grid_min_cluster_size": sorted(int(v) for v in
                                                    history["min_cluster_size"].unique()),
                })
    return pd.DataFrame(rows)


def hdbscan_table(cycle):
    """LaTeX tabular of the HDBSCAN* inner cycle, with the chosen configuration in bold."""
    lines = [
        "\\begin{tabular}{L{2.6cm} L{2.3cm} C{1.1cm} C{1.4cm} C{1.7cm} C{1.7cm} C{1.7cm}}",
        "    \\toprule",
        "    \\textbf{Distância} & \\textbf{Base} & $\\boldsymbol{m_{pts}}$ & \\textbf{Grupos} & "
        "\\textbf{DISCO médio} & \\textbf{DISCO máximo} & \\textbf{AUPRC média} \\\\",
        "    \\midrule",
    ]
    for d_index, (distance, distance_label, _) in enumerate(DISTANCES):
        for b_index, (base, _, base_label) in enumerate(BASES):
            block = cycle[(cycle["distance"] == distance) & (cycle["base"] == base)]
            for i, (_, row) in enumerate(block.iterrows()):
                cells = [str(row["m_pts"]), decimal(row["clusters"], 1), decimal(row["disco"]),
                         decimal(row["disco_max"]), decimal(row["auprc"])]
                if row["chosen"]:
                    cells = [f"$\\boldsymbol{{{c[1:-1]}}}$" if c.startswith("$")
                             else f"\\textbf{{{c}}}" for c in cells]
                distance_cell = (f"\\multirow{{8}}{{2.6cm}}{{\\raggedright {distance_label}}}"
                                 if b_index == 0 and i == 0 else "")
                base_cell = (f"\\multirow{{4}}{{2.3cm}}{{\\raggedright {base_label}}}"
                             if i == 0 else "")
                lines.append(f"    {distance_cell} & {base_cell} & " + " & ".join(cells) + " \\\\")
            if b_index == 0:
                lines.append("    \\cmidrule{2-7}")
        if d_index < len(DISTANCES) - 1:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def figure_hdbscan_selection(cycle):
    """DISCO and inner AUPRC against m_pts, one column per distance, one colour per base."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, len(DISTANCES), figsize=(ps.TEXT_WIDTH_IN, 4.2),
                             sharex=True, sharey="row")
    rows = (("disco", "DISCO"), ("auprc", "AUPRC interna"))
    for col, (distance, distance_label, _) in enumerate(DISTANCES):
        for row_index, (metric, metric_label) in enumerate(rows):
            ax = axes[row_index, col]
            for base, _, base_label in BASES:
                data = cycle[(cycle["distance"] == distance) & (cycle["base"] == base)]
                color = BASE_COLORS[base]
                ax.plot(data["m_pts"], data[metric], color=color, linewidth=1.6, zorder=2)
                ax.scatter(data["m_pts"], data[metric], s=22, marker=BASE_MARKERS[base],
                           facecolor=ps.SURFACE, edgecolor=color, linewidth=1.2, zorder=3)
                chosen = data[data["chosen"]]
                ax.scatter(chosen["m_pts"], chosen[metric], s=40, marker=BASE_MARKERS[base],
                           facecolor=color, edgecolor=ps.SURFACE, linewidth=1.2, zorder=4)
            if row_index == 0:
                ax.set_title(distance_label)
                ax.axhline(0, color=ps.AXIS, linewidth=0.8, zorder=1)
            if col == 0:
                ax.set_ylabel(metric_label)
            if row_index == len(rows) - 1:
                ax.set_xlabel("$m_{pts}$")
            ax.set_xticks(sorted(cycle["m_pts"].unique()))
            ps.decimal_comma(ax.yaxis)
    handles = [Line2D([], [], color=BASE_COLORS[base], marker=BASE_MARKERS[base],
                      markerfacecolor=ps.SURFACE, markersize=5, linewidth=1.6,
                      label=f"Base {label.lower()}")
               for base, _, label in BASES]
    handles.append(Line2D([], [], color=ps.INK_SECONDARY, marker="o", linestyle="none",
                          markerfacecolor=ps.INK_SECONDARY, markersize=6,
                          label="Configuração escolhida pelo DISCO"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    return fig


# ------------------------------------------------------------------------------------- quadros

QUADRO_HEADER = [
    "    \\hline",
    "    \\multicolumn{1}{|c|}{\\multirow{2}{*}{\\textbf{Parâmetro}}} & "
    "\\multicolumn{1}{c|}{\\multirow{2}{*}{\\textbf{Descrição}}} & "
    "\\multicolumn{1}{c|}{\\multirow{2}{*}{\\textbf{Listagem}}} & "
    "\\multicolumn{2}{c|}{\\textbf{Escolhido}} \\\\",
    "    \\cline{4-5}",
    "    & & & \\textbf{Completa} & \\textbf{Subamostrada} \\\\",
    "    \\hline",
]
QUADRO_COLUMNS = "\\begin{tabular}{|L{2.9cm}|L{4.2cm}|C{2.4cm}|C{1.9cm}|C{2.4cm}|}"


def quadro(rows):
    """LaTeX tabular of a quadro with closed borders: parameter, description, grid and choices."""
    lines = [QUADRO_COLUMNS, *QUADRO_HEADER]
    for name, description, values, full, under in rows:
        escaped = name.replace("_", "\\_")
        lines.append(f"    \\textit{{{escaped}}} & {description} & {values} & {full} & {under} \\\\")
        lines.append("    \\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def supervised_quadro(selection, model):
    """Quadro of the grid of a supervised model and the configuration chosen in each base."""
    block = selection[selection["model"] == model].set_index("base")
    grid = block.loc["full", "grid"]
    if block.loc["undersampled", "grid"] != grid:
        raise SystemExit(f"As grades de {model} diferem entre as bases")
    return quadro([(name, DESCRIPTIONS[model][name], listing(values),
                    latex_value(block.loc["full", "chosen"][name]),
                    latex_value(block.loc["undersampled", "chosen"][name]))
                   for name, values in grid.items()])


def hdbscan_quadro(cycle):
    """Quadro of the HDBSCAN* grid, one min_samples row per distance, and min_cluster_size."""
    chosen = cycle[cycle["chosen"]].set_index(["distance", "base"])
    values = listing(sorted(cycle["m_pts"].unique()))
    rows = []
    for distance, _, inline in DISTANCES:
        article = "" if distance == "euclidean" else "de "
        rows.append(("min_samples",
                     f"Número mínimo de pontos, $m_{{pts}}$, com a distância {article}{inline}",
                     values, chosen.loc[(distance, "full"), "m_pts"],
                     chosen.loc[(distance, "undersampled"), "m_pts"]))
    sizes = {tuple(grid) for grid in cycle["grid_min_cluster_size"]}
    if len(sizes) != 1:
        raise SystemExit(f"As grades de min_cluster_size diferem entre as rodadas: {sizes}")
    size_grid = sizes.pop()
    full_sizes = set(chosen.xs("full", level="base")["min_cluster_size"])
    under_sizes = set(chosen.xs("undersampled", level="base")["min_cluster_size"])
    rows.append(("min_cluster_size", "Tamanho mínimo de agrupamento, $m_{clSize}$",
                 listing(size_grid), ", ".join(map(str, sorted(full_sizes))),
                 ", ".join(map(str, sorted(under_sizes)))))
    return quadro(rows)


# ------------------------------------------------------------------------------------- summary

def write_summary(outer, rates, selection, cycle, path):
    """Markdown summary with the numbers of the tables and figures."""
    lines = ["# Resultados: capítulo 5 e apêndice A", "",
             "Gerado por `src/analysis/results_analysis.py` a partir das rodadas finais.", "",
             "## Teste externo: média e desvio padrão nos cinco folds", "",
             "| Base | Modelo | " + " | ".join(label for _, label in METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    for (base, model), folds in outer.groupby(["base", "model"], sort=False):
        lines.append(f"| {base} | {model} | " + " | ".join(
            f"{folds[metric].mean():.3f} ± {folds[metric].std():.3f}" for metric, _ in METRICS)
            + " |")
    sums = alert_counts(outer)
    lines += ["", "## Alertas somados nos cinco testes externos", "",
              "| Base | Modelo | Alertas | VP | FP | FN |", "|---|---|---|---|---|---|"]
    for (base, model), row in sums.iterrows():
        lines.append(f"| {base} | {model} | {row['alerts']} | {row['TP']} | {row['FP']} | "
                     f"{row['FN']} |")
    lines += ["", f"## Fração de alertas do HDBSCAN* com {ALERT_DISTANCE}", "",
              f"Fração adotada nas tabelas: {ADOPTED_RATE}. Fração dos alertas das rodadas: "
              f"{rates[0]}.", "",
              "| Base | Fração | Precisão | Revocação | F1 |", "|---|---|---|---|---|"]
    for _, row in alert_curve(outer, rates).iterrows():
        lines.append(f"| {row['base']} | {row['rate']} | {row['precision']:.3f} | "
                     f"{row['recall']:.3f} | {row['f1']:.3f} |")
    lines += ["", "## Escores de referência do HDBSCAN* (AUPRC)", "",
              "| Base | Distância | " + " | ".join(label for _, label in REFERENCE_SCORES) + " |",
              "|---|---|" + "---|" * len(REFERENCE_SCORES)]
    for base, _, _ in BASES:
        for key, _, _ in DISTANCES:
            folds = outer[(outer["model"] == f"hdbscan_{key}") & (outer["base"] == base)]
            lines.append(f"| {base} | {key} | " + " | ".join(
                f"{folds[column].mean():.3f} ± {folds[column].std():.3f}"
                for column, _ in REFERENCE_SCORES) + " |")
    lines += ["", "## Ciclo interno dos modelos supervisionados", "",
              "| Modelo | Base | Configuração | Fold que a indicou | AUPRC no fold | AUPRC média "
              "| Maior média da grade | Combinações a menos de 0,01 dela "
              "| Combinações distintas indicadas |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, row in selection.iterrows():
        lines.append(f"| {row['model']} | {row['base']} | {row['params']} | {row['indicating_fold']}"
                     f" | {row['auprc_fold']:.3f} | {row['auprc_mean']:.3f} | {row['best_mean']:.3f}"
                     f" | {row['within_001']} de {row['grid_size']} | {row['distinct_winners']} de 5 |")
    lines += ["", "## Ciclo interno do HDBSCAN*", "",
              "| Distância | Base | m_pts | Grupos | DISCO médio | DISCO máximo | AUPRC | Escolhida |",
              "|---|---|---|---|---|---|---|---|"]
    for _, row in cycle.iterrows():
        lines.append(f"| {row['distance']} | {row['base']} | {row['m_pts']} | {row['clusters']:.1f}"
                     f" | {row['disco']:.3f} | {row['disco_max']:.3f} | {row['auprc']:.3f}"
                     f" | {'sim' if row['chosen'] else ''} |")
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")
    return path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps.apply_style()
    os.makedirs(TABLES, exist_ok=True)

    rates = alert_rates()
    outer = outer_reports(rates)
    selection = supervised_selection()
    cycle = hdbscan_inner_cycle()

    tables = {
        "outer-results": results_table(outer),
        "outer-alerts": alerts_table(outer),
        "reference-scores": reference_table(outer),
        "rf-hyperparameters": supervised_quadro(selection, "rf"),
        "xgb-hyperparameters": supervised_quadro(selection, "xgb"),
        "hdbscan-hyperparameters": hdbscan_quadro(cycle),
        "supervised-selection": supervised_table(selection),
        "hdbscan-selection": hdbscan_table(cycle),
    }
    for name, content in tables.items():
        print("   ", write_table(name, content))

    figures = {
        "01-outer-metrics": figure_outer_metrics(outer),
        "02-alert-fraction": figure_alert_fraction(outer, rates),
        "03-hdbscan-selection": figure_hdbscan_selection(cycle),
    }
    for name, fig in figures.items():
        print("   ", ps.save(fig, f"{OUTPUT}/{name}")[0])
        plt.close(fig)
    print("   ", write_summary(outer, rates, selection, cycle,
                               f"{OUTPUT}/evaluation-statistics.md"))


if __name__ == "__main__":
    main()
