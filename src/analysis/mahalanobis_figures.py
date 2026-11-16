"""
Figures of the Mahalanobis distance section of the theoretical background.

Both figures use simulated data, so that the shape of each distance shows without the anonymised
attributes of the dataset. The centre and the covariance are estimated as in the pipeline
(src/models/whitening.py): the sample mean and np.cov for the empirical covariance, and the
MinCovDet of scikit-learn, with its default support fraction, for the MCD.

  1. 01-euclidean-vs-mahalanobis: points at Euclidean distances 1, 2 and 3 from the centre
     (circles) and at Mahalanobis distances 1, 2 and 3 (ellipses), for a sample of a correlated
     bivariate normal. Points A and B are at the same Euclidean distance from the centre, A along
     the direction in which the data vary the most and B across it.
  2. 02-classical-vs-robust-covariance: 97.5% tolerance ellipses of the empirical covariance and
     of the MCD for a sample with a group of outliers, the masking effect described by Hubert and
     Debruyne (2010).

The numbers the monograph quotes (the Mahalanobis distances of A and B and how many outliers each
ellipse holds) go to mahalanobis-statistics.md.

Run from the repository root:

    python src/analysis/mahalanobis_figures.py
"""

import os
import sys

import numpy as np
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # noqa: E402

OUTPUT = "results/figures/theory"
SEED = 42

# Figure 1: correlated bivariate normal, with standard deviations 3 and 1 and correlation 0.8.
SAMPLE_SIZE = 150
SPREAD = (3.0, 1.0)
CORRELATION = 0.8
LEVELS = (1, 2, 3)  # distances of the circles and ellipses
AB_DISTANCE = 3.0  # Euclidean distance of points A and B from the centre

# Figure 2: regular points and a group of outliers away from their main direction.
REGULAR_SIZE = 60
REGULAR_SPREAD = (1.0, 2.0)
REGULAR_CORRELATION = 0.5
OUTLIER_SIZE = 10  # with fewer, the farthest one already falls outside the classical ellipse
OUTLIER_CENTRE = (4.5, -2.0)
OUTLIER_SPREAD = 0.4
TOLERANCE = 0.975  # coverage of the tolerance ellipse, chi-squared quantile with p = 2

COLOR_EUCLIDEAN = ps.SLATE
COLOR_EMPIRICAL = ps.NAVY
COLOR_MCD = ps.CYAN


def covariance_matrix(spread, correlation):
    """Covariance of two variables with the given standard deviations and correlation."""
    sx, sy = spread
    return np.array([[sx**2, correlation * sx * sy], [correlation * sx * sy, sy**2]])


def ellipse(centre, covariance, level, points=400):
    """Points at Mahalanobis distance `level` from `centre`, for the given covariance."""
    angle = np.linspace(0, 2 * np.pi, points)
    circle = np.vstack([np.cos(angle), np.sin(angle)])
    return (centre[:, None] + level * np.linalg.cholesky(covariance) @ circle).T


def draw_closed(ax, curve, color, linewidth):
    """Closed outline, without the seam a line plot leaves where the curve starts and ends."""
    from matplotlib.patches import Polygon

    ax.add_patch(
        Polygon(curve, closed=True, fill=False, edgecolor=color, linewidth=linewidth, zorder=3)
    )


def mahalanobis(x, centre, covariance):
    """Mahalanobis distance of each row of x to the centre (Equation 2.3 of the monograph)."""
    diff = np.atleast_2d(x) - centre
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, np.linalg.inv(covariance), diff))


def euclidean_vs_mahalanobis():
    """Figure 1 and the Mahalanobis distances of points A and B."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(SEED)
    data = rng.multivariate_normal([0, 0], covariance_matrix(SPREAD, CORRELATION), SAMPLE_SIZE)
    centre, covariance = data.mean(axis=0), np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending order
    point_a = centre + AB_DISTANCE * eigenvectors[:, 1]  # direction of largest variance
    point_b = centre + AB_DISTANCE * eigenvectors[:, 0]  # direction of smallest variance
    distance_a, distance_b = (mahalanobis(p, centre, covariance)[0] for p in (point_a, point_b))

    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH_IN, 2.6))
    ax.scatter(
        data[:, 0],
        data[:, 1],
        s=12,
        facecolor="none",
        edgecolor=ps.SLATE_LIGHT,
        linewidth=0.7,
        zorder=2,
    )
    for level in LEVELS:
        # With the identity as covariance, the Mahalanobis distance is the Euclidean one.
        draw_closed(ax, ellipse(centre, np.eye(2), level), COLOR_EUCLIDEAN, 1.2)
        draw_closed(ax, ellipse(centre, covariance, level), COLOR_EMPIRICAL, 1.2)
    ax.scatter(*centre, s=28, color=ps.INK, zorder=4)
    for name, point, distance, offset, align in (
        ("A", point_a, distance_a, (-7, 5), "right"),
        ("B", point_b, distance_b, (8, 2), "left"),
    ):
        ax.scatter(*point, s=34, color=ps.INK, marker="D", zorder=4)
        label = f"{name} ($D_M$ = {distance:.1f})".replace(".", ",")
        ax.annotate(
            label,
            point,
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            color=ps.INK,
            ha=align,
            # White box, so that the label stays legible where it crosses an ellipse.
            bbox={"facecolor": ps.SURFACE, "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
            zorder=5,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ps.decimal_comma(ax.xaxis, ax.yaxis)
    handles = [
        Line2D([], [], color=COLOR_EUCLIDEAN, linewidth=1.2, label="Distância euclidiana 1, 2 e 3"),
        Line2D(
            [], [], color=COLOR_EMPIRICAL, linewidth=1.2, label="Distância de Mahalanobis 1, 2 e 3"
        ),
        Line2D([], [], color=ps.INK, marker="o", linewidth=0, markersize=5, label="Centro"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    return fig, {"distance_a": distance_a, "distance_b": distance_b}


def classical_vs_robust():
    """Figure 2 and how many outliers each tolerance ellipse holds."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(SEED)
    regular = rng.multivariate_normal(
        [0, 0], covariance_matrix(REGULAR_SPREAD, REGULAR_CORRELATION), REGULAR_SIZE
    )
    outliers = rng.normal(OUTLIER_CENTRE, OUTLIER_SPREAD, (OUTLIER_SIZE, 2))
    data = np.vstack([regular, outliers])
    estimates = {
        "empirical": (data.mean(axis=0), np.cov(data, rowvar=False)),
        "mcd": (lambda mcd: (mcd.location_, mcd.covariance_))(
            MinCovDet(random_state=SEED).fit(data)
        ),
    }
    cutoff = np.sqrt(chi2.ppf(TOLERANCE, df=2))
    inside = {
        name: int(np.sum(mahalanobis(outliers, *estimate) <= cutoff))
        for name, estimate in estimates.items()
    }

    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH_IN, 3.1))
    ax.scatter(
        regular[:, 0],
        regular[:, 1],
        s=12,
        facecolor="none",
        edgecolor=ps.SLATE_LIGHT,
        linewidth=0.7,
        zorder=2,
    )
    # Outliers are anomalies, so they wear the accent reserved for them in every figure.
    ax.scatter(
        outliers[:, 0],
        outliers[:, 1],
        s=16,
        color=ps.ACCENT,
        edgecolor=ps.SURFACE,
        linewidth=0.4,
        zorder=2,
    )
    for name, color in (("empirical", COLOR_EMPIRICAL), ("mcd", COLOR_MCD)):
        draw_closed(ax, ellipse(*estimates[name], cutoff), color, 1.4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ps.decimal_comma(ax.xaxis, ax.yaxis)
    handles = [
        Line2D([], [], color=COLOR_EMPIRICAL, linewidth=1.4, label="Covariância empírica"),
        Line2D([], [], color=COLOR_MCD, linewidth=1.4, label="Covariância robusta (MCD)"),
        Line2D(
            [],
            [],
            color=ps.ACCENT,
            marker="o",
            linewidth=0,
            markersize=4.5,
            label="Observações discrepantes",
        ),
    ]
    fig.legend(handles=handles, loc="outside right center", ncol=1)
    return fig, {"cutoff": cutoff, "outliers_inside": inside}


def write_summary(first, second, path):
    lines = [
        "# Figuras da distância de Mahalanobis",
        "",
        f"Gerado por `src/analysis/mahalanobis_figures.py`, com dados simulados (semente {SEED}).",
        "",
        "## 01-euclidean-vs-mahalanobis",
        "",
        f"- {SAMPLE_SIZE} pontos de uma normal bivariada com desvios padrão {SPREAD[0]:g} e "
        f"{SPREAD[1]:g} e correlação {CORRELATION:g}.",
        f"- A e B ficam à distância euclidiana {AB_DISTANCE:g} do centro.",
        f"- Distância de Mahalanobis de A: {first['distance_a']:.2f}.",
        f"- Distância de Mahalanobis de B: {first['distance_b']:.2f}.",
        "",
        "## 02-classical-vs-robust-covariance",
        "",
        f"- {REGULAR_SIZE} observações regulares e {OUTLIER_SIZE} discrepantes.",
        f"- Elipse de tolerância de {TOLERANCE:.1%}: distância de Mahalanobis até "
        f"{second['cutoff']:.3f}.",
        f"- Discrepantes dentro da elipse clássica: {second['outliers_inside']['empirical']} de "
        f"{OUTLIER_SIZE}.",
        f"- Discrepantes dentro da elipse robusta: {second['outliers_inside']['mcd']} de "
        f"{OUTLIER_SIZE}.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps.apply_style()
    first_fig, first = euclidean_vs_mahalanobis()
    second_fig, second = classical_vs_robust()
    for name, fig in (
        ("01-euclidean-vs-mahalanobis", first_fig),
        ("02-classical-vs-robust-covariance", second_fig),
    ):
        print("   ", ps.save(fig, f"{OUTPUT}/{name}")[0])
        plt.close(fig)
    print("   ", write_summary(first, second, f"{OUTPUT}/mahalanobis-statistics.md"))


if __name__ == "__main__":
    main()
