"""Part (b) figures and the metric table, from the JIPNet outputs in ``experiments/runs``.

Usage (from the repository root): ``python src/report_figs.py``. Writes
``experiments/summary.csv`` and the ``b_*.png`` figures in ``reports/figures``.
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

matplotlib.use("Agg")
from analyze import FIG, ORDER, eer, load, pose_err, summary, tar_at  # noqa: E402
from make_pairs import load_patch  # noqa: E402

plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
#: Databases made by the SFinGe generator.
KIND = {"2002_DB4": "synthetic", "2004_DB4": "synthetic"}
#: Axis label per database (year, database, sensor).
LABEL = {
    "2004_DB1": "2004 DB1\noptical",
    "2004_DB2": "2004 DB2\noptical",
    "2002_DB1": "2002 DB1\noptical",
    "2002_DB3": "2002 DB3\ncapacitive",
    "2002_DB2": "2002 DB2\noptical",
    "2004_DB3": "2004 DB3\nthermal",
    "2002_DB4": "2002 DB4\nsynthetic",
    "2004_DB4": "2004 DB4\nsynthetic",
}
#: Role of each database in the JIPNet paper; the rest were not used by the authors.
GROUPOF = {
    "2004_DB1": "seen in training",
    "2004_DB2": "seen in training",
    "2002_DB1": "paper test set",
    "2002_DB3": "paper test set",
}
GC = {"seen in training": "#c0392b", "paper test set": "#d9822b", "never seen": "#3c9a5f"}
MC = {"JIPNet": "#d9822b", "AFRNet": "#2f6db5"}
FULL_RUN = 3200  # 800 genuine + 2,400 impostor pairs per database


def is_full(D: pd.DataFrame, m: str, db: str) -> bool:
    """Whether a model has scored every pair of a database.

    Args:
        D: Joined model outputs.
        m: Model name.
        db: Database key.

    Returns:
        True if all pairs are present.
    """
    return ((D.model == m) & (D.db == db)).sum() >= FULL_RUN


def kind_of(db: str) -> str:
    """``"synthetic"`` or ``"real"`` for a database key."""
    return KIND.get(db, "real")


def plot_score_hills(D: pd.DataFrame, models: list[str]) -> None:
    """Figure ``b_hills_<model>.png``: genuine and impostor score histograms per database.

    Args:
        D: Joined model outputs.
        models: Models to plot.
    """
    for m in models:
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True)
        for ax, db in zip(axes.ravel(), ORDER):
            d = D[(D.model == m) & (D.db == db)]
            if len(d) < 100:
                ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes, color="#999")
                ax.set_title(LABEL[db].replace("\n", " "), fontsize=10)
                continue
            g, i = d[d.label == 1].score, d[d.label == 0].score
            bins = np.linspace(min(d.score.min(), 0), 1, 41)
            ax.hist(i, bins, color="#c0392b", alpha=0.55, density=True, label="different fingers")
            ax.hist(g, bins, color="#3c9a5f", alpha=0.55, density=True, label="same finger")
            e, t = eer(g.values, i.values)
            ax.axvline(t, color="#222", ls="--", lw=1)
            ax.set_title(
                f"{LABEL[db].replace(chr(10), ' ')} · EER {e:.1%}"
                + ("" if is_full(D, m, db) else " (partial)"),
                fontsize=10,
                color=GC[GROUPOF.get(db, "never seen")],
            )
            ax.set_yticks([])
        axes[0, 0].legend(frameon=False, fontsize=9)
        for a in axes[1]:
            a.set_xlabel("match score")
        fig.suptitle(
            f"{m}: score distributions per database (dashed line = EER threshold). Title colour: red = used in JIPNet training, orange = paper's test set, green = not used by the authors",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(f"{FIG}/b_hills_{m}.png", dpi=90)
        plt.close(fig)


def plot_metric_bars(D: pd.DataFrame, S: pd.DataFrame, models: list[str]) -> None:
    """Figure ``b_bars.png``: EER and TAR at FAR 0.1% per database with bootstrap intervals.

    Args:
        D: Joined model outputs.
        S: Metric table from :func:`analyze.summary`.
        models: Models to plot.
    """
    Sd = S[~S.db.str.startswith("ALL")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    w = 0.38
    for ax, col, lo, hi, title, fmt in [
        (axes[0], "EER", "EER_lo", "EER_hi", "EER (lower is better)", "{:.1%}"),
        (axes[1], "TAR_0p1", "TAR_lo", "TAR_hi", "TAR at FAR 0.1% (higher is better)", "{:.0%}"),
    ]:
        for k, m in enumerate(models):
            x = np.arange(len(ORDER)) + (k - (len(models) - 1) / 2) * w
            for xi, db in zip(x, ORDER):
                r = Sd[(Sd.model == m) & (Sd.db == db)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.bar(xi, r[col], w * 0.92, color=MC[m], alpha=1 if is_full(D, m, db) else 0.45)
                ax.errorbar(
                    xi, r[col], yerr=[[r[col] - r[lo]], [r[hi] - r[col]]], color="#333", lw=1, capsize=2
                )
                ax.text(
                    xi,
                    r[hi] + (0.004 if col == "EER" else 0.02),
                    fmt.format(r[col]),
                    ha="center",
                    fontsize=7.5,
                )
        ax.set_xticks(range(len(ORDER)), [LABEL[d] for d in ORDER], fontsize=8.5)
        ax.set_title(title, fontsize=11)
        for t, db in zip(ax.get_xticklabels(), ORDER):
            t.set_color(GC[GROUPOF.get(db, "never seen")])
    axes[1].set_ylim(0, 1.12)
    axes[0].legend(handles=[Patch(color=MC[m], label=m) for m in models], frameon=False)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    fig.suptitle(
        "Whiskers: 90% range across random re-draws of the test pairs.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/b_bars.png", dpi=95)
    plt.close(fig)


def plot_pose_curves(D: pd.DataFrame, dbs: list[str]) -> None:
    """Figure ``b_pose.png``: share of genuine pairs whose JIPNet pose error is within X.

    Args:
        D: Joined model outputs.
        dbs: Databases to plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), squeeze=False)
    for r_, m in enumerate(["JIPNet"]):
        for db in dbs:
            g = D[(D.model == m) & (D.db == db) & (D.label == 1)]
            if len(g) < 100:
                continue
            turn, shift = pose_err(g)
            ls = "--" if KIND.get(db) else "-"
            c = GC[GROUPOF.get(db, "never seen")]
            for ax, v, mx in [(axes[r_, 0], turn, 60), (axes[r_, 1], shift, 60)]:
                xs = np.linspace(0, mx, 121)
                ax.plot(
                    xs,
                    [(v <= x).mean() for x in xs],
                    ls=ls,
                    color=c,
                    lw=1.8,
                    label=LABEL[db].replace("\n", " "),
                )
        axes[r_, 0].set_xlabel("rotation error (degrees)")
        axes[r_, 1].set_xlabel("translation error (pixels; 20 px ≈ 1 mm)")
        for ax in axes[r_]:
            ax.set_ylabel("share of genuine pairs\nwithin this error")
            ax.set_ylim(0, 1.01)
            ax.axhline(0.5, color="#bbb", lw=0.8)
            ax.axhline(0.9, color="#bbb", lw=0.8)
        axes[r_, 0].set_title(f"{m}: share of genuine pairs within a given rotation error", fontsize=10.5)
        axes[r_, 1].set_title(f"{m}: share of genuine pairs within a given translation error", fontsize=10.5)
        axes[r_, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Grey lines at 50% and 90% of pairs. Dashed lines: synthetic databases.", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/b_pose.png", dpi=90)
    plt.close(fig)


def plot_pose_vs_overlap(D: pd.DataFrame) -> None:
    """Figure ``b_pose_overlap.png``: median JIPNet pose error by patch overlap, real vs synthetic.

    Args:
        D: Joined model outputs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    bins = [0, 0.3, 0.45, 0.6, 0.75, 1.01]
    mids = ["<30%", "30–45%", "45–60%", "60–75%", ">75%"]
    for m in ["JIPNet"]:
        for kind, ls in [("real", "-"), ("synthetic", "--")]:
            g = D[(D.model == m) & (D.label == 1) & (D.db.map(kind_of) == kind)]
            if len(g) < 100:
                continue
            turn, shift = pose_err(g)
            b = pd.cut(g.overlap, bins)
            for ax, err in [(axes[0], turn), (axes[1], shift)]:
                ax.plot(
                    mids,
                    err.groupby(b, observed=False).median().values,
                    ls=ls,
                    marker="o",
                    color=MC[m],
                    label=f"{m}, {kind}",
                )
    axes[0].set_ylabel("median rotation error (degrees)")
    axes[1].set_ylabel("median translation error (pixels)")
    for ax in axes:
        ax.set_xlabel("overlap between the two patches")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Pose error by overlap between the patches (median per overlap band, all databases pooled)",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/b_pose_overlap.png", dpi=95)
    plt.close(fig)


def plot_coverage(D: pd.DataFrame, models: list[str]) -> None:
    """Figure ``b_coverage.png``: EER and TAR when only pairs with enough fingerprint area are kept.

    Args:
        D: Joined model outputs.
        models: Models to plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ths = [0, 0.5, 0.6, 0.7, 0.8]
    for m in models:
        for kind, ls in [("real", "-"), ("synthetic", "--")]:
            d = D[(D.model == m) & (D.db.map(kind_of) == kind)].copy()
            d["mc"] = np.minimum(d.cov1, d.cov2)
            if len(d) < 400:
                continue
            E, T = [], []
            for th in ths:
                s = d[d.mc >= th]
                E.append(eer(s[s.label == 1].score.values, s[s.label == 0].score.values)[0])
                T.append(tar_at(s[s.label == 1].score.values, s[s.label == 0].score.values)[0])
            axes[0].plot([f"≥{t:.0%}" for t in ths], E, ls=ls, marker="o", color=MC[m], label=f"{m}, {kind}")
            axes[1].plot([f"≥{t:.0%}" for t in ths], T, ls=ls, marker="o", color=MC[m])
    axes[0].set_ylabel("EER")
    axes[1].set_ylabel("TAR at FAR 0.1%")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel("minimum fingerprint share of both patches")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Accuracy by fingerprint coverage of the patches (databases pooled: real / synthetic)", fontsize=10.5
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/b_coverage.png", dpi=95)
    plt.close(fig)


def plot_failure_gallery(D: pd.DataFrame, m: str, db: str, n: int = 6, tag: str = "") -> None:
    """Figure ``b_fail_<model>_<db>.png``: lowest genuine, highest impostor and worst-pose pairs.

    Args:
        D: Joined model outputs.
        m: Model name.
        db: Database key.
        n: Number of examples per row.
        tag: Optional suffix for the file name.
    """
    d = D[(D.model == m) & (D.db == db)]
    if len(d) < 400:
        return
    g = d[d.label == 1].nsmallest(n, "score")
    i = d[d.label == 0].nlargest(n, "score")
    turn, _ = pose_err(d[d.label == 1])
    pw = d[d.label == 1].assign(turn=turn).nlargest(n, "turn")
    sets = [(g, "Genuine pair,\nlow score"), (i, "Impostor pair,\nhigh score")] + (
        [(pw, "Genuine pair, largest\nrotation error")] if m == "JIPNet" else []
    )
    fig, axes = plt.subplots(len(sets), n, figsize=(2.7 * n, 3.07 * len(sets)))
    for row, (sel, lab) in enumerate(sets):
        for k, (_, x) in enumerate(sel.iterrows()):
            p1 = load_patch(db, x.img1, x.r1, x.c1, x.a1)
            p2 = load_patch(db, x.img2, x.r2, x.c2, x.a2)
            axes[row, k].imshow(
                np.hstack([p1, np.full((160, 4), 0, np.uint8), p2]), cmap="gray", vmin=0, vmax=255
            )
            t = f"{lab}\nscore {x.score:.2f} · print {min(x.cov1, x.cov2):.0%}"
            if row == 2:
                t += f"\nrotation error {x.turn:.0f}° · overlap {x.overlap:.0%}"
            axes[row, k].set_title(t, fontsize=7.5)
            axes[row, k].axis("off")
    fig.suptitle(
        f"{m} · {LABEL[db].replace(chr(10), ' ')} · lowest-scoring and highest-scoring cases (left patch | right patch). 'print' = fingerprint share of the patch with less fingerprint",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/b_fail_{m}_{db}{tag}.png", dpi=85)
    plt.close(fig)


def main() -> None:
    """Compute the metric table, save it, and draw all part (b) figures."""
    D = load()
    S = summary(D)
    S.to_csv(ROOT + "/experiments/summary.csv", index=False)
    models = [m for m in ["JIPNet", "AFRNet"] if m in set(D.model)]
    dbs = [d for d in ORDER if d in set(D.db)]
    plot_score_hills(D, models)
    plot_metric_bars(D, S, models)
    plot_pose_curves(D, dbs)
    plot_pose_vs_overlap(D)
    plot_coverage(D, models)
    for m in models:
        for db in ["2002_DB1", "2002_DB4", "2004_DB3"]:
            plot_failure_gallery(D, m, db)
    print(S[["model", "db", "n_gen", "EER", "TAR_0p1", "turn_med", "shift_med"]].round(3).to_string())


if __name__ == "__main__":
    main()
