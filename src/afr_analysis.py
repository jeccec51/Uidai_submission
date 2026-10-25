"""AFR-Net-style baseline: metrics, comparison with JIPNet, failure cases and rotation effect.

Uses the template similarity matrices from ``run_afr_embed.py`` (``experiments/runs_afr``) on
FVC2002 DB1 and DB3.

Usage (from the repository root): ``python src/afr_analysis.py``. Writes
``experiments/afr_summary.json``, ``experiments/afr_rotation.json`` and the figures
``b_afr_vs_jip_hills.png``, ``b_fail_AFR_<db>.png`` and ``b_afr_rotation.png``.
"""

import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

matplotlib.use("Agg")
from analyze import boot, eer, load, tar_at  # noqa: E402
from make_pairs import load_patch  # noqa: E402

P = ROOT
FIG = f"{P}/reports/figures"
TURN_BANDS = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 181)]
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)


def load_scores(db: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the AFR-Net similarity matrix of one database.

    Args:
        db: Database key, e.g. ``"2002_DB1"``.

    Returns:
        ``(S, diff, pair_idx)``: similarity matrix (patch A rows x patch B columns, genuine pairs
        on the diagonal), mask of different-finger cells, and the pair index of each row.
    """
    z = np.load(f"{P}/experiments/runs_afr/{db}.npz")
    return z["S"], z["diff"], z["pair_idx"]


def compare_with_jipnet(D: pd.DataFrame, dbs: list[str]) -> dict:
    """Baseline metrics per database, JIPNet on the same genuine pairs, and the score-histogram figure.

    Args:
        D: Joined JIPNet outputs from :func:`analyze.load`.
        dbs: Database keys with AFR-Net results.

    Returns:
        Per-database dictionary of EER and TAR at FAR 0.1% (with bootstrap intervals) for both
        models, pair counts and template time per patch.
    """
    res = {}
    fig, axes = plt.subplots(2, len(dbs), figsize=(6.5 * len(dbs), 7), squeeze=False)
    for k, db in enumerate(dbs):
        S, diff, pidx = load_scores(db)
        sec_per_patch = float(np.load(f"{P}/experiments/runs_afr/{db}.npz")["sec_per_patch"])
        gen = np.diag(S)
        imp = S[diff]
        e, et = eer(gen, imp)
        t, tt, na = tar_at(gen, imp)
        (elo, ehi), (tlo, thi) = boot(gen, imp, n=200)
        J = D[(D.model == "JIPNet") & (D.db == db)]
        jg = J[J.idx.isin(pidx) & (J.label == 1)].score.values
        ji = J[J.label == 0].score.values
        je, _ = eer(jg, ji)
        jt, _, jna = tar_at(jg, ji)
        res[db] = dict(
            afr_EER=e,
            afr_EER_lo=elo,
            afr_EER_hi=ehi,
            afr_TAR=t,
            afr_TAR_lo=tlo,
            afr_TAR_hi=thi,
            afr_fa_allowed=na,
            n_gen=len(gen),
            n_imp=len(imp),
            jip_same400_EER=je,
            jip_same400_TAR=jt,
            sec_per_patch=sec_per_patch,
        )
        for row, (name, g, i, c) in enumerate(
            [("AFR-Net-style", gen, imp, "#2f6db5"), ("JIPNet (same 400 genuine pairs)", jg, ji, "#d9822b")]
        ):
            ax = axes[row, k]
            bins = np.linspace(min(g.min(), i.min(), 0), 1, 41)
            ax.hist(i, bins, color="#c0392b", alpha=0.55, density=True, label="different fingers")
            ax.hist(g, bins, color="#3c9a5f", alpha=0.55, density=True, label="same finger")
            ee, tt2 = eer(g, i)
            ax.axvline(tt2, color="#222", ls="--", lw=1)
            ax.set_yticks([])
            ax.set_title(f"{name} · FVC{db[:4]} {db[5:]} · EER {ee:.1%}", fontsize=10.5, color=c)
        axes[1, k].set_xlabel("match score")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Score distributions on the paper's two test sets: AFR-Net-style baseline (top) vs JIPNet (bottom)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/b_afr_vs_jip_hills.png", dpi=90)
    plt.close(fig)
    return res


def plot_failures(db: str) -> None:
    """Figure: the six lowest-scoring genuine pairs and six highest-scoring impostor pairs.

    Args:
        db: Database key, e.g. ``"2002_DB1"``.
    """
    S, diff, pidx = load_scores(db)
    pairs = pd.read_csv(f"{P}/data/partial/{db}_pairs.csv")
    G = pairs.loc[pidx].reset_index(drop=True)
    low = np.argsort(np.diag(S))[:6]
    Si = np.where(diff, S, -9)
    hi = np.dstack(np.unravel_index(np.argsort(Si, axis=None)[::-1][:6], S.shape))[0]
    fig, axes = plt.subplots(2, 6, figsize=(16.2, 6.2))
    for kk, i in enumerate(low):
        g = G.iloc[i]
        p1 = load_patch(db, g.img1, g.r1, g.c1, g.a1)
        p2 = load_patch(db, g.img2, g.r2, g.c2, g.a2)
        axes[0, kk].imshow(np.hstack([p1, np.zeros((160, 4), np.uint8), p2]), cmap="gray", vmin=0, vmax=255)
        axes[0, kk].set_title(
            f"MISSED: same finger\nscore {S[i, i]:.2f} · print {min(g.cov1, g.cov2):.0%} · overlap {g.overlap:.0%}",
            fontsize=7.5,
        )
    for kk, (i, j) in enumerate(hi):
        a, b = G.iloc[i], G.iloc[j]
        p1 = load_patch(db, a.img1, a.r1, a.c1, a.a1)
        p2 = load_patch(db, b.img2, b.r2, b.c2, b.a2)
        axes[1, kk].imshow(np.hstack([p1, np.zeros((160, 4), np.uint8), p2]), cmap="gray", vmin=0, vmax=255)
        axes[1, kk].set_title(
            f"FALSE ALARM: different fingers\nscore {S[i, j]:.2f} · {a.img1} vs {b.img2}", fontsize=7.5
        )
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"AFR-Net-style baseline · FVC{db[:4]} {db[5:]} · worst cases", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/b_fail_AFR_{db}.png", dpi=85)
    plt.close(fig)


def rotation_effect(D: pd.DataFrame, dbs: list[str]) -> dict:
    """Genuine-pair scores of both models by true relative turn, with the bar-chart figure.

    Args:
        D: Joined JIPNet outputs from :func:`analyze.load`.
        dbs: Database keys with AFR-Net results.

    Returns:
        Per-database list of turn bands with pair count, median genuine score of each model and
        the baseline's EER within the band.
    """
    fig, axes = plt.subplots(1, len(dbs), figsize=(6.5 * len(dbs), 3.8), squeeze=False)
    rot = {}
    for k, db in enumerate(dbs):
        S, diff, pidx = load_scores(db)
        Pp = pd.read_csv(f"{P}/data/partial/{db}_pairs.csv").loc[pidx].reset_index(drop=True)
        gen = np.diag(S)
        imp = S[diff]
        turn = Pp.gt_angle.abs().values
        J = D[(D.model == "JIPNet") & (D.db == db)]
        Jg = J[J.idx.isin(pidx) & (J.label == 1)].set_index("idx").loc[pidx]
        rows = []
        for lo, hi in TURN_BANDS:
            m = (turn >= lo) & (turn < hi)
            rows.append(
                dict(
                    band=f"{lo}–{min(hi, 180)}°",
                    n=int(m.sum()),
                    afr_med=float(np.median(gen[m])),
                    afr_eer=float(eer(gen[m], imp)[0]),
                    jip_med=float(np.median(Jg.score.values[m])),
                )
            )
        rot[db] = rows
        ax = axes[0, k]
        x = np.arange(len(rows))
        ax.bar(
            x - 0.2,
            [r["afr_med"] for r in rows],
            0.4,
            color="#2f6db5",
            label="AFR-Net-style: typical same-finger score",
        )
        ax.bar(
            x + 0.2,
            [r["jip_med"] for r in rows],
            0.4,
            color="#d9822b",
            label="JIPNet: typical same-finger score",
        )
        ax.axhline(
            np.median(imp),
            color="#c0392b",
            ls="--",
            lw=1.2,
            label="AFR-Net-style: typical different-finger score",
        )
        ax.set_xticks(x, [f'{r["band"]}\n(n={r["n"]})' for r in rows])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("true turn between the two patches")
        ax.set_title(f"FVC{db[:4]} {db[5:]}", fontsize=10.5)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(
        "The baseline's code does not survive rotation; JIPNet does (it estimates the turn itself)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/b_afr_rotation.png", dpi=95)
    plt.close(fig)
    return rot


def main() -> None:
    """Run the baseline analysis and write its JSON summaries and figures."""
    D = load()
    dbs = [d for d in ["2002_DB1", "2002_DB3"] if os.path.exists(f"{P}/experiments/runs_afr/{d}.npz")]
    res = compare_with_jipnet(D, dbs)
    if dbs:
        plot_failures(dbs[0])
    with open(f"{P}/experiments/afr_summary.json", "w") as handle:
        json.dump(res, handle, indent=1)
    print(json.dumps(res, indent=1))
    rot = rotation_effect(D, dbs)
    with open(f"{P}/experiments/afr_rotation.json", "w") as handle:
        json.dump(rot, handle, indent=1)


if __name__ == "__main__":
    main()
