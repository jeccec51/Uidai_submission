"""Metrics from the model outputs in ``experiments/runs/*.csv``.

* EER: the threshold where the share of impostors accepted equals the share of genuine pairs rejected.
* TAR at FAR 0.1%: threshold set so that at most 0.1% of impostor pairs pass; share of genuine pairs
  that pass.
* Pose error (genuine pairs only): rotation error in degrees, translation error in pixels
  (1 px = 0.0508 mm at 500 dpi).

Usage (from the repository root): ``python src/analyze.py``. Writes ``experiments/summary.csv``.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

P = ROOT
FIG = f"{P}/reports/figures"
ORDER = ["2004_DB1", "2004_DB2", "2002_DB1", "2002_DB3", "2002_DB2", "2002_DB4", "2004_DB3", "2004_DB4"]


def load() -> pd.DataFrame:
    """Load all model outputs and join them with their test-pair definitions.

    Returns:
        One row per (model, pair): pair definition columns from ``data/partial/<db>_pairs.csv``
        plus the model's score, predicted pose and run time, and a ``model`` column.
    """
    out = []
    for f in glob.glob(f"{P}/experiments/runs/*.csv"):
        model, db = os.path.basename(f)[:-4].split("_", 1)
        r = pd.read_csv(f)
        pairs = pd.read_csv(f"{P}/data/partial/{db}_pairs.csv")
        d = pairs.loc[r.idx].reset_index(drop=True)
        d = pd.concat([d, r.drop(columns="idx").reset_index(drop=True)], axis=1)
        d["model"] = model
        d["idx"] = r.idx.values
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def eer(gen: np.ndarray, imp: np.ndarray) -> tuple[float, float]:
    """Equal error rate: the threshold where false accepts equal false rejects.

    Args:
        gen: Scores of genuine (same-finger) pairs.
        imp: Scores of impostor (different-finger) pairs.

    Returns:
        ``(eer, threshold)``.
    """
    t = np.unique(np.r_[gen, imp])
    gs, is_ = np.sort(gen), np.sort(imp)
    far = 1 - np.searchsorted(is_, t, side="left") / len(is_)  # impostors with score >= t
    frr = np.searchsorted(gs, t, side="left") / len(gs)  # genuine with score < t
    k = np.argmin(np.abs(far - frr))
    return (far[k] + frr[k]) / 2, t[k]


def boot(gen: np.ndarray, imp: np.ndarray, n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """90% bootstrap intervals of EER and TAR at FAR 0.1%, by re-drawing pairs with replacement.

    Args:
        gen: Scores of genuine pairs.
        imp: Scores of impostor pairs.
        n: Number of bootstrap re-draws.
        seed: Random seed.

    Returns:
        ``(eer_interval, tar_interval)``, each the 5th and 95th percentiles.
    """
    rng = np.random.default_rng(seed)
    E, T = [], []
    for _ in range(n):
        gg, ii = rng.choice(gen, len(gen)), rng.choice(imp, len(imp))
        E.append(eer(gg, ii)[0])
        T.append(tar_at(gg, ii)[0])
    return np.percentile(E, [5, 95]), np.percentile(T, [5, 95])


def tar_at(gen: np.ndarray, imp: np.ndarray, far: float = 1e-3) -> tuple[float, float, int]:
    """True-accept rate at a fixed false-accept rate.

    The threshold is set so that at most ``far`` of the impostor pairs score above it.

    Args:
        gen: Scores of genuine pairs.
        imp: Scores of impostor pairs.
        far: Target false-accept rate (default 0.1%).

    Returns:
        ``(tar, threshold, false_accepts_allowed)``.
    """
    s = np.sort(imp)[::-1]
    n_allowed = int(np.floor(far * len(imp)))  # false accepts allowed
    thr = (
        s[n_allowed] if n_allowed < len(s) else -np.inf
    )  # strictly above the (n_allowed+1)-th highest impostor
    return (gen > thr).mean(), thr, n_allowed


def pose_err(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Rotation and translation error of the predicted pose against the ground truth.

    Args:
        d: Genuine pairs with predicted (``p_cos``, ``p_sin``, ``p_tx``, ``p_ty``) and true
            (``gt_angle``, ``gt_tx``, ``gt_ty``) pose columns.

    Returns:
        ``(rotation_error_degrees, translation_error_pixels)``.
    """
    pa = np.rad2deg(np.arctan2(d.p_sin, d.p_cos))
    turn = np.abs((pa - d.gt_angle + 180) % 360 - 180)
    shift = np.hypot(d.p_tx - d.gt_tx, d.p_ty - d.gt_ty)
    return turn, shift


def summary(D: pd.DataFrame) -> pd.DataFrame:
    """Per-database and per-group metrics for every model.

    Args:
        D: Joined model outputs from :func:`load`.

    Returns:
        One row per (model, database) and per (model, group) with EER, TAR at FAR 0.1%
        (with bootstrap intervals), pose-error statistics and median run time.
    """
    rows = []
    for (m, db), d in D.groupby(["model", "db"]):
        g, i = d[d.label == 1], d[d.label == 0]
        if len(g) < 20 or len(i) < 20:
            continue
        e, et = eer(g.score.values, i.score.values)
        t, tt, na = tar_at(g.score.values, i.score.values)
        turn, shift = pose_err(g)
        (e_lo, e_hi), (t_lo, t_hi) = boot(g.score.values, i.score.values)
        rows.append(
            dict(
                EER_lo=e_lo,
                EER_hi=e_hi,
                TAR_lo=t_lo,
                TAR_hi=t_hi,
                model=m,
                db=db,
                group=d.group.iloc[0],
                n_gen=len(g),
                n_imp=len(i),
                EER=e,
                TAR_0p1=t,
                thr_0p1=tt,
                fa_allowed=na,
                turn_med=turn.median(),
                turn_p90=turn.quantile(0.9),
                shift_med=shift.median(),
                shift_p90=shift.quantile(0.9),
                sec_med=d.sec.median(),
            )
        )
    for (m, grp), d in D.groupby(["model", "group"]):  # pooled per group
        g, i = d[d.label == 1], d[d.label == 0]
        if len(g) < 20:
            continue
        e, _ = eer(g.score.values, i.score.values)
        t, tt, na = tar_at(g.score.values, i.score.values)
        turn, shift = pose_err(g)
        rows.append(
            dict(
                model=m,
                db="ALL " + grp,
                group=grp,
                n_gen=len(g),
                n_imp=len(i),
                EER=e,
                TAR_0p1=t,
                thr_0p1=tt,
                fa_allowed=na,
                turn_med=turn.median(),
                turn_p90=turn.quantile(0.9),
                shift_med=shift.median(),
                shift_p90=shift.quantile(0.9),
                sec_med=d.sec.median(),
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Print the metric table and save it to ``experiments/summary.csv``."""
    D = load()
    S = summary(D)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string())
    S.to_csv(f"{P}/experiments/summary.csv", index=False)


if __name__ == "__main__":
    main()
