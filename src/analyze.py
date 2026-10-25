"""Metrics + plots from experiments/runs/*.csv.
EER: cut-off where 'impostors let in' = 'genuine rejected'.
TAR@FAR=0.1%: cut-off set so that at most 0.1% of impostor pairs pass; share of genuine pairs that pass.
Pose error (genuine pairs only): turn error in degrees, shift error in pixels (1 px = 0.0508 mm at 500 dpi)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import ROOT
import glob, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = ROOT + ""; FIG = f"{P}/reports/figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
ORDER = ["2004_DB1", "2004_DB2", "2002_DB1", "2002_DB3", "2002_DB2", "2002_DB4", "2004_DB3", "2004_DB4"]
GCOL = {"seen in training": "#c0392b", "paper test set": "#d9822b", "never seen": "#3c9a5f"}
MCOL = {"JIPNet": "#d9822b", "AFRNet": "#2f6db5"}

def load():
    out = []
    for f in glob.glob(f"{P}/experiments/runs/*.csv"):
        model, db = os.path.basename(f)[:-4].split("_", 1)
        r = pd.read_csv(f); pairs = pd.read_csv(f"{P}/data/partial/{db}_pairs.csv")
        d = pairs.loc[r.idx].reset_index(drop=True); d = pd.concat([d, r.drop(columns="idx").reset_index(drop=True)], axis=1)
        d["model"] = model; d["idx"] = r.idx.values; out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def eer(gen, imp):
    t = np.unique(np.r_[gen, imp]); gs, is_ = np.sort(gen), np.sort(imp)
    far = 1 - np.searchsorted(is_, t, side="left") / len(is_)      # impostors with score >= t
    frr = np.searchsorted(gs, t, side="left") / len(gs)             # genuine with score < t
    k = np.argmin(np.abs(far - frr)); return (far[k] + frr[k]) / 2, t[k]

def boot(gen, imp, n=300, seed=0):
    rng = np.random.default_rng(seed); E, T = [], []
    for _ in range(n):
        gg, ii = rng.choice(gen, len(gen)), rng.choice(imp, len(imp)); E.append(eer(gg, ii)[0]); T.append(tar_at(gg, ii)[0])
    return np.percentile(E, [5, 95]), np.percentile(T, [5, 95])

def tar_at(gen, imp, far=1e-3):
    s = np.sort(imp)[::-1]; n_allowed = int(np.floor(far * len(imp)))   # false accepts allowed
    thr = s[n_allowed] if n_allowed < len(s) else -np.inf                # strictly above the (n_allowed+1)-th highest impostor
    return (gen > thr).mean(), thr, n_allowed

def pose_err(d):
    pa = np.rad2deg(np.arctan2(d.p_sin, d.p_cos))
    turn = np.abs((pa - d.gt_angle + 180) % 360 - 180); shift = np.hypot(d.p_tx - d.gt_tx, d.p_ty - d.gt_ty)
    return turn, shift

def summary(D):
    rows = []
    for (m, db), d in D.groupby(["model", "db"]):
        g, i = d[d.label == 1], d[d.label == 0]
        if len(g) < 20 or len(i) < 20: continue
        e, et = eer(g.score.values, i.score.values); t, tt, na = tar_at(g.score.values, i.score.values)
        turn, shift = pose_err(g); (e_lo, e_hi), (t_lo, t_hi) = boot(g.score.values, i.score.values)
        rows.append(dict(EER_lo=e_lo, EER_hi=e_hi, TAR_lo=t_lo, TAR_hi=t_hi, model=m, db=db, group=d.group.iloc[0], n_gen=len(g), n_imp=len(i), EER=e, TAR_0p1=t, thr_0p1=tt, fa_allowed=na,
                         turn_med=turn.median(), turn_p90=turn.quantile(.9), shift_med=shift.median(), shift_p90=shift.quantile(.9),
                         sec_med=d.sec.median()))
    for (m, grp), d in D.groupby(["model", "group"]):   # pooled per group
        g, i = d[d.label == 1], d[d.label == 0]
        if len(g) < 20: continue
        e, _ = eer(g.score.values, i.score.values); t, tt, na = tar_at(g.score.values, i.score.values); turn, shift = pose_err(g)
        rows.append(dict(model=m, db="ALL " + grp, group=grp, n_gen=len(g), n_imp=len(i), EER=e, TAR_0p1=t, thr_0p1=tt, fa_allowed=na,
                         turn_med=turn.median(), turn_p90=turn.quantile(.9), shift_med=shift.median(), shift_p90=shift.quantile(.9), sec_med=d.sec.median()))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    D = load(); S = summary(D)
    pd.set_option("display.width", 220); print(S.round(3).to_string())
    S.to_csv(f"{P}/experiments/summary.csv", index=False)
