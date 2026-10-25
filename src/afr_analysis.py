"""AFR-Net-style baseline metrics + comparison with JIPNet on FVC2002 DB1/DB3."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import ROOT
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ROOT + "/src")
from analyze import load, eer, tar_at, boot
from make_pairs import load_patch
P = ROOT + ""; FIG = f"{P}/reports/figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
D = load(); res = {}
dbs = [d for d in ["2002_DB1", "2002_DB3"] if __import__("os").path.exists(f"{P}/experiments/runs_afr/{d}.npz")]
fig, axes = plt.subplots(2, len(dbs), figsize=(6.5 * len(dbs), 7), squeeze=False)
for k, db in enumerate(dbs):
    z = np.load(f"{P}/experiments/runs_afr/{db}.npz"); S, diff, pidx = z["S"], z["diff"], z["pair_idx"]
    gen = np.diag(S); imp = S[diff]
    e, et = eer(gen, imp); t, tt, na = tar_at(gen, imp); (elo, ehi), (tlo, thi) = boot(gen, imp, n=200)
    J = D[(D.model == "JIPNet") & (D.db == db)]; jg = J[J.idx.isin(pidx) & (J.label == 1)].score.values; ji = J[J.label == 0].score.values
    je, _ = eer(jg, ji); jt, _, jna = tar_at(jg, ji)
    res[db] = dict(afr_EER=e, afr_EER_lo=elo, afr_EER_hi=ehi, afr_TAR=t, afr_TAR_lo=tlo, afr_TAR_hi=thi, afr_fa_allowed=na, n_gen=len(gen), n_imp=len(imp),
                   jip_same400_EER=je, jip_same400_TAR=jt, sec_per_patch=float(z["sec_per_patch"]))
    for row, (name, g, i, c) in enumerate([("AFR-Net-style", gen, imp, "#2f6db5"), ("JIPNet (same 400 genuine pairs)", jg, ji, "#d9822b")]):
        ax = axes[row, k]; bins = np.linspace(min(g.min(), i.min(), 0), 1, 41)
        ax.hist(i, bins, color="#c0392b", alpha=.55, density=True, label="different fingers"); ax.hist(g, bins, color="#3c9a5f", alpha=.55, density=True, label="same finger")
        ee, tt2 = eer(g, i); ax.axvline(tt2, color="#222", ls="--", lw=1); ax.set_yticks([])
        ax.set_title(f"{name} · FVC{db[:4]} {db[5:]} · EER {ee:.1%}", fontsize=10.5, color=c)
    axes[1, k].set_xlabel("match score")
axes[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle("Score distributions on the paper's two test sets: AFR-Net-style baseline (top) vs JIPNet (bottom)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/b_afr_vs_jip_hills.png", dpi=90); plt.close(fig)
# failure gallery for AFR on DB1
if dbs:
    db = dbs[0]; z = np.load(f"{P}/experiments/runs_afr/{db}.npz"); S, diff, pidx = z["S"], z["diff"], z["pair_idx"]
    pairs = pd.read_csv(f"{P}/data/partial/{db}_pairs.csv"); G = pairs.loc[pidx].reset_index(drop=True)
    low = np.argsort(np.diag(S))[:6]; Si = np.where(diff, S, -9); hi = np.dstack(np.unravel_index(np.argsort(Si, axis=None)[::-1][:6], S.shape))[0]
    fig, axes = plt.subplots(2, 6, figsize=(16.2, 6.2))
    for kk, i in enumerate(low):
        g = G.iloc[i]; p1 = load_patch(db, g.img1, g.r1, g.c1, g.a1); p2 = load_patch(db, g.img2, g.r2, g.c2, g.a2)
        axes[0, kk].imshow(np.hstack([p1, np.zeros((160, 4), np.uint8), p2]), cmap="gray", vmin=0, vmax=255)
        axes[0, kk].set_title(f"MISSED: same finger\nscore {S[i, i]:.2f} · print {min(g.cov1, g.cov2):.0%} · overlap {g.overlap:.0%}", fontsize=7.5)
    for kk, (i, j) in enumerate(hi):
        a, b = G.iloc[i], G.iloc[j]; p1 = load_patch(db, a.img1, a.r1, a.c1, a.a1); p2 = load_patch(db, b.img2, b.r2, b.c2, b.a2)
        axes[1, kk].imshow(np.hstack([p1, np.zeros((160, 4), np.uint8), p2]), cmap="gray", vmin=0, vmax=255)
        axes[1, kk].set_title(f"FALSE ALARM: different fingers\nscore {S[i, j]:.2f} · {a.img1} vs {b.img2}", fontsize=7.5)
    for ax in axes.ravel(): ax.axis("off")
    fig.suptitle(f"AFR-Net-style baseline · FVC{db[:4]} {db[5:]} · worst cases", fontsize=10.5); fig.tight_layout(); fig.savefig(f"{FIG}/b_fail_AFR_{db}.png", dpi=85); plt.close(fig)
json.dump(res, open(f"{P}/experiments/afr_summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
# rotation diagnosis: baseline genuine score vs true relative turn
fig, axes = plt.subplots(1, len(dbs), figsize=(6.5 * len(dbs), 3.8), squeeze=False)
bands = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 181)]; rot = {}
for k, db in enumerate(dbs):
    z = np.load(f"{P}/experiments/runs_afr/{db}.npz"); S, diff, pidx = z["S"], z["diff"], z["pair_idx"]
    Pp = pd.read_csv(f"{P}/data/partial/{db}_pairs.csv").loc[pidx].reset_index(drop=True); gen = np.diag(S); imp = S[diff]; turn = Pp.gt_angle.abs().values
    J = D[(D.model == "JIPNet") & (D.db == db)]; Jg = J[J.idx.isin(pidx) & (J.label == 1)].set_index("idx").loc[pidx]
    rows = []
    for lo, hi in bands:
        m = (turn >= lo) & (turn < hi); rows.append(dict(band=f"{lo}–{min(hi,180)}°", n=int(m.sum()), afr_med=float(np.median(gen[m])), afr_eer=float(eer(gen[m], imp)[0]), jip_med=float(np.median(Jg.score.values[m]))))
    rot[db] = rows; ax = axes[0, k]; x = np.arange(len(rows))
    ax.bar(x - .2, [r["afr_med"] for r in rows], .4, color="#2f6db5", label="AFR-Net-style: typical same-finger score")
    ax.bar(x + .2, [r["jip_med"] for r in rows], .4, color="#d9822b", label="JIPNet: typical same-finger score")
    ax.axhline(np.median(imp), color="#c0392b", ls="--", lw=1.2, label="AFR-Net-style: typical different-finger score")
    ax.set_xticks(x, [f'{r["band"]}\n(n={r["n"]})' for r in rows]); ax.set_ylim(0, 1.05); ax.set_xlabel("true turn between the two patches")
    ax.set_title(f"FVC{db[:4]} {db[5:]}", fontsize=10.5)
axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper right")
fig.suptitle("The baseline's code does not survive rotation; JIPNet does (it estimates the turn itself)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/b_afr_rotation.png", dpi=95); plt.close(fig)
json.dump(rot, open(f"{P}/experiments/afr_rotation.json", "w"), indent=1)
