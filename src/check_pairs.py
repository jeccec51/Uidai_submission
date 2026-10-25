"""Proof figures for pair construction: (1) masks, (2) GT pose re-aligns patch B onto A,
(3) quick JIPNet sanity run on 60 genuine pairs."""

import json
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

matplotlib.use("Agg")
from make_pairs import DBS, RAW, fg_mask, load_patch, square
from models_cpu import JIP, REPO

sys.path.insert(0, REPO)
from models.utils import AffinePatch

FIG = ROOT + "/reports/figures"
aff = AffinePatch()
# 1 masks + patch boxes
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for ax, db in zip(axes.ravel(), DBS):
    df = pd.read_csv(f"{ROOT}/data/partial/{db}_pairs.csv")
    g = df[df.label == 1].iloc[10]
    im = cv2.imread(f"{RAW}/{DBS[db]}/{g.img1}.tif", 0)
    m = fg_mask(im)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    rgb[m == 0] = (rgb[m == 0] * 0.45 + np.array([255, 120, 120]) * 0.55).astype(np.uint8)
    ax.imshow(rgb)
    for (r, c, a), col in [((g.r1, g.c1, g.a1), "#2f6db5"), ((g.r2, g.c2, g.a2), "#d9822b")]:
        p = square((r, c), a)
        ax.add_patch(plt.Polygon(p, fill=False, ec=col, lw=2.2))
    ax.set_title(f"{db}  ({g.img1})  overlap {g.overlap:.0%}", fontsize=10)
    ax.axis("off")
fig.suptitle(
    "Pink = not fingerprint (ignored). Blue = patch A, orange = patch B, cut from the same impression.",
    fontsize=11,
)
fig.savefig(f"{FIG}/s2_masks.png", dpi=85, bbox_inches="tight")
plt.close(fig)


# 2 GT re-alignment check on 4 pairs + numeric check on 200 pairs
def realign(p2, gt, pad=100):
    out = (
        aff(torch.tensor(np.float32(p2)[None, None] / 255.0), torch.tensor(np.float32([gt])), pad_width=pad)
        .squeeze()
        .numpy()
    )
    return out


df = pd.read_csv(ROOT + "/data/partial/2002_DB1_pairs.csv")
G = df[df.label == 1].reset_index(drop=True)
fig, axes = plt.subplots(3, 4, figsize=(13, 10))
for k, i in enumerate([3, 50, 222, 600]):
    g = G.iloc[i]
    p1 = load_patch(g.db, g.img1, g.r1, g.c1, g.a1)
    p2 = load_patch(g.db, g.img2, g.r2, g.c2, g.a2)
    gt = [g.gt_cos, g.gt_sin, g.gt_tx, g.gt_ty]
    # work in "ridge = bright" space like the repo: invert, warp, overlay in colour
    w2 = realign(255 - p2, gt)
    a1 = np.pad(255 - p1, 100)
    ov = np.zeros(a1.shape + (3,))
    ov[..., 2] = a1 / 255
    ov[..., 0] = w2
    ov[..., 1] = np.minimum(a1 / 255, w2)
    axes[0, k].imshow(np.hstack([p1, np.full((160, 6), 255, np.uint8), p2]), cmap="gray")
    axes[0, k].set_title(f"pair {i}: A | B  (turn {g.gt_angle:.0f}°)", fontsize=9)
    axes[1, k].imshow(np.clip(ov, 0, 1))
    axes[1, k].set_title("B moved by TRUE pose over A\nwhite = ridges agree", fontsize=9)
    mask_ov = (a1 > 0) & (w2 > 0.02)
    corr = np.corrcoef(a1[mask_ov].ravel() / 255, w2[mask_ov].ravel())[0, 1]
    wrong = realign(255 - p2, [g.gt_cos, -g.gt_sin, g.gt_tx, g.gt_ty])
    ov2 = np.zeros_like(ov)
    ov2[..., 2] = a1 / 255
    ov2[..., 0] = wrong
    ov2[..., 1] = np.minimum(a1 / 255, wrong)
    axes[2, k].imshow(np.clip(ov2, 0, 1))
    axes[2, k].set_title("control: turn sign flipped (wrong)", fontsize=9)
    axes[1, k].set_xlabel(f"ridge agreement in overlap: {corr:.2f}", fontsize=9)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig(f"{FIG}/s2_gt_check.png", dpi=90, bbox_inches="tight")
plt.close(fig)
cs = []
for i in range(200):
    g = G.iloc[i]
    if g.overlap < 0.25:
        continue
    p1 = load_patch(g.db, g.img1, g.r1, g.c1, g.a1)
    p2 = load_patch(g.db, g.img2, g.r2, g.c2, g.a2)
    w2 = realign(255 - p2, [g.gt_cos, g.gt_sin, g.gt_tx, g.gt_ty])
    a1 = np.pad(255 - p1, 100) / 255
    m = (w2 > 0.02) & (np.pad(np.ones_like(p1), 100) > 0)
    m &= np.pad(np.ones((160, 160), bool), 100)
    m[: 100 + 8] = 0
    m[-108:] = 0
    m[:, :108] = 0
    m[:, -108:] = 0
    if m.sum() > 500:
        cs.append(np.corrcoef(a1[m], w2[m])[0, 1])
print(
    "GT ridge agreement over",
    len(cs),
    "pairs: median %.3f, 5th pct %.3f" % (np.median(cs), np.percentile(cs, 5)),
)
# 3 JIPNet sanity
j = JIP()
err = []
for i in range(60):
    g = G.iloc[i]
    p1 = load_patch(g.db, g.img1, g.r1, g.c1, g.a1)
    p2 = load_patch(g.db, g.img2, g.r2, g.c2, g.a2)
    s, p = j(p1, p2)
    da = abs((np.rad2deg(np.arctan2(p[1], p[0])) - g.gt_angle + 180) % 360 - 180)
    err.append((s, da, np.hypot(p[2] - g.gt_tx, p[3] - g.gt_ty), g.overlap))
e = np.array(err)
print(
    "JIPNet on 60 genuine: median P(same) %.2f, median turn err %.1f deg, median shift err %.1f px"
    % tuple(np.median(e[:, :3], 0))
)
json.dump(dict(gt_agreement=cs, jip_sanity=e.tolist()), open(ROOT + "/experiments/step2_checks.json", "w"))
