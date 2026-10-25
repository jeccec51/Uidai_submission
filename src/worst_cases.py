"""Show the worst genuine pairs (lowest score) and worst impostor pairs (highest score)."""

import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

matplotlib.use("Agg")
from analyze import load
from make_pairs import load_patch

model, db, n = sys.argv[1], sys.argv[2], 6
D = load()
d = D[(D.model == model) & (D.db == db)]
g = d[d.label == 1].nsmallest(n, "score")
i = d[d.label == 0].nlargest(n, "score")
fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 6))
for row, (sel, lab) in enumerate([(g, "genuine, LOW score"), (i, "impostor, HIGH score")]):
    for k, (_, x) in enumerate(sel.iterrows()):
        p1 = load_patch(db, x.img1, x.r1, x.c1, x.a1)
        p2 = load_patch(db, x.img2, x.r2, x.c2, x.a2)
        axes[row, k].imshow(
            np.hstack([p1, np.full((160, 4), 0, np.uint8), p2]), cmap="gray", vmin=0, vmax=255
        )
        t = f"{lab}\nscore {x.score:.2f}  {x.img1} | {x.img2}"
        if x.label == 1:
            t += f"\noverlap on print {x.overlap_print:.0%}"
        axes[row, k].set_title(t, fontsize=7.5)
        axes[row, k].axis("off")
out = f"{ROOT}/reports/figures/worst_{model}_{db}.png"
fig.savefig(out, dpi=90, bbox_inches="tight")
print(out)
