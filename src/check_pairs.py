"""Checks on the test-pair construction.

1. Fingerprint masks and patch positions (figure ``s2_masks.png``).
2. The ground-truth pose moves patch B back onto patch A (figure ``s2_gt_check.png`` and a
   ridge-agreement statistic over 200 genuine pairs).
3. A quick JIPNet run on 60 genuine pairs.

Usage (from the repository root): ``python src/check_pairs.py``. Writes
``experiments/step2_checks.json``.
"""

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
AFFINE = AffinePatch()


def plot_masks() -> None:
    """Figure: fingerprint mask and the two patch outlines for one genuine pair per database."""
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


def realign(p2: np.ndarray, gt: list[float], pad: int = 100) -> np.ndarray:
    """Move patch B by a pose so that it lies on patch A.

    Args:
        p2: Patch B, ridges bright (inverted greyscale), 0-255.
        gt: Pose as (cos, sin, tx, ty).
        pad: Padding in pixels added around the patch before warping.

    Returns:
        The warped patch, values in [0, 1], size (160 + 2 * pad) squared.
    """
    out = (
        AFFINE(
            torch.tensor(np.float32(p2)[None, None] / 255.0), torch.tensor(np.float32([gt])), pad_width=pad
        )
        .squeeze()
        .numpy()
    )
    return out


def overlay(a1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Colour overlay of patch A (blue) and warped patch B (red); white where both have ridges.

    Args:
        a1: Padded patch A, ridges bright, 0-255.
        w2: Warped patch B, values in [0, 1].

    Returns:
        RGB image, values in [0, 1].
    """
    ov = np.zeros(a1.shape + (3,))
    ov[..., 2] = a1 / 255
    ov[..., 0] = w2
    ov[..., 1] = np.minimum(a1 / 255, w2)
    return np.clip(ov, 0, 1)


def plot_gt_check(G: pd.DataFrame) -> None:
    """Figure: four genuine pairs re-aligned with the true pose, and with a wrong pose as control.

    Args:
        G: Genuine pairs of 2002 DB1.
    """
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for k, i in enumerate([3, 50, 222, 600]):
        g = G.iloc[i]
        p1 = load_patch(g.db, g.img1, g.r1, g.c1, g.a1)
        p2 = load_patch(g.db, g.img2, g.r2, g.c2, g.a2)
        gt = [g.gt_cos, g.gt_sin, g.gt_tx, g.gt_ty]
        # work in "ridge = bright" space like the repository: invert, warp, overlay in colour
        w2 = realign(255 - p2, gt)
        a1 = np.pad(255 - p1, 100)
        axes[0, k].imshow(np.hstack([p1, np.full((160, 6), 255, np.uint8), p2]), cmap="gray")
        axes[0, k].set_title(f"pair {i}: A | B  (turn {g.gt_angle:.0f}°)", fontsize=9)
        axes[1, k].imshow(overlay(a1, w2))
        axes[1, k].set_title("B moved by TRUE pose over A\nwhite = ridges agree", fontsize=9)
        mask_ov = (a1 > 0) & (w2 > 0.02)
        corr = np.corrcoef(a1[mask_ov].ravel() / 255, w2[mask_ov].ravel())[0, 1]
        wrong = realign(255 - p2, [g.gt_cos, -g.gt_sin, g.gt_tx, g.gt_ty])
        axes[2, k].imshow(overlay(a1, wrong))
        axes[2, k].set_title("control: turn sign flipped (wrong)", fontsize=9)
        axes[1, k].set_xlabel(f"ridge agreement in overlap: {corr:.2f}", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(f"{FIG}/s2_gt_check.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def gt_agreement(G: pd.DataFrame, n: int = 200) -> list[float]:
    """Ridge correlation between patch A and the re-aligned patch B, away from the patch borders.

    Args:
        G: Genuine pairs.
        n: Number of pairs to check (pairs with overlap below 25% are skipped).

    Returns:
        Correlation per checked pair.
    """
    cs = []
    for i in range(n):
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
    return cs


def jipnet_sanity(G: pd.DataFrame, n: int = 60) -> np.ndarray:
    """Run JIPNet on the first genuine pairs.

    Args:
        G: Genuine pairs.
        n: Number of pairs.

    Returns:
        n x 4 array of (score, rotation error in degrees, translation error in pixels, overlap).
    """
    j = JIP()
    err = []
    for i in range(n):
        g = G.iloc[i]
        p1 = load_patch(g.db, g.img1, g.r1, g.c1, g.a1)
        p2 = load_patch(g.db, g.img2, g.r2, g.c2, g.a2)
        s, p = j(p1, p2)
        da = abs((np.rad2deg(np.arctan2(p[1], p[0])) - g.gt_angle + 180) % 360 - 180)
        err.append((s, da, np.hypot(p[2] - g.gt_tx, p[3] - g.gt_ty), g.overlap))
    return np.array(err)


def main() -> None:
    """Run the three checks, print their summaries and save the raw values."""
    plot_masks()
    df = pd.read_csv(ROOT + "/data/partial/2002_DB1_pairs.csv")
    G = df[df.label == 1].reset_index(drop=True)
    plot_gt_check(G)
    cs = gt_agreement(G)
    print(
        "GT ridge agreement over",
        len(cs),
        "pairs: median %.3f, 5th pct %.3f" % (np.median(cs), np.percentile(cs, 5)),
    )
    e = jipnet_sanity(G)
    print(
        "JIPNet on 60 genuine: median P(same) %.2f, median turn err %.1f deg, median shift err %.1f px"
        % tuple(np.median(e[:, :3], 0))
    )
    with open(ROOT + "/experiments/step2_checks.json", "w") as handle:
        json.dump(dict(gt_agreement=cs, jip_sanity=e.tolist()), handle)


if __name__ == "__main__":
    main()
