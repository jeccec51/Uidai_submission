"""Quick look at the lowest-scoring genuine pairs and highest-scoring impostor pairs.

Usage (from the repository root): ``python src/worst_cases.py JIPNet 2002_DB1``. Writes
``reports/figures/worst_<model>_<db>.png``.
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

matplotlib.use("Agg")
from analyze import load  # noqa: E402
from make_pairs import load_patch  # noqa: E402


def plot_worst(model: str, db: str, n: int = 6) -> str:
    """Draw the ``n`` worst genuine and ``n`` worst impostor pairs of one model and database.

    Args:
        model: Model name, e.g. ``"JIPNet"``.
        db: Database key, e.g. ``"2002_DB1"``.
        n: Number of pairs per row.

    Returns:
        Path of the saved figure.
    """
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
    plt.close(fig)
    return out


def main() -> None:
    """Parse model and database and draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="model name, e.g. JIPNet")
    parser.add_argument("db", help="database key, e.g. 2002_DB1")
    args = parser.parse_args()
    print(plot_worst(args.model, args.db))


if __name__ == "__main__":
    main()
