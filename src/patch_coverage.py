"""Add the fingerprint coverage of each patch (share of the patch on the print) to the pair tables.

Usage (from the repository root): ``python src/patch_coverage.py``. Adds the columns ``cov1``
and ``cov2`` to every ``data/partial/<db>_pairs.csv``.
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402
from make_pairs import DBS, PATCH, RAW, fg_mask  # noqa: E402

sys.path.insert(0, ROOT + "/ext/JIPNet/make_data")
from generate_patch import cut_patch  # noqa: E402


def coverage(db: str) -> pd.DataFrame:
    """Compute patch coverage for every pair of one database and save the updated table.

    The mask is inverted (0 = print) before cutting, because ``cut_patch`` pads with 255 and
    padding must not count as print.

    Args:
        db: Database key, e.g. ``"2002_DB1"``.

    Returns:
        The pair table with ``cov1`` and ``cov2`` added.
    """
    df = pd.read_csv(f"{ROOT}/data/partial/{db}_pairs.csv")
    masks: dict[str, np.ndarray] = {}
    c1, c2 = [], []
    for _, x in df.iterrows():
        for (n, r, c, a), lst in [((x.img1, x.r1, x.c1, x.a1), c1), ((x.img2, x.r2, x.c2, x.a2), c2)]:
            if n not in masks:
                masks[n] = 255 - fg_mask(cv2.imread(f"{RAW}/{DBS[db]}/{n}.tif", 0)).astype(np.float32) * 255
            lst.append(float((cut_patch(masks[n], [r, c, a], PATCH) < 128).mean()))
    df["cov1"], df["cov2"] = c1, c2
    df.to_csv(f"{ROOT}/data/partial/{db}_pairs.csv", index=False)
    print(db, "median print coverage patch1 %.2f patch2 %.2f" % (np.median(c1), np.median(c2)), flush=True)
    return df


def main() -> None:
    """Add coverage columns for all eight databases."""
    for db in DBS:
        coverage(db)


if __name__ == "__main__":
    main()
