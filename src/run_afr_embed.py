"""AFR-Net-style baseline in its intended form: one fixed-length template per patch.

Each patch gets a 768-number template (CNN 384 + attention 384, with AFR-Net's own built-in
alignment module); templates are compared by cosine, score = 0.2 x CNN + 0.8 x attention
(weights as in the JIPNet authors' ``inference_AFRNet.py``). No pairwise alignment network.

Subsample: every other genuine pair of the database (400). Impostor pairs: patch A of each
finger against patch B of every other finger (all combinations).

Usage (from the repository root): ``python src/run_afr_embed.py 2002_DB1 2002_DB3``. Writes
``experiments/runs_afr/<db>.npz``.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402
from make_pairs import load_patch  # noqa: E402
from models_cpu import AFR  # noqa: E402

OUT = ROOT + "/experiments/runs_afr"


def unit_rows(x: np.ndarray) -> np.ndarray:
    """Scale each row to unit length."""
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run_database(m: AFR, db: str) -> None:
    """Compute templates for the subsampled pairs of one database and save the similarity matrix.

    Args:
        m: Loaded AFR-Net model.
        db: Database key, e.g. ``"2002_DB1"``.
    """
    df = pd.read_csv(f"{ROOT}/data/partial/{db}_pairs.csv")
    G = df[df.label == 1].iloc[::2].reset_index()  # 400 genuine
    t0 = time.perf_counter()
    FA, FB = [], []
    with torch.no_grad():
        for _, g in G.iterrows():
            a = torch.tensor((load_patch(db, g.img1, g.r1, g.c1, g.a1) / 255.0)[None, None]).float()
            b = torch.tensor((load_patch(db, g.img2, g.r2, g.c2, g.a2) / 255.0)[None, None]).float()
            f = m.embed(torch.cat([a, b]))
            FA.append(f[0])
            FB.append(f[1])
    FA, FB = np.array(FA), np.array(FB)
    sec = (time.perf_counter() - t0) / (2 * len(G))
    S = (
        0.2 * unit_rows(FA[:, :384]) @ unit_rows(FB[:, :384]).T
        + 0.8 * unit_rows(FA[:, 384:]) @ unit_rows(FB[:, 384:]).T
    )
    finger = G.img1.str.split("_").str[0].values
    diff = finger[:, None] != finger[None, :]
    np.savez(f"{OUT}/{db}.npz", S=S, diff=diff, pair_idx=G["index"].values, sec_per_patch=sec)
    print(db, "genuine", len(G), "impostor", int(diff.sum()), "sec/patch %.2f" % sec, flush=True)


def main() -> None:
    """Parse the database list and run the baseline on each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("databases", nargs="+", help="database keys, e.g. 2002_DB1")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    m = AFR()
    for db in args.databases:
        run_database(m, db)


if __name__ == "__main__":
    main()
