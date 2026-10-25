"""Score every test pair of the given databases with JIPNet (or the pairwise AFR-Net chain).

Usage (from the repository root)::

    python src/run_pairs.py JIPNet 2002_DB1 2002_DB3 ...
    python src/run_pairs.py AFRNet 2002_DB1 ...

Results are appended to ``experiments/runs/<model>_<db>.csv``; pairs already scored are
skipped, so a run can be resumed. The AFR-Net results in the report come from the faster
template-based ``run_afr_embed.py``; the ``AFRNet`` option here is the authors' pairwise chain.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402
from make_pairs import load_patch
from models_cpu import JIP, AFR, norm_pred

RUNS = ROOT + "/experiments/runs"
os.makedirs(RUNS, exist_ok=True)
COLS = ["idx", "score", "p_cos", "p_sin", "p_tx", "p_ty", "sec"]


def done_set(path: str) -> set[int]:
    """Pair indices already present in a results file.

    Args:
        path: Results CSV path.

    Returns:
        Set of completed pair indices (empty if the file does not exist).
    """
    return set(pd.read_csv(path).idx) if os.path.exists(path) else set()


def run_jip(dbs: list[str]) -> None:
    """Score all pairs of the given databases with JIPNet and record score, pose and run time.

    Args:
        dbs: Database keys, e.g. ``["2002_DB1", "2004_DB2"]``.
    """
    m = JIP()
    for db in dbs:
        df = pd.read_csv(f"{ROOT}/data/partial/{db}_pairs.csv")
        out = f"{RUNS}/JIPNet_{db}.csv"
        done = done_set(out)
        new = not os.path.exists(out)
        with open(out, "a", newline="") as fh:
            w = csv.writer(fh)
            new and w.writerow(COLS)
            for i, g in df.iterrows():
                if i in done:
                    continue
                p1 = load_patch(db, g.img1, g.r1, g.c1, g.a1)
                p2 = load_patch(db, g.img2, g.r2, g.c2, g.a2)
                t = time.perf_counter()
                s, p = m(p1, p2)
                dt = time.perf_counter() - t
                w.writerow([i, s, *[float(x) for x in p], round(dt, 4)])
                if i % 50 == 0:
                    fh.flush()
        print("done", db, flush=True)


@torch.no_grad()
def run_afr(dbs: list[str]) -> None:
    """Score all pairs with the authors' pairwise AFR-Net chain (enhance, align, embed, score).

    Patch A's template is computed once per block of pairs that share it. Not used for the
    report figures, which use ``run_afr_embed.py``.

    Args:
        dbs: Database keys, e.g. ``["2002_DB1"]``.
    """
    m = AFR()

    def to_t(x: np.ndarray) -> torch.Tensor:
        """8-bit patch to a (1, 1, H, W) tensor in [0, 1]."""
        return torch.tensor((x / 255.0)[None, None]).float()

    for db in dbs:
        df = pd.read_csv(f"{ROOT}/data/partial/{db}_pairs.csv")
        out = f"{RUNS}/AFRNet_{db}.csv"
        done = done_set(out)
        new = not os.path.exists(out)
        with open(out, "a", newline="") as fh:
            w = csv.writer(fh)
            new and w.writerow(COLS)
            for key, blk in df.groupby(
                ["img1", "r1", "c1", "a1"], sort=False
            ):  # 1 genuine + K impostors share patch A
                todo = [i for i in blk.index if i not in done]
                if not todo:
                    continue
                g0 = blk.iloc[0]
                t0 = time.perf_counter()
                a = to_t(load_patch(db, g0.img1, g0.r1, g0.c1, g0.a1))
                fa = m.embed(a)
                ta = (time.perf_counter() - t0) / len(blk)  # patch A template computed once per block
                for i in todo:
                    g = df.loc[i]
                    t = time.perf_counter()
                    b = to_t(load_patch(db, g.img2, g.r2, g.c2, g.a2))
                    ea, eb = m.enh([a, b])
                    al = m.align([ea, eb])  # as in authors' code
                    b2 = m.affine(b, m.tap(al))
                    s = float(m.score(fa, m.embed(b2)).squeeze())
                    p = norm_pred(al.cpu().numpy().reshape(1, -1))[0]
                    w.writerow([i, s, *[float(x) for x in p], round(time.perf_counter() - t + ta, 4)])
                fh.flush()
        print("done", db, flush=True)


def main() -> None:
    """Parse the command line and run the selected model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=["JIPNet", "AFRNet"], help="model to run")
    parser.add_argument("databases", nargs="+", help="database keys, e.g. 2002_DB1")
    args = parser.parse_args()
    {"JIPNet": run_jip, "AFRNet": run_afr}[args.model](args.databases)


if __name__ == "__main__":
    main()
