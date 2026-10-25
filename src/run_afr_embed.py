"""AFR-Net-style baseline as AFR-Net is designed: one fixed-length code per patch (CNN 384 + attention 384,
AFR-Net's own built-in alignment module), compare codes by cosine; score = 0.2*CNN + 0.8*attention
(weights as in the JIPNet authors' inference_AFRNet.py). No pairwise alignment network.
Genuine: first 400 genuine pairs of the DB (every other impression). Impostor: patch A of one finger vs
patch B of every other finger (all combinations)."""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402
from make_pairs import load_patch
from models_cpu import AFR

m = AFR()
out = ROOT + "/experiments/runs_afr"

os.makedirs(out, exist_ok=True)
for db in sys.argv[1:]:
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
    n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    S = 0.2 * n(FA[:, :384]) @ n(FB[:, :384]).T + 0.8 * n(FA[:, 384:]) @ n(FB[:, 384:]).T
    finger = G.img1.str.split("_").str[0].values
    diff = finger[:, None] != finger[None, :]
    np.savez(f"{out}/{db}.npz", S=S, diff=diff, pair_idx=G["index"].values, sec_per_patch=sec)
    print(db, "genuine", len(G), "impostor", int(diff.sum()), "sec/patch %.2f" % sec, flush=True)
