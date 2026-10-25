"""Fraction of each patch that lies on the fingerprint (foreground mask), for every pair."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import ROOT
import sys, numpy as np, pandas as pd, cv2
sys.path.insert(0, ROOT + "/src")
from make_pairs import DBS, RAW, fg_mask, PATCH
sys.path.insert(0, ROOT + "/ext/JIPNet/make_data"); from generate_patch import cut_patch
for db in DBS:
    df = pd.read_csv(f"{ROOT}/data/partial/{db}_pairs.csv"); M = {}
    def cov(n, r, c, a):
        if n not in M: M[n] = fg_mask(cv2.imread(f"{RAW}/{DBS[db]}/{n}.tif", 0)).astype(np.float32) * 255
        pm = cut_patch(M[n], [r, c, a], PATCH)   # padding value 255 would count as print -> use inverted mask
        return pm
    c1, c2 = [], []
    for _, x in df.iterrows():
        for (n, r, c, a), lst in [((x.img1, x.r1, x.c1, x.a1), c1), ((x.img2, x.r2, x.c2, x.a2), c2)]:
            if n not in M: M[n] = 255 - fg_mask(cv2.imread(f"{RAW}/{DBS[db]}/{n}.tif", 0)).astype(np.float32) * 255  # 0 = print
            lst.append(float((cut_patch(M[n], [r, c, a], PATCH) < 128).mean()))
    df["cov1"], df["cov2"] = c1, c2; df.to_csv(f"{ROOT}/data/partial/{db}_pairs.csv", index=False)
    print(db, "median print coverage patch1 %.2f patch2 %.2f" % (np.median(c1), np.median(c2)), flush=True)
