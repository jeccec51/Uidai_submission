"""Run JIPNet / AFR-Net rebuild over all test pairs. Resumable: appends to
experiments/runs/<model>_<db>.csv and skips rows already done.
Usage: python run_pairs.py JIPNet 2002_DB1 2002_DB3 ...   (or AFRNet)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import ROOT
import sys, os, time, csv, numpy as np, pandas as pd, torch
sys.path.insert(0, ROOT + "/src")
from make_pairs import load_patch
from models_cpu import JIP, AFR, norm_pred
RUNS = ROOT + "/experiments/runs"; os.makedirs(RUNS, exist_ok=True)
COLS = ["idx", "score", "p_cos", "p_sin", "p_tx", "p_ty", "sec"]

def done_set(path):
    return set(pd.read_csv(path).idx) if os.path.exists(path) else set()

def run_jip(dbs):
    m = JIP()
    for db in dbs:
        df = pd.read_csv(f"{ROOT}/data/partial/{db}_pairs.csv"); out = f"{RUNS}/JIPNet_{db}.csv"
        done = done_set(out); new = not os.path.exists(out)
        with open(out, "a", newline="") as fh:
            w = csv.writer(fh); new and w.writerow(COLS)
            for i, g in df.iterrows():
                if i in done: continue
                p1 = load_patch(db, g.img1, g.r1, g.c1, g.a1); p2 = load_patch(db, g.img2, g.r2, g.c2, g.a2)
                t = time.perf_counter(); s, p = m(p1, p2); dt = time.perf_counter() - t
                w.writerow([i, s, *[float(x) for x in p], round(dt, 4)])
                if i % 50 == 0: fh.flush()
        print("done", db, flush=True)

@torch.no_grad()
def run_afr(dbs):
    m = AFR()
    to_t = lambda x: torch.tensor((x / 255.0)[None, None]).float()
    for db in dbs:
        df = pd.read_csv(f"{ROOT}/data/partial/{db}_pairs.csv"); out = f"{RUNS}/AFRNet_{db}.csv"
        done = done_set(out); new = not os.path.exists(out)
        with open(out, "a", newline="") as fh:
            w = csv.writer(fh); new and w.writerow(COLS)
            for key, blk in df.groupby(["img1", "r1", "c1", "a1"], sort=False):   # 1 genuine + K impostors share patch A
                todo = [i for i in blk.index if i not in done]
                if not todo: continue
                g0 = blk.iloc[0]; t0 = time.perf_counter()
                a = to_t(load_patch(db, g0.img1, g0.r1, g0.c1, g0.a1))
                fa = m.embed(a); ta = (time.perf_counter() - t0) / len(blk)   # patch A template computed once per block
                for i in todo:
                    g = df.loc[i]; t = time.perf_counter()
                    b = to_t(load_patch(db, g.img2, g.r2, g.c2, g.a2))
                    ea, eb = m.enh([a, b]); al = m.align([ea, eb])   # as in authors' code
                    b2 = m.affine(b, m.tap(al)); s = float(m.score(fa, m.embed(b2)).squeeze())
                    p = norm_pred(al.cpu().numpy().reshape(1, -1))[0]
                    w.writerow([i, s, *[float(x) for x in p], round(time.perf_counter() - t + ta, 4)])
                fh.flush()
        print("done", db, flush=True)

if __name__ == "__main__":
    {"JIPNet": run_jip, "AFRNet": run_afr}[sys.argv[1]](sys.argv[2:])
