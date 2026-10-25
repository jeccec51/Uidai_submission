"""Build partial-print test pairs from FVC (Track A: known shift + turn).
Genuine pair: both 160x160 patches cut from the SAME impression, exactly with the JIPNet
authors' own recipe (make_data/generate_patch.py): patch-1 centre near the print centre,
patch-2 centre on a ring of radius r (0..100 px) around it, each patch turned by a random
angle in [-180, 180]. True relative pose = authors' reverse_info() -> (cos, sin, tx, ty).
Impostor pair: the same patch-1 against a patch from a random OTHER finger (K per genuine).
Only the recipe is stored (image, centre, angle); patches are re-cut on the fly."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import ROOT
import sys, os, glob, random, json, numpy as np, cv2, pandas as pd
REPO = ROOT + "/ext/JIPNet"; sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/make_data")
from generate_patch import generate_pos, generate_pos_single, cut_patch
from data_loader import reverse_info
RAW = ROOT + "/data/raw/74034_3_En_4_MOESM1_ESM"
OUT = ROOT + "/data/partial"
DBS = {"2002_DB1": "FVC2002/Dbs/Db1_a", "2002_DB2": "FVC2002/Dbs/Db2_a", "2002_DB3": "FVC2002/Dbs/Db3_a", "2002_DB4": "FVC2002/Dbs/Db4_a",
       "2004_DB1": "FVC2004/Dbs/DB1_A", "2004_DB2": "FVC2004/Dbs/DB2_A", "2004_DB3": "FVC2004/Dbs/DB3_A", "2004_DB4": "FVC2004/Dbs/DB4_A"}
GROUP = {"2004_DB1": "seen in training", "2004_DB2": "seen in training", "2002_DB1": "paper test set", "2002_DB3": "paper test set"}
PATCH, RING_MAX, K_IMP = 160, 100, 3

def fg_mask(img, blk=16, erode=12):
    """Foreground = blocks with ridge texture (high local spread of grey values)."""
    f = img.astype(np.float32)
    mu = cv2.blur(f, (blk, blk)); sd = np.sqrt(np.maximum(cv2.blur(f*f, (blk, blk)) - mu*mu, 0))
    m = (sd > max(12.0, 0.35 * np.percentile(sd, 95))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)            # keep largest blob
    if n > 1: m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return cv2.erode(m, np.ones((erode, erode), np.uint8))

def square(c, ang, s=PATCH):
    """corners of a patch centred at c=(row,col), turned by ang deg (cv2 convention) -> (x,y) polygon."""
    th = np.deg2rad(ang); h = s / 2
    pts = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])   # inverse of cv2's rotation, image coords
    return (pts @ R.T + [c[1], c[0]]).astype(np.float32)

def overlap(info1, info2, mask):
    H, W = mask.shape; pad = PATCH
    c1 = np.zeros((H + 2*pad, W + 2*pad), np.uint8); c2 = c1.copy()
    cv2.fillPoly(c1, [np.round(square(info1[:2], info1[2]) + pad).astype(np.int32)], 1)
    cv2.fillPoly(c2, [np.round(square(info2[:2], info2[2]) + pad).astype(np.int32)], 1)
    mp = np.pad(mask, pad)
    inter = c1 & c2
    return inter.sum() / c1.sum(), (inter & mp).sum() / c1.sum()   # geometric overlap, overlap on print

def build(db, seed=2026):
    random.seed(seed); np.random.seed(seed)
    files = sorted(glob.glob(f"{RAW}/{DBS[db]}/*.tif"), key=lambda p: tuple(map(int, os.path.basename(p)[:-4].split("_"))))
    names = [os.path.basename(p)[:-4] for p in files]
    masks = {n: fg_mask(cv2.imread(p, 0)) for n, p in zip(names, files)}
    rows = []
    for n in names:
        finger = n.split("_")[0]; mask = masks[n]
        r = random.randint(0, RING_MAX)
        try: pos1, pos2 = generate_pos(mask, ring_r=r)
        except Exception: continue
        a1, a2 = random.randint(-180, 180), random.randint(-180, 180)
        info1 = np.array([pos1[0], pos1[1], a1], float); info2 = np.array([pos2[0], pos2[1], a2], float)
        rv = reverse_info(info1, info2)
        gt = [np.cos(np.deg2rad(rv[2])), np.sin(np.deg2rad(rv[2])), rv[1], rv[0]]
        ov, ovp = overlap(info1, info2, mask)
        base = dict(db=db, img1=n, r1=info1[0], c1=info1[1], a1=a1)
        rows.append({**base, "label": 1, "img2": n, "r2": info2[0], "c2": info2[1], "a2": a2, "ring": r,
                     "gt_cos": gt[0], "gt_sin": gt[1], "gt_tx": gt[2], "gt_ty": gt[3],
                     "gt_angle": float(np.rad2deg(np.arctan2(gt[1], gt[0]))), "overlap": ov, "overlap_print": ovp})
        others = [m for m in names if m.split("_")[0] != finger]
        for m in random.sample(others, K_IMP):
            p2 = generate_pos_single(masks[m]); a = random.randint(-180, 180)
            rows.append({**base, "label": 0, "img2": m, "r2": float(p2[0]), "c2": float(p2[1]), "a2": a})
    df = pd.DataFrame(rows); df["group"] = GROUP.get(db, "never seen")
    os.makedirs(OUT, exist_ok=True); df.to_csv(f"{OUT}/{db}_pairs.csv", index=False)
    return df, masks

def load_patch(db, name, r, c, a, cache={}):
    key = (db, name)
    if key not in cache:
        if len(cache) > 300: cache.clear()
        cache[key] = cv2.imread(f"{RAW}/{DBS[db]}/{name}.tif", 0).astype(np.float32)
    return cut_patch(cache[key], [r, c, a], PATCH).astype(np.uint8)

if __name__ == "__main__":
    for db in (sys.argv[1:] or DBS):
        df, _ = build(db)
        g = df[df.label == 1]
        print(db, "genuine", len(g), "impostor", (df.label == 0).sum(), "median overlap %.2f" % g.overlap.median(), "on-print %.2f" % g.overlap_print.median())
