"""Proof of setup: run both models on the 4 example pairs shipped in the repo and compare
with the authors' saved outputs (examples/results/*/N.txt)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import ROOT
import json, re, time, cv2, numpy as np, sys
sys.path.insert(0, ROOT + "/src")
from models_cpu import JIP, AFR, REPO
def read_ref(p):
    t = open(p).read(); nums = [float(x) for x in re.findall(r"-?\d+\.\d+", t)]
    return nums[0], nums[1:5]
out = {}
for name, M in [("JIPNet", JIP), ("AFRNet", AFR)]:
    m = M(); out[name] = []
    for i in range(4):
        a = cv2.imread(f"{REPO}/examples/data/{i}_1.png", 0); b = cv2.imread(f"{REPO}/examples/data/{i}_2.png", 0)
        t = time.perf_counter(); s, pose = m(a, b); dt = time.perf_counter() - t
        rs, rp = read_ref(f"{ROOT}/ext/reference_results/{name}/{i}.txt")
        out[name].append(dict(pair=i, ours=round(s, 4), ref=rs, pose_ours=[round(float(x), 3) for x in pose], pose_ref=rp, sec=round(dt, 3), shape=list(a.shape)))
        print(name, out[name][-1])
json.dump(out, open(ROOT + "/experiments/step1_examples.json", "w"), indent=1)
