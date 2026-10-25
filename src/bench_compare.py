"""Time ONE template comparison (fixed-length embedding) on ONE CPU core.
One comparison = dot product of the new template with one stored template
(templates are scaled to length 1, so the dot product IS the cosine similarity)
+ checking the score against a threshold.
Making the template itself (running the network) is NOT included - that is done
once per print, not once per comparison."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import ROOT
import os
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]: os.environ[v]="1"
import numpy as np, time, json
rng=np.random.default_rng(0); res={}
M=1_000_000; THR=0.5
for d in [768,192,32]:
    G=rng.standard_normal((M,d),dtype=np.float32); G/=np.linalg.norm(G,axis=1,keepdims=True)
    q=rng.standard_normal(d,dtype=np.float32); q/=np.linalg.norm(q)
    best=1e9
    for _ in range(6):
        t=time.perf_counter()
        score=G@q                 # 1 million cosine similarities
        hits=np.flatnonzero(score>THR)   # threshold check
        best=min(best,time.perf_counter()-t)
    res[d]=best/M
    print(f"{d:4d} numbers: {best*1e3:.1f} ms for 1 million comparisons -> {best/M*1e9:.1f} ns each")
json.dump(res,open(ROOT + "/experiments/bench.json","w"))
