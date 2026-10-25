"""Time one fixed-length template comparison on one CPU core.

One comparison is the dot product of a query template with one stored template (templates are
unit length, so this is the cosine similarity) plus the threshold check. Computing the template
itself is not included; it is done once per print, not once per comparison.

Usage (from the repository root): ``python src/bench_compare.py``. Writes
``experiments/bench.json`` (seconds per comparison, by template length).
"""

import json
import os
import sys
import time

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_var] = "1"  # must be set before numpy is imported

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

GALLERY = 1_000_000
THRESHOLD = 0.5
REPEATS = 6


def time_comparison(d: int, rng: np.random.Generator) -> float:
    """Best-of-``REPEATS`` time per comparison against a random gallery of unit templates.

    Args:
        d: Template length.
        rng: Random generator.

    Returns:
        Seconds per comparison.
    """
    G = rng.standard_normal((GALLERY, d), dtype=np.float32)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    q = rng.standard_normal(d, dtype=np.float32)
    q /= np.linalg.norm(q)
    best = 1e9
    for _ in range(REPEATS):
        t = time.perf_counter()
        score = G @ q  # one million cosine similarities
        np.flatnonzero(score > THRESHOLD)  # threshold check
        best = min(best, time.perf_counter() - t)
    return best / GALLERY


def main() -> None:
    """Time template lengths 768, 192 and 32 and save the results."""
    rng = np.random.default_rng(0)
    res = {}
    for d in [768, 192, 32]:
        res[d] = time_comparison(d, rng)
        print(
            f"{d:4d} numbers: {res[d] * GALLERY * 1e3:.1f} ms for 1 million comparisons -> {res[d] * 1e9:.1f} ns each"
        )
    with open(ROOT + "/experiments/bench.json", "w") as handle:
        json.dump(res, handle)


if __name__ == "__main__":
    main()
