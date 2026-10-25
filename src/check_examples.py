"""Set-up check: run both models on the four example pairs shipped with the JIPNet repository.

The outputs are compared with the authors' saved results (``ext/reference_results``).

Usage (from the repository root): ``python src/check_examples.py``. Writes
``experiments/step1_examples.json``.
"""

import json
import os
import re
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402
from models_cpu import AFR, JIP, REPO  # noqa: E402


def read_ref(path: str) -> tuple[float, list[float]]:
    """Read one of the authors' saved results.

    Args:
        path: Result text file.

    Returns:
        ``(score, pose)`` with pose as (cos, sin, tx, ty).
    """
    with open(path) as handle:
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+", handle.read())]
    return nums[0], nums[1:5]


def main() -> None:
    """Run both models on the example pairs, print and save our and the authors' outputs."""
    out = {}
    for name, M in [("JIPNet", JIP), ("AFRNet", AFR)]:
        m = M()
        out[name] = []
        for i in range(4):
            a = cv2.imread(f"{REPO}/examples/data/{i}_1.png", 0)
            b = cv2.imread(f"{REPO}/examples/data/{i}_2.png", 0)
            t = time.perf_counter()
            s, pose = m(a, b)
            dt = time.perf_counter() - t
            rs, rp = read_ref(f"{ROOT}/ext/reference_results/{name}/{i}.txt")
            out[name].append(
                dict(
                    pair=i,
                    ours=round(s, 4),
                    ref=rs,
                    pose_ours=[round(float(x), 3) for x in pose],
                    pose_ref=rp,
                    sec=round(dt, 3),
                    shape=list(a.shape),
                )
            )
            print(name, out[name][-1])
    with open(ROOT + "/experiments/step1_examples.json", "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
