"""Machine-independent cost: floating-point operations and parameter counts of both networks.

Counts are taken with ``torch.utils.flop_counter.FlopCounterMode`` on 160x160 inputs.

Usage (from the repository root): ``python src/count_flops.py``. Writes ``experiments/flops.json``.
"""

import json
from typing import Any, Callable
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402
from models_cpu import AFR, JIP  # noqa: E402


def count_gflop(fn: Callable, *args: Any) -> float:
    """Floating-point operations of one call, in units of 10^9.

    Args:
        fn: Callable to measure.
        *args: Arguments passed to ``fn``.

    Returns:
        GFLOP of the call.
    """
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn(*args)
    return fc.get_total_flops() / 1e9


def main() -> None:
    """Count JIPNet per pair and AFR-Net per template, and save the results."""
    j = JIP()
    m = AFR()
    out = {
        "JIPNet_pair_GFLOP": count_gflop(j.m, [torch.rand(1, 1, 160, 160), torch.rand(1, 1, 160, 160)]),
        "AFRNet_template_GFLOP": count_gflop(m.embed, torch.rand(1, 1, 160, 160)),
        "JIPNet_params_M": sum(p.numel() for p in j.m.parameters()) / 1e6,
        "AFRNet_params_M": sum(p.numel() for p in m.net.parameters()) / 1e6,
    }
    with open(ROOT + "/experiments/flops.json", "w") as handle:
        json.dump(out, handle, indent=1)
    print(out)


if __name__ == "__main__":
    main()
