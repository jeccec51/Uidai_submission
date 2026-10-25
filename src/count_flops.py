"""Machine-independent cost: floating-point operations per network run (torch FlopCounterMode)."""

import json
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402
from models_cpu import JIP, AFR

out = {}
j = JIP()
a = torch.rand(1, 1, 160, 160)
b = torch.rand(1, 1, 160, 160)
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    j.m([a, b])
out["JIPNet_pair_GFLOP"] = fc.get_total_flops() / 1e9
m = AFR()
x = torch.rand(1, 1, 160, 160)
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    m.embed(x)
out["AFRNet_template_GFLOP"] = fc.get_total_flops() / 1e9
out["JIPNet_params_M"] = sum(p.numel() for p in j.m.parameters()) / 1e6
out["AFRNet_params_M"] = sum(p.numel() for p in m.net.parameters()) / 1e6
json.dump(out, open(ROOT + "/experiments/flops.json", "w"), indent=1)
print(out)
