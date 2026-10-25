"""Load JIPNet and the authors' AFR-Net re-implementation on CPU, using the official repo code.
Repo: github.com/XiongjunGuan/JIPNet (MIT), commit 40d8445. Weights from the repo's Drive link."""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

REPO = os.environ.get("JIPNET_REPO", ROOT + "/ext/JIPNet")
sys.path.insert(0, REPO)
from models.JIPNet import JIPNet
from other_models.AFRNet.AFRNet import AFRNet
from other_models.PFVNet.PFVNet_AlignNet import AlignNet
from other_models.PFVNet.utils import AffinePatch as PAffine, TransformAffinePred, load_model
from other_models.RidgeNet.RidgeNet import RidgeNet
import inference_AFRNet as _afr  # for load_afrnet helper only (guarded by __main__)

CPU = torch.device("cpu")
torch.set_num_threads(os.cpu_count())


def norm_pred(p, eps=1e-6):
    n = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) + eps
    return np.stack([p[:, 0] / n, p[:, 1] / n, p[:, 2], p[:, 3]], 1)


class JIP:
    def __init__(self, patch_size=160):
        cfg = yaml.safe_load(open(f"{REPO}/ckpts/JIPNet/config.yaml"))["model_cfg"]
        self.m = JIPNet(
            input_size=patch_size,
            img_channel=cfg["img_channel"],
            num_classes=cfg["num_classes"],
            width=cfg["width"],
            enc_blk_nums=cfg["enc_blk_nums"],
            dw_expand=cfg["dw_expand"],
            ffn_expand=cfg["ffn_expand"],
            mid_blk_nums=cfg["mid_blk_nums"],
            mid_blk_strides=cfg["mid_blk_strides"],
            mid_embed_dims=cfg["mid_embed_dims"],
            dec_hidden_dim=cfg["dec_hidden_dim"],
            dec_nhead=cfg["dec_nhead"],
            dec_local_num=cfg["dec_local_num"],
        )
        sd = torch.load(f"{REPO}/ckpts/JIPNet/best.pth", map_location=CPU)
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
        self.m.load_state_dict(sd)
        self.m.eval()

    @torch.no_grad()
    def __call__(self, img1, img2):
        """img: uint8 grayscale patch (dark ridges on light background), same as repo examples.
        returns (match probability, [cos, sin, tx, ty])"""
        a = torch.tensor(((255.0 - img1) / 255.0)[None, None]).float()
        b = torch.tensor(((255.0 - img2) / 255.0)[None, None]).float()
        cla, al = self.m([a, b])
        return float(cla.squeeze()), norm_pred(al.cpu().numpy().reshape(1, -1))[0]


class AFR:
    """Authors' AFR-Net re-implementation: RidgeNet enhance -> PFVNet AlignNet -> AFR-Net embeddings (CNN 384 + ViT 384).
    score = 0.2*cos(CNN part) + 0.8*cos(ViT part), exactly as in inference_AFRNet.py."""

    def __init__(self):
        self.align = AlignNet()
        self.align.load_state_dict(torch.load(f"{REPO}/ckpts/PFVNet/alignnet.pth", map_location=CPU))
        self.net = AFRNet(input_size=224, pretrained=False, num_classes=384, is_stn=True)
        _afr.load_afrnet(self.net, f"{REPO}/ckpts/AFRNet/best.pth")
        self.enh = RidgeNet()
        load_model(self.enh, f"{REPO}/ckpts/RidgeNet/best.pth")
        for m in (self.align, self.net, self.enh):
            m.eval()
        self.affine, self.tap = PAffine(), TransformAffinePred()

    @torch.no_grad()
    def embed(self, x):
        pad_h, pad_w = max(0, 224 - x.shape[2]), max(0, 224 - x.shape[3])
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        x = (x - 0.5) / 0.5
        c, v = self.net(torch.cat([x, x, x], 1))
        return torch.cat([c, v], -1).cpu().numpy()

    @staticmethod
    def score(f1, f2):
        cos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)).clip(1e-6)
        return 0.2 * cos(f1[:, :384], f2[:, :384]) + 0.8 * cos(f1[:, 384:], f2[:, 384:])

    @torch.no_grad()
    def __call__(self, img1, img2):
        a = torch.tensor((img1 / 255.0)[None, None]).float()
        b = torch.tensor((img2 / 255.0)[None, None]).float()
        e1, e2 = self.enh([a, b])
        al = self.align([e1, e2])
        b2 = self.affine(b, self.tap(al))
        s = self.score(self.embed(a), self.embed(b2))
        return float(s.squeeze()), norm_pred(al.cpu().numpy().reshape(1, -1))[0]
