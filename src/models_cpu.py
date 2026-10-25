"""JIPNet and the authors' AFR-Net re-implementation, loaded on CPU from the official code.

Source: github.com/XiongjunGuan/JIPNet (MIT licence), commit 40d8445; weights from the
repository's download link, stored under ``ext/JIPNet/ckpts``.
"""

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


def norm_pred(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalise predicted (cos, sin) so they describe a valid rotation.

    Args:
        p: N x 4 array of (cos, sin, tx, ty) predictions.
        eps: Small constant to avoid division by zero.

    Returns:
        N x 4 array with unit-length (cos, sin).
    """
    n = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) + eps
    return np.stack([p[:, 0] / n, p[:, 1] / n, p[:, 2], p[:, 3]], 1)


class JIP:
    """JIPNet (official code and weights) running on CPU."""

    def __init__(self, patch_size: int = 160) -> None:
        """Build JIPNet from the official configuration and load the released weights.

        Args:
            patch_size: Input patch size in pixels (160 in all experiments).
        """
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
    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compare two patches.

        Args:
            img1: First patch, 8-bit greyscale (dark ridges on a light background).
            img2: Second patch, same format.

        Returns:
            ``(match_probability, pose)`` where pose is (cos, sin, tx, ty) of patch 2 relative to patch 1.
        """
        a = torch.tensor(((255.0 - img1) / 255.0)[None, None]).float()
        b = torch.tensor(((255.0 - img2) / 255.0)[None, None]).float()
        cla, al = self.m([a, b])
        return float(cla.squeeze()), norm_pred(al.cpu().numpy().reshape(1, -1))[0]


class AFR:
    """AFR-Net as re-implemented by the JIPNet authors, running on CPU.

    Chain used by the authors: RidgeNet enhancement, PFVNet AlignNet, AFR-Net embedding
    (CNN 384 + attention 384). Score = 0.2 x CNN cosine + 0.8 x attention cosine, as in their
    ``inference_AFRNet.py``. The report uses :meth:`embed` and :meth:`score` (fixed-length
    template); :meth:`__call__` reproduces the authors' full pairwise chain.
    """

    def __init__(self) -> None:
        """Build the ridge-enhancement, alignment and AFR-Net networks and load their weights."""
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
    def embed(self, x: torch.Tensor) -> np.ndarray:
        """Compute the fixed-length AFR-Net template of a patch.

        Args:
            x: Patch tensor of shape (1, 1, H, W), values in [0, 1].

        Returns:
            1 x 768 array: CNN part (384) followed by attention part (384).
        """
        pad_h, pad_w = max(0, 224 - x.shape[2]), max(0, 224 - x.shape[3])
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        x = (x - 0.5) / 0.5
        c, v = self.net(torch.cat([x, x, x], 1))
        return torch.cat([c, v], -1).cpu().numpy()

    @staticmethod
    def score(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """Similarity of two templates: 0.2 x CNN cosine + 0.8 x attention cosine.

        Args:
            f1: N x 768 templates.
            f2: N x 768 templates.

        Returns:
            N similarity scores.
        """
        cos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)).clip(1e-6)
        return 0.2 * cos(f1[:, :384], f2[:, :384]) + 0.8 * cos(f1[:, 384:], f2[:, 384:])

    @torch.no_grad()
    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compare two patches with the full pairwise chain (enhance, align, embed, score).

        Args:
            img1: First patch, 8-bit greyscale.
            img2: Second patch, same format.

        Returns:
            ``(score, pose)`` where pose is AlignNet's (cos, sin, tx, ty) estimate.
        """
        a = torch.tensor((img1 / 255.0)[None, None]).float()
        b = torch.tensor((img2 / 255.0)[None, None]).float()
        e1, e2 = self.enh([a, b])
        al = self.align([e1, e2])
        b2 = self.affine(b, self.tap(al))
        s = self.score(self.embed(a), self.embed(b2))
        return float(s.squeeze()), norm_pred(al.cpu().numpy().reshape(1, -1))[0]
