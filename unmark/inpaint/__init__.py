import cv2
import numpy as np
import torch

from ..hw import device
from ..weights import fetch


class Lama:
    """Big-LaMa TorchScript (same weights IOPaint ships), on CUDA, MPS or CPU."""

    def __init__(self, on_download=None):
        p = fetch("big-lama.pt", "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
                  md5="e3aa4aaa15225a33ec84f9f4bc47e500", progress=on_download)
        self.dev = device()
        self.model = torch.jit.load(str(p), map_location="cpu").to(self.dev).eval()

    def __call__(self, imgs: np.ndarray, mask: np.ndarray, max_side: int = 0) -> np.ndarray:
        """imgs: NxHxWx3 uint8, mask: HxW uint8 (255 = fill). Returns NxHxWx3 uint8.
        max_side>0 runs on a downscaled copy (LaMa cost ~ pixels; fp16 isn't an option, its FFT rejects it)."""
        h, w = mask.shape
        s = max_side / max(h, w) if max_side else 1
        if s >= 1:
            return self._run(imgs, mask)
        size = (round(w * s), round(h * s))
        small = np.stack([cv2.resize(i, size, interpolation=cv2.INTER_AREA) for i in imgs])
        out = self._run(small, cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))
        return np.stack([cv2.resize(o, (w, h), interpolation=cv2.INTER_CUBIC) for o in out])

    @torch.inference_mode()
    def _run(self, imgs, mask):
        n, h, w, _ = imgs.shape
        ph, pw = -h % 8, -w % 8  # LaMa needs /8 dims
        x = np.pad(imgs, ((0, 0), (0, ph), (0, pw), (0, 0)), mode="symmetric")
        m = np.pad(mask, ((0, ph), (0, pw)), mode="symmetric")
        x = torch.from_numpy(x).to(self.dev).permute(0, 3, 1, 2).float() / 255
        m = (torch.from_numpy(m).to(self.dev) > 127).float()[None, None].expand(n, 1, -1, -1)
        out = self.model(x, m)[:, :, :h, :w]
        return (out.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
