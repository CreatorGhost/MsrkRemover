import json
import threading
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import log
from .config import CACHE, cfg
from .io import Writer, probe, read_frames
from .localize import detect_static

_lama = None
_SPEED = CACHE / "speed.json"  # measured seconds per crop-megapixel-frame, per device


class Cancelled(Exception):
    pass


def lama(on_download=None):
    global _lama
    if _lama is None:
        from .inpaint import Lama
        _lama = Lama(on_download)
    return _lama


def analyze(src) -> np.ndarray:
    return detect_static(src, probe(src), cfg["static_samples"])


def prepare(mask: np.ndarray, info):
    """Dilated mask, feathered alpha and the padded crop box (y0, y1, x0, x1) that gets inpainted."""
    hard = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg["dilate_px"] + 1,) * 2))
    f = cfg["feather_px"]
    alpha = cv2.GaussianBlur(hard.astype(np.float32) / 255, (0, 0), f) if f else hard / 255.0
    alpha = np.maximum(alpha, hard / 255.0)[..., None]  # feather outward only, never expose the logo
    ys, xs = np.nonzero(hard)
    pad = cfg["crop_pad"] + 3 * f
    box = (max(0, ys.min() - pad), min(info.height, ys.max() + pad + 1),
           max(0, xs.min() - pad), min(info.width, xs.max() + pad + 1))
    return hard, alpha, box


def _work_mp(box) -> float:
    """Megapixels LaMa actually processes per frame (after the max-side downscale)."""
    y0, y1, x0, x1 = box
    h, w, m = y1 - y0, x1 - x0, cfg["lama_max_side"]
    s = min(1.0, m / max(h, w)) if m else 1.0
    return h * w * s * s / 1e6


def estimate_seconds(mask: np.ndarray, info, frames: int) -> float | None:
    """From the last measured run on this device; None until one render has finished."""
    from .hw import device
    try:
        rate = json.loads(_SPEED.read_text())[device()]
    except (OSError, KeyError, ValueError):
        return None
    return frames * _work_mp(prepare(mask, info)[2]) * rate


def remove(src, dst, mask: np.ndarray, start: float = 0, duration: float | None = None,
           progress=None, cancel: threading.Event | None = None, on_download=None) -> Path:
    """Inpaint `mask` (HxW uint8, 255 = overlay) out of every frame. Streams; memory stays flat.
    progress(done_frames, total_frames)."""
    info = probe(src)
    if not mask.any():
        raise ValueError("Empty mask: nothing to remove. Mark the watermark first.")
    hard, alpha, box = prepare(mask, info)
    y0, y1, x0, x1 = box
    cmask, calpha = hard[y0:y1, x0:x1], alpha[y0:y1, x0:x1]
    ring = (cmask == 0)  # pixels outside the mask, used for motion detection

    total = round((duration or info.duration - start) * Fraction(info.fps))
    model, prev_fill, prev_crop = lama(on_download), None, None
    smooth = cfg["smooth"]
    log.info("Inpainting %dx%d crop over ~%d frames", x1 - x0, y1 - y0, total)

    def flush(batch):
        nonlocal prev_fill, prev_crop
        crops = np.stack([b[y0:y1, x0:x1] for b in batch])
        fills = model(crops, cmask, cfg["lama_max_side"]).astype(np.float32)
        for frame, crop, fill in zip(batch, crops, fills):
            # Anti-flicker on still shots only; on motion it would smear.
            if prev_fill is not None and np.abs(crop[ring].astype(np.int16) - prev_crop[ring]).mean() < cfg["smooth_max_motion"]:
                fill = (1 - smooth) * fill + smooth * prev_fill
            prev_fill, prev_crop = fill, crop
            out = frame.copy()
            out[y0:y1, x0:x1] = (calpha * fill + (1 - calpha) * crop).round().astype(np.uint8)
            w.write(out)

    t0, done, batch = time.time(), 0, []
    with Writer(dst, src, info, start, duration) as w:
        for frame in read_frames(src, info, start, duration):
            if cancel and cancel.is_set():
                raise Cancelled()
            batch.append(frame)
            if len(batch) == cfg["batch"]:
                flush(batch)
                done += len(batch)
                batch = []
                if progress:
                    progress(min(done, total), total)
        if batch:
            flush(batch)
            done += len(batch)
    if done >= 100:  # remember speed for next time's up-front estimate
        from .hw import device
        try:
            speeds = json.loads(_SPEED.read_text())
        except (OSError, ValueError):
            speeds = {}
        speeds[device()] = (time.time() - t0) / done / _work_mp(box)
        _SPEED.write_text(json.dumps(speeds))
    if progress:
        progress(total, total)
    return Path(dst)
