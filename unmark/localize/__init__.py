import cv2
import numpy as np

from ..io import grab


def static_mask(frames: list[np.ndarray]) -> np.ndarray:
    """Overlays that never move keep their edges in every frame; the scene doesn't.
    Median of per-frame gradient magnitude isolates persistent edges -> Otsu -> blobs."""
    h, w = frames[0].shape[:2]
    s = min(1.0, 1280 / w)  # detect at <=1280 wide, memory stays small for 4K
    grads = []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        g = g.astype(np.float32)
        grads.append(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)))
    med = np.median(np.stack(grads), axis=0)
    med = np.clip(med / max(np.percentile(med, 99.9), 1e-6) * 255, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    k = max(3, int(0.012 * m.shape[1]))  # merge letters of a wordmark into one blob
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    out = np.zeros_like(m)
    mh, mw = m.shape
    for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # ponytail: size heuristics drop specks, letterbox edges and static backgrounds;
        # the brain stage replaces this guesswork with per-overlay labels.
        if area < 1e-4 * mh * mw or cw > 0.5 * mw or ch > 0.5 * mh:
            continue
        cv2.drawContours(out, [c], -1, 255, -1)  # filled -> holes inside logos closed
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)


def detect_static(path, info, n: int) -> np.ndarray:
    ts = np.linspace(0, info.duration, n + 2)[1:-1]
    return static_mask([grab(path, t) for t in ts])
