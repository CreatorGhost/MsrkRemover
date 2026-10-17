"""Synthetic test clip: panning textured scene + static "logo" + moving fake logo + tone audio."""
import subprocess
import sys

import cv2
import numpy as np

W, H, FPS, N = 640, 360, 24, 72
STATIC_BOX = (470, 20, 150, 50)  # x, y, w, h


def scene(i: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    big = cv2.GaussianBlur(rng.integers(0, 255, (H + 40, W * 2, 3), dtype=np.uint8), (0, 0), 6)
    big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
    x = i * 4  # camera pan
    return np.ascontiguousarray(big[20:20 + H, x:x + W])


def frame(i: int, logos=True) -> np.ndarray:
    f = scene(i)
    if logos:
        x, y, w, h = STATIC_BOX
        cv2.rectangle(f, (x, y), (x + w, y + h), (255, 255, 255), 3)
        cv2.putText(f, "LOGO", (x + 15, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        mx = 40 + i * 6  # moving bug (stage 2 / SAM 2 target)
        cv2.circle(f, (mx, 300), 18, (255, 220, 0), -1)
    return f


def write(path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-f", "lavfi", "-i", f"sine=frequency=440:duration={N / FPS}",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                          "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                          "-c:a", "aac", "-shortest", str(path)], stdin=subprocess.PIPE)
    for i in range(N):
        p.stdin.write(frame(i).tobytes())
    p.stdin.close()
    assert p.wait() == 0


if __name__ == "__main__":
    write(sys.argv[1] if len(sys.argv) > 1 else "synthetic.mp4")
