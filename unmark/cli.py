import argparse
from pathlib import Path

import numpy as np

from . import log
from .config import cfg
from .io import probe
from .pipeline import analyze, remove


def main():
    ap = argparse.ArgumentParser(prog="unmark", description="Remove visible overlays from a video you own.")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--mode", choices=["quality", "fast"], default="quality")
    ap.add_argument("--brain", choices=["local", "openai", "codex"], default="local")
    ap.add_argument("--box", action="append", default=[], metavar="X,Y,W,H",
                    help="manual target in pixels (repeatable); skips auto-detection")
    ap.add_argument("--preview", action="store_true", help=f"render only {cfg['preview_seconds']}s")
    a = ap.parse_args()
    if a.mode == "quality":
        log.warning("quality mode (ProPainter) arrives in stage 3; using fast (LaMa)")
    log.info("brain=%s not wired yet (stage 4); using static-overlay detector", a.brain)

    info = probe(a.input)
    if a.box:
        mask = np.zeros((info.height, info.width), np.uint8)
        for b in a.box:
            x, y, w, h = map(int, b.split(","))
            mask[y:y + h, x:x + w] = 255
    else:
        mask = analyze(a.input)
        if not mask.any():
            raise SystemExit("No static overlay found. Pass --box X,Y,W,H.")
    out = a.output or a.input.with_name(a.input.stem + "_unmarked.mp4")
    remove(a.input, out, mask, duration=cfg["preview_seconds"] if a.preview else None,
           progress=lambda d, t: print(f"\r{d}/{t} frames", end="", flush=True))
    print(f"\n{out}")


if __name__ == "__main__":
    main()
