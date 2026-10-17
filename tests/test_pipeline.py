import numpy as np

from synth import FPS, N, STATIC_BOX, frame, write
from unmark.io import probe, read_frames
from unmark.pipeline import analyze, remove


def test_static_logo_end_to_end(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    write(src)

    mask = analyze(src)
    x, y, w, h = STATIC_BOX
    assert (mask[y:y + h, x:x + w] > 0).mean() > 0.9, "static logo not detected"
    assert (mask[270:330, :] > 0).mean() < 0.05, "moving bug / scene wrongly masked"

    remove(src, dst, mask)
    a, b = probe(src), probe(dst)
    assert (b.width, b.height, b.fps, b.has_audio) == (a.width, a.height, f"{FPS}/1", True)
    assert b.color.get("color_space") == "bt709"
    outs = list(read_frames(dst, b))
    ins = list(read_frames(src, a))
    assert len(outs) == N

    sl = np.s_[y - 4:y + h + 4, x - 4:x + w + 4]
    err_in = np.mean([np.abs(ins[i][sl].astype(int) - frame(i, False)[sl]).mean() for i in range(N)])
    err_out = np.mean([np.abs(outs[i][sl].astype(int) - frame(i, False)[sl]).mean() for i in range(N)])
    print(f"logo-region MAE vs clean: input {err_in:.1f} -> output {err_out:.1f}")
    # Logo is pure white; the textured scene almost never is. Count what survives.
    white = lambda fs: sum(int((f[sl].min(axis=2) > 235).sum()) for f in fs)
    w_in, w_out = white(ins), white(outs)
    print(f"near-white logo pixels: {w_in} -> {w_out}")
    assert w_out < 0.02 * w_in
    assert err_out < 0.7 * err_in  # loose: nothing can recreate random noise exactly
