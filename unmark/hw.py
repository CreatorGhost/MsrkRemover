"""Pick compute device and video encoder for whatever machine we're on (CUDA, Apple MPS, or CPU)."""
import platform
import shutil
import subprocess
from functools import cache

from .config import cfg

if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    raise SystemExit("ffmpeg/ffprobe not found on PATH. Install FFmpeg first (see README).")


@cache
def device() -> str:
    import torch
    if cfg["device"] != "auto":
        return cfg["device"]
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # crop size is fixed per job, so autotune pays off
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@cache
def device_name() -> str:
    d = device()
    if d == "cuda":
        import torch
        return torch.cuda.get_device_name(0)
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        return out.stdout.strip() or "Apple Silicon"
    return platform.processor() or "CPU"


@cache
def _works(enc: str) -> bool:
    # ffmpeg lists nvenc/videotoolbox even when the hardware is absent; a 1-frame encode tells the truth.
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.1",
                        "-frames:v", "1", "-c:v", enc, "-f", "null", "-"], capture_output=True)
    return r.returncode == 0


@cache
def encoder(codec: str = "h264") -> str:
    """codec: 'h264' or 'hevc'. Hardware first, software last (always works)."""
    if cfg["encoder"] != "auto":
        return cfg["encoder"]
    sw = {"h264": "libx264", "hevc": "libx265"}[codec]
    for enc in (f"{codec}_nvenc", f"{codec}_videotoolbox", sw):
        if _works(enc):
            return enc
    return "libx264"


def encoder_args(enc: str, bitrate: int) -> list[str]:
    if enc.startswith("lib"):
        return ["-c:v", enc, "-crf", "18", "-preset", "medium"]
    extra = ["-preset", "p5"] if enc.endswith("nvenc") else []
    return ["-c:v", enc, "-b:v", str(bitrate), *extra]
