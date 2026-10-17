"""FFmpeg pipe IO: frames stream through memory, never dumped to disk."""
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..hw import encoder, encoder_args

# ffprobe colorspace -> swscale matrix name, so RGB->YUV uses the source's matrix.
_MATRIX = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "bt2020nc": "bt2020"}


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: str          # exact rational, e.g. "30000/1001"
    duration: float
    codec: str
    has_audio: bool
    bit_rate: int | None
    color: dict

    @property
    def frames(self) -> int:
        return round(self.duration * float(Fraction(self.fps)))


def probe(path) -> VideoInfo:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                          "-show_format", str(path)], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    w, h = v["width"], v["height"]
    # ffmpeg autorotates on decode, so phone videos come out with swapped dims.
    rot = next((int(d.get("rotation", 0)) for d in v.get("side_data_list", []) if "rotation" in d), 0)
    if abs(rot) % 180 == 90:
        w, h = h, w
    # r_frame_rate is exact for CFR; for VFR it's a timebase (e.g. 90000/1), so fall back to the average.
    fps, avg = v["r_frame_rate"], v.get("avg_frame_rate", "0/0")
    if avg not in ("0/0", "0/1") and float(Fraction(fps)) > 2 * float(Fraction(avg)):
        fps = avg
    br = v.get("bit_rate") or info["format"].get("bit_rate")
    return VideoInfo(
        width=w, height=h, fps=fps,
        duration=float(v.get("duration") or info["format"]["duration"]),
        codec=v["codec_name"],
        has_audio=any(s["codec_type"] == "audio" for s in info["streams"]),
        bit_rate=int(br) if br else None,
        color={k: v[k] for k in ("color_primaries", "color_transfer", "color_space", "color_range")
               if v.get(k) and v[k] != "unknown"},
    )


def read_frames(path, info: VideoInfo, start: float = 0, duration: float | None = None):
    """Yield HxWx3 uint8 RGB frames, one at a time."""
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-map", "0:v:0", "-r", info.fps, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = info.width * info.height * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        try:
            while len(buf := p.stdout.read(n)) == n:
                yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
        finally:
            p.kill()


def grab(path, t: float) -> np.ndarray:
    """Single RGB frame at time t (fast input seek)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    import cv2
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class Writer:
    """Encode RGB frames with the best available encoder, copying the source's audio untouched."""

    def __init__(self, dst, src, info: VideoInfo, start: float = 0, duration: float | None = None):
        enc = encoder("hevc" if info.codec == "hevc" else "h264")
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{info.width}x{info.height}", "-r", info.fps, "-i", "-"]
        if info.has_audio:
            cmd += ["-ss", str(start)] + (["-t", str(duration)] if duration else []) + ["-i", str(src)]
            cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "copy", "-shortest"]
        if m := _MATRIX.get(info.color.get("color_space")):
            cmd += ["-vf", f"scale=out_color_matrix={m}"]
        flags = {"color_primaries": "-color_primaries", "color_transfer": "-color_trc",
                 "color_space": "-colorspace", "color_range": "-color_range"}
        for k, v in info.color.items():
            cmd += [flags[k], v]
        # ponytail: 8-bit RGB pipeline; 10-bit/HDR sources come out 8-bit SDR-tagged-as-source.
        cmd += encoder_args(enc, int((info.bit_rate or 8_000_000) * 1.3))
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if "hevc" in enc or "265" in enc:
            cmd += ["-tag:v", "hvc1"]  # QuickTime/Finder playback
        self.p = subprocess.Popen(cmd + [str(dst)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def __enter__(self):
        return self

    def __exit__(self, exc, *_):
        self.p.stdin.close()
        if exc:
            self.p.kill()
        elif self.p.wait():
            raise RuntimeError("ffmpeg encode failed")
