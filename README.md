# MsrkRemover (Unmark)

Removes visible logos, watermarks and channel bugs from videos **you own**, entirely on your own computer.
Runs on **NVIDIA GPUs (CUDA)**, **Apple Silicon (MPS)**, or CPU. Invisible/provenance watermarks (SynthID, C2PA)
are out of scope by design.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and FFmpeg:

   | Platform | FFmpeg |
   |---|---|
   | macOS | `brew install ffmpeg` (VideoToolbox encoder included) |
   | Windows (NVIDIA) | `winget install Gyan.FFmpeg` (NVENC included); NVIDIA driver ≥ 580 |
   | Linux (NVIDIA) | distro `ffmpeg` built with `--enable-nvenc`, or a static build; NVIDIA driver ≥ 580 |

2. Install dependencies:

   ```bash
   uv sync
   ```

   On Windows/Linux this pulls the CUDA 13.0 build of PyTorch (RTX 20xx–50xx). For an older driver, change
   `cu130` to `cu126` in `pyproject.toml` (that build doesn't support RTX 50xx).

The AI model (Big-LaMa, 196 MB, md5-verified) downloads to `~/.cache/unmark` on first use.

## Web app

```bash
uv run unmark-ui                     # opens http://127.0.0.1:7860
uv run unmark-ui --host 0.0.0.0      # let other devices on your network use it
```

Drop a video, and Unmark detects the watermark and starts removing it. You get live progress (frames, speed,
time left, finish time). Renders run on the server, so you can close the tab and reopen it later from **Your videos**.
Untick *Start removing as soon as the watermark is found* to review first: click detected boxes to skip them,
drag on the frame to mark anything missed, or render a 5-second preview with a before/after slider.
Videos are rendered one at a time; extra uploads queue. Results are saved to `~/Movies/Unmark` (macOS) or
`~/Videos/Unmark`.

> `--host 0.0.0.0` has no login: anyone on your network can upload. Use it only on networks you trust.

## CLI

```bash
uv run unmark input.mp4 -o out.mp4
uv run unmark input.mp4 --box 1650,40,220,80   # manual target, x,y,w,h in pixels
uv run unmark input.mp4 --preview               # 5 s only
```

## How it works

1. **Detect**: 24 frames sampled across the video; the median of per-frame edge strength keeps only edges that
   never move (logos), then Otsu threshold → filled blobs. Moving scene content and changing subtitles are ignored.
2. **Remove**: only a padded crop around the watermark goes through Big-LaMa on the GPU, then gets pasted back with
   feathered edges. The rest of each frame is untouched.
3. **Encode**: frames stream through FFmpeg pipes (memory stays flat for any length) into the best working encoder:
   NVENC → VideoToolbox → x264. Original audio is stream-copied; fps, resolution and color tags preserved.

Device and encoder are chosen automatically; override in `unmark/config.yaml` (`device`, `encoder`) or point
`UNMARK_CONFIG` at your own YAML.

## Why no Docker

Docker on macOS can't reach the Apple GPU (MPS) or VideoToolbox, so it would fall back to CPU and run many times
slower. On NVIDIA it would need nvidia-container-toolkit for no gain over `uv sync`. Plain `uv` runs natively everywhere.

## Benchmarks

Apple M4 Pro (48 GB), fast mode:

| Clip | Watermark crop | Time | Speed |
|---|---|---|---|
| 1080p, 5:00, 9,000 frames (real channel logo) | 501×258 | 11 min 57 s | 12.6 fps · 2.4 min per video minute |
| 1080p, 0:30, 720 frames (synthetic) | 606×312 | ~55 s | ~13 fps |

NVIDIA numbers: not measured yet. After the first render, the app records your machine's speed and shows an
up-front estimate for every new video.

## Tests

```bash
uv run pytest -s   # synthetic clip with a static + moving fake logo: pipeline end to end + web API
```

## Roadmap

- [x] Streaming IO, static-logo detection, LaMa, web app with live progress, CLI, CUDA/MPS/CPU
- [ ] SAM 2 tracking (moving overlays)
- [ ] ProPainter quality mode (`--mode quality` currently uses LaMa)
- [ ] Vision-model "brain" (local Qwen2.5-VL / OpenAI / codex) — `--brain` is accepted but ignored
