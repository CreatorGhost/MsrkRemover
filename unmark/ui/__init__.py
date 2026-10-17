"""Local web app: upload -> detect -> remove, with live progress. FastAPI + one static page."""
import argparse
import re
import shutil
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .. import hw, log
from ..config import CACHE, cfg
from ..io import VideoInfo, grab, probe
from ..pipeline import Cancelled, analyze, estimate_seconds, remove

JOBS_DIR = CACHE / "jobs"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
BUSY = {"analyzing", "queued", "downloading", "rendering", "finalizing"}
gpu = threading.Lock()  # one render at a time; the GPU is the bottleneck anyway


@dataclass
class Job:
    id: str
    dir: Path
    src: Path
    name: str
    info: VideoInfo
    state: str = "analyzing"
    error: str = ""
    labels: np.ndarray | None = None
    regions: list = field(default_factory=list)
    estimate: float | None = None
    preview: bool = False
    preview_start: float = 0.0
    done: int = 0
    total: int = 0
    download: float = 0.0
    started: float = 0.0
    t_first: float = 0.0  # first progress tick (excludes model load from the speed)
    d_first: int = 0
    finished: float = 0.0
    output: Path | None = None
    cancel: threading.Event = field(default_factory=threading.Event)


jobs: dict[str, Job] = {}
app = FastAPI(title="Unmark")
allowed_hosts = {"127.0.0.1", "localhost"}  # main() adds the --host value; "*" when shared on the LAN


@app.middleware("http")
async def same_origin_only(request: Request, call_next):
    """Block DNS rebinding (unexpected Host) and CSRF from other sites (foreign Origin on POST)."""
    host = request.headers.get("host", "").rsplit(":", 1)[0].strip("[]")
    if "*" not in allowed_hosts and host not in allowed_hosts:
        return JSONResponse({"detail": "Host not allowed"}, 403)
    origin = request.headers.get("origin")
    if request.method != "GET" and origin and urlsplit(origin).netloc != request.headers.get("host"):
        return JSONResponse({"detail": "Cross-site request blocked"}, 403)
    return await call_next(request)


def get(job_id: str) -> Job:
    if (j := jobs.get(job_id)) is None:
        raise HTTPException(404, "Unknown job (the server may have restarted).")
    return j


def status(j: Job) -> dict:
    now = time.time()
    speed = (j.done - j.d_first) / (now - j.t_first) if j.t_first and j.done > j.d_first else None
    if j.state not in BUSY and j.finished:
        speed = j.total / max(j.finished - j.started, 1e-6)
    return {
        "id": j.id, "name": j.name, "state": j.state, "error": j.error,
        "width": j.info.width, "height": j.info.height, "duration": j.info.duration,
        "fps": round(float(eval_fps(j.info.fps)), 3), "frames": j.info.frames,
        "regions": j.regions, "estimate": j.estimate,
        "preview": j.preview, "preview_start": j.preview_start,
        "done": j.done, "total": j.total, "download": j.download, "speed": speed,
        "elapsed": ((j.finished or now) - j.started) if j.started else 0,
        "eta": (j.total - j.done) / speed if speed and j.state == "rendering" else None,
        "saved": str(j.output) if j.output and not j.preview and j.state == "done" else None,
    }


def eval_fps(fps: str) -> float:
    n, _, d = fps.partition("/")
    return float(n) / float(d or 1)


def _analyze(j: Job):
    try:
        mask = analyze(j.src)
        n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
        j.labels = labels
        j.regions = [{"id": i, "x": int(s[0]), "y": int(s[1]), "w": int(s[2]), "h": int(s[3])}
                     for i, s in enumerate(stats) if i > 0]
        if j.regions:
            j.estimate = estimate_seconds(mask, j.info, j.info.frames)
        frame = grab(j.src, j.info.duration / 2)
        cv2.imwrite(str(j.dir / "frame.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        j.state = "ready"
    except Exception as e:
        log.exception("analyze failed")
        j.state, j.error = "error", f"Could not analyze this video: {e}"


def _render(j: Job, mask: np.ndarray, start: float, duration: float | None, dst: Path):
    def on_progress(done, total):
        if not j.t_first:
            j.t_first, j.d_first = time.time(), done
        j.done, j.total = done, total
        j.state = "finalizing" if done >= total else "rendering"

    def on_download(done, total):
        j.state, j.download = "downloading", done / total if total else 0

    with gpu:
        if j.cancel.is_set():
            j.state = "cancelled"
            return
        j.state, j.started = "rendering", time.time()
        try:
            remove(j.src, dst, mask, start, duration, progress=on_progress, cancel=j.cancel, on_download=on_download)
            j.output, j.state = dst, "done"
        except Cancelled:
            j.state = "cancelled"
            dst.unlink(missing_ok=True)
        except Exception as e:
            log.exception("render failed")
            j.state, j.error = "error", str(e)
            dst.unlink(missing_ok=True)
        j.finished = time.time()


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/api/system")
def system():
    return {"device": hw.device(), "device_name": hw.device_name(), "encoder": hw.encoder("h264"),
            "out_dir": str(cfg["out_dir"]), "preview_seconds": cfg["preview_seconds"]}


@app.post("/api/upload")
def upload(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext or '?'}'. Use {', '.join(sorted(VIDEO_EXT))}.")
    jid = uuid.uuid4().hex
    d = JOBS_DIR / jid
    d.mkdir(parents=True)
    src = d / f"input{ext}"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f, 1 << 20)
    try:
        info = probe(src)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "That file isn't a readable video.")
    name = re.sub(r"[^\w\- ]+", "_", Path(file.filename).stem)[:80] or "video"
    j = jobs[jid] = Job(jid, d, src, name, info)
    threading.Thread(target=_analyze, args=(j,), daemon=True).start()
    return status(j)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return status(get(job_id))


class RenderRequest(BaseModel):
    regions: list[int] = []
    boxes: list[tuple[int, int, int, int]] = []  # x, y, w, h in video pixels
    preview: bool = False


@app.post("/api/jobs/{job_id}/render")
def render(job_id: str, req: RenderRequest):
    j = get(job_id)
    if j.state in BUSY:
        raise HTTPException(409, "This video is already being processed.")
    W, H = j.info.width, j.info.height
    mask = np.zeros((H, W), np.uint8)
    ids = [r["id"] for r in j.regions if r["id"] in req.regions]
    if ids:
        mask[np.isin(j.labels, ids)] = 255
    for x, y, w, h in req.boxes:
        x0, y0 = max(0, x), max(0, y)
        mask[y0:min(H, y + h), x0:min(W, x + w)] = 255
    if not mask.any():
        raise HTTPException(400, "Nothing selected. Drag a box over the watermark.")

    secs = cfg["preview_seconds"]
    j.preview = req.preview
    j.preview_start = max(0.0, j.info.duration / 2 - secs / 2) if req.preview else 0.0
    if req.preview:
        dst = j.dir / "preview.mp4"
    else:
        cfg["out_dir"].mkdir(parents=True, exist_ok=True)
        # Job id in the name: two jobs can never write (or cancel-delete) the same file.
        dst = cfg["out_dir"] / f"{j.name}_unmarked_{j.id[:6]}.mp4"
    j.cancel.clear()
    j.state, j.error, j.output = "queued", "", None
    j.done, j.total, j.download, j.started, j.t_first, j.finished = 0, 0, 0.0, 0.0, 0.0, 0.0
    threading.Thread(target=_render, args=(j, mask, j.preview_start, secs if req.preview else None, dst),
                     daemon=True).start()
    return status(j)


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    get(job_id).cancel.set()
    return {"ok": True}


@app.get("/api/jobs/{job_id}/frame.jpg")
def frame(job_id: str):
    p = get(job_id).dir / "frame.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/jobs/{job_id}/input")
def input_video(job_id: str):
    return FileResponse(get(job_id).src)


@app.get("/api/jobs/{job_id}/output")
def output_video(job_id: str, download: bool = False):
    j = get(job_id)
    if j.state != "done" or not j.output:
        raise HTTPException(404, "No output yet.")
    name = f"{j.name}_{'preview' if j.preview else 'unmarked'}.mp4"
    return FileResponse(j.output, media_type="video/mp4", filename=name if download else None)


def _cleanup(max_age_h: float = 24):
    """Uploads live in ~/.cache/unmark/jobs; drop ones from old sessions."""
    if JOBS_DIR.exists():
        for d in JOBS_DIR.iterdir():
            if time.time() - d.stat().st_mtime > max_age_h * 3600:
                shutil.rmtree(d, ignore_errors=True)


def main():
    import uvicorn
    ap = argparse.ArgumentParser(prog="unmark-ui")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let other devices on your network use it")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    _cleanup()
    log.info("Device: %s (%s), encoder: %s", hw.device_name(), hw.device(), hw.encoder("h264"))
    allowed_hosts.add("*" if a.host in ("0.0.0.0", "::") else a.host)
    if a.host not in ("127.0.0.1", "localhost"):
        log.warning("Listening on %s: anyone on your network can upload videos to this machine.", a.host)
    url = f"http://127.0.0.1:{a.port}"
    if not a.no_browser:
        threading.Timer(1.2, webbrowser.open, [url]).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
