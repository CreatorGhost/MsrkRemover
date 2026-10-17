import hashlib
import urllib.request

from . import log
from .config import CACHE


def fetch(name: str, url: str, md5: str | None = None, progress=None):
    """Download once to ~/.cache/unmark, verify checksum. progress(done_bytes, total_bytes)."""
    path = CACHE / name
    if path.exists():
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s (first run only)", name)
    tmp, h = path.with_suffix(".part"), hashlib.md5()
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
            elif total:
                print(f"\r  {name}: {done * 100 // total}%", end="", flush=True)
    if md5 and h.hexdigest() != md5:
        tmp.unlink()
        raise RuntimeError(f"{name}: checksum mismatch, download deleted")
    tmp.rename(path)
    return path
