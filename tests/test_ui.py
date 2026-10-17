import time

from fastapi.testclient import TestClient

from synth import write
from unmark.ui import app

c = TestClient(app)


def wait(jid, busy):
    for _ in range(300):
        s = c.get(f"/api/jobs/{jid}").json()
        if s["state"] not in busy:
            return s
        time.sleep(0.2)
    raise TimeoutError(s)


def test_upload_detect_render_cancel(tmp_path):
    assert c.post("/api/upload", files={"file": ("x.txt", b"hi")}).status_code == 400
    assert c.post("/api/upload", files={"file": ("x.mp4", b"not a video")}).status_code == 400
    assert c.get("/api/jobs/nope").status_code == 404

    write(tmp_path / "in.mp4")
    s = c.post("/api/upload", files={"file": ("my clip.mp4", (tmp_path / "in.mp4").read_bytes())}).json()
    s = wait(s["id"], {"analyzing"})
    assert s["state"] == "ready" and s["regions"], s
    assert c.get(f"/api/jobs/{s['id']}/frame.jpg").status_code == 200
    assert c.post(f"/api/jobs/{s['id']}/render", json={}).status_code == 400  # nothing selected

    r = c.post(f"/api/jobs/{s['id']}/render", json={"regions": [s["regions"][0]["id"]], "preview": True})
    assert r.status_code == 200
    s = wait(s["id"], {"queued", "downloading", "rendering", "finalizing"})
    assert s["state"] == "done", s
    out = c.get(f"/api/jobs/{s['id']}/output", headers={"Range": "bytes=0-99"})
    assert out.status_code == 206  # range requests -> video seeking works

    c.post(f"/api/jobs/{s['id']}/render", json={"boxes": [[0, 0, 50, 50]]})
    c.post(f"/api/jobs/{s['id']}/cancel")
    s = wait(s["id"], {"queued", "downloading", "rendering", "finalizing"})
    assert s["state"] in ("cancelled", "done")  # tiny clip may finish before cancel lands
