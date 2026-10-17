import os
import sys
from pathlib import Path

import yaml

CACHE = Path.home() / ".cache" / "unmark"

cfg = yaml.safe_load((Path(__file__).with_name("config.yaml")).read_text())
if os.environ.get("UNMARK_CONFIG"):
    cfg.update(yaml.safe_load(Path(os.environ["UNMARK_CONFIG"]).read_text()) or {})
if cfg["out_dir"] == "auto":
    cfg["out_dir"] = Path.home() / ("Movies" if sys.platform == "darwin" else "Videos") / "Unmark"
cfg["out_dir"] = Path(cfg["out_dir"]).expanduser()
