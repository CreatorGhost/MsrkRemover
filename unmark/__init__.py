import logging
import os
import warnings

# Must be set before torch is imported anywhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

log = logging.getLogger("unmark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

_show = warnings.showwarning


def _flag_cpu_fallback(message, category, filename, lineno, file=None, line=None):
    if "fall back to run on the CPU" in str(message):
        log.warning("MPS->CPU fallback: %s", message)
    else:
        _show(message, category, filename, lineno, file, line)


warnings.showwarning = _flag_cpu_fallback
