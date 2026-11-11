import logging
import math
import os

LOG = logging.getLogger("harness")


def number(name: str, default: int | float, *, minimum: float = 0, maximum: float = math.inf):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw) if type(default) is int else float(raw)
        if math.isfinite(value) and minimum <= value <= maximum:
            return value
    except ValueError:
        pass
    LOG.warning("invalid %s; using %s", name, default)
    return default
