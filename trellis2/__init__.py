import os
from typing import Optional


def _is_truthy(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


# Ensure import-time attention backend selection stays Windows-safe when
# low-VRAM mode is requested via environment variable.
if _is_truthy(os.environ.get("ZYREX_LOW_VRAM")):
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")


from . import models
from . import modules
from . import pipelines
from . import renderers
from . import representations
from . import utils
