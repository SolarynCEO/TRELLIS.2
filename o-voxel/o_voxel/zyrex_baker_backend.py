from __future__ import annotations

import os
import warnings


SUPPORTED_BAKER_BACKENDS = ("nvdiffrast", "zyrex")
DEFAULT_BAKER_BACKEND = "nvdiffrast"


def resolve_baker_backend() -> str:
    backend = os.getenv("ZYREX_BAKER", DEFAULT_BAKER_BACKEND).strip().lower()
    if not backend:
        return DEFAULT_BAKER_BACKEND
    if backend not in SUPPORTED_BAKER_BACKENDS:
        warnings.warn(
            f"Unsupported ZYREX_BAKER={backend!r}; falling back to {DEFAULT_BAKER_BACKEND!r}. "
            f"Supported values: {SUPPORTED_BAKER_BACKENDS}",
            stacklevel=2,
        )
        return DEFAULT_BAKER_BACKEND
    return backend
