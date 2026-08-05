"""Narrow OOM classification shared by adaptive execution wrappers."""

_RUNTIME_OOM_MARKERS = (
    "out of memory",
    "cuda error: memory allocation",
    "cublas_status_alloc_failed",
    "cannot allocate memory",
    "failed to allocate memory",
)


def is_oom_error(error: BaseException) -> bool:
    """Return whether an exception is a recoverable allocation OOM."""

    if isinstance(error, MemoryError):
        return True
    if "outofmemory" in error.__class__.__name__.replace("_", "").lower():
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return any(marker in message for marker in _RUNTIME_OOM_MARKERS)
