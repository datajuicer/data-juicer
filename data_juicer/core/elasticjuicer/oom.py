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


def snapshot_is_oom(snapshot) -> bool:
    """Read producer classification, with a narrow fallback for old samples."""

    def attr(name, default=None):
        return snapshot.get(name, default) if isinstance(snapshot, dict) else getattr(snapshot, name, default)

    if attr("succeeded", True):
        return False
    classified = attr("oom")
    if isinstance(classified, bool):
        return classified
    name = str(attr("error_type", "")).replace("_", "").lower()
    return "outofmemory" in name or name.rsplit(".", 1)[-1] == "memoryerror"
