from data_juicer.utils.lazy_loader import LazyLoader

ffmpeg = LazyLoader("ffmpeg", "ffmpeg-python")

# keep error messages readable: FFmpeg prints the actual cause at the end
STDERR_TAIL_CHARS = 2000


def run_ffmpeg(stream, **kwargs):
    """
    Run an ffmpeg-python stream and turn its failures into a picklable error.

    ``ffmpeg.Error`` keeps ``stdout`` and ``stderr`` outside of ``args``, so it
    cannot be rebuilt after crossing a process boundary. In a multiprocess
    ``Dataset.map`` that kills the result thread and the pipeline hangs. This
    helper re-raises it as a ``RuntimeError`` that carries the tail of FFmpeg's
    stderr, which survives pickling and still tells the user what went wrong.

    :param stream: the ffmpeg-python stream to run.
    :param kwargs: extra arguments for ``stream.run``, e.g. ``capture_stderr``.
    :return: whatever ``stream.run`` returns.
    """
    try:
        return stream.run(**kwargs)
    except ffmpeg.Error as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace").strip()
        message = "ffmpeg error"
        if stderr:
            message += f": {stderr[-STDERR_TAIL_CHARS:]}"
        raise RuntimeError(message) from e
