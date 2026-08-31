"""Build API docs from real modules and verify their Sphinx inventory."""

import re
from pathlib import Path

import pyximport
from sphinx.errors import ExtensionError
from sphinx.util import logging

logger = logging.getLogger(__name__)


def verify_api(app, exception):
    if exception is not None:
        return
    modules = {
        name
        for source in (Path(app.srcdir) / "api").glob("*.rst")
        for name in re.findall(r"^\.\. automodule:: (\S+)", source.read_text(), re.MULTILINE)
    }
    documented = {name for name, _, kind, *_ in app.env.get_domain("py").get_objects() if kind == "module"}
    missing = modules - documented
    if not modules or missing:
        raise ExtensionError(f"API documentation is incomplete: {sorted(missing) or 'no API modules generated'}")
    logger.info("Verified API documentation for %d modules", len(modules))


def setup(app):
    # Each historical version is imported from its own source worktree. Compile
    # its Cython module on demand, using that version's source.
    pyximport.install(build_dir=str(Path(app.doctreedir) / "cython"), language_level=3)

    try:
        from data_juicer.utils.lazy_loader import LazyLoader
    except ModuleNotFoundError as exc:
        if exc.name != "data_juicer.utils.lazy_loader":
            raise
    else:
        original_getattr = LazyLoader.__getattr__

        def docs_getattr(self, name):
            # Sphinx probes __sphinx_mock__, __spec__, etc. on module members.
            # These metadata queries should leave optional imports deferred.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return original_getattr(self, name)

        def require_installed_dependency(cls, package_spec, pip_args=None):
            raise ImportError(f"Install API build dependency explicitly: {package_spec}")

        LazyLoader.__getattr__ = docs_getattr
        LazyLoader._install_package = classmethod(require_installed_dependency)

    app.connect("build-finished", verify_api)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
