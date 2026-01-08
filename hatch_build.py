"""Hatchling build hook for compiling C++ extensions."""

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to compile C++ extensions."""

    def initialize(self, version, build_data):
        """Compile C++ extensions before building the wheel."""
        if self.target_name not in ("wheel", "sdist"):
            return

        # Only compile for wheel builds
        if self.target_name == "wheel":
            self._build_extensions()

    def _build_extensions(self):
        """Build C++ extensions using setuptools."""
        import pybind11
        from Cython.Build import cythonize
        from setuptools import Extension
        from setuptools.command.build_ext import build_ext

        class BuildExt(build_ext):
            """Custom build_ext to add compiler-specific options."""

            def build_extensions(self):
                ct = self.compiler.compiler_type
                opts = []

                if ct == "unix":
                    opts.append("-std=c++11")
                    opts.append("-fvisibility=hidden")
                elif ct == "msvc":
                    opts.append("/std:c++11")

                for ext in self.extensions:
                    ext.extra_compile_args = opts

                build_ext.build_extensions(self)

        # Define extensions
        ext_modules = [
            Extension(
                "data_juicer.ops.deduplicator.minhash",
                sources=["data_juicer/ops/deduplicator/minhash.cpp"],
                include_dirs=[pybind11.get_include()],
                extra_compile_args=["-fopenmp", "-O3"],
                extra_link_args=["-fopenmp"],
                language="c++",
            ),
            Extension(
                "data_juicer.ops.deduplicator.tokenize",
                sources=["data_juicer/ops/deduplicator/tokenize.pyx"],
                extra_compile_args=["-O3"],
            ),
        ]

        # Cythonize
        ext_modules = cythonize(
            ext_modules,
            compiler_directives={
                "language_level": "3",
                "embedsignature": True,
            },
        )

        # Build extensions
        from setuptools.dist import Distribution

        dist = Distribution(attrs={"ext_modules": ext_modules, "cmdclass": {"build_ext": BuildExt}})

        build_ext_cmd = dist.get_command_obj("build_ext")
        build_ext_cmd.ensure_finalized()

        # Set build directory to the package directory
        build_ext_cmd.inplace = 1
        build_ext_cmd.run()

        print("C++ extensions built successfully!")
