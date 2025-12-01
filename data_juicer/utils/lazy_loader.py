"""A LazyLoader class for on-demand module loading with uv integration."""

import importlib
import importlib.resources
import inspect
import subprocess
import sys
import traceback
import types

import tomli
from loguru import logger


def get_toml_file_path():
    """Get the path to pyproject.toml file."""
    try:
        # First try to find it in the installed package data
        with importlib.resources.path("py_data_juicer", "pyproject.toml") as toml_path:
            return toml_path
    except (ImportError, FileNotFoundError):
        # If not found in package data, try project root
        with importlib.resources.path("data_juicer", "__init__.py") as init_path:
            project_root = init_path.parent.parent
            return project_root / "pyproject.toml"


def get_uv_lock_path():
    """Get the path to uv.lock file."""
    try:
        # First try to find it in the installed package data
        with importlib.resources.path("py_data_juicer", "uv.lock") as lock_path:
            return lock_path
    except (ImportError, FileNotFoundError):
        # If not found in package data, try project root
        with importlib.resources.path("data_juicer", "__init__.py") as init_path:
            project_root = init_path.parent.parent
            return project_root / "uv.lock"


class LazyLoader(types.ModuleType):
    """
    Lazily import a module, mainly to avoid pulling in large dependencies.
    Uses uv for fast dependency installation when available.
    """

    # Class variable to cache dependencies
    _dependencies = None

    # Mapping of module names to their corresponding package names
    _module_to_package = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "bs4": "beautifulsoup4",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
        "git": "gitpython",
    }

    @classmethod
    def get_package_name(cls, module_name: str) -> str:
        """Convert a module name to its corresponding package name.

        Args:
            module_name: The name of the module (e.g., 'cv2', 'PIL')

        Returns:
            str: The corresponding package name (e.g., 'opencv-python', 'Pillow')
        """
        # Try to get the package name from the mapping
        if module_name in cls._module_to_package:
            return cls._module_to_package[module_name]

        # If not in mapping, return the module name as is
        return module_name

    @classmethod
    def reset_dependencies_cache(cls):
        """Reset the dependencies cache."""
        cls._dependencies = None

    @classmethod
    def get_all_dependencies(cls):
        """
        Get all dependencies, prioritizing uv.lock if available.
        Falls back to pyproject.toml if uv.lock is not found or fails to parse.

        Returns:
            dict: A dictionary mapping module names to their full package specifications
                 e.g. {'numpy': 'numpy>=1.26.4,<2.0.0', 'pandas': 'pandas>=2.0.0'}
        """
        # Return cached dependencies if available
        if cls._dependencies is not None:
            return cls._dependencies

        # Try to get dependencies from uv.lock first
        try:
            lock_path = get_uv_lock_path()
            if lock_path.exists():
                with open(lock_path, "rb") as f:
                    try:
                        lock_data = tomli.load(f)
                    except Exception as e:
                        logger.debug(f"Failed to parse uv.lock: {str(e)}")
                        # Don't return empty dict here, fall back to pyproject.toml
                        pass
                    else:
                        result = {}
                        # Extract package versions from uv.lock
                        if "package" in lock_data:
                            for pkg in lock_data["package"]:
                                if "name" in pkg and "version" in pkg:
                                    name = pkg["name"]
                                    version = pkg["version"]
                                    result[name] = f"{name}=={version}"

                        if result:
                            cls._dependencies = result
                            return cls._dependencies
        except Exception as e:
            logger.debug(f"Failed to read dependencies from uv.lock: {str(e)}")

        # Fall back to pyproject.toml if uv.lock is not available or empty
        try:
            pyproject_path = get_toml_file_path()

            if not pyproject_path.exists():
                logger.debug("pyproject.toml not found")
                cls._dependencies = {}
                return cls._dependencies

            with open(pyproject_path, "rb") as f:
                try:
                    pyproject = tomli.load(f)
                except Exception as e:
                    logger.debug(f"Failed to parse pyproject.toml: {str(e)}")
                    cls._dependencies = {}
                    return cls._dependencies

            result = {}

            # Get main dependencies
            if "project" in pyproject and "dependencies" in pyproject["project"]:
                for dep in pyproject["project"]["dependencies"]:
                    if ">=" in dep or "<=" in dep or "==" in dep or ">" in dep or "<" in dep:
                        # Find the first occurrence of any version operator
                        for op in [">=", "<=", "==", ">", "<"]:
                            if op in dep:
                                name, version = dep.split(op, 1)
                                name = name.strip()
                                result[name] = f"{name}{op}{version.strip()}"
                                break
                    else:
                        name = dep.strip()
                        result[name] = name

            # Get optional dependencies
            if "project" in pyproject and "optional-dependencies" in pyproject["project"]:
                for group in pyproject["project"]["optional-dependencies"].values():
                    for dep in group:
                        if ">=" in dep or "<=" in dep or "==" in dep or ">" in dep or "<" in dep:
                            # Find the first occurrence of any version operator
                            for op in [">=", "<=", "==", ">", "<"]:
                                if op in dep:
                                    name, version = dep.split(op, 1)
                                    name = name.strip()
                                    result[name] = f"{name}{op}{version.strip()}"
                                    break
                        else:
                            name = dep.strip()
                            result[name] = name

            # Cache the dependencies
            cls._dependencies = result
            return cls._dependencies

        except Exception as e:
            logger.debug(f"Failed to read dependencies from pyproject.toml: {str(e)}")
            cls._dependencies = {}
            return cls._dependencies

    @classmethod
    def check_packages(cls, package_specs, pip_args=None):
        """
        Check if packages are installed and install them if needed.

        Args:
            package_specs: A list of package specifications to check/install.
                          Can be package names or URLs (e.g., 'torch' or 'git+https://github.com/...')
            pip_args: Optional list of additional arguments to pass to pip install command
                     (e.g., ['--no-deps', '--upgrade'])
        """

        def _is_package_installed(package_name):
            """Check if a package is installed by attempting to import it."""
            if "@" in package_name:
                package_name = package_name.split("@")[0]
            if "[" in package_name:
                package_name = package_name.split("[")[0]
            if "/" in package_name:  # Handle GitHub URLs
                package_name = package_name.split("/")[-1].replace(".git", "")
            try:
                importlib.import_module(package_name)
                return True
            except ImportError:
                return False

        # Convert pip_args to list if it's a string
        if isinstance(pip_args, str):
            pip_args = [pip_args]

        for package_spec in package_specs:
            if not _is_package_installed(package_spec):
                logger.info(f"Package [{package_spec}] not found, installing...")
                try:
                    cls._install_package(package_spec, pip_args)
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Failed to install {package_spec}. This package may "
                        f"require system-level dependencies. Please try "
                        f"installing it manually with: pip install {package_spec}\n"
                        f"Error details: {str(e)}"
                    )
            else:
                logger.info(f"Package [{package_spec}] already installed, carry on..")

    def __init__(self, module_name: str, package_name: str = None, package_url: str = None, auto_install: bool = True):
        """
        Initialize the LazyLoader.

        Args:
            module_name: The name of the module to import (e.g., 'cv2', 'ray.data', 'torchvision.models')
            package_name: The name of the pip package to install (e.g., 'opencv-python', 'ray', 'torchvision')
                        If None, will use the base module name (e.g., 'ray' for 'ray.data')
            package_url: The URL to install the package from (e.g., git+https://github.com/...)
            auto_install: Whether to automatically install missing dependencies
        """
        self._module_name = module_name

        # For installation, use the provided package_name or get it from mapping
        if package_name is None:
            base_module = module_name.split(".")[0]
            self._package_name = self.get_package_name(base_module)
        else:
            self._package_name = package_name

        # Standardize package_url to use git+ format
        if package_url and "@" in package_url:
            # Convert from package@git+ format to git+ format
            self._package_url = package_url.split("@", 1)[1]
        else:
            self._package_url = package_url

        self._auto_install = auto_install

        frame = inspect.currentframe().f_back
        self._parent_module_globals = frame.f_globals
        self._module = None

        # Print trace information
        # logger.debug(
        #     f'Initialized LazyLoader for module: {module_name} '
        #     f'(package: {self._package_name}' +
        #     (f', url: {self._package_url}' if self._package_url else '') + ')')
        # # Get last 3 frames of the stack trace
        # stack = traceback.extract_stack(frame)[-3:]
        # logger.debug('LazyLoader called from:\n' +
        #              ''.join(traceback.format_list(stack)))

        super(LazyLoader, self).__init__(module_name)

    @classmethod
    def _install_package(cls, package_spec, pip_args=None):
        """Install a package using uv if available, otherwise pip."""
        # Print trace information for package installation
        logger.debug(f"Installing package: {package_spec}")
        # Get last 3 frames of the stack trace
        stack = traceback.extract_stack()[-3:]
        logger.debug("Package installation triggered from:\n" + "".join(traceback.format_list(stack)))

        # Convert pip_args to list if it's a string
        if isinstance(pip_args, str):
            pip_args = [pip_args]

        # For GitHub repositories, clone only to get dependencies
        if package_spec.startswith(("git+", "https://github.com/")):
            import os
            import shutil
            import tempfile

            import git

            # Create a temporary directory for cloning
            temp_dir = tempfile.mkdtemp()
            try:
                # Clone the repository
                logger.info(f"Cloning {package_spec} to get dependencies...")
                if package_spec.startswith("git+"):
                    repo_url = package_spec[4:]  # Remove 'git+' prefix
                else:
                    repo_url = package_spec
                git.Repo.clone_from(repo_url, temp_dir)

                # Define all possible dependency files
                dep_files = {
                    "requirements.txt": "Installing requirements from requirements.txt...",
                    "pyproject.toml": "Installing dependencies from pyproject.toml...",
                    "setup.py": "Installing dependencies from setup.py...",
                    "setup.cfg": "Installing dependencies from setup.cfg...",
                    "Pipfile": "Installing dependencies from Pipfile...",
                    "poetry.lock": "Installing dependencies from poetry.lock...",
                }

                # Try to install dependencies from each file if it exists
                for dep_file, log_msg in dep_files.items():
                    dep_path = os.path.join(temp_dir, dep_file)
                    if os.path.exists(dep_path):
                        logger.info(log_msg)
                        try:
                            # Try uv first
                            if dep_file in ["pyproject.toml", "setup.py", "setup.cfg"]:
                                # For these files, install dependencies only
                                cmd = [sys.executable, "-m", "uv", "pip", "install", temp_dir]
                            elif dep_file == "Pipfile":
                                # For Pipfile, use pipenv
                                cmd = [sys.executable, "-m", "pipenv", "install", "--deploy", "--skip-lock"]
                            elif dep_file == "poetry.lock":
                                # For poetry.lock, use poetry
                                cmd = [sys.executable, "-m", "poetry", "install", "--no-root", "--no-sync"]
                            else:
                                # For requirements.txt, use standard pip install
                                cmd = [sys.executable, "-m", "uv", "pip", "install", "-r", dep_path]

                            if pip_args:
                                cmd.extend(pip_args)
                            subprocess.check_call(cmd)
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            logger.warning("uv not found or failed, falling back to pip...")
                            if dep_file in ["pyproject.toml", "setup.py", "setup.cfg"]:
                                cmd = [sys.executable, "-m", "pip", "install", temp_dir]
                            elif dep_file == "Pipfile":
                                cmd = [sys.executable, "-m", "pipenv", "install", "--deploy", "--skip-lock"]
                            elif dep_file == "poetry.lock":
                                cmd = [sys.executable, "-m", "poetry", "install", "--no-root", "--no-sync"]
                            else:
                                cmd = [sys.executable, "-m", "pip", "install", "-r", dep_path]
                            if pip_args:
                                cmd.extend(pip_args)
                            subprocess.check_call(cmd)

                # Install the package directly from remote
                try:
                    logger.info(f"Installing {package_spec} directly from remote...")
                    cmd = [sys.executable, "-m", "uv", "pip", "install", "--force-reinstall", package_spec]
                    if pip_args:
                        cmd.extend(pip_args)
                    subprocess.check_call(cmd)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("uv not found or failed, falling back to pip...")
                    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", package_spec]
                    if pip_args:
                        cmd.extend(pip_args)
                    subprocess.check_call(cmd)
                return True
            finally:
                # Clean up the temporary directory
                shutil.rmtree(temp_dir)
        else:
            # Get the full package spec from dependencies
            deps = cls.get_all_dependencies()
            package_name = package_spec.split("@")[0] if "@" in package_spec else package_spec
            if "[" in package_name:
                package_name = package_name.split("[")[0]
            if "/" in package_name:  # Handle GitHub URLs
                package_name = package_name.split("/")[-1].replace(".git", "")

            # Use the version from dependencies if available and not a URL
            is_url = package_spec.startswith(("git+", "https://"))
            if package_name in deps and not is_url:
                package_spec = deps[package_name]
                logger.info(f"Using version from dependencies: {package_spec}")
            else:
                logger.warning(
                    f"No version constraint found in pyproject.toml for {package_name}, "
                    f"using original spec: {package_spec}"
                )

            # For non-GitHub packages, use direct installation
            try:
                logger.info(f"Installing {package_spec} using uv...")
                cmd = [sys.executable, "-m", "uv", "pip", "install", package_spec]
                if pip_args:
                    cmd.extend(pip_args)
                subprocess.check_call(cmd)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("uv not found or failed, falling back to pip...")
                cmd = [sys.executable, "-m", "pip", "install", package_spec]
                if pip_args:
                    cmd.extend(pip_args)
                subprocess.check_call(cmd)
                return True

    def _load(self):
        """Load the module and handle any missing dependencies."""
        logger.debug(f"Loading {self._module_name}...")

        if self._module is not None:
            return self._module

        try:
            # Try to import the module directly first
            self._module = importlib.import_module(self._module_name)
        except ImportError:
            if not self._auto_install:
                raise

            # Prepare the package spec for installation
            package_spec = self._package_url if self._package_url else self._package_name

            # Install the package
            try:
                self._install_package(package_spec)
            except subprocess.CalledProcessError as e:
                raise ImportError(
                    f"Failed to install {package_spec}. This package may "
                    f"require system-level dependencies. Please try "
                    f"installing it manually with: pip install {package_spec}\n"
                    f"Error details: {str(e)}"
                )

            # Try importing again
            try:
                self._module = importlib.import_module(self._module_name)
            except ImportError as import_error:
                raise ImportError(
                    f"Failed to import {self._module_name} after "
                    f"installing {package_spec}. "
                    f"Error details: {str(import_error)}"
                )

        # Update the parent module's globals with the loaded module
        self._parent_module_globals[self._module_name] = self._module
        self.__dict__.update(self._module.__dict__)
        return self._module

    def __getattr__(self, item):
        """Handle attribute access, including submodule imports."""
        if self._module is None:
            self._load()

        # Try to get the attribute directly
        try:
            return getattr(self._module, item)
        except AttributeError:
            # If not found, try importing it as a submodule
            try:
                submodule = importlib.import_module(f"{self._module_name}.{item}")
                setattr(self._module, item, submodule)
                return submodule
            except ImportError:
                raise AttributeError(f"module '{self._module_name}' has no attribute '{item}'")

    def __dir__(self):
        if self._module is None:
            self._load()
        return dir(self._module)
