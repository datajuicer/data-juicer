import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple

from loguru import logger

from data_juicer.config import init_configs
from data_juicer.utils.lazy_loader import LazyLoader


def is_local_import(module_name: str) -> bool:
    """Check if a module name is a relative or local import."""
    logger.info(f"Checking if {module_name} is a relative or local import")
    return module_name.startswith("data_juicer") or module_name.startswith("__")  # local imports  # special imports


def get_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all imports from a Python file."""
    imports = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if not is_local_import(name.name):
                        imports.add(name.name)
            elif isinstance(node, ast.ImportFrom):
                # For ImportFrom, we need to check both the module and the level
                # level > 0 means it's a relative import
                if node.level > 0:
                    continue
                if node.module and not is_local_import(node.module):
                    imports.add(node.module)
    except Exception as e:
        logger.warning(f"Failed to parse imports from {file_path}: {e}")
    return imports


def find_lazy_loaders(file_path: Path) -> List[Tuple[str, str]]:
    """Find all LazyLoader instances in a file and their package names."""
    lazy_loaders = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "LazyLoader":
                    # Get the module name (first argument)
                    if node.args and isinstance(node.args[0], ast.Constant):
                        module_name = node.args[0].value
                        # Get the package name (second argument) if provided
                        package_name = None
                        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                            package_name = node.args[1].value
                        elif len(node.keywords) > 0:
                            for kw in node.keywords:
                                if kw.arg == "package_name" and isinstance(kw.value, ast.Constant):
                                    package_name = kw.value.value
                                    break
                        lazy_loaders.append((module_name, package_name))
    except Exception as e:
        logger.warning(f"Failed to parse LazyLoader instances from {file_path}: {e}")
    return lazy_loaders


def get_operator_imports(op_name: str) -> Set[str]:
    """Get all imports needed by an operator."""
    # Try to find the operator file in all op subdirectories
    op_dirs = ["filter", "mapper", "deduplicator", "selector", "aggregrator", "grouper"]
    op_paths = []
    for op_dir in op_dirs:
        op_paths.extend(
            [
                Path(__file__).parent.parent / "data_juicer" / "ops" / op_dir / f"{op_name}.py",
                Path(__file__).parent.parent / "data_juicer" / "ops" / op_dir / op_name / "__init__.py",
            ]
        )

    imports = set()
    for path in op_paths:
        if path.exists():
            # Get regular imports
            imports.update(get_imports_from_file(path))

            # Get LazyLoader dependencies
            lazy_loaders = find_lazy_loaders(path)
            for module_name, package_name in lazy_loaders:
                if package_name:
                    imports.add(package_name)
                else:
                    # If no package name specified, use the base module name
                    imports.add(module_name.split(".")[0])

    return imports


def get_package_name(module_name: str) -> str:
    """Convert a module name to its corresponding package name."""
    return LazyLoader.get_package_name(module_name)


def main():
    # Initialize config and get operators
    cfg = init_configs()
    op_names = [list(op.keys())[0] for op in cfg.process]
    logger.info(f"Operators: {op_names}")

    # Get all required imports
    all_imports = set()
    for op_name in op_names:
        op_imports = get_operator_imports(op_name)
        all_imports.update(op_imports)

    logger.info(f"All imports: {all_imports}")

    # Convert imports to package names
    required_packages = {get_package_name(imp) for imp in all_imports}
    logger.info(f"Required packages: {required_packages}")

    if not required_packages:
        logger.warning("No dependencies found for the specified operators.")
        return

    # Install dependencies using LazyLoader
    try:
        logger.info("Resolving dependencies...")
        LazyLoader.check_packages(list(required_packages))
        logger.info("Dependencies resolved successfully.")
    except Exception as e:
        logger.error(f"Failed to resolve dependencies: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
