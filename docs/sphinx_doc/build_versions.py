#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
import argparse
from pathlib import Path
from packaging import version as pv

# Repository structure and build configuration
REPO_ROOT = Path(__file__).resolve().parents[2]   
SITE_DIR = REPO_ROOT / "docs" / "sphinx_doc" / "build"            # Build output directory
TEST_DATA_REL = Path("tests/ops/data")
WORKTREES_DIR = REPO_ROOT / ".worktrees"          # Temporary worktree directory for version builds
DOCS_REL = Path("docs/sphinx_doc")
LANGS = ["en", "zh_CN"]                           # Supported documentation languages
MIN_TAG = "v1.4.0"                               # Minimum version tag to build
REMOTE = "origin"                                 # Git remote name

# Build options
KEEP_WORKTREES = False     # Whether to keep worktrees after build (default: cleanup)
HAS_SUBMODULES = False     # Set True if repo uses submodules and needs initialization

def run(cmd, cwd=None, env=None, check=True):
    """Execute shell command with logging"""
    print(f"[RUN] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, cwd=cwd, env=env, check=check)

def is_valid_tag(tag: str) -> bool:
    """Check if tag matches version pattern and meets minimum version requirement"""
    if not re.match(r"^v\d+\.\d+\.\d+$", tag):
        return False
    try:
        return pv.parse(tag) >= pv.parse(MIN_TAG)
    except Exception:
        return False

def get_tags():
    """Fetch and filter valid version tags from remote repository"""
    run(["git", "fetch", "--tags", "--force", REMOTE])
    out = subprocess.check_output(["git", "tag"], text=True).strip()
    tags = [t for t in out.splitlines() if t]
    return [t for t in tags if is_valid_tag(t)]

def ensure_clean_worktree(path: Path):
    """Remove existing worktree if present to ensure clean state"""
    if path.exists():
        try:
            run(["git", "worktree", "remove", "--force", str(path)])
        except Exception:
            shutil.rmtree(path, ignore_errors=True)

def copy_docs_source_to(wt_root: Path):
    """Copy current docs source to worktree to unify templates and extensions"""
    src = REPO_ROOT / DOCS_REL
    dst = wt_root / DOCS_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[COPY] {src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", "build", ".pyc"))

def maybe_init_submodules(wt_root: Path):
    """Initialize submodules in worktree if repository uses them"""
    if HAS_SUBMODULES:
        try:
            run(["git", "submodule", "update", "--init", "--recursive"], cwd=wt_root)
        except Exception as e:
            print(f"[WARN] submodule init failed: {e}")

def copy_markdown_files(wt_root: Path):
    for md_file in wt_root.rglob("*.md"):
        exclude_paths = ["outputs", "sphinx_doc", ".github"]
        if any(path in str(md_file) for path in exclude_paths):
            continue
        target = wt_root / DOCS_REL / "source" / md_file.relative_to(wt_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(md_file, target)
    
    shutil.copytree(wt_root / TEST_DATA_REL, wt_root / DOCS_REL / "source" / "extra" / TEST_DATA_REL, dirs_exist_ok=True)

def build_one(ref: str, ref_label: str, available_versions: list[str], enable_api_doc: bool = True):
    """Build documentation for a single version/branch"""
    # Create and setup worktree for the specific git reference
    wt = WORKTREES_DIR / ref_label
    ensure_clean_worktree(wt)
    run(["git", "worktree", "add", "--force", str(wt), ref])
    maybe_init_submodules(wt)

    # Override docs/sphinx_doc with current repo version for unified templates
    copy_docs_source_to(wt)
    copy_markdown_files(wt)

    src = wt / DOCS_REL / "source"
    if not src.exists():
        print(f"[SKIP] {ref_label}: {src} not found")
        if not KEEP_WORKTREES:
            run(["git", "worktree", "remove", "--force", str(wt)])
        return

    # Build documentation for each supported language
    for lang in LANGS:
        out_dir = SITE_DIR / lang / ref_label
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup environment variables for Sphinx build
        env = os.environ.copy()
        env["DOCS_VERSION"] = ref_label              # Documentation version label (e.g., latest, v1.5.0)
        env["GIT_REF_FOR_LINKS"] = ref               # Git reference for GitHub links
        env["AVAILABLE_VERSIONS"] = ",".join(available_versions)  # All available versions for switcher
        env["REPO_ROOT"] = str(wt)                   # Version-specific repo root for copying markdown files
        env["CODE_ROOT"] = str(wt)                   # Version-specific code root for autodoc imports
        
        # Generate the API rst files (only if enabled)
        if enable_api_doc:
            api_cmd = [
                "sphinx-apidoc",
                "-o", str(wt / DOCS_REL / "source" / "api"),
                str(wt / "data_juicer"),
                "-t", "_templates",
                "-e"
            ]
            run(api_cmd, env=env)
        
        # Execute Sphinx build command
        cmd = [
            "sphinx-build",
            "-b", "html",                            # HTML builder
            "-D", f"language={lang}",                # Set language for this build
            "-j", "auto",
            str(src),                                # Source directory
            str(out_dir),                            # Output directory
        ]
        run(cmd, env=env)

    # Cleanup worktree after successful build
    if not KEEP_WORKTREES:
        run(["git", "worktree", "remove", "--force", str(wt)])
        try:
            run(["git", "worktree", "prune"])        # Clean up worktree references
        except Exception:
            pass

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Build multi-version documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s                                    # Default: build main branch with tags and API docs enabled
  %(prog)s --no-api-doc                       # Disable API documentation generation
  %(prog)s --no-tags                          # Build branches only, exclude tags
  %(prog)s --branches main dev feature/new   # Specify branch list to build
  %(prog)s --no-tags --branches main dev     # Build specified branches only, exclude tags
        """
    )
    
    parser.add_argument(
        "--no-api-doc",
        action="store_true",
        help="Disable API documentation generation (default: API docs enabled)"
    )
    
    parser.add_argument(
        "--no-tags",
        action="store_true", 
        help="Exclude version tags, build branches only (default: include tags)"
    )
    
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["main"],
        help="Specify branch list to build (default: ['main'])"
    )
    
    return parser.parse_args()

def main():
    """Main entry point: build documentation for all versions"""
    args = parse_args()
    
    print(f"[CONFIG] API documentation generation: {'Disabled' if args.no_api_doc else 'Enabled'}")
    print(f"[CONFIG] Include version tags: {'No' if args.no_tags else 'Yes'}")
    print(f"[CONFIG] Build branches: {args.branches}")
    
    WORKTREES_DIR.mkdir(exist_ok=True)
    
    # Get version list
    versions = list(args.branches)  # Start with branches specified from command line
    
    # If tags are enabled, get and add tags
    if not args.no_tags:
        tags = get_tags()
        tags.sort(key=pv.parse, reverse=True)
        versions.extend(tags)
        print(f"[INFO] Found {len(tags)} valid tags")
    
    print(f"[INFO] Total {len(versions)} versions to build: {versions}")
    
    enable_api_doc = not args.no_api_doc
    
    # Build all specified branches
    for branch in args.branches:
        print(f"[BUILD] Building branch: {branch}")
        build_one(branch, branch, versions, enable_api_doc)
    
    # If tags are enabled, build all tag versions
    if not args.no_tags:
        tags = get_tags()
        tags.sort(key=pv.parse, reverse=True)
        for t in tags:
            print(f"[BUILD] Building tag: {t}")
            build_one(t, t, versions, enable_api_doc)

if __name__ == "__main__":
    main()