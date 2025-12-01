#!/bin/bash

# Get project root directory (assuming we're in docs/sphinx_doc)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKTREES_DIR="$PROJECT_ROOT/.worktrees"

# Cleanup function: handle git worktree related issues
cleanup_worktrees() {
    echo "Cleaning up git worktrees..."
    echo "Project root: $PROJECT_ROOT"
    
    # Change to project root for git operations
    cd "$PROJECT_ROOT"
    
    # 1. Prune invalid worktree references
    git worktree prune 2>/dev/null || true
    
    # 2. Force remove all worktrees in .worktrees directory
    if [ -d "$WORKTREES_DIR" ]; then
        echo "Found .worktrees directory at: $WORKTREES_DIR"
        for wt_dir in "$WORKTREES_DIR"/*; do
            if [ -d "$wt_dir" ]; then
                wt_name=$(basename "$wt_dir")
                echo "  Removing worktree: $wt_name"
                # Try normal removal first
                git worktree remove --force "$wt_dir" 2>/dev/null || {
                    # If normal removal fails, force delete directory
                    echo "  Force deleting directory: $wt_dir"
                    rm -rf "$wt_dir"
                }
            fi
        done
        # Remove empty .worktrees directory
        rmdir "$WORKTREES_DIR" 2>/dev/null || true
    fi
    
    # 3. Prune worktree references again
    git worktree prune 2>/dev/null || true
    
    echo "Worktree cleanup completed"
    
    # Return to original directory
    cd - > /dev/null
}

# Error handling function
handle_error() {
    echo "Error occurred during build process, cleaning up..."
    cleanup_worktrees
    exit 1
}

# Set up error handling
trap handle_error ERR

# Store current directory
ORIGINAL_DIR=$(pwd)

# Pre-cleanup: ensure clean environment before starting
echo "Pre-cleanup before build..."
echo "Current directory: $ORIGINAL_DIR"
cleanup_worktrees

# Execute original build process (back in docs/sphinx_doc)
echo "Starting build..."
make clean
python build_versions.py

# # Post-build cleanup (optional, as build_versions.py already has cleanup logic)
# echo "Build completed, performing final cleanup..."
# cleanup_worktrees

echo "All operations completed successfully!"