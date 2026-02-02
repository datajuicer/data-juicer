#!/bin/bash

# Get the cache directory from data_juicer
CACHE_DIR=$(python -c "from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE; print(DATA_JUICER_ASSETS_CACHE)")
DROID_CALIB_PATH="$CACHE_DIR/DroidCalib"

echo "Target installation path: $DROID_CALIB_PATH"

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR"

# Clone DroidCalib if it doesn't exist
if [ ! -d "$DROID_CALIB_PATH" ]; then
    echo "Cloning DroidCalib..."
    git clone https://github.com/1van2ha0/DroidCalib.git "$DROID_CALIB_PATH"
else
    echo "DroidCalib repo already exists at $DROID_CALIB_PATH"
fi

cd "$DROID_CALIB_PATH" || exit

# Clean up existing egg-info to avoid "Multiple .egg-info directories found" error
if ls *.egg-info 1> /dev/null 2>&1; then
    echo "Cleaning up existing egg-info..."
    rm -rf *.egg-info
fi

echo "Installing DroidCalib..."
python setup.py install

echo "Installation complete."
