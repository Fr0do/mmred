#!/bin/bash

# Set the OUTLINES_CACHE_DIR variable to a directory named "model_name" inside the current working directory
OUTLINES_CACHE_DIR="$(pwd)/cache/model_name"

# Example usage: Create the directory if it doesn't exist
mkdir -p "$OUTLINES_CACHE_DIR"

# Print the path for verification
echo "OUTLINES_CACHE_DIR is set to: $OUTLINES_CACHE_DIR"