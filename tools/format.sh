#!/bin/bash
set -exo pipefail

# Use autoflake to fix F401
python3 -m autoflake --in-place \
    --remove-all-unused-imports \
    --remove-unused-variables \
    --ignore-init-module-imports \
    --recursive ./
python3 -m isort ./
python3 -m black ./