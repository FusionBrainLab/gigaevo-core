#!/bin/bash

set -exo pipefail

python3 -m isort -c ./
python3 -m black --check ./
python3 -m flake8 ./