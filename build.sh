#!/usr/bin/env bash

set -euo pipefail

# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
trap 'cd "$original_dir"' EXIT
cd "$script_dir"

# Remove old dist file, build files, and install
rm -rf build dist
rm -rf *.egg-info
python setup.py bdist_wheel



