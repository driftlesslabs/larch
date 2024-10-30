#!/usr/bin/env bash

# exit this script immediately upon any failure
set -e

# default values
TARGET_DIR="${HOME}/driftless"
TARGET_KERNEL="LARIX"
RUN_TESTS=false
NO_KERNEL=false

# function to display help
show_help() {
  echo "Usage: cmd [-d target_directory] [-k kernel_name] [-t] [--no-kernel] [-h]"
  echo ""
  echo "Options:"
  echo "  -d    Define a target directory"
  echo "  -k    Name a kernel"
  echo "  -t    Run tests"
  echo "  --no-kernel  Do not create a Jupyter kernel"
  echo "  -h    Show help"
}

# parse options
while getopts "d:k:th-:" opt; do
  case ${opt} in
    d )
      TARGET_DIR=$OPTARG
      ;;
    k )
      TARGET_KERNEL=$OPTARG
      ;;
    t )
      RUN_TESTS=true
      ;;
    h )
      show_help
      exit 0
      ;;
    - )
      case "${OPTARG}" in
        no-kernel)
          NO_KERNEL=true
          ;;
        *)
          show_help
          exit 1
          ;;
      esac
      ;;
    \? )
      show_help
      exit 1
      ;;
  esac
done

# check that uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv is not installed, installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# check that gh is installed
if ! command -v gh &> /dev/null
then
    echo "gh is not installed, installing it..."
    curl -sS https://webi.sh/gh | sh
fi

# make the target working directory if it doesn't already exist
mkdir -p "${TARGET_DIR}"

# change to the target working directory
cd -- "${TARGET_DIR}"

# clone the various repositories
gh repo clone driftlesslabs/larch -- --recurse-submodules

# change to the larch directory
cd larch

# create/sync the UV python venv
# this will install larch in editable mode
uv sync --all-extras

# install sharrow in editable mode in the UV virtual environment
uv pip install -e ./subs/sharrow

# rip examples to loadable modules
uv run ./tools/rip_examples.py

# make this environment available to jupyter if --no-kernel option is not provided
if [ "$NO_KERNEL" = false ]; then
    uv add --dev ipykernel
    uv run ipython kernel install --user --name=$TARGET_KERNEL
fi

# run tests if -t option is provided
if [ "$RUN_TESTS" = true ]; then
    uv run pytest
fi
