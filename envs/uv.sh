#!/usr/bin/env zsh

# exit this script immediately upon any failure
set -e

# check that uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv is not installed, install it with 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
    exit 1
fi

# check that gh is installed
if ! command -v gh &> /dev/null
then
    echo "gh is not installed, install it with 'brew install gh'"
    exit 1
fi

# if provided, take the first shell argument as the target directory
# otherwise default HOME/driftless
if [ -n "$1" ]; then
    export TARGET_DIR="$1"
else
    export TARGET_DIR="${HOME}/driftless"
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
uv sync


# install larch in editable mode in a UV virtual environment

# install the conda development environment
#mkdir -p .env/LARIX
#conda env update -p .env/LARIX -f larch/envs/development.yaml
#conda activate .env/LARIX

# make this environment available to jupyter
#ipython kernel install --user --name=LARIX

# rip examples to loadable modules
uv run ./tools/rip_examples.py

## prep sharrow submodule
#uv pip install -e ./larch/sharrow
#
## compile and install
#uv pip install -e ./larch
#
## run unit tests (optional)
#mkdir sandbox
#cd sandbox
#python -m pytest -v ../larch/sharrow/sharrow/tests
#python -m pytest -v ../larch/tests
