#!/usr/bin/env bash

###############################################################################
# This script is used to create a development environment for activitysim's
# estimation mode.  It installs the activitysim and larch repos in the target
# directory, creates a conda environment, and installs the repos in the
# environment in editable mode.  It also installs pre-commit hooks for both
# repos. If the --no-kernel option is not provided, it also makes the
# environment available to Jupyter as a kernel.
###############################################################################

# exit this script immediately upon any failure
set -e

# default values
TARGET_DIR="${HOME}/driftless"
TARGET_KERNEL="ESTER"
RUN_TESTS=false
NO_KERNEL=false
ORIGINAL_PWD=$(pwd)

# function to display help
show_help() {
  echo "Usage: cmd [-d target_directory] [-k kernel_name] [-t] [--no-kernel] [--local-repo path] [-h]"
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

# check that conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda is not installed, please install it."
    exit 1
fi

# check that uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv is not installed, installing it ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# check that gh is installed
if ! command -v gh &> /dev/null
then
    echo "gh is not installed, installing it ..."
    curl -sS https://webi.sh/gh | sh
fi

# make the target working directory if it doesn't already exist
mkdir -p "${TARGET_DIR}"

# change to the target working directory
cd -- "${TARGET_DIR}"

# initialize conda in the shell
eval "$(conda shell.bash hook)"

# create a conda environment
conda create -p .env/${TARGET_KERNEL} python=3.10 --yes

# activate the conda environment
conda activate .env/${TARGET_KERNEL}

# install pydot for rendering nesting trees
conda install pydot --yes

# clone the repos
gh repo clone driftlesslabs/activitysim
gh repo clone driftlesslabs/larch -- --recurse-submodules

# install the repos
uv pip install -e ./larch/subs/sharrow
uv run ./larch/tools/rip_examples.py
uv pip install -e ./larch
uv pip install -e ./activitysim

# install development dependencies
uv pip install altair asv "black<23" cytoolz dask filelock geopandas h5py isort \
  matplotlib myst-parser nbconvert nbformat nbmake numpydoc psutil pyarrow \
  pycodestyle pydata-sphinx-theme pyinstrument pypyr pytest pytest-regressions \
  rich ruby setuptools_scm scikit-learn simwrapper sparse sphinx sphinx_rtd_theme \
  sphinx-argparse zarr zstandard xlsxwriter

cd larch
uvx pre-commit install
cd ..

cd activitysim
uvx pre-commit install
cd ..

# make this environment available to jupyter if --no-kernel option is not provided
if [ "$NO_KERNEL" = false ]; then
    echo "Making this environment available to Jupyter ..."
    uv pip install ipykernel
    ipython kernel install --user --name=${TARGET_KERNEL}
fi

if [ "$RUN_TESTS" = true ]; then
    echo "Running sharrow tests ..."
    uv run python -m pytest larch/subs/sharrow/sharrow/tests
fi
