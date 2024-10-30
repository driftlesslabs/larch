#!/usr/bin/env bash

# exit this script immediately upon any failure
set -e

# default values
TARGET_DIR="${HOME}/driftless"
TARGET_KERNEL="LARIX"
RUN_TESTS=false
NO_KERNEL=false
LOCAL_REPO_PATH=""
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
  echo "  --local-repo  Path to a local repository to clone from"
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
        local-repo)
          LOCAL_REPO_PATH="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
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

# check if the target directory is a clone of the larch repo
if [ -d ".git" ]; then
  REMOTE_URL=$(git config --get remote.origin.url)
  NORMALIZED_REMOTE_URL=${REMOTE_URL%.git}
  if [ "$NORMALIZED_REMOTE_URL" = "https://github.com/driftlesslabs/larch" ]; then
    echo "The target directory is already a clone of the larch repository."
    echo "Updating submodules ..."
    git submodule update --init --recursive
  else
    echo "The target directory is a git repository with remote URL: $REMOTE_URL"
    echo "This is not the expected remote URL."
    exit 1
  fi
else
  # clone the repository
  if [ -n "$LOCAL_REPO_PATH" ]; then
    if [[ "$LOCAL_REPO_PATH" != /* ]]; then
      LOCAL_REPO_PATH="file://$ORIGINAL_PWD/$LOCAL_REPO_PATH"
    elif [[ "$LOCAL_REPO_PATH" != file://* ]]; then
      LOCAL_REPO_PATH="file://$LOCAL_REPO_PATH"
    fi
    echo "Cloning the larch repository from $LOCAL_REPO_PATH ..."
    git clone --recurse-submodules "$LOCAL_REPO_PATH" larch
  else
    echo "Cloning the larch repository from GitHub: driftlesslabs/larch ..."
    gh repo clone driftlesslabs/larch -- --recurse-submodules
  fi
  cd larch
fi

# create/sync the UV python venv
# this will install larch in editable mode
echo "Creating/syncing the UV python venv ..."
uv sync --all-extras

# install sharrow in editable mode in the UV virtual environment
echo "Installing sharrow in editable mode ..."
uv pip install -e ./subs/sharrow

# rip examples to loadable modules
echo "Ripping examples to loadable modules ..."
uv run ./tools/rip_examples.py

# ensure pre-commit hooks are installed
echo "Ensuring pre-commit hooks are installed ..."
uvx pre-commit install

# make this environment available to jupyter if --no-kernel option is not provided
if [ "$NO_KERNEL" = false ]; then
    echo "Making this environment available to Jupyter ..."
    uv add --dev ipykernel
    uv run ipython kernel install --user --name=$TARGET_KERNEL
fi

# run tests if -t option is provided
if [ "$RUN_TESTS" = true ]; then
    echo "Running tests ..."
    uv run pytest
fi
