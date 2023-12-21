# Larch v6

Discrete choice models using numba and JAX


# Developer's Installation

Before you start with the installation, you need have the following tools already:
- [miniforge for Mac or Linux](https://github.com/conda-forge/miniforge#unix-like-platforms-mac-os--linux)
  or [miniforge for Windows](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe)
- [gh](https://cli.github.com), the github command line tool (`mamba install gh` should work)

For now, you also need to have a github account and have authenticated 
with `gh` using `gh auth login`.  Once this repository is public, this
will no longer be necessary.

```zsh
#!/usr/bin/env zsh

# exit this script immediately upon any failure
set -e

# Change this directory to where you want to install this whole mess
export TARGET_DIR="${HOME}/driftless"

# make the target working directory if it doesn't already exist
mkdir -p "${TARGET_DIR}"

# change to the target working directory
cd -- "${TARGET_DIR}"

# clone the various repositories
gh repo clone driftlesslabs/larch
gh repo clone driftlesslabs/sharrow

# install the mamba development environment
mkdir -p .env/LARIX
mamba env update -p .env/LARIX -f larch/envs/development.yaml
conda activate .env/LARIX

# rip examples to loadable modules
python larch/tools/rip_examples.py

# compile and install 
python -m pip install -e ./sharrow
python -m pip install -e ./larch

# run unit tests (optional)
python -m pytest -v ./sharrow/sharrow/tests
python -m pytest -v ./larch/tests
```
