# Larch v6

Discrete choice models using numba and JAX


# Developer's Installation

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

# compile and install 
python -m pip install -e ./sharrow
python -m pip install -e ./larch

# run unit tests (optional)
python -m pytest -v ./sharrow/sharrow/tests
python -m pytest -v ./larch/tests
```
