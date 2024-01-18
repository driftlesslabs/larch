# Larch v6

This repository contains the under-development next generation of the Larch
package, for estimating and applying discrete choice models. Version 6
is a substantial rewrite of the package, changing to a platform that allows
swapping out the underlying computational engine, so that the same model
can run in numba or JAX.

:warning: **This is a work in progress.**  A lot of things are working, but not
everything.  The interface is quite similar to Larch v5 and existing users will
likely find it familiar. If you want to try it out, please do, and feel free to
open issues in the issue tracker.  But, please don't expect it to work perfectly
yet, especially for more advanced models.


# Quick Start Guide

You can install Larch v6 with pip:

```shell
python -m pip install larch6
```

This will install the package and all of its required dependencies.  Note that
while the installation name is "larch6", the package import name is "larch", and
you cannot install both Larch v5 and Larch v6 in the same environment.

Or you can install it using `mamba` to create a new environment:

```shell
mamba env create -p ARBORETUM -f https://raw.githubusercontent.com/driftlesslabs/larch/main/envs/arboretum.yml
conda activate ./ARBORETUM
```


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

# make this environment available to jupyter
ipython kernel install --user --name=LARIX

# rip examples to loadable modules
python larch/tools/rip_examples.py

# compile and install
python -m pip install -e ./sharrow
python -m pip install -e ./larch

# run unit tests (optional)
python -m pytest -v ./sharrow/sharrow/tests
python -m pytest -v ./larch/tests
```


## Windows Installation

The above script should *mostly* work on Windows as well, but one minor modification
is required, as the JAX library is not yet available for Windows via conda.  You need to
install it with pip instead.  So, replace the `mamba env update` line in the script
above with the following:

```
mamba env update -p .env/LARIX -f larch/envs/windows.yaml
```
