# Developer's Installation for ActivitySim Estimation Mode

These instructions are for developers who want to install the ActivitySim
package and work on the Larch-ActivitySim interface (a.k.a. estimation mode).

Before you start with the installation, you need have the following tools already:
- [miniforge for Mac or Linux](https://github.com/conda-forge/miniforge#unix-like-platforms-mac-os--linux)
  or [miniforge for Windows](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe)
- [gh](https://cli.github.com), the github command line tool (`conda install gh` should work)


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
gh repo clone driftlesslabs/activitysim

# install the conda development environment
mkdir -p .env/ASIM-LARIX
conda env update -p .env/ASIM-LARIX -f larch/envs/activitysim/environment.yaml
conda activate .env/ASIM-LARIX

# make this environment available to jupyter
ipython kernel install --user --name=ASIM-LARIX

# rip examples to loadable modules
python larch/tools/rip_examples.py

# compile and install
python -m pip install -e ./sharrow
python -m pip install -e ./larch
python -m pip install -e ./activitysim

# run unit tests (optional)
mkdir testbed
cd testbed
python -m pytest -v ../sharrow/sharrow/tests
python -m pytest -v ../larch/tests
python -m pytest -v ../activitysim/activitysim/estimation/test
cd ..
rm -rf testbed
```
