#!/usr/bin/env zsh

# exit this script immediately upon any failure
set -e

# change to the docs directory
cd "$(dirname "$0")"

# create a new conda environment for the docs build
mamba create -p ../.env/DOCBUILD "python=3.11" \
  pip \
  addicty \
  dask \
  filelock \
  networkx \
  numba \
  numexpr \
  numpy \
  pandas \
  pyarrow \
  rich \
  scipy \
  seaborn \
  sharrow \
  sparse \
  xarray \
  xlsxwriter \
  xmle \
  "sphinx>=7.0" \
  myst-nb pyyaml sphinx sphinx-comments sphinx-copybutton \
  sphinx-design sphinx-thebe sphinxcontrib-bibtex \
  linkify-it-py sphinx-togglebutton pydata-sphinx-theme \
  nbformat \
  "jax[cpu]" \
  ruamel.yaml \
  geopandas \
  -c conda-forge \
  --yes

# activate the new environment
eval "$(conda shell.bash hook)"
conda activate ../.env/DOCBUILD

# install larch
python -m pip install ..

conda info

conda list

# hide all jupyter notebook cells tagged with "TEST"
#python _scripts/hide_test_cells.py

# build the docs
#jb build .
#jb config sphinx . >> CONF.py

sphinx-build -b html . _build/html
