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
  sharrow \
  sparse \
  xarray \
  xmle \
  jupyter-book \
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
python _scripts/hide_test_cells.py
python _scripts/developer_doc_title.py

# build the docs
jb build .
