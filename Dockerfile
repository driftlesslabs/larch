FROM mambaorg/micromamba:1.5.6 AS build-stage
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# listing all the builder environment packages literally in this dockerfile
# instead of using a seperate definitions file ensures that the image does
# not need to be rebuilt unless the packages actually change (i.e. not if
# the external file appears to have a recent file modification date but wasn't
# actually edited).  This list also should only included the minimum set of
# packages to actually build and test the core, not to run everything a developer
# might touch (e.g. skip docs, jupyter, and add-on tools).
RUN micromamba install -y -n base -c conda-forge \
    "python=3.10" \
    pip \
    "numpy>=1.19" \
    "pandas>=2.1" \
    pyarrow \
    xarray \
    dask \
    networkx \
    "numba>=0.53" \
    numexpr \
    sparse \
    filelock \
    addicty \
    xmle \
    rich \
    altair \
    altair_saver \
    pydot \
    platformdirs \
    tabulate \
    pytest \
    pytest-regressions \
    pytest-xdist \
    nbmake \
    openmatrix \
    zarr \
    geopandas \
    sharrow \
    && \
    micromamba clean --all --yes \
    && \
    python -m pip install build jax jaxlib xlogit --no-cache-dir

ENV PYTHONUNBUFFERED=1

# Custom cache invalidation
ARG CACHEBUST=1

# copy the repo contents into the docker container
COPY --chown=$MAMBA_USER:$MAMBA_USER . /tmp/larix/larch

# Compile wheel
RUN python -m build --outdir=/tmp/wheelhouse /tmp/larix/larch

# Install from wheel
RUN python -m pip install --no-index --find-links=/tmp/wheelhouse larch6

# Run tests
RUN python -m pytest -v /tmp/larix/larch --ignore=/tmp/larix/larch/sandbox
