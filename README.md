# Larch

This repository contains the Larch package, for estimating and applying discrete
choice models. Version 6 is a substantial rewrite of the package, changing to a
platform that allows swapping out the underlying computational engine, so that
the same model can run in numba or JAX.


# Quick Start Guide

You can install Larch with pip:

```shell
python -m pip install larch
```

# Developer's Installation

You can install larch for development by running the following script:

```shell
curl -LsSf https://driftless.xyz/larch-dev.sh | bash -s -- -d ~/Git/larix
```

You can change the install directory by changing the `-d` argument.  The script
will clone the larch repository and install it in the specified directory.  It
will also install the required dependencies and create a new uv virtual
environment for development.

To run the test suite, you'll need to run with the `--extra test` flag, which
will ensure that all the extra dependencies needed for testing are installed.
To do this, from the `larch` directory, run the following command:

```shell
uv run --extra test pytest .
```
