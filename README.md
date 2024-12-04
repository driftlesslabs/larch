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

Or you can install it using `conda` to create a new environment:

```shell
conda env create -p ARBORETUM -f https://raw.githubusercontent.com/driftlesslabs/larch/main/envs/arboretum.yml
conda activate ./ARBORETUM
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
