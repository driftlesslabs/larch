(larch-installation)=
# Installing Larch


## Quick Start

Larch v6 is available via `pip`.  Since it is not yet feature complete, it is not
the default "Larch" for most users, and must be installed as the `larch6` package.

```shell
python -m pip install larch6
```

This will install the package and all of its required dependencies.  Note that
while the installation name (using pip) is "larch6", the package import name
(from within Python) is still just "larch", and you cannot install both
Larch v5 and Larch v6 in the same environment.

Once installed, use larch as you normally would, import it in Python as just `larch`:

```python
import larch as lx
```

Version 6 is a work in progress, not a production-ready tool. A lot of things are
working, but not everything. The interface is quite similar to Larch v5 and existing
users will likely find it familiar. If you want to try it out, please do, and feel
free to open issues in the issue tracker. But, please don't expect it to work
perfectly yet, especially for more advanced models.


## New Users

Larch is a Python package, so you'll need to also have Python installed.  If you
don't already have Python installed, we recommend installing the latest version
via [MiniForge](https://github.com/conda-forge/miniforge#download).  Once you have
MiniForge installed, you can create an environment for using Larch with the
following command:

```shell
mamba env create -p ./ARBORETUM -f https://raw.githubusercontent.com/driftlesslabs/larch/main/envs/arboretum.yml
```

This will create a new environment called `arboretum` with Larch and all of its
dependencies installed.  To activate the environment and use Larch in Jupyter, run:

```shell
conda activate ./ARBORETUM
jupyter lab
```


## Google Colab

Users of the Google Colab platform can install Larch by running the following cell

```shell
!python -m pip install larch6
```
