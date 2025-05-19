from __future__ import annotations

import os

from xmle import NumberedCaption, Reporter
from xmle import load_metadata as read_metadata

from . import examples
from .dataset import DataArray, Dataset, DataTree
from .examples import example, example_file
from .model import mixtures
from .model.basemodel import BaseModel
from .model.jaxmodel import Model
from .model.latent_class import LatentClass, MixedLatentClass
from .model.model_group import ModelGroup
from .model.param_core import ParameterBucket
from .model.saving import load_model
from .model.tree import NestingTree
from .shorts import PX, P, X
from .util import figures

try:
    from ._version import version as _install_version
except ImportError:
    _install_version = "999.404"

# Get decorated version when in development
try:
    top_package = __import__(__name__.split(".")[0])
    import setuptools_scm

    __version__ = setuptools_scm.get_version(
        os.path.dirname(top_package.__path__[0]), fallback_version=_install_version
    )
except Exception:
    __version__ = _install_version

__all__ = [
    "NumberedCaption",
    "Reporter",
    "read_metadata",
    "examples",
    "example",
    "example_file",
    "DataArray",
    "Dataset",
    "DataTree",
    "mixtures",
    "BaseModel",
    "Model",
    "ModelGroup",
    "LatentClass",
    "MixedLatentClass",
    "ParameterBucket",
    "load_model",
    "NestingTree",
    "PX",
    "P",
    "X",
    "figures",
    "versions",
]


def versions() -> dict:
    """Versions of Larch and its key dependencies."""
    import numba as nb
    import numpy as np
    import pandas as pd
    import sharrow as sh
    import xarray as xr

    try:
        import jax
    except ImportError:
        jax_version = "not installed"
    else:
        jax_version = jax.__version__

    return {
        "larch": __version__,
        "sharrow": sh.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xarray": xr.__version__,
        "numba": nb.__version__,
        "jax": jax_version,
    }
