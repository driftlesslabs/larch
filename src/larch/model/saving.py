from __future__ import annotations

import importlib
import os.path
from collections.abc import Mapping

import numpy as np
from addicty import Dict
from yaml import SafeDumper

from .constraints import ParametricConstraintList
from .linear import DictOfAlts, DictOfLinearFunction, LinearComponent, LinearFunction
from .tree import NestingTree

SafeDumper.add_representer(
    np.float64, lambda dumper, data: dumper.represent_float(float(data))
)
SafeDumper.add_representer(
    np.float32, lambda dumper, data: dumper.represent_float(float(data))
)
SafeDumper.add_representer(
    np.int32, lambda dumper, data: dumper.represent_int(int(data))
)
SafeDumper.add_representer(
    np.int64, lambda dumper, data: dumper.represent_int(int(data))
)
SafeDumper.add_representer(
    np.bool_, lambda dumper, data: dumper.represent_bool(bool(data))
)
SafeDumper.add_representer(
    np.str_, lambda dumper, data: dumper.represent_str(str(data))
)
SafeDumper.add_representer(
    np.ndarray, lambda dumper, data: dumper.represent_list(list(data))
)
SafeDumper.add_representer(
    LinearComponent, lambda dumper, data: dumper.represent_dict(data.to_dict())
)
SafeDumper.add_representer(
    LinearFunction, lambda dumper, data: dumper.represent_list(list(data))
)
SafeDumper.add_representer(
    DictOfLinearFunction, lambda dumper, data: dumper.represent_dict(dict(data))
)
SafeDumper.add_representer(
    DictOfAlts, lambda dumper, data: dumper.represent_dict(dict(data))
)
SafeDumper.add_representer(
    ParametricConstraintList, lambda dumper, data: dumper.represent_list(list(data))
)
SafeDumper.add_representer(
    ParametricConstraintList, lambda dumper, data: dumper.represent_list(list(data))
)
SafeDumper.add_representer(
    NestingTree, lambda dumper, data: dumper.represent_dict(data.to_dict())
)


def save_model(m, filename=None, format="yaml", overwrite=False):
    if format == "yaml":
        if overwrite and os.path.isfile(filename):
            os.unlink(filename)
        return _save_model_yaml(m, filename)
    elif format == "raw":
        return _save_model_yaml(m, None, raw=True)
    else:
        raise NotImplementedError(f"saving in {format} format")


def load_model(filename_or_content):
    if isinstance(filename_or_content, str) and os.path.exists(filename_or_content):
        if filename_or_content.endswith(".html") or filename_or_content.endswith(
            ".xhtml"
        ):
            from .. import read_metadata

            try:
                y = read_metadata(filename_or_content)
            except Exception:
                raise
            else:
                if isinstance(y, str) and "\n" in y:
                    y = Dict.load(y)
                if y is not None:
                    filename_or_content = y
    return _load_model_yaml(filename_or_content)


def _save_model_yaml(m, filename, skipping=(), raw=False):
    x = Dict()

    def savethis(attr, transformer=None, wrapper=None):
        if attr not in skipping:
            i = getattr(m, attr, None)
            if i:
                try:
                    if transformer is not None:
                        j = getattr(i, transformer)
                        if callable(j):
                            i = j()
                        else:
                            i = j
                    elif wrapper is not None:
                        i = wrapper(i)
                except AttributeError:
                    pass
                else:
                    x[attr] = i

    x.model_type = f"{m.__class__.__module__}.{m.__class__.__name__}"
    savethis("availability_any")
    savethis("availability_ca_var")
    savethis("availability_co_vars", "to_dict")
    savethis("choice_any")
    savethis("choice_ca_var")
    savethis("choice_co_code")
    savethis("choice_co_vars", "to_dict")
    savethis("common_draws")
    savethis("compute_engine")
    savethis("constraint_intensity")
    savethis("constraint_sharpness")
    savethis("constraints")
    savethis("float_dtype", "__name__")
    savethis("graph", "to_dict")
    savethis("groupid")
    savethis("index_name")
    savethis("logsum_parameter")
    savethis("mixtures", "to_list")
    savethis("n_draws")
    savethis("parameters", "to_dict")
    savethis("prerolled_draws")
    savethis("quantity_ca")
    savethis("quantity_scale")
    savethis("title")
    savethis("utility_ca")
    savethis("utility_co")
    savethis("weight_co_var")
    savethis("weight_normalization")

    _models = getattr(m, "_models", None)
    if _models is not None:
        submodels = Dict()
        for k, v in _models.items():
            submodels[k] = _save_model_yaml(v, None, skipping=("parameters"), raw=True)
        x["_models"] = submodels

    if raw:
        return x
    return x.dump(filename)


def _load_model_yaml(filename_or_content):
    if isinstance(filename_or_content, str) and os.path.exists(filename_or_content):
        content = Dict.load(filename_or_content)
    elif isinstance(filename_or_content, Mapping):
        content = filename_or_content
    else:
        raise TypeError("filename_or_content must be an existing file or a Dict")

    model_type = content.get("model_type").split(".")
    model_module = ".".join(model_type[:-1])
    module = importlib.import_module(model_module)
    cls = getattr(module, model_type[-1])
    return cls.from_dict(content)
