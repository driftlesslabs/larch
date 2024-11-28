from __future__ import annotations

import os
import os.path

import numpy as np
import pandas as pd

from ..dataset import DataArray, Dataset

try:
    from .generated import ex
except ImportError:
    ex = None


def _exec_example_n(n, *arg, **kwarg):
    if ex is None:
        raise ImportError("No examples are available")
    return ex[n](*arg, **kwarg)


def example(n, extract="m", estimate=False, output_file=None):
    """Run an example code section (from the documentation) and give the result.

    Parameters
    ----------
    n : int
        The number of the example to reproduce.
    extract : str or Sequence[str]
        The name of the object of the example to extract.  By default `m`
        but it any named object that exists in the example namespace can
        be returned.  Give a sequence of `str` to get multiple objects.
    estimate : bool, default False
        Whether to run the `maximize_loglike` command (and all subsequent
        steps) on the example.
    output_file : str, optional
        If given, check whether this named file exists.  If so,
        return the filename, otherwise run the example code section
        (which should as a side effect create this file).  Then
        check that the file now exists, raising a FileNotFoundError
        if it still does not exist.
    """
    if output_file is not None:
        if os.path.exists(output_file):
            return output_file
        _exec_example_n(n, extract=(), estimate=estimate)
        if os.path.exists(output_file):
            return output_file
        else:
            raise FileNotFoundError(output_file)
    else:
        return _exec_example_n(n, extract=extract, estimate=estimate)


def example_file(filename):
    warehouse_file = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data_warehouse", filename)
    )
    if os.path.exists(warehouse_file):
        return warehouse_file
    raise FileNotFoundError(
        f"there is no example data file '{warehouse_file}' in data_warehouse"
    )


def MTC(**kwargs):
    ca = pd.read_csv(
        example_file("MTCwork.csv.gz"),
        index_col=("casenum", "altnum"),
        dtype={"chose": np.float32},
    )
    dataset = Dataset.construct.from_idca(
        ca.rename_axis(index=("caseid", "altid")),
        altnames=["DA", "SR2", "SR3+", "Transit", "Bike", "Walk"],
        fill_missing=0,
    )
    dataset["avail"] = DataArray(
        dataset["_avail_"].values, dims=["caseid", "altid"], coords=dataset.coords
    )
    dataset["chose"] = dataset["chose"].fillna(0.0)
    return dataset


def ARTIFICIAL(**kwargs):
    df = pd.read_csv(example_file("artificial_long.csv.gz"))
    df["panel"] = np.ceil(df["id"] / 10).astype(int)
    return Dataset.construct.from_idca(df.set_index(["id", "alt"]))
