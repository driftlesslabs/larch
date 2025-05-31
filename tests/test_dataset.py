from __future__ import annotations

import numpy as np
import pandas as pd
from pytest import fixture, raises

import larch as lx
from larch.examples import example_file


@fixture(scope="module")
def mtc_dataset_idca():
    ca = pd.read_csv(
        example_file("MTCwork.csv.gz"),
        index_col=("casenum", "altnum"),
        dtype={"chose": np.float32},
    )
    return lx.Dataset.dc.from_idca(ca)


def test_datarray_dc_accessor(mtc_dataset_idca):
    ivtt = mtc_dataset_idca.dc["ivtt"]
    assert ivtt.shape == (5029, 6)
    assert ivtt.dc.n_cases == 5029
    assert ivtt.dc.n_alts == 6

    dist = mtc_dataset_idca.dc["dist"]
    assert dist.shape == (5029,)
    assert dist.dc.n_cases == 5029
    with raises(ValueError):
        _ = dist.dc.n_alts


def test_dataset_dc_accessor(mtc_dataset_idca):
    assert mtc_dataset_idca.dc.n_cases == 5029
    assert mtc_dataset_idca.dc.n_alts == 6
    assert mtc_dataset_idca.dc.CASEID == "casenum"
    assert mtc_dataset_idca.dc.ALTID == "altnum"
    assert mtc_dataset_idca.dc.CASEALT is None
    assert mtc_dataset_idca.dc.CASEPTR is None
    assert mtc_dataset_idca.dc.ALTIDX is None
    assert mtc_dataset_idca.dc.GROUPID is None
    assert mtc_dataset_idca.dc.INGROUP is None


def test_dataset_dtypes():
    ca = pd.read_csv(
        example_file("MTCwork.csv.gz"),
        index_col=("casenum", "altnum"),
        dtype={"chose": np.float32},
    )

    # set some variables with different dtypes
    ca["ovtt_round"] = ca["ovtt"].round().astype(np.int8)
    ca["ivtt_round"] = ca["ivtt"].round().astype(np.uint8)
    ca["male"] = ca["femdum"] == 0

    # initialize the Dataset from the DataFrame
    ds = lx.Dataset.dc.from_idca(ca)

    # check the dtypes
    assert ds["ovtt_round"].dtype == ca["ovtt_round"].dtype
    assert ds["ivtt_round"].dtype == ca["ivtt_round"].dtype
    assert ds["male"].dtype == ca["male"].dtype
    assert ds["chose"].dtype == ca["chose"].dtype
    assert ds["ivtt"].dtype == ca["ivtt"].dtype
    assert ds["hhid"].dtype == ca["hhid"].dtype
