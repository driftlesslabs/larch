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
