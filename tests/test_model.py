from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest import approx

import larch as lx
from larch import P, X
from larch.exceptions import MissingDataError


def test_bad_avail_declaration():
    d = lx.examples.MTC(format="dataset")
    m = lx.Model(d)
    with pytest.raises(TypeError):
        m.unknown_attr = 123
    with pytest.raises(TypeError):
        m.availability_var = "avail"


def test_model_has_no_data():
    m = lx.Model()
    with pytest.raises(MissingDataError):
        m.maximize_loglike()


def test_model_with_altcode_zero():
    raw_data = pd.read_csv(lx.example_file("swissmetro.csv.gz")).rename_axis(
        index="CASEID"
    )
    ds = lx.Dataset.construct.from_idco(raw_data, alts={0: "Ticketed", 1: "GA"})
    m = lx.Model(ds, compute_engine="numba")
    m.title = "Swiss GA Travelcard (binary logit)"
    m.choice_co_code = "GA"
    m.utility_co[1] = (
        +P.intercept + P.age * X.AGE + P.male * X.MALE + P.income * X.INCOME
    )
    m.maximize_loglike()
    assert m._most_recent_estimation_result["loglike"] == approx(-4258.78631691331)
    assert m._most_recent_estimation_result["x"].to_dict() == approx(
        {
            "age": -0.27650147517278006,
            "income": -0.05339095646626831,
            "intercept": -0.4818265416235945,
            "male": -0.6078366425546569,
        }
    )


def test_model_choice_avail():
    d = lx.examples.MTC(format="dataset")
    m = lx.Model(d, compute_engine="numba")
    m.availability_ca_var = "avail"
    m.choice_ca_var = "chose"
    cas = m.choice_avail_summary()
    np.testing.assert_array_equal(cas["name"].values[:-1], d.coords["alt_names"].data)
    np.testing.assert_array_equal(
        cas["chosen"].values, [3637, 517, 161, 498, 50, 166, 5029]
    )
    np.testing.assert_array_equal(
        cas["available"].values[:-1], [4755, 5029, 5029, 4003, 1738, 1479]
    )


def test_model_choice_avail_weighted():
    d = lx.examples.MTC(format="dataset")
    m = lx.Model(d, compute_engine="numba")
    m.availability_ca_var = "avail"
    m.choice_ca_var = "chose"
    m.weight_co_var = "famtype"
    cas = m.choice_avail_summary()
    np.testing.assert_array_equal(cas["name"].values[:-1], d.coords["alt_names"].data)
    np.testing.assert_array_equal(
        cas["chosen unweighted"].values, [3637, 517, 161, 498, 50, 166, 5029]
    )
    np.testing.assert_array_equal(
        cas["available unweighted"].values[:-1], [4755, 5029, 5029, 4003, 1738, 1479]
    )
    np.testing.assert_array_almost_equal(
        cas["chosen weighted"].values,
        [5851.0, 2563.0, 8512.0, 758.0, 2760.0, 59820.0, 39376.0],
    )
    np.testing.assert_array_almost_equal(
        cas["available weighted"].values,
        [59820.0, 59820.0, 48023.0, 22370.0, 19273.0, 59820.0, np.nan],
    )
