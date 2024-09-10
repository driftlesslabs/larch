from __future__ import annotations

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
