from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy import random
from pytest import approx

import larch as lx
from larch import P, X


@pytest.mark.parametrize(
    "compute_engine,use_streaming",
    [
        ("jax", False),
        ("numba", False),
        # ("numba", True),
    ],
)
def test_destination_only(compute_engine, use_streaming):
    hh, pp, tour, skims, emp = lx.example(200, ["hh", "pp", "tour", "skims", "emp"])
    hh["INCOME_GRP"] = pd.qcut(hh.INCOME, 3)
    co = lx.Dataset.construct(
        tour.set_index("TOURID"),
        caseid="TOURID",
        alts=skims.TAZ_ID,
    )
    tree = lx.DataTree(
        base=co,
        hh=hh.set_index("HHID"),
        person=pp.set_index("PERSONID"),
        emp=emp,
        skims=lx.Dataset.construct.from_omx(skims),
        relationships=(
            "base.TAZ_ID @ emp.TAZ",
            "base.HHID @ hh.HHID",
            "base.PERSONID @ person.PERSONID",
            "hh.HOMETAZ @ skims.otaz",
            "base.TAZ_ID @ skims.dtaz",
        ),
    )
    m = lx.Model(
        datatree=tree, compute_engine=compute_engine, use_streaming=use_streaming
    )
    m.title = "Exampville Tour Destination Choice v2"
    m.quantity_ca = (
        +P.EmpRetail_HighInc * X("RETAIL_EMP * (INCOME>50000)")
        + P.EmpNonRetail_HighInc * X("NONRETAIL_EMP") * X("INCOME>50000")
        + P.EmpRetail_LowInc * X("RETAIL_EMP") * X("INCOME<=50000")
        + P.EmpNonRetail_LowInc * X("NONRETAIL_EMP") * X("INCOME<=50000")
    )
    m.quantity_scale = P.Theta
    m.utility_ca = +P.distance * X.AUTO_DIST
    m.choice_co_code = "base.DTAZ"
    m.plock(EmpRetail_HighInc=0, EmpRetail_LowInc=0)
    assert m.loglike() == approx(-77777.17321427424)
    assert m.d_loglike() == approx(
        [-223.950165, -682.110256, 0.0, 0.0, -7406.389922, -34762.91256], rel=1e-2
    )

    result = m.maximize_loglike(stderr=True)
    assert result.loglike == approx(-70650.07578452416)
    assert result.success
    assert result.n_cases == 20739
    assert result.logloss == approx(3.4066288531040145)
    x = result.x
    if not isinstance(x, pd.Series):
        x = pd.Series(x, index=m.pnames)
    pd.testing.assert_series_equal(
        x.sort_index(),
        pd.Series(
            {
                "EmpNonRetail_HighInc": 1.2453335020460703,
                "EmpNonRetail_LowInc": -1.0893594261458912,
                "EmpRetail_HighInc": 0.0,
                "EmpRetail_LowInc": 0.0,
                "Theta": 0.676440163641688,
                "distance": -0.3347118435209836,
            }
        ).sort_index(),
        rtol=1e-2,
    )
    assert m.pstderr == approx(
        np.array([0.145749, 0.052355, 0.0, 0.0, 0.009012, 0.003812]),
        rel=1e-2,
    )


@pytest.mark.parametrize("compute_engine", ["jax", "numba"])
def test_destination_some_zero_q(compute_engine):
    hh, pp, tour, skims, emp = lx.example(200, ["hh", "pp", "tour", "skims", "emp"])
    tour = tour.join(hh[["HHID", "HOMETAZ"]].set_index("HHID"), on="HHID")
    emp["samp_prob"] = emp.TOTAL_EMP / emp.TOTAL_EMP.sum()
    distance = xr.DataArray(
        skims["AUTO_DIST"][:],
        dims=["otaz", "dtaz"],
        coords={"otaz": skims["TAZ_ID"][:], "dtaz": skims["TAZ_ID"][:]},
    )
    rng = random.default_rng(42)
    samps = {}
    for i, dtaz in zip(tour.index, tour.DTAZ):
        samp = rng.choice(emp.index, 10, replace=True, p=emp["samp_prob"])
        samp[0] = dtaz
        samps[i] = pd.Series(samp)
    df = (
        pd.concat(samps, axis=0, names=["tour_id", "samp_id"])
        .rename("sampled_taz")
        .reset_index()
    )
    df["fake_data"] = rng.normal(size=len(df))
    df = df.set_index(["tour_id", "samp_id"])
    df = df.join(emp, on="sampled_taz")
    df = df.join(tour[["TOURID", "HOMETAZ"]].set_index("TOURID"), on="tour_id")
    distances = []
    for i in range(len(df)):
        distances.append(
            distance.data[df.HOMETAZ.iloc[i] - 1, df.sampled_taz.iloc[i] - 1]
        )
    df["distance"] = pd.Series(distances, index=df.index)
    ds = lx.Dataset.construct.from_idca(df)
    m = lx.Model(ds, compute_engine="jax")
    m.quantity_ca = (
        +P.RETAIL_EMP * X("RETAIL_EMP")
        # + P.NONRETAIL_EMP * X('NONRETAIL_EMP')
    )
    m.lock_value(P.RETAIL_EMP, 0)
    m.quantity_scale = P.Theta
    m.utility_ca = m.utility_ca = +P.fake_data * X.fake_data + P.distance * X.distance
    m.choice_ca_var = "samp_id == 0"
    assert m.probability().sum(0) == approx(
        [
            1981.24053581,
            2089.47147447,
            2121.30911427,
            2053.22970856,
            2045.50005845,
            2078.48652599,
            2084.46825967,
            2057.77903425,
            2106.36456315,
            2121.15072539,
        ]
    )

    m.doctor(repair_ch_zq="-")
    # some cases the choice is a zone with zero quantity
    # this is intentional, and we need to repair the model for it

    assert m.d_loglike() == approx([0.0, -31148.736, -22003.318, 84.19797])
    assert m.loglike() == approx(-62083.82421875)
    result = m.maximize_loglike()
    assert result.loglike == approx(-39720.1015625)


def test_query_on_subspace():
    hh, pp, tour, skims = lx.example(200, ["hh", "pp", "tour", "skims"])

    from addicty import Dict

    Mode = Dict(
        DA=1,
        SR=2,
        Walk=3,
        Bike=4,
        Transit=5,
    ).freeze()

    tour_dataset = lx.Dataset.construct.from_idco(tour.set_index("TOURID"), alts=Mode)
    od_skims = lx.Dataset.construct.from_omx(skims)
    dt = lx.DataTree(
        tour=tour_dataset,
        hh=hh.set_index("HHID"),
        person=pp.set_index("PERSONID"),
        od=od_skims,
        do=od_skims,
        relationships=(
            "tours.HHID @ hh.HHID",
            "tours.PERSONID @ person.PERSONID",
            "hh.HOMETAZ @ od.otaz",
            "tours.DTAZ @ od.dtaz",
            "hh.HOMETAZ @ do.dtaz",
            "tours.DTAZ @ do.otaz",
        ),
    )
    assert dt.n_cases == 20739

    # query cases on a variable in the root dataset
    dt_work = dt.query_cases("TOURPURP == 1")

    # check that the query worked, and that the result from pandas is the same
    assert dt_work.n_cases == 7564
    # straightforward pandas
    assert tour.eval("TOURPURP == 1").sum() == 7564

    # query cases on a IDCO variable in the household dataset
    dt_lowincome = dt.query_cases("INCOME < 25_000")

    # check that the query worked, and that the result from pandas is the same
    assert dt_lowincome.n_cases == 6911
    # more complex pandas eval-on-join
    assert (
        tour.join(hh.set_index("HHID").INCOME, on="HHID").eval("INCOME < 25_000")
    ).sum() == 6911

    # query cases on a variable in the skims dataset
    dt_long = dt.query_cases("AUTO_DIST > 10")

    # check that the query worked, and that the result from pandas is the same
    assert dt_long.n_cases == 94
    # and here is why DataTree exists ...
    assert (
        tour.join(
            hh.set_index("HHID").HOMETAZ,
            on="HHID",
        )
        .join(
            (
                skims.get_dataframe("AUTO_DIST", "TAZ_ID", "TAZ_ID")
                .stack()
                .rename("AUTO_DIST")
                .rename_axis(["otaz", "dtaz"])
            ),
            on=["HOMETAZ", "DTAZ"],
        )
        .eval("AUTO_DIST > 10")
    ).sum() == 94
