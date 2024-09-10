from __future__ import annotations

import pandas as pd
import pytest
from pytest import approx

import larch as lx
from larch import P, X


@pytest.fixture
def swissmetro_data():
    raw = pd.read_csv(lx.example_file("swissmetro.csv.gz"))
    raw["SM_COST"] = raw["SM_CO"] * (raw["GA"] == 0)
    raw["TRAIN_COST"] = raw.eval("TRAIN_CO * (GA == 0)")
    raw["TRAIN_COST_SCALED"] = raw["TRAIN_COST"] / 100
    raw["TRAIN_TT_SCALED"] = raw["TRAIN_TT"] / 100

    raw["SM_COST_SCALED"] = raw["SM_COST"] / 100
    raw["SM_TT_SCALED"] = raw["SM_TT"] / 100

    raw["CAR_CO_SCALED"] = raw["CAR_CO"] / 100
    raw["CAR_TT_SCALED"] = raw["CAR_TT"] / 100
    raw["CAR_AV_SP"] = raw.eval("CAR_AV * (SP!=0)")
    raw["TRAIN_AV_SP"] = raw.eval("TRAIN_AV * (SP!=0)")

    raw["keep"] = raw.eval("PURPOSE in (1,3) and CHOICE != 0")
    data = lx.Dataset.construct.from_idco(raw).dc.query_cases("keep")
    return data


def test_panel_latent_class(swissmetro_data):
    availability_co_vars = {
        1: "TRAIN_AV_SP",
        2: "SM_AV",
        3: "CAR_AV_SP",
    }

    m1 = lx.Model(title="Model1")
    m1.availability_co_vars = availability_co_vars
    m1.choice_co_code = "CHOICE"
    m1.utility_co[1] = P("ASC_TRAIN") + X("TRAIN_COST_SCALED") * P("B_COST")
    m1.utility_co[2] = X("SM_COST_SCALED") * P("B_COST")
    m1.utility_co[3] = P("ASC_CAR") + X("CAR_CO_SCALED") * P("B_COST")

    m2 = lx.Model(title="Model2")
    m2.availability_co_vars = availability_co_vars
    m2.choice_co_code = "CHOICE"
    m2.utility_co[1] = (
        P("ASC_TRAIN")
        + X("TRAIN_TT_SCALED") * P("B_TIME")
        + X("TRAIN_COST_SCALED") * P("B_COST")
    )
    m2.utility_co[2] = X("SM_TT_SCALED") * P("B_TIME") + X("SM_COST_SCALED") * P(
        "B_COST"
    )
    m2.utility_co[3] = (
        P("ASC_CAR")
        + X("CAR_TT_SCALED") * P("B_TIME")
        + X("CAR_CO_SCALED") * P("B_COST")
    )

    m3 = lx.Model(title="Model3")
    m3.availability_co_vars = availability_co_vars
    m3.choice_co_code = "CHOICE"
    m3.utility_co[1] = X("TRAIN_COST_SCALED") * P("Z_COST")
    m3.utility_co[2] = X("SM_COST_SCALED") * P("Z_COST")
    m3.utility_co[3] = X("CAR_CO_SCALED") * P("Z_COST")

    mk = lx.Model(title="ModelC")
    mk.utility_co[102] = P("W_OTHER")
    mk.utility_co[103] = P("W_COST")
    mk.groupid = "ID"

    # print("hello")

    b = lx.LatentClass(
        mk,
        {101: m1, 102: m2, 103: m3},
        datatree=swissmetro_data.dc.set_altids([1, 2, 3]),
        groupid="ID",
    )

    # print("hello2")

    # for k, m in b._models.items():
    #     print(f"==========={k=}===========\n{m.dataset}")

    b.lock_value(Z_COST=-10000)
    # for k, m in b._models.items():
    #     print(f"~=========={k=}===========\n{m.dataset}")

    assert b.loglike() == approx(-6867.245, rel=1e-4)
    # print(f"{b.loglike()=}")
    assert b.d_loglike() == approx(
        [
            -1.104770e02,
            -1.545916e03,
            -2.189165e01,
            -9.183448e02,
            -1.658521e02,
            8.292606e01,
            -1.490116e-08,
        ],
        abs=1e-7,
        rel=1e-3,
    )
    # print(f"{b.d_loglike()=}")

    result = b.maximize_loglike(method="slsqp")
    assert result.loglike == approx(-4474.478515625)
    # print(f"{result.loglike=}")


if __name__ == "__main__":
    swissmetro_data_ = swissmetro_data()
    test_panel_latent_class(swissmetro_data_)
