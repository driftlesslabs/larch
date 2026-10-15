from __future__ import annotations

import jax
import pandas as pd
from pytest import approx

import larch as lx
from larch import P, X


def test_swissmetro_101():
    # Float32 likelihood plateaus can stop SLSQP before this coefficient
    # benchmark's tolerances are met. Keep precision local to this test.
    with jax.experimental.enable_x64():
        m = lx.Model()
        m.title = "swissmetro example 01 (simple logit)"
        m.availability_co_vars = {
            1: "TRAIN_AV * (SP!=0)",
            2: "SM_AV",
            3: "CAR_AV * (SP!=0)",
        }
        m.choice_co_code = "CHOICE"
        m.utility_co[1] = P("ASC_TRAIN")
        m.utility_co[2] = 0
        m.utility_co[3] = P("ASC_CAR")
        m.utility_co[1] += X("TRAIN_TT") * P("B_TIME")
        m.utility_co[2] += X("SM_TT") * P("B_TIME")
        m.utility_co[3] += X("CAR_TT") * P("B_TIME")
        m.utility_co[1] += X("TRAIN_CO*(GA==0)") * P("B_COST")
        m.utility_co[2] += X("SM_CO*(GA==0)") * P("B_COST")
        m.utility_co[3] += X("CAR_CO") * P("B_COST")
        m.ordering = [("ASCs", "ASC.*"), ("LOS", "B_.*")]
        raw_data = pd.read_csv(lx.example_file("swissmetro.csv.gz")).rename_axis(
            index="CASEID"
        )
        raw_data.head()
        keep = raw_data.eval("PURPOSE in (1,3) and CHOICE != 0")
        selected_data = raw_data[keep]
        ds = lx.Dataset.construct.from_idco(
            selected_data, alts={1: "Train", 2: "SM", 3: "Car"}
        )
        m.datatree = ds
        m.set_cap(15)
        result = m.maximize_loglike(method="SLSQP", options={"ftol": 1e-9})
        assert result.success
        assert abs(m.d_loglike()).max() / m.total_weight() < 1e-5
        assert result.loglike == approx(-5331.252006971916)
        m.calculate_parameter_covariance()
        m.parameter_summary()
        assert m.pvals == approx(
            [-0.1546324225, -0.7011867125, -0.01083790651, -0.01277860255], rel=1e-3
        )
        assert m.pstderr == approx(
            [0.04323547174, 0.05487393317, 0.0005183019169, 0.0005688334527],
            rel=1e-3,
        )
        assert m.parameter_summary().data["t Stat"].values.astype(float) == approx(
            [-3.58, -12.79, -20.92, -22.46], rel=1e-2
        )
