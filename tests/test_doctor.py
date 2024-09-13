from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest import approx

import larch as lx
from larch import P, X


@pytest.fixture(params=["numba", "jax"])
def ref_model(request):
    m = lx.example(1)
    m.compute_engine = request.param
    m.graph.new_node(parameter="mu_motor", children=[1, 2, 3, 4], name="Motorized")
    m.graph.new_node(parameter="mu_nonmotor", children=[5, 6], name="Nonmotorized")
    m.unmangle()
    return m


def test_chosen_but_not_available_query(ref_model: lx.Model):
    m = ref_model

    # change first 3 observations to WALK
    m.dataset = m.dataset.assign(ch=m.dataset["ch"].copy())
    m.dataset["ch"][:3, :] = 0
    m.dataset["ch"][:3, -1] = 1

    # test just checking
    m, problems = m.doctor(repair_ch_av="?")
    assert len(problems) == 1
    assert "chosen_but_not_available" in problems
    ll = m.loglike_casewise()
    assert ll[0] < -9e9
    assert ll[1] < -9e9
    assert ll[2] < -9e9

    # test raising error
    with pytest.raises(ValueError):
        m, problems = m.doctor(repair_ch_av="!")


def test_chosen_but_not_available_plus(ref_model: lx.Model):
    m = ref_model

    # change first 3 observations to WALK
    m.dataset = m.dataset.assign(ch=m.dataset["ch"].copy())
    m.dataset["ch"][:3, :] = 0
    m.dataset["ch"][:3, -1] = 1

    m, problems = m.doctor(repair_ch_av="+")
    assert (m.dataset["av"][:3, -1] == 1).all()
    ll = m.loglike_casewise()
    assert ll[:3] == approx([-1.79175947, -1.79175947, -1.60943791])


def test_chosen_but_not_available_minus(ref_model: lx.Model):
    m = ref_model

    # change first 3 observations to WALK
    m.dataset = m.dataset.assign(ch=m.dataset["ch"].copy())
    m.dataset["ch"][:3, :] = 0
    m.dataset["ch"][:3, -1] = 1

    m, problems = m.doctor(repair_ch_av="-", repair_noch_nzwt=None)
    assert (m.dataset["av"][:3, -1] == 0).all()
    assert (m.dataset["ch"][:3, -1] == 0).all()
    ll = m.loglike_casewise()
    assert ll[:3] == approx([0, 0, 0])
    assert m.loglike() == approx(-7304.995801563645)


def test_nothing_chosen_but_nonzero_weight(ref_model: lx.Model):
    m = ref_model
    m.weight_co_var = "famtype"

    # change first 3 observations to no choice
    m.dataset = m.dataset.assign(ch=m.dataset["ch"].copy())
    m.dataset["ch"][:3, :] = 0

    m, problems = m.doctor(repair_noch_nzwt="?")
    assert "nothing_chosen_but_nonzero_weight" in problems


def test_nothing_chosen_but_nonzero_weight_minus(ref_model: lx.Model):
    m = ref_model
    m.weight_co_var = "famtype"

    # change first 3 observations to no choice
    m.dataset = m.dataset.assign(ch=m.dataset["ch"].copy())
    m.dataset["ch"][:3, :] = 0

    m, problems = m.doctor(repair_noch_nzwt="-")
    assert "nothing_chosen_but_nonzero_weight" in problems
    assert (m.dataset["wt"][:3] == 0).all()
    ll = m.loglike_casewise()
    assert ll[:3] == approx([0, 0, 0])


def test_nan_in_data_co(ref_model: lx.Model):
    m = ref_model

    # change first 3 observations to have NaN for income
    m.dataset = m.dataset.assign(co=m.dataset["co"].copy())
    m.dataset["co"][:3, 0] = np.nan
    m._rebuild_data_arrays()

    assert np.isnan(m.dataset["co"][:3, 0]).all()
    assert np.isnan(m.loglike())

    # test raising error
    with pytest.raises(ValueError):
        m, problems = m.doctor(repair_nan_data_co="!")

    # test just checking
    m, problems = m.doctor(repair_nan_data_co="?")
    assert len(problems) == 1
    assert "nan_data_co" in problems


def test_low_variance_data_co(ref_model: lx.Model):
    m = ref_model

    # change first 3 observations to have NaN for income
    m.dataset = m.dataset.assign(co=m.dataset["co"].copy())
    m.dataset["co"][:, 0] = 42.0
    m._rebuild_data_arrays()

    # test raising error
    with pytest.raises(ValueError):
        m, problems = m.doctor(check_low_variance_data_co="!")

    # test just checking
    m, problems = m.doctor(check_low_variance_data_co="?")
    assert len(problems) == 1
    assert "low_variance_data_co" in problems


def test_nan_in_weight(ref_model: lx.Model):
    m = ref_model
    m.weight_co_var = "famtype"

    # change some observations to have NaN for weight
    m.dataset = m.dataset.assign(wt=m.dataset["wt"].copy())
    m.dataset["wt"][3:9] = np.nan
    m._rebuild_data_arrays()

    assert np.isnan(m.dataset["wt"][3:9]).all()
    assert np.isnan(m.loglike())

    # test raising error
    with pytest.raises(ValueError):
        m, problems = m.doctor(repair_nan_wt="!")

    # test just checking
    m, problems = m.doctor(repair_nan_wt="?")
    assert len(problems) == 1
    assert "nan_weight" in problems


def test_chosen_but_zero_quantity():
    hh, pp, tour, skims, emp = lx.example(200, ["hh", "pp", "tour", "skims", "emp"])
    base = lx.Dataset.construct.new_idca(tour.TOURID, skims.TAZ_ID)
    hh["INCOME_GRP"] = pd.qcut(hh.INCOME, 3)

    # set employment in a couple zones to zero
    emp.loc[2] = 0
    emp.loc[9] = 0

    tree = lx.DataTree(
        base=base,
        tour=tour.rename_axis(index="TOUR_ID"),
        hh=hh.set_index("HHID"),
        person=pp.set_index("PERSONID"),
        emp=emp,
        skims=lx.Dataset.construct.from_omx(skims),
        relationships=(
            "base.TAZ_ID @ emp.TAZ",
            "base.TOURID -> tour.TOUR_ID",
            "tour.HHID @ hh.HHID",
            "tour.PERSONID @ person.PERSONID",
            "hh.HOMETAZ @ skims.otaz",
            "base.TAZ_ID @ skims.dtaz",
        ),
    )

    m = lx.Model(datatree=tree)
    m.quantity_ca = (
        +P.EmpRetail_HighInc * X("RETAIL_EMP * (INCOME>50000)")
        + P.EmpNonRetail_HighInc * X("NONRETAIL_EMP") * X("INCOME>50000")
        + P.EmpRetail_LowInc * X("RETAIL_EMP") * X("INCOME<=50000")
        + P.EmpNonRetail_LowInc * X("NONRETAIL_EMP") * X("INCOME<=50000")
    )
    m.quantity_scale = P.Theta
    m.choice_co_code = "tour.DTAZ"

    # test raising error
    with pytest.raises(ValueError):
        m, problems = m.doctor(repair_ch_zq="!")

    # test just checking
    m, problems = m.doctor(repair_ch_zq="?")
    assert len(problems) == 1
    assert "chosen_but_zero_quantity" in problems

    the_problem = pd.DataFrame.from_records(
        np.rec.array(
            [(2, 528, "302, 413, 415"), (9, 557, "357, 373, 388")],
            dtype=[("TAZ_ID", "<i8"), ("n", "<i8"), ("example rows", "O")],
        )
    ).set_index("TAZ_ID")

    pd.testing.assert_frame_equal(problems.chosen_but_zero_quantity, the_problem)
