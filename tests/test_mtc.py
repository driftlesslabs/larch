import re

import numpy as np
import pandas as pd
import pytest
from pytest import approx

import larch as lx
from larch import PX, P, X
from larch.util.testing import assert_same_text


@pytest.mark.parametrize("compute_engine", ["jax", "numba"])
def test_mtc_1(compute_engine):
    d = lx.examples.MTC(format="dataset")
    m = lx.Model(d)
    m.compute_engine = compute_engine
    m.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
    m.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
    m.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
    m.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
    m.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
    m.utility_ca = PX("tottime") + PX("totcost")
    m.availability_ca_var = "avail"
    m.choice_ca_var = "chose"
    m.title = "MTC Example 1 (Simple MNL)"
    m.choice_avail_summary()
    # TEST
    s = """            name  chosen available
    altid
    1                    DA    3637      4755
    2                   SR2     517      5029
    3                  SR3+     161      5029
    4               Transit     498      4003
    5                  Bike      50      1738
    6                  Walk     166      1479
    < Total All Alternatives > 5029
    """
    mash = lambda x: re.sub("\s+", " ", x).strip()
    assert mash(s) == mash(str(m.choice_avail_summary()))
    m.set_cap(20)
    # TEST
    assert dict(m.required_data()) == {
        "ca": ["totcost", "tottime"],
        "co": ["hhinc"],
        "choice_ca": "chose",
        "avail_ca": "avail",
    }
    assert m.loglike() == approx(-7309.600971749634)
    assert m.compute_engine == compute_engine
    result = m.maximize_loglike(stderr=True)
    assert result.loglike == approx(-3626.18625551293)
    assert result.logloss == approx(0.7210551313408093)
    assert result.message == "Optimization terminated successfully"
    assert m.total_weight() == 5029.0
    m.parameter_summary()
    m.ordering = (
        (
            "LOS",
            "totcost",
            "tottime",
        ),
        (
            "ASCs",
            "ASC.*",
        ),
        (
            "Income",
            "hhinc.*",
        ),
    )
    m.parameter_summary()
    es = m.estimation_statistics()
    assert es[0][1][0][1].text == "5029"
    assert "|".join(i.text for i in es[0][1][0]) == "Number of Cases|5029"
    assert (
        "|".join(i.text for i in es[0][1][1])
        == "Log Likelihood at Convergence|-3626.19|-0.72"
    )
    assert (
        "|".join(i.text for i in es[0][1][2])
        == "Log Likelihood at Null Parameters|-7309.60|-1.45"
    )
    assert (
        "|".join(i.text for i in es[0][1][3])
        == "Rho Squared w.r.t. Null Parameters|0.504"
    )


@pytest.mark.parametrize("compute_engine", ["jax", "numba"])
def test_mtc_17(compute_engine):
    d = lx.examples.MTC()
    m = lx.Model(d, compute_engine=compute_engine)
    m.availability_ca_var = "avail"
    m.choice_ca_var = "chose"
    m.utility_ca = (
        +X("totcost/hhinc") * P("costbyincome")
        + X("tottime * (altid <= 4)") * P("motorized_time")
        + X("tottime * (altid >= 5)") * P("nonmotorized_time")
        + X("ovtt/dist * (altid <= 4)") * P("motorized_ovtbydist")
    )
    for a in [4, 5, 6]:
        m.utility_co[a] += X("hhinc") * P("hhinc#{}".format(a))
    for i in d["alt_names"][1:3]:
        name = str(i.values)
        a = int(i.altid)
        m.utility_co[a] += (
            +X("vehbywrk") * P("vehbywrk_SR")
            + X("wkccbd+wknccbd") * P("wkcbd_" + name)
            + X("wkempden") * P("wkempden_" + name)
            + P("ASC_" + name)
        )

    for i in d["alt_names"][3:]:
        name = str(i.values)
        a = int(i.altid)
        m.utility_co[a] += (
            +X("vehbywrk") * P("vehbywrk_" + name)
            + X("wkccbd+wknccbd") * P("wkcbd_" + name)
            + X("wkempden") * P("wkempden_" + name)
            + P("ASC_" + name)
        )
    m.ordering = (
        (
            "LOS",
            ".*cost.*",
            ".*time.*",
            ".*dist.*",
        ),
        (
            "Zonal",
            "wkcbd.*",
            "wkempden.*",
        ),
        (
            "Household",
            "hhinc.*",
            "vehbywrk.*",
        ),
        (
            "ASCs",
            "ASC.*",
        ),
    )
    m.set_cap(25)
    r = m.maximize_loglike(stderr=True, options={"maxiter": 1000, "ftol": 1e-10})
    assert m.compute_engine == compute_engine
    assert r.loglike == approx(-3444.185100565835)
    assert r.n_cases == 5029
    assert "success" in r.message.lower()
    revealed_x = dict(zip(m.pnames, r.x))
    expected_x = {
        "ASC_Bike": -1.6287240376911014,
        "ASC_SR2": -1.8078232312767688,
        "ASC_SR3+": -3.4337524579776892,
        "ASC_Transit": -0.6848167683017945,
        "ASC_Walk": 0.06789809699436829,
        "costbyincome": -0.052418170888803996,
        "hhinc#4": -0.00532360665236578,
        "hhinc#5": -0.008644071235285034,
        "hhinc#6": -0.005996620294658138,
        "motorized_ovtbydist": -0.13286982431826339,
        "motorized_time": -0.020186966819292145,
        "nonmotorized_time": -0.04544348563829685,
        "vehbywrk_Bike": -0.7022733847559917,
        "vehbywrk_SR": -0.31663168406855235,
        "vehbywrk_Transit": -0.9462442341551858,
        "vehbywrk_Walk": -0.7217116484777637,
        "wkcbd_Bike": 0.4894649328283836,
        "wkcbd_SR2": 0.2598446761352557,
        "wkcbd_SR3+": 1.0692960216140865,
        "wkcbd_Transit": 1.3088289983522288,
        "wkcbd_Walk": 0.10193954307129499,
        "wkempden_Bike": 0.0019275473535991215,
        "wkempden_SR2": 0.0015776233147650747,
        "wkempden_SR3+": 0.002256812396741901,
        "wkempden_Transit": 0.003132410500039168,
        "wkempden_Walk": 0.0028901474800512262,
    }

    print(compute_engine, "====>", m.loglike(expected_x))

    for k in expected_x:
        assert revealed_x[k] == approx(
            expected_x[k], 1.1e-2
        ), f"{k}, {revealed_x[k] / expected_x[k]}"
    m.parameter_summary()


def test_mtc_22():

    m = lx.example(17)
    m.compute_engine = "numba"

    m.graph.new_node(parameter="mu_motor", children=[1, 2, 3, 4], name="Motorized")
    m.graph.new_node(parameter="mu_nonmotor", children=[5, 6], name="Nonmotorized")

    m.ordering = (
        (
            "CostbyInc",
            "costbyincome",
        ),
        (
            "TravelTime",
            ".*time.*",
            ".*dist.*",
        ),
        (
            "Household",
            "hhinc.*",
            "vehbywrk.*",
        ),
        (
            "Zonal",
            "wkcbd.*",
            "wkempden.*",
        ),
        (
            "ASCs",
            "ASC.*",
        ),
    )
    mj = m.copy()
    mj.compute_engine = "jax"

    result = m.maximize_loglike(method="bhhh")
    r = result
    from pytest import approx

    assert r.loglike == approx(-3441.6725273276093, rel=1e-4)
    assert dict(zip(m.pnames, r.x)) == approx(
        {
            "ASC_Bike": -1.2024073703132545,
            "ASC_SR2": -1.3257247399188727,
            "ASC_SR3+": -2.506936202874175,
            "ASC_Transit": -0.4041442821263078,
            "ASC_Walk": 0.3447191316975722,
            "costbyincome": -0.03864412094110262,
            "hhinc#4": -0.003929601755651791,
            "hhinc#5": -0.010035112486480783,
            "hhinc#6": -0.006205979434206748,
            "motorized_ovtbydist": -0.11389194935358538,
            "motorized_time": -0.014518697124195973,
            "mu_motor": 0.7261696323637531,
            "mu_nonmotor": 0.7690391629698575,
            "nonmotorized_time": -0.046200685225646534,
            "vehbywrk_Bike": -0.7347520730837045,
            "vehbywrk_SR": -0.22598417504565899,
            "vehbywrk_Transit": -0.7075038510739201,
            "vehbywrk_Walk": -0.7641289876632265,
            "wkcbd_Bike": 0.4077477599180845,
            "wkcbd_SR2": 0.1930608067123969,
            "wkcbd_SR3+": 0.7814124041724411,
            "wkcbd_Transit": 0.9217986579385763,
            "wkcbd_Walk": 0.11364443225208345,
            "wkempden_Bike": 0.0016747777566393732,
            "wkempden_SR2": 0.0011502120827767475,
            "wkempden_SR3+": 0.0016390812178071399,
            "wkempden_Transit": 0.0022379922179423173,
            "wkempden_Walk": 0.0021706844461508662,
        },
        rel=1e-2,
    )
    m.calculate_parameter_covariance()
    m.parameter_summary()
    expected_se = pd.Series(
        {
            "ASC_Bike": 0.416852464751687,
            "ASC_SR2": 0.2545998857743335,
            "ASC_SR3+": 0.4749098839206808,
            "ASC_Transit": 0.2211891608590185,
            "ASC_Walk": 0.35780565829941885,
            "costbyincome": 0.010368875452431911,
            "hhinc#4": 0.0016122509691149048,
            "hhinc#5": 0.004650739659998643,
            "hhinc#6": 0.00302148217700312,
            "motorized_ovtbydist": 0.021102031065567364,
            "motorized_time": 0.003865662496250571,
            "mu_motor": 0.13491012665162105,
            "mu_nonmotor": 0.1785021270767945,
            "nonmotorized_time": 0.005396871957201883,
            "vehbywrk_Bike": 0.22879172995664934,
            "vehbywrk_SR": 0.06504869465180056,
            "vehbywrk_Transit": 0.14983034511610385,
            "vehbywrk_Walk": 0.1633867246456,
            "wkcbd_Bike": 0.3276503369966752,
            "wkcbd_SR2": 0.09619096122973834,
            "wkcbd_SR3+": 0.19983327839835419,
            "wkcbd_Transit": 0.2218432314826066,
            "wkcbd_Walk": 0.23643542277462148,
            "wkempden_Bike": 0.0010873335879477298,
            "wkempden_SR2": 0.00035425322602890654,
            "wkempden_SR3+": 0.0004487422174289541,
            "wkempden_Transit": 0.0005072868584578029,
            "wkempden_Walk": 0.0007623255600411431,
        },
        name="t_stat",
    )
    pd.testing.assert_series_equal(
        m.parameters.std_err.to_series(), expected_se, rtol=5.0e-2, check_names=False
    )
    resultj = mj.maximize_loglike(stderr=True)
    r = resultj
    assert r.loglike == approx(-3441.6725273276093)
    assert dict(zip(mj.pnames, r.x)) == approx(
        {
            "ASC_Bike": -1.2024073703132545,
            "ASC_SR2": -1.3257247399188727,
            "ASC_SR3+": -2.506936202874175,
            "ASC_Transit": -0.4041442821263078,
            "ASC_Walk": 0.3447191316975722,
            "costbyincome": -0.03864412094110262,
            "hhinc#4": -0.003929601755651791,
            "hhinc#5": -0.010035112486480783,
            "hhinc#6": -0.006205979434206748,
            "motorized_ovtbydist": -0.11389194935358538,
            "motorized_time": -0.014518697124195973,
            "mu_motor": 0.7261696323637531,
            "mu_nonmotor": 0.7690391629698575,
            "nonmotorized_time": -0.046200685225646534,
            "vehbywrk_Bike": -0.7347520730837045,
            "vehbywrk_SR": -0.22598417504565899,
            "vehbywrk_Transit": -0.7075038510739201,
            "vehbywrk_Walk": -0.7641289876632265,
            "wkcbd_Bike": 0.4077477599180845,
            "wkcbd_SR2": 0.1930608067123969,
            "wkcbd_SR3+": 0.7814124041724411,
            "wkcbd_Transit": 0.9217986579385763,
            "wkcbd_Walk": 0.11364443225208345,
            "wkempden_Bike": 0.0016747777566393732,
            "wkempden_SR2": 0.0011502120827767475,
            "wkempden_SR3+": 0.0016390812178071399,
            "wkempden_Transit": 0.0022379922179423173,
            "wkempden_Walk": 0.0021706844461508662,
        },
        rel=1e-2,
    )
    assert mj.pstderr == approx(
        np.array(
            [
                4.168558e-01,
                2.545745e-01,
                4.749014e-01,
                2.212087e-01,
                3.577894e-01,
                1.036857e-02,
                1.612386e-03,
                4.650531e-03,
                3.021172e-03,
                2.110203e-02,
                3.866330e-03,
                1.349138e-01,
                1.784958e-01,
                5.396660e-03,
                2.287765e-01,
                6.506080e-02,
                1.498349e-01,
                1.633791e-01,
                3.276509e-01,
                9.619242e-02,
                1.998420e-01,
                2.218623e-01,
                2.364309e-01,
                1.087425e-03,
                3.542897e-04,
                4.488141e-04,
                5.073145e-04,
                7.623227e-04,
            ],
            dtype=np.float32,
        ),
        rel=1e-2,
    )


@pytest.mark.parametrize("compute_engine", ["numba"])
def test_mtc_30(compute_engine):
    m = lx.example(1)
    m.title = "MTC Example 30 (Constrained Simple MNL)"
    m.compute_engine = compute_engine
    m_explicit = m.copy()
    m_explicit.utility_ca = P.tottime * X.tottime + P.tottime * 3 * X.totcost
    m_explicit.remove_unused_parameters()
    result_explicit = m_explicit.maximize_loglike(
        stderr=True, options={"maxiter": 1000, "ftol": 1e-10}
    )
    revealed_x = dict(zip(m.pnames, result_explicit.x))
    print(revealed_x)
    assert result_explicit.logloss == approx(0.7533452255366115)
    assert result_explicit.loglike == approx(-3788.5731392236194)
    assert result_explicit.message == "Optimization terminated successfully"
    expected_x = {
        "ASC_BIKE": -3.1600042371469477,
        "ASC_SR2": -2.4559743104337053,
        "ASC_SR3P": -4.079065844612338,
        "ASC_TRAN": -1.894005040588427,
        "ASC_WALK": -1.9902266629268957,
        "hhinc#2": -0.0019829834793173512,
        "hhinc#3": 0.00044418076641711556,
        "hhinc#4": -0.006153651399668838,
        "hhinc#5": -0.014112829960368308,
        "hhinc#6": -0.009343128298864946,
        "totcost": -0.0017390713018709733,
    }
    revealed_x = {k: v for k, v in revealed_x.items() if k in expected_x}
    assert revealed_x == approx(expected_x, rel=0.1)

    from larch.model.constraints import RatioBound

    m.pmaximum = {"totcost": 0, "tottime": 0}

    m.constraints = [
        RatioBound("totcost", "tottime", min_ratio=3.0, max_ratio=999.0, scale=100),
    ]
    assert dict(m.required_data()) == {
        "ca": ["totcost", "tottime"],
        "co": ["hhinc"],
        "choice_ca": "chose",
        "avail_ca": "avail",
    }
    assert m.loglike() == approx(-7309.600971749634)
    result = m.maximize_loglike(stderr=True, options={"maxiter": 1000, "ftol": 1e-10})
    assert result.loglike == approx(-3788.573358)
    assert result.logloss == approx(0.753345269140234)
    assert result.message == "Optimization terminated successfully"
    assert m.total_weight() == 5029.0
    m.parameter_summary()
    summary = m.parameter_summary()
    # assert_same_text(
    #     summary.data.to_markdown(),
    #     """
    # | Parameter   |     Value |   Std Err |   t Stat | Signif   |   Null Value | Constrained             |
    # |:------------|----------:|----------:|---------:|:---------|-------------:|:------------------------|
    # | ASC_BIKE    | -3.16     |  0.309    |   -10.23 | ***      |            0 |                         |
    # | ASC_SR2     | -2.46     |  0.103    |   -23.94 | ***      |            0 |                         |
    # | ASC_SR3P    | -4.08     |  0.175    |   -23.26 | ***      |            0 |                         |
    # | ASC_TRAN    | -1.89     |  0.112    |   -16.87 | ***      |            0 |                         |
    # | ASC_WALK    | -1.99     |  0.169    |   -11.77 | ***      |            0 |                         |
    # | hhinc#2     | -0.00199  |  0.00154  |    -1.3  |          |            0 |                         |
    # | hhinc#3     |  0.000462 |  0.00252  |     0.18 |          |            0 |                         |
    # | hhinc#4     | -0.00616  |  0.0018   |    -3.42 | ***      |            0 |                         |
    # | hhinc#5     | -0.0141   |  0.0055   |    -2.57 | *        |            0 |                         |
    # | hhinc#6     | -0.00941  |  0.00306  |    -3.08 | **       |            0 |                         |
    # | totcost     | -0.00522  |  0.000243 |   -21.5  | ***      |            0 | totcost / tottime ≥ 3.0 |
    # | tottime     | -0.00174  |  8.09e-05 |   -21.5  | ***      |            0 | totcost / tottime ≥ 3.0 |
    #     """,
    # )
    m.ordering = (
        (
            "LOS",
            "totcost",
            "tottime",
        ),
        (
            "ASCs",
            "ASC.*",
        ),
        (
            "Income",
            "hhinc.*",
        ),
    )
    m.parameter_summary()
    summary2 = m.parameter_summary()
    # assert_same_text(
    #     summary2.data.to_markdown(),
    #     """
    # |                       |     Value |   Std Err |   t Stat | Signif   |   Null Value | Constrained             |
    # |:----------------------|----------:|----------:|---------:|:---------|-------------:|:------------------------|
    # | ('LOS', 'totcost')    | -0.00522  |  0.000243 |   -21.5  | ***      |            0 | totcost / tottime ≥ 3.0 |
    # | ('LOS', 'tottime')    | -0.00174  |  8.09e-05 |   -21.5  | ***      |            0 | totcost / tottime ≥ 3.0 |
    # | ('ASCs', 'ASC_BIKE')  | -3.16     |  0.309    |   -10.23 | ***      |            0 |                         |
    # | ('ASCs', 'ASC_SR2')   | -2.46     |  0.103    |   -23.94 | ***      |            0 |                         |
    # | ('ASCs', 'ASC_SR3P')  | -4.08     |  0.175    |   -23.26 | ***      |            0 |                         |
    # | ('ASCs', 'ASC_TRAN')  | -1.89     |  0.112    |   -16.87 | ***      |            0 |                         |
    # | ('ASCs', 'ASC_WALK')  | -1.99     |  0.169    |   -11.77 | ***      |            0 |                         |
    # | ('Income', 'hhinc#2') | -0.00199  |  0.00154  |    -1.3  |          |            0 |                         |
    # | ('Income', 'hhinc#3') |  0.000462 |  0.00252  |     0.18 |          |            0 |                         |
    # | ('Income', 'hhinc#4') | -0.00616  |  0.0018   |    -3.42 | ***      |            0 |                         |
    # | ('Income', 'hhinc#5') | -0.0141   |  0.0055   |    -2.57 | *        |            0 |                         |
    # | ('Income', 'hhinc#6') | -0.00941  |  0.00306  |    -3.08 | **       |            0 |                         |
    #     """,
    # )
