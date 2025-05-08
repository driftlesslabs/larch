from __future__ import annotations

import numpy as np
import xarray as xr
from pytest import approx, fixture

import larch as lx

inf = np.inf


@fixture
def mtc7():
    d = lx.examples.MTC()
    return d.icase[:7]


@fixture
def model7(mtc7):
    m = lx.Model(mtc7)
    P, X, PX = lx.P, lx.X, lx.PX
    m.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
    m.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
    m.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
    m.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
    m.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
    m.utility_ca = PX("tottime") + PX("totcost")
    m.availability_ca_var = "avail"
    m.choice_ca_var = "chose"
    m.title = "MTC Example 1 (Simple MNL)"
    return m


def test_mtc_data(mtc7):
    assert mtc7.dc.n_cases == 7
    xr.testing.assert_equal(
        mtc7["avail"],
        xr.DataArray(
            np.array(
                [
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ],
                dtype=np.int8,
            ),
            dims=("caseid", "altid"),
            coords={
                "caseid": np.arange(1, 8),
                "altid": np.arange(1, 7),
                "alt_names": xr.DataArray(
                    ["DA", "SR2", "SR3+", "Transit", "Bike", "Walk"],
                    dims="altid",
                ),
            },
        ),
    )
    xr.testing.assert_equal(
        mtc7["chose"],
        xr.DataArray(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.int8,
            ),
            dims=("caseid", "altid"),
            coords={
                "caseid": np.arange(1, 8),
                "altid": np.arange(1, 7),
                "alt_names": xr.DataArray(
                    ["DA", "SR2", "SR3+", "Transit", "Bike", "Walk"],
                    dims="altid",
                ),
            },
        ),
    )


def test_basic_utility(model7):
    m = model7
    u = m.jax_utility(m.pvals)
    target = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, -inf, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -inf, 0.0],
            [0.0, 0.0, 0.0, 0.0, -inf, -inf, 0.0],
            [0.0, 0.0, 0.0, 0.0, -inf, -inf, 0.0],
            [-inf, 0.0, 0.0, 0.0, 0.0, -inf, 0.0],
            [-inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert u == approx(target)

    # An IDCA parameter
    m.pvals = "null"
    m.pvals = {"tottime": -1}
    u = m.jax_utility(m.pvals)
    target = np.array(
        [
            [-15.38, -20.38, -22.38, -41.1, -42.5, -inf, 0.0],
            [-39.92, -44.92, -31.92, -37.16, -68.95, -inf, 0.0],
            [-14.6, -19.6, -21.6, -38.27, -inf, -inf, 0.0],
            [-39.1, -44.2, -31.2, -33.27, -inf, -inf, 0.0],
            [-inf, -26.04, -28.04, -48.54, -37.35, -inf, 0.0],
            [-inf, -14.64, -16.64, -20.19, -19.55, -60.1, 0.0],
            [-11.8, -16.8, -18.8, -69.88, -21.15, -71.8, 0.0],
        ],
        dtype=np.float32,
    )
    assert u == approx(target)

    # An IDCO constant
    m.pvals = "null"
    m.pvals = {"ASC_SR2": -1}
    u = m.jax_utility(m.pvals)
    target = np.array(
        [
            [0.0, -1.0, 0.0, 0.0, 0.0, -inf, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, -inf, 0.0],
            [0.0, -1.0, 0.0, 0.0, -inf, -inf, 0.0],
            [0.0, -1.0, 0.0, 0.0, -inf, -inf, 0.0],
            [-inf, -1.0, 0.0, 0.0, 0.0, -inf, 0.0],
            [-inf, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert u == approx(target)

    # An IDCO variable
    m.pvals = "null"
    m.pvals = {"hhinc#5": -1}
    u = m.jax_utility(m.pvals)
    target = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, -42.5, -inf, 0.0],
            [0.0, 0.0, 0.0, 0.0, -17.5, -inf, 0.0],
            [0.0, 0.0, 0.0, 0.0, -inf, -inf, 0.0],
            [0.0, 0.0, 0.0, 0.0, -inf, -inf, 0.0],
            [-inf, 0.0, 0.0, 0.0, -87.5, -inf, 0.0],
            [-inf, 0.0, 0.0, 0.0, -87.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -12.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert u == approx(target)


def test_basic_loglike_mnl(model7):
    m = model7
    ll = m.jax_loglike_casewise(m.pvals)
    target = np.array(
        [-1.609438, -1.609438, -1.386294, -1.386294, -1.386294, -1.609438, -1.791759],
        dtype=np.float32,
    )
    assert ll == approx(target, rel=1e-4)
    assert m.jax_loglike(m.pvals) == approx(-10.778956)

    m.pvals = "null"
    m.pvals = {"tottime": -1}
    ll = m.jax_loglike_casewise(m.pvals)
    target = np.array(
        [-0.007621, -5.245622, -0.007621, -2.189167, -0.126939, -5.686797, -0.007707],
        dtype=np.float32,
    )
    assert ll == approx(target, rel=1e-4)
    assert m.jax_loglike(m.pvals) == approx(-13.271474)

    m.pvals = "null"
    m.pvals = {"ASC_SR2": -1}
    ll = m.jax_loglike_casewise(m.pvals)
    target = np.array(
        [-1.474278, -1.474278, -1.214283, -1.214283, -2.214283, -1.474278, -1.680433],
        dtype=np.float32,
    )
    assert ll == approx(target, rel=1e-4)
    assert m.jax_loglike(m.pvals) == approx(-10.746116)

    m.pvals = "null"
    m.pvals = {"hhinc#5": -1}
    ll = m.jax_loglike_casewise(m.pvals)
    target = np.array(
        [-1.386294, -1.386294, -1.386294, -1.386294, -1.098612, -1.386294, -1.609439],
        dtype=np.float32,
    )
    assert ll == approx(target, rel=1e-4)
    assert m.jax_loglike(m.pvals) == approx(-9.639524)


@fixture
def xlogit_artificial():
    varnames = [
        "price",
        "time",
        "conven",
        "comfort",
        "meals",
        "petfr",
        "emipp",
        "nonsig1",
        "nonsig2",
        "nonsig3",
    ]
    d = lx.examples.ARTIFICIAL()
    m = lx.Model(d)
    m.utility_ca = sum(lx.PX(i) for i in varnames)
    m.choice_ca_var = "choice"
    randvars = {"meals": "n", "petfr": "n", "emipp": "n"}
    m.mixtures = [lx.mixtures.Normal(k, f"{k}_s") for k in randvars]
    return m


def test_prerolled_common(xlogit_artificial):
    m = xlogit_artificial
    m.n_draws = 100
    m.seed = 42
    m.common_draws = True
    m.make_random_draws(engine="jax")
    r = m.jax_maximize_loglike()
    assert m._draws.shape == (100, 3)
    expected = {
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2281.146240234375,
    }
    for k, v in expected.items():
        if v is True:
            assert r[k], k
        else:
            assert r[k] == approx(v, rel=1e-4), k
    assert m.d_loglike(
        [1.05262625, 0.86398761, -1.95256468, -0.91285049, 1.68754496]
        + [0.65481684, 0.04932408, 0.03698133, 0.02225039, 3.78042743]
        + [-1.30948931, -1.02678108, -1.43896373]
    ) == approx(
        np.array(
            [-0.01108778, -0.00438261, 0.0271306, -0.01761484, 0.01830649]
            + [0.01826873, 0.03595978, -0.05165243, 0.06293976, 0.00445962]
            + [-0.00531793, -0.02016234, 0.04396705]
        ),
        rel=1e-1,
    )


def test_prerolled_notcommon(xlogit_artificial):
    m = xlogit_artificial
    m.n_draws = 100
    m.seed = 42
    m.common_draws = False
    m.make_random_draws(engine="jax")
    assert m._draws.shape == (4000, 100, 3)
    r = m.jax_maximize_loglike()
    expected = {
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2282.65625,
    }
    for k, v in expected.items():
        if v is True:
            assert r[k], k
        else:
            assert r[k] == approx(v, rel=1e-4), k


def test_prerolled_common_panel(xlogit_artificial):
    m = xlogit_artificial
    m.n_draws = 100
    m.seed = 42
    m.common_draws = True
    m.groupid = "panel"
    m.make_random_draws(engine="jax")
    r = m.jax_maximize_loglike()
    assert m._draws.shape == (100, 3)
    expected = {
        # "x": np.array(
        #     [
        #         0.30750718,
        #         0.36996433,
        #         -0.69783579,
        #         -0.04693729,
        #         0.55090686,
        #         0.12666472,
        #         -0.04562635,
        #         0.01679837,
        #         -0.15892525,
        #         1.47167849,
        #         -0.01195818,
        #         -0.25290716,
        #         -0.62790667,
        #     ]
        # ),
        # "jac": np.array(
        #     [
        #         -0.05060759,
        #         0.05077744,
        #         0.10762198,
        #         0.12020826,
        #         -0.08236539,
        #         -0.04637361,
        #         -0.00720203,
        #         -0.02035937,
        #         0.0121733,
        #         0.04453276,
        #         0.02297267,
        #         0.06204722,
        #         0.0245111,
        #     ]
        # ),
        # "nit": 42,
        # "nfev": 103,
        # "njev": 42,
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2410.298828125,
    }
    for k, v in expected.items():
        if v is True:
            assert r[k], k
        else:
            assert r[k] == approx(v, rel=1e-4), k


def test_prerolled_notcommon_panel(xlogit_artificial):
    m = xlogit_artificial
    m.mangle()
    m.n_draws = 100
    m.seed = 42
    m.common_draws = False
    m.groupid = "panel"
    m.make_random_draws(engine="jax")
    r = m.jax_maximize_loglike()
    assert m._draws.shape == (400, 100, 3)
    expected = {
        # "x": np.array(
        #     [
        #         0.30678348,
        #         0.36968052,
        #         -0.696866,
        #         0.04446525,
        #         0.54800504,
        #         0.14261559,
        #         -0.04638611,
        #         0.01656364,
        #         -0.15919519,
        #         1.47039525,
        #         -0.01553508,
        #         -0.25281922,
        #         -0.6285691,
        #     ]
        # ),
        # "jac": np.array(
        #     [
        #         -0.25035292,
        #         0.37478894,
        #         0.73430878,
        #         1.01120996,
        #         -0.15879673,
        #         0.00166333,
        #         0.40691006,
        #         -0.25099784,
        #         0.34652871,
        #         0.29867542,
        #         0.32027781,
        #         0.36472762,
        #         0.15102315,
        #     ]
        # ),
        # "nit": 26,
        # "nfev": 82,
        # "njev": 26,
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2409.8564453125,
    }
    for k, v in expected.items():
        if v is True:
            assert r[k], k
        else:
            assert r[k] == approx(v, rel=1e-4), k


def test_prerolled_common_numpy(xlogit_artificial):
    m = xlogit_artificial
    m.n_draws = 100
    m.seed = 42
    m.common_draws = True
    m.make_random_draws(engine="numpy")
    r = m.jax_maximize_loglike()
    assert m._draws.shape == (100, 3)
    expected = {
        # "x": np.array(
        #     [
        #         1.01824077,
        #         0.86954868,
        #         -1.92307968,
        #         0.87533738,
        #         1.64613896,
        #         0.65090859,
        #         0.03953801,
        #         0.03554835,
        #         0.03748745,
        #         3.76680678,
        #         -1.31003218,
        #         -0.97908175,
        #         -1.40014785,
        #     ]
        # ),
        # "jac": np.array(
        #     [
        #         0.02463913,
        #         -0.02158892,
        #         -0.05098915,
        #         -0.09312749,
        #         0.02326787,
        #         -0.00692827,
        #         0.05335683,
        #         -0.01071632,
        #         0.01512003,
        #         -0.00716937,
        #         0.03487635,
        #         -0.07052767,
        #         -0.00752079,
        #     ]
        # ),
        # "nit": 56,
        # "nfev": 113,
        # "njev": 56,
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2280.888916015625,
    }
    for k, v in expected.items():
        if v is True:
            assert r[k], k
        else:
            assert r[k] == approx(v, rel=1e-4), k


def test_prerolled_notcommon_numpy(xlogit_artificial):
    m = xlogit_artificial
    m.n_draws = 100
    m.seed = 42
    m.common_draws = False
    m.make_random_draws(engine="numpy")
    assert m._draws.shape == (4000, 100, 3)
    r = m.jax_maximize_loglike()
    expected = {
        # "x": np.array(
        #     [
        #         1.05478002,
        #         0.91677321,
        #         -1.98208336,
        #         1.01215019,
        #         1.6889608,
        #         -0.68103816,
        #         0.03213415,
        #         0.00387927,
        #         0.02359222,
        #         3.83056018,
        #         1.19754625,
        #         -1.04429401,
        #         -1.43690736,
        #     ]
        # ),
        # "jac": np.array(
        #     [
        #         -0.01412737,
        #         -0.05148172,
        #         0.00548744,
        #         -0.02089691,
        #         0.06817544,
        #         0.00233963,
        #         0.04330277,
        #         -0.04035771,
        #         0.07260931,
        #         -0.01422405,
        #         0.00388414,
        #         -0.07043552,
        #         0.06703633,
        #     ]
        # ),
        # "nit": 57,
        # "nfev": 117,
        # "njev": 57,
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2278.027587890625,
    }
    for k, v in expected.items():
        if v is True:
            assert r[k], k
        else:
            assert r[k] == approx(v, rel=1.2e-4), k


def test_prerolled_common_panel_numpy(xlogit_artificial):
    m = xlogit_artificial
    m.n_draws = 100
    m.seed = 42
    m.common_draws = True
    m.groupid = "panel"
    m.make_random_draws(engine="numpy")
    r = m.jax_maximize_loglike()
    assert m._draws.shape == (100, 3)
    expected = {
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2410.3818,
    }
    for k, v in expected.items():
        if v is True:
            assert r[k], k
        else:
            assert r[k] == approx(v, rel=1.3e-4), k


def test_prerolled_notcommon_panel_numpy(xlogit_artificial):
    m = xlogit_artificial
    m.mangle()
    m.n_draws = 100
    m.seed = 42
    m.common_draws = False
    m.groupid = "panel"
    m.make_random_draws(engine="numpy")
    r = m.jax_maximize_loglike()
    assert m._draws.shape == (400, 100, 3)
    expected = {
        # "x": np.array(
        #     [
        #         0.30815274,
        #         0.37004427,
        #         -0.6982185,
        #         0.05149314,
        #         0.54701746,
        #         0.12900519,
        #         -0.04664678,
        #         0.01746336,
        #         -0.15724227,
        #         1.47156871,
        #         -0.00690103,
        #         -0.25272569,
        #         -0.62692483,
        #     ]
        # ),
        # "jac": np.array(
        #     [
        #         0.28615081,
        #         -0.16590273,
        #         -0.61982584,
        #         -1.31703365,
        #         0.03213306,
        #         -0.50487339,
        #         -0.35227597,
        #         0.35448289,
        #         -0.43845224,
        #         -0.19293058,
        #         -0.41268432,
        #         -0.17269456,
        #         -0.24641925,
        #     ]
        # ),
        # "nit": 25,
        # "nfev": 81,
        # "njev": 25,
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2409.903076171875,
    }
    for k, v in expected.items():
        if v is True:
            assert r[k], k
        else:
            assert r[k] == approx(v, rel=1e-4), k
