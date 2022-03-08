import jax.numpy as jnp
import numpy as np
import xarray as xr
from pytest import approx, fixture

import larix as lx

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
    m.availability_var = "avail"
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
                "altnames": xr.DataArray(
                    ["DA", "SR2", "SR3+", "Transit", "Bike", "Walk"], dims="altid",
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
                "altnames": xr.DataArray(
                    ["DA", "SR2", "SR3+", "Transit", "Bike", "Walk"], dims="altid",
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
    for k in randvars:
        m.mix_parameter(k, f"{k}_s")
    return m


def test_prerolled_common(xlogit_artificial):
    m = xlogit_artificial
    m.n_draws = 100
    m.seed = 42
    m.prerolled = True
    m.common_draws = True
    m._make_random_draws()
    r = m.jax_maximize_loglike()
    assert m._draws.shape == (100, 3)
    expected = {
        "x": np.array(
            [
                0.974191,
                0.803419,
                -1.882733,
                0.910499,
                1.623802,
                -0.555167,
                0.033868,
                0.035736,
                0.012158,
                3.591912,
                1.148798,
                -0.977542,
                -1.362563,
            ]
        ),
        "jac": np.array(
            [
                0.036815,
                -0.065251,
                -0.076691,
                -0.044403,
                0.008265,
                -0.004813,
                -0.048743,
                0.011079,
                -0.022234,
                -0.058469,
                -0.012695,
                -0.017211,
                -0.008998,
            ]
        ),
        "nit": 48,
        "nfev": 103,
        "njev": 48,
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2281.3876953125,
    }
    for k, v in expected.items():
        assert r[k] == approx(v, rel=1e-4), k


def test_prerolled_notcommon(xlogit_artificial):
    m = xlogit_artificial
    m.n_draws = 100
    m.seed = 42
    m.prerolled = True
    m.common_draws = False
    m._make_random_draws()
    assert m._draws.shape == (4000, 100, 3)
    r = m.jax_maximize_loglike()
    expected = {
        "x": np.array(
            [
                0.998088,
                0.86163,
                -1.8877,
                -0.996309,
                1.662274,
                -0.4996,
                0.075712,
                -0.003768,
                -0.006356,
                3.628446,
                -1.069672,
                -0.964553,
                -1.380425,
            ]
        ),
        "jac": np.array(
            [
                -0.00057065,   # -0.000571,
                0.02387673,    # 0.023877,
                0.04541352,    # 0.045414,
                -0.03415966,   # -0.03416,
                -0.00718924,   # -0.007189,
                0.00062239,    # 0.000622,
                -0.00588775,   # -0.005888,
                0.00627816,    # 0.006278,
                0.01326609,    # 0.013266,
                0.01735568,    # 0.017356,
                -0.00986013,   # -0.00986,
                0.05631149,    # 0.056311,
                0.01436687     # 0.014367,
            ]
        ),
        "nit": 48,
        "nfev": 99,
        "njev": 48,
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2278.027587890625,
    }
    for k, v in expected.items():
        assert r[k] == approx(v, rel=1e-4), k


def test_prerolled_common_panel(xlogit_artificial):
    m = xlogit_artificial
    m.n_draws = 100
    m.seed = 42
    m.prerolled = True
    m.common_draws = True
    m.groupid = "panel"
    m._make_random_draws()
    r = m.jax_maximize_loglike()
    assert m._draws.shape == (100, 3)
    expected = {
        "x": np.array(
            [
                0.30750718, 0.36996433, -0.69783579, -0.04693729, 0.55090686,
                0.12666472, -0.04562635, 0.01679837, -0.15892525, 1.47167849,
                -0.01195818, -0.25290716, -0.62790667
                # 0.306783,
                # 0.369681,
                # -0.696866,
                # 0.044465,
                # 0.548005,
                # 0.142616,
                # -0.046386,
                # 0.016564,
                # -0.159195,
                # 1.470395,
                # -0.015535,
                # -0.252819,
                # -0.628569,
            ]
        ),
        "jac": np.array(
            [
                -0.05060759, 0.05077744, 0.10762198, 0.12020826, -0.08236539,
                -0.04637361, -0.00720203, -0.02035937, 0.0121733, 0.04453276,
                0.02297267, 0.06204722, 0.0245111,
                # -0.250353,
                # 0.374789,
                # 0.734309,
                # 1.01121,
                # -0.158797,
                # 0.001663,
                # 0.40691,
                # -0.250998,
                # 0.346529,
                # 0.298675,
                # 0.320278,
                # 0.364728,
                # 0.151023,
            ]
        ),
        "nit": 42,
        "nfev": 103,
        "njev": 42,
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2410.3095703125,
    }
    for k, v in expected.items():
        assert r[k] == approx(v, rel=1e-4), k


def test_prerolled_notcommon_panel(xlogit_artificial):
    m = xlogit_artificial
    m.mangle()
    m.n_draws = 100
    m.seed = 42
    m.prerolled = True
    m.common_draws = False
    m.groupid = "panel"
    m._make_random_draws()
    r = m.jax_maximize_loglike()
    assert m._draws.shape == (400, 100, 3)
    expected = {
        "x": np.array(
            [
                # 0.30750718, 0.36996433, -0.69783579, -0.04693729, 0.55090686,
                # 0.12666472, -0.04562635, 0.01679837, -0.15892525, 1.47167849,
                # -0.01195818, -0.25290716, -0.62790667,
                0.30678348, 0.36968052, -0.696866, 0.04446525, 0.54800504,
                0.14261559, -0.04638611, 0.01656364, -0.15919519, 1.47039525,
                -0.01553508, -0.25281922, -0.6285691
            ]
        ),
        "jac": np.array(
            [
                -0.25035292, 0.37478894, 0.73430878, 1.01120996, -0.15879673,
                0.00166333, 0.40691006, -0.25099784, 0.34652871, 0.29867542,
                0.32027781, 0.36472762, 0.15102315,
            ]
        ),
        "nit": 26,
        "nfev": 82,
        "njev": 26,
        "status": 0,
        "message": "Optimization terminated successfully",
        "success": True,
        "loglike": -2410.3095703125,
    }
    for k, v in expected.items():
        assert r[k] == approx(v, rel=1e-4), k
