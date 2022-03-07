from pytest import approx, fixture
import larix as lx
import jax.numpy as jnp
import numpy as np
import xarray as xr

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
    m.availability_var = 'avail'
    m.choice_ca_var = 'chose'
    m.title = "MTC Example 1 (Simple MNL)"
    return m


def test_mtc_data(mtc7):
    assert mtc7.dc.n_cases == 7
    xr.testing.assert_equal(
        mtc7['avail'],
        xr.DataArray(
            np.array([
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ], dtype=np.int8),
            dims=('caseid', 'altid'),
            coords={
                'caseid': np.arange(1,8),
                'altid': np.arange(1,7),
                'altnames': xr.DataArray(
                    ['DA', 'SR2', 'SR3+', 'Transit', 'Bike', 'Walk'],
                    dims='altid',
                )
            }
        )
    )
    xr.testing.assert_equal(
        mtc7['chose'],
        xr.DataArray(
            np.array([
                [1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0.],
                [1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0.],
                [0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0.],
                [1., 0., 0., 0., 0., 0.],
            ], dtype=np.int8),
            dims=('caseid', 'altid'),
            coords={
                'caseid': np.arange(1,8),
                'altid': np.arange(1,7),
                'altnames': xr.DataArray(
                    ['DA', 'SR2', 'SR3+', 'Transit', 'Bike', 'Walk'],
                    dims='altid',
                )
            }
        )
    )


def test_basic_utility(model7):
    m = model7
    u = m.jax_utility(m.pvals)
    target = np.array([
        [0., 0., 0., 0., 0., -inf, 0.],
        [0., 0., 0., 0., 0., -inf, 0.],
        [0., 0., 0., 0., -inf, -inf, 0.],
        [0., 0., 0., 0., -inf, -inf, 0.],
        [-inf, 0., 0., 0., 0., -inf, 0.],
        [-inf, 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
    ], dtype=np.float32)
    assert u == approx(target)

    # An IDCA parameter
    m.pvals = 'null'
    m.pvals = {'tottime': -1}
    u = m.jax_utility(m.pvals)
    target = np.array([
        [-15.38, -20.38, -22.38, -41.1, -42.5, -inf, 0.],
        [-39.92, -44.92, -31.92, -37.16, -68.95, -inf, 0.],
        [-14.6, -19.6, -21.6, -38.27, -inf, -inf, 0.],
        [-39.1, -44.2, -31.2, -33.27, -inf, -inf, 0.],
        [-inf, -26.04, -28.04, -48.54, -37.35, -inf, 0.],
        [-inf, -14.64, -16.64, -20.19, -19.55, -60.1, 0.],
        [-11.8, -16.8, -18.8, -69.88, -21.15, -71.8, 0.],
    ], dtype=np.float32)
    assert u == approx(target)

    # An IDCO constant
    m.pvals = 'null'
    m.pvals = {'ASC_SR2': -1}
    u = m.jax_utility(m.pvals)
    target = np.array([
        [0., -1., 0., 0., 0., -inf, 0.],
        [0., -1., 0., 0., 0., -inf, 0.],
        [0., -1., 0., 0., -inf, -inf, 0.],
        [0., -1., 0., 0., -inf, -inf, 0.],
        [-inf, -1., 0., 0., 0., -inf, 0.],
        [-inf, -1., 0., 0., 0., 0., 0.],
        [0., -1., 0., 0., 0., 0., 0.],
    ], dtype=np.float32)
    assert u == approx(target)

    # An IDCO variable
    m.pvals = 'null'
    m.pvals = {"hhinc#5": -1}
    u = m.jax_utility(m.pvals)
    target = np.array([
        [0., 0., 0., 0., -42.5, -inf, 0.],
        [0., 0., 0., 0., -17.5, -inf, 0.],
        [0., 0., 0., 0., -inf, -inf, 0.],
        [0., 0., 0., 0., -inf, -inf, 0.],
        [-inf, 0., 0., 0., -87.5, -inf, 0.],
        [-inf, 0., 0., 0., -87.5, 0., 0.],
        [0., 0., 0., 0., -12.5, 0., 0.],
    ], dtype=np.float32)
    assert u == approx(target)
