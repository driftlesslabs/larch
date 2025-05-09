from __future__ import annotations

import jax  # noqa: F401
import numpy as np
import pandas as pd
from pytest import approx, fixture

import larch as lx
from larch import PX, P, X


@fixture(scope="module")
def raw_data() -> pd.DataFrame:
    return pd.read_parquet(lx.example_file("vehicle_choice.parquet"))


@fixture
def simple_model(raw_data: pd.DataFrame) -> lx.Model:
    data = lx.Dataset.construct.from_idca(raw_data)
    simple = lx.Model(data)
    # price and opcost are scaled to improve numerical stability
    simple.utility_ca = (
        P("price") * X("price / 10000")
        + P("opcost") * X("opcost / 10")
        + PX("max_range")
        + PX("ev")
        + PX("hybrid")
        + PX("hiperf")
        + PX("medhiperf")
    )
    simple.choice_ca_var = "chosen"
    return simple


def test_vehicle_choice(simple_model: lx.Model):
    simple = simple_model
    simple.maximize_loglike(stderr=True, options={"ftol": 1e-9})
    assert simple.most_recent_estimation_result["loglike"] == approx(-1399.1932)
    expected_value = {
        "price": -0.4167,
        "opcost": -0.1288,
        "max_range": 0.4770,
        "ev": -1.3924,
        "hybrid": 0.3555,
        "hiperf": 0.1099,
        "medhiperf": 0.3841,
    }
    assert simple.parameters["value"].to_series().to_dict() == approx(
        expected_value, rel=1e-3
    )
    expected_stderr = {
        "price": 0.0332,
        "opcost": 0.0353,
        "max_range": 0.1765,
        "ev": 0.2766,
        "hybrid": 0.1218,
        "hiperf": 0.0838,
        "medhiperf": 0.0855,
    }
    assert simple.parameters["std_err"].to_series().to_dict() == approx(
        expected_stderr, rel=2e-3
    )


def test_mixed_logit(simple_model: lx.Model):
    mixed = simple_model.copy()
    mixed.mixtures = [
        lx.mixtures.Normal(k, f"s_{k}")
        for k in ["opcost", "max_range", "ev", "hybrid", "hiperf", "medhiperf"]
    ]
    mixed.n_draws = 200
    mixed.seed = 42
    assert mixed.pvals == approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    )
    mixed.pvals = "null"
    assert mixed.pvals == approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert mixed.compute_engine == "jax"
    assert mixed.n_draws == 200
    assert mixed.seed == 42
    assert len(mixed.mixtures) == 6
    assert isinstance(mixed.mixtures[0], lx.mixtures.Normal)
    assert mixed.mixtures[0].to_dict() == {
        "type": "Normal",
        "mean": "opcost",
        "std": "s_opcost",
    }

    # Check initial values are a good match
    assert mixed.loglike() == approx(-1630.34033203125)
    np.testing.assert_allclose(
        mixed.d_loglike(),
        np.array(
            [
                -2.056667e02,
                5.766666e01,
                6.666666e01,
                -2.477668e02,
                8.300000e01,
                2.914683e02,
                -7.866664e02,
                -3.433672e-02,
                -2.093602e-02,
                -2.493681e-02,
                1.647819e-02,
                9.754598e-04,
                2.504729e-01,
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
    )

    result0 = mixed.maximize_loglike(stderr=True, options={"maxiter": 2})
    assert not result0["success"]
    assert result0["nit"] == 2
    assert result0["loglike"] == approx(-1565.2197265625)
    expected_value = {
        "ev": -0.14029760159652338,
        "hiperf": 0.03933788979469556,
        "hybrid": 0.04547732430870773,
        "max_range": -0.1690165820789483,
        "medhiperf": 0.05661927740375723,
        "opcost": 0.19882802193266583,
        "price": -0.5366323560619928,
        "s_ev": -2.3423133069298892e-05,
        "s_hiperf": -1.4281714556392845e-05,
        "s_hybrid": -1.7010893622241297e-05,
        "s_max_range": 1.124076043977622e-05,
        "s_medhiperf": 6.654196362547049e-07,
        "s_opcost": 0.0001708625878473434,
    }
    assert mixed.parameters["value"].to_series().to_dict() == approx(
        expected_value, rel=5e-2
    )

    mixed.maximize_loglike(stderr=True, options={"ftol": 1e-9})
    # TEST
    assert mixed.float_dtype == np.float64
    assert mixed.most_recent_estimation_result["success"]
    assert mixed.most_recent_estimation_result["total_weight"] == approx(1484.0)
    assert mixed.most_recent_estimation_result["loglike"] == approx(-1385.494384765625)
    expected_value = {
        "ev": -3.0386047026095038,
        "hiperf": 0.1759331797395676,
        "hybrid": 0.5469828878951994,
        "max_range": 0.8913475142526257,
        "medhiperf": 0.734468888360476,
        "opcost": -0.22007540332571882,
        "price": -0.648648076713321,
        "s_ev": -2.810148184360984,
        "s_hiperf": -0.5264974590918174,
        "s_hybrid": 1.1459421445296165,
        "s_max_range": -0.12631587788430482,
        "s_medhiperf": 1.5788576158691365,
        "s_opcost": 0.5213287408633901,
    }
    assert mixed.parameters["value"].to_series().to_dict() == approx(
        expected_value, rel=5e-2
    )
    expected_stderr = {
        "ev": 0.7826659083366394,
        "hiperf": 0.12497398257255554,
        "hybrid": 0.19386492669582367,
        "max_range": 0.32467690110206604,
        "medhiperf": 0.2096310406923294,
        "opcost": 0.0717368870973587,
        "price": 0.10241813212633133,
        "s_ev": 0.7595077753067017,
        "s_hiperf": 0.8704573512077332,
        "s_hybrid": 0.6614100337028503,
        "s_max_range": 1.0017218589782715,
        "s_medhiperf": 0.6141265034675598,
        "s_opcost": 0.1859593540430069,
    }
    assert mixed.parameters["std_err"].to_series().to_dict() == approx(
        expected_stderr, rel=5e-2
    )

    # make sure we can copy the model
    mixed2 = mixed.copy()
    assert mixed2.compute_engine == "jax"
    assert mixed2.n_draws == 200
    assert mixed2.seed == 42
    assert len(mixed2.mixtures) == 6
    assert isinstance(mixed2.mixtures[0], lx.mixtures.Normal)
    assert mixed2.mixtures[0].to_dict() == {
        "type": "Normal",
        "mean": "opcost",
        "std": "s_opcost",
    }
    assert mixed.parameters["value"].to_series().to_dict() == approx(
        expected_value, rel=5e-2
    )
    assert mixed.parameters["std_err"].to_series().to_dict() == approx(
        expected_stderr, rel=5e-2
    )
