from __future__ import annotations

import jax  # noqa: F401
import numpy as np
import pandas as pd
from pytest import approx, fixture

import larch as lx
from larch import PX


@fixture(scope="module")
def raw_data() -> pd.DataFrame:
    return pd.read_parquet(lx.example_file("vehicle_choice.parquet"))


@fixture
def simple_model(raw_data: pd.DataFrame) -> lx.Model:
    data = lx.Dataset.construct.from_idca(raw_data)
    simple = lx.Model(data)
    simple.utility_ca = (
        PX("price")
        + PX("opcost")
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
        "price": -0.4167 / 10000,
        "opcost": -0.1288 / 10,
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
        "price": 0.0332 / 10000,
        "opcost": 0.0353 / 10,
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
                2.914685e03,
                -7.866661e06,
                -3.433672e-02,
                -2.093602e-02,
                -2.493681e-02,
                1.647819e-02,
                9.754598e-04,
                2.504725e00,
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
    )

    result0 = mixed.maximize_loglike(stderr=True, options={"maxiter": 2})
    assert not result0["success"]
    assert result0["nit"] == 2
    assert result0["loglike"] == approx(-8264.240234375)

    mixed.maximize_loglike(stderr=True, options={"ftol": 1e-9})
    # TEST
    assert mixed.float_dtype == np.float64
    assert mixed.most_recent_estimation_result["success"]
    assert mixed.most_recent_estimation_result["total_weight"] == approx(1484.0)
    # assert mixed.most_recent_estimation_result["nit"] == 69
    assert mixed.most_recent_estimation_result["loglike"] == approx(-1385.39794921875)
    expected_value = {
        "ev": -3.277402013573714,
        "hiperf": 0.16966497775326497,
        "hybrid": 0.5611881591133725,
        "max_range": 0.910611685813643,
        "medhiperf": 0.7898127484131651,
        "opcost": -0.02352086247664143,
        "price": -6.837570351215833e-05,
        "s_ev": -3.123950469481568,
        "s_hiperf": 0.9649833582498155,
        "s_hybrid": -1.3488156988733033,
        "s_max_range": 0.38154871637880616,
        "s_medhiperf": -1.6749211275719094,
        "s_opcost": 0.05002188876036286,
    }
    assert mixed.parameters["value"].to_series().to_dict() == approx(
        expected_value, rel=5e-2
    )
    expected_stderr = {
        "ev": 0.9645621180534363,
        "hiperf": 0.13538870215415955,
        "hybrid": 0.20663860440254211,
        "max_range": 0.37424299120903015,
        "medhiperf": 0.2417304515838623,
        "opcost": 0.007898632436990738,
        "price": 1.2206787687318865e-05,
        "s_ev": 0.960139811038971,
        "s_hiperf": 0.7722559571266174,
        "s_hybrid": 0.6910321116447449,
        "s_max_range": 0.7266789078712463,
        "s_medhiperf": 0.6885043382644653,
        "s_opcost": 0.019354678690433502,
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
