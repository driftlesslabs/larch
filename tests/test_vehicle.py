from __future__ import annotations

import jax  # noqa: F401
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
    mixed.maximize_loglike(stderr=True, options={"ftol": 1e-9})
    # TEST
    assert mixed.most_recent_estimation_result["loglike"] == approx(-1385.4436)
    expected_value = {
        "ev": -3.116113749980346,
        "hiperf": 0.16526011648413752,
        "hybrid": 0.5468359000231515,
        "max_range": 0.893907686323169,
        "medhiperf": 0.7444412811439853,
        "opcost": -0.022305055541141636,
        "price": -6.53728508122224e-05,
        "s_ev": 2.8852968025136363,
        "s_hiperf": 0.7816346849525359,
        "s_hybrid": -1.2110724172285334,
        "s_max_range": -0.10080141887433067,
        "s_medhiperf": -1.5364071517526834,
        "s_opcost": 0.04830038015701635,
    }
    assert mixed.parameters["value"].to_series().to_dict() == approx(
        expected_value, rel=5e-2
    )
    expected_stderr = {
        "ev": 0.8305132389068604,
        "hiperf": 0.12845700979232788,
        "hybrid": 0.20117034018039703,
        "max_range": 0.33233582973480225,
        "medhiperf": 0.21790540218353271,
        "opcost": 0.007205564994364977,
        "price": 1.0975929399137385e-05,
        "s_ev": 0.7865121364593506,
        "s_hiperf": 0.8348018527030945,
        "s_hybrid": 0.6305373907089233,
        "s_max_range": 0.9989855885505676,
        "s_medhiperf": 0.6614137887954712,
        "s_opcost": 0.0211756881326437,
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
