from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest import approx
from scipy.optimize import LinearConstraint, NonlinearConstraint

import larch as lx
from larch.model.constraints import OrderingBound


def _choice_model(compute_engine, counts=(10, 30, 60), *, shared=False):
    choices = np.repeat(np.arange(1, len(counts) + 1), counts)
    frame = pd.DataFrame({"choice": choices}).rename_axis("caseid")
    dataset = lx.Dataset.construct.from_idco(frame, alts=np.arange(1, len(counts) + 1))
    model = lx.Model(dataset, compute_engine=compute_engine)
    model.choice_co_code = "choice"
    for alt in range(1, len(counts)):
        model.utility_co[alt] = lx.P("shared" if shared else f"asc_{alt}")
    model.utility_co[len(counts)] = 0
    model.set_cap(5)
    return model


def _estimate(model, **kwargs):
    return model.maximize_loglike(
        method="slsqp", stderr=True, quiet=True, options={"ftol": 1e-8}, **kwargs
    )


def test_ordering_constraint_estimation_and_covariance_parity():
    # The observed shares prefer ASC_2 > ASC_1; the constraint binds them together.
    models = [_choice_model(engine) for engine in ("numba", "jax")]
    explicit = _choice_model("numba", shared=True)
    explicit_result = _estimate(explicit)
    expected_value = np.log(40 / (2 * 60))
    expected_covariance = np.full((2, 2), 1 / (100 * 0.4 * 0.6))
    for model in models:
        model.constraints = [OrderingBound("asc_2", "asc_1")]
        result = _estimate(model)
        assert result.success
        assert result.loglike == approx(explicit_result.loglike, abs=2e-4)
        np.testing.assert_allclose(model.pvals, expected_value, atol=2e-4)
        covariance = model.parameters["covariance_matrix"].values
        np.testing.assert_allclose(covariance, expected_covariance, rtol=2e-4)
        np.testing.assert_allclose(model.pstderr, np.sqrt(covariance.diagonal()))
        assert (model.parameters["constrained"].values != "").all()
        if model.compute_engine == "jax":
            np.testing.assert_allclose(result.stderr, model.pstderr)
            np.testing.assert_allclose(result.stderr, np.sqrt(1 / 24), rtol=2e-4)

        # Evaluate both Hessians and projections at exactly the same optimum.
        model.pvals = np.full(2, expected_value)
        model.calculate_parameter_covariance()

    np.testing.assert_allclose(
        models[0].parameters["covariance_matrix"],
        models[1].parameters["covariance_matrix"],
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        expected_covariance,
        np.full((2, 2), explicit.pstderr[0] ** 2),
        rtol=2e-5,
    )


@pytest.mark.parametrize("compute_engine", ["numba", "jax"])
def test_active_parameter_bound_and_holdfast_covariance(compute_engine):
    model = _choice_model(compute_engine, counts=(10, 30, 20, 40))
    model.lock_value(asc_1=-2)
    model.pmaximum = {"asc_2": -1}
    result = _estimate(model)
    assert result.success
    assert model.get_value("asc_1") == -2
    assert model.get_value("asc_2") == approx(-1, abs=1e-6)
    assert model.get_value("asc_3") == approx(
        np.log((0.2 / 0.8) * (1 + np.exp(-2) + np.exp(-1))), abs=2e-4
    )
    covariance = model.parameters["covariance_matrix"].values
    np.testing.assert_allclose(covariance, np.diag([0, 0, 1 / 16]), atol=1e-6)
    np.testing.assert_allclose(model.pstderr, [0, 0, 0.25], atol=1e-5)
    assert model.parameters["constrained"].sel(param_name="asc_1").item() == "fixed value"
    assert model.parameters["constrained"].sel(param_name="asc_2").item()
    if compute_engine == "jax":
        np.testing.assert_allclose(result.stderr, model.pstderr)
        np.testing.assert_allclose(result.stderr, [0, 0, 0.25], atol=1e-5)


@pytest.mark.parametrize(
    "case_slice",
    [slice(None), slice(5, 90, 2), slice(None, -1)],
    ids=["all", "slice", "exclude-last"],
)
def test_numba_objective_gradients_match_observations_and_finite_differences(case_slice):
    model = _choice_model("numba")
    # Equal initial probabilities give gradients directly from observed counts.
    choices = np.repeat([1, 2, 3], [10, 30, 60])[case_slice]
    expected_gradient = np.bincount(choices, minlength=4)[1:3] - len(choices) / 3
    slice_kwargs = (
        {}
        if case_slice == slice(None)
        else {
            "start_case": case_slice.start,
            "stop_case": case_slice.stop,
            "step_case": case_slice.step,
        }
    )
    np.testing.assert_allclose(model.neg_d_loglike(**slice_kwargs), -expected_gradient)
    np.testing.assert_allclose(model.d_logloss(**slice_kwargs), -expected_gradient / 100)

    params = np.asarray([-0.2, 0.3])
    epsilon = 1e-6
    for objective, gradient in [
        (model.neg_loglike, model.neg_d_loglike),
        (model.logloss, model.d_logloss),
    ]:
        finite_difference = []
        for displacement in np.eye(model.n_params) * epsilon:
            finite_difference.append(
                (
                    objective(params + displacement, **slice_kwargs)
                    - objective(params - displacement, **slice_kwargs)
                )
                / (2 * epsilon)
            )
        np.testing.assert_allclose(
            gradient(params, **slice_kwargs), finite_difference, rtol=1e-6, atol=1e-7
        )


@pytest.mark.parametrize("constraint_type", ["dict", "linear", "nonlinear", "sequence"])
def test_jax_preserves_caller_constraints_alongside_model_constraints(constraint_type):
    model = _choice_model("jax")
    model.constraints = [OrderingBound("asc_2", "asc_1")]
    a = model.get_param_loc("asc_1")
    direction = np.eye(model.n_params)[a]
    if constraint_type == "dict":
        constraint = {
            "type": "ineq",
            "fun": lambda params: -1.5 - params[a],
            "jac": lambda params: -direction,
        }
    elif constraint_type == "nonlinear":
        constraint = NonlinearConstraint(
            lambda params: params[a], -np.inf, -1.5, jac=lambda params: direction
        )
    else:
        constraint = LinearConstraint(direction, -np.inf, -1.5)
        if constraint_type == "sequence":
            constraint = [constraint]
    result = model.maximize_loglike(
        method="slsqp", quiet=True, constraints=constraint, options={"ftol": 1e-8}
    )
    assert result.success
    # Dropping either constraint gives a different, less restrictive optimum.
    np.testing.assert_allclose(model.pvals, [-1.5, -1.5], atol=2e-4)


def test_jax_rejects_optimizer_that_cannot_apply_model_constraints():
    model = _choice_model("jax")
    model.constraints = [OrderingBound("asc_2", "asc_1")]
    with pytest.raises(ValueError, match="constraint"):
        model.maximize_loglike(method="l-bfgs-b", quiet=True)


def test_jax_unconstrained_fit_with_readonly_double_precision_gradient():
    jax = pytest.importorskip("jax")
    with jax.experimental.enable_x64():
        model = _choice_model("jax")
        result = model.maximize_loglike(
            method="slsqp", quiet=True, options={"ftol": 1e-10}
        )
        assert result.success
        np.testing.assert_allclose(
            model.pvals, np.log(np.asarray([10, 30]) / 60), atol=2e-5
        )
        expected_loglike = np.asarray([10, 30, 60]) @ np.log([0.1, 0.3, 0.6])
        assert result.loglike == approx(expected_loglike, abs=1e-8)
        assert result.jac.dtype == np.float64
        np.testing.assert_allclose(result.jac, model.d_loglike(), atol=1e-8)


@pytest.mark.parametrize("method", ["trust-constr", "cobyla"])
def test_jax_constrained_solver_applies_parameter_bounds_and_preserves_metadata(method):
    model = _choice_model("jax")
    model.constraints = [OrderingBound("asc_2", "asc_1")]
    model.pmaximum = {"asc_1": -1.5}
    if method == "trust-constr":
        options = {
            "gtol": 1e-8,
            "barrier_tol": 1e-9,
            "initial_barrier_parameter": 1e-3,
            "maxiter": 300,
        }
    else:
        options = {"tol": 1e-5, "catol": 1e-6, "maxiter": 300}
    result = model.maximize_loglike(method=method, quiet=True, options=options)
    assert result.success
    assert model.get_value("asc_1") <= -1.5 + 1e-6
    assert model.get_value("asc_2") <= model.get_value("asc_1") + 1e-6
    np.testing.assert_allclose(model.pvals, [-1.5, -1.5], atol=2e-3)
    assert np.isfinite(result.loglike)
    if method == "trust-constr":
        assert isinstance(result.jac, list)
        assert all(np.isfinite(jacobian).all() for jacobian in result.jac)
        assert all(np.isfinite(multiplier).all() for multiplier in result.v)
        assert np.isfinite(result.lagrangian_grad).all()
        np.testing.assert_allclose(result.grad, -model.d_loglike(), atol=2e-5)
