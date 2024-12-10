from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from rich.table import Table
from scipy.optimize import Bounds, minimize

from ._optional import jax, jnp
from .compiled import jitmethod
from .exceptions import MissingDataError
from .util.simple_attribute import SimpleAttribute

if TYPE_CHECKING:
    from xarray import Dataset


class BucketAccess(ABC):
    @abstractmethod
    def jax_loglike(self, params):
        raise NotImplementedError

    @property
    @abstractmethod
    def pnames(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def pvals(self) -> np.ndarray:
        raise NotImplementedError

    @pvals.setter
    @abstractmethod
    def pvals(self, x):
        raise NotImplementedError

    @property
    @abstractmethod
    def pholdfast(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def pbounds(self) -> Bounds:
        raise NotImplementedError

    @property
    @abstractmethod
    def pstderr(self) -> np.ndarray:
        raise NotImplementedError

    @pstderr.setter
    @abstractmethod
    def pstderr(self, x):
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def add_parameter_array(self, name, values):
        raise NotImplementedError

    @property
    @abstractmethod
    def pretty_table(self):
        raise NotImplementedError


class OptimizeMixin(BucketAccess):
    _cached_loglike_best = None
    log_nans = False

    @abstractmethod
    def jax_loglike(self, params):
        raise NotImplementedError

    @jitmethod
    def jax_d_loglike(self, params):
        return jax.grad(self.jax_loglike)(params)

    @jitmethod
    def jax_hvp_loglike(self, params, vector):
        return jax.grad(lambda x: jnp.vdot(self.jax_d_loglike(x), vector))(params)

    @jitmethod
    def jax_d2_loglike(self, params):
        return jax.jacfwd(jax.jacrev(self.jax_neg_loglike))(params)

    @jitmethod
    def jax_invhess_loglike(self, params):
        hess = self.jax_d2_loglike(params)
        return jnp.linalg.inv(hess)

    def __rich_table(self, params, std_errs=None):
        rich_table = Table()
        rich_table.add_column("Parameter", width=30)
        if std_errs is not None:
            rich_table.add_column("Estimate", width=14)
            rich_table.add_column("Std. Error", width=14)
            rich_table.add_column("t-Stat", width=10)
            rich_table.add_column("Null Val", width=8)
            tstats = (self.pvals - self.pnullvals) / std_errs
            for i in range(params.size):
                rich_table.add_row(
                    str(self.pnames[i]),
                    f"{params[i]: .8g}",
                    " locked" if self.pholdfast[i] else f"{std_errs[i]: .8g}",
                    " locked" if self.pholdfast[i] else f"{tstats[i]: .4g}",
                    f"{self.pnullvals[i]: .2g}",
                )
        else:
            rich_table.add_column("Value", width=14)
            rich_table.add_column("Gradient", width=14)
            rich_table.add_column("Best", width=14)
            if "best" in self.parameters:
                best = self.parameters["best"].to_numpy()
            else:
                best = np.full_like(self.pvals, np.nan)
            for i in range(params.size):
                rich_table.add_row(
                    str(self.pnames[i]),
                    f"{params[i]: .8g}",
                    " locked"
                    if self.pholdfast[i]
                    else f"{self._latest_gradient[i]: .8g}",
                    f"{best[i]: .8g}",
                )
        return rich_table

    def jax_neg_loglike(self, params):
        return -self.jax_loglike(params)

    def __neg_loglike(self, params):
        result = -self.jax_loglike(params)
        self._check_if_best(-result, params)
        if np.isnan(result):
            result = np.inf
        if self.dashboard is not None:
            if self.dashboard.throttle():
                self.dashboard.update_content(
                    loglike=-result,
                    params=self.__rich_table(params),
                    bestloglike=self._cached_loglike_best,
                )
        return result

    def jax_neg_d_loglike(self, *args, **kwargs):
        return -self.jax_d_loglike(*args, **kwargs)

    def __neg_d_loglike(self, *args, **kwargs):
        result = self.jax_neg_d_loglike(*args, **kwargs)
        result = result.at[self.pholdfast != 0].set(0)  # zero out holdfast parameters
        self._latest_gradient = -result
        return result

    def _check_if_best(self, computed_ll, pvalues=None):
        if np.isnan(computed_ll):
            if self.log_nans:
                logging.error("<Detected NaN>")
                if pvalues is None:
                    logging.error(f"{self.pvals}")
                    for z1, z2 in zip(self.pnames, self.pvals):
                        logging.error(f"  {z1:20s} {z2}")
                else:
                    logging.error(f"{pvalues}")
                    for z1, z2 in zip(self.pnames, pvalues):
                        logging.error(f"  {z1:20s} {z2}")
                logging.error("</NaN>")
        elif (
            self._cached_loglike_best is None or computed_ll > self._cached_loglike_best
        ):
            self._cached_loglike_best = computed_ll
            if pvalues is None:
                self.add_parameter_array("best", self.pvals)
            else:
                self.add_parameter_array("best", pvalues)

    dashboard = SimpleAttribute()

    @property
    @abstractmethod
    def dataset(self) -> Dataset:
        raise NotImplementedError

    def jax_maximize_loglike(
        self,
        method="slsqp",
        stderr=False,
        dashboard=True,
        dashboard_update=1.0,
        quiet: bool = False,
        **kwargs,
    ):
        from .model.dashboard import Dashboard

        if self.dataset is None:
            raise MissingDataError("No dataset attached to model")

        if quiet:
            dashboard = False
        self._latest_gradient = np.full_like(self.pvals, np.nan)
        self.dashboard = Dashboard(
            status="[yellow]compiling objective function...",
            params=self.__rich_table(self.pvals),
            show=dashboard,
            throttle=dashboard_update if dashboard else 9999,
        )
        try:
            if method.lower() in {
                "nelder-mead",
                "l-bfgs-b",
                "tnc",
                "slsqp",
                "powell",
                "trust-constr",
            }:
                kwargs["bounds"] = self.pbounds
            self.__neg_loglike(self.pvals)
            self.dashboard.update_content(
                status="[yellow]compiling gradient function..."
            )
            self.__neg_d_loglike(self.pvals)
            self.dashboard.update_content(status="[yellow]optimizing parameters...")
            result = minimize(
                self.__neg_loglike,
                self.pvals,
                jac=self.__neg_d_loglike,
                method=method,
                **kwargs,
            )

            try:
                result["n_cases"] = self.n_cases
            except AttributeError:
                pass
            if hasattr(self, "total_weight"):
                result["total_weight"] = self.total_weight()
            self.pvals = result.x
            if "fun" in result:
                if hasattr(self, "total_weight"):
                    result["logloss"] = result["fun"] / self.total_weight()
                result["loglike"] = -result["fun"]
                del result["fun"]
            if "jac" in result:
                result["jac"] = -result["jac"]
            self.dashboard.update_content(
                loglike=result["loglike"],
                params=self.__rich_table(result.x),
                bestloglike=self._cached_loglike_best,
            )
            if stderr:
                self.dashboard.update_content(
                    status="[yellow]computing standard errors..."
                )
                se, hess, ihess = self.jax_param_cov(result.x)
                result["stderr"] = np.asarray(se)
                hess = np.asarray(hess).copy()
                hess[self.pholdfast.astype(bool), :] = 0
                hess[:, self.pholdfast.astype(bool)] = 0
                ihess = np.asarray(ihess).copy()
                ihess[self.pholdfast.astype(bool), :] = 0
                ihess[:, self.pholdfast.astype(bool)] = 0
                self.add_parameter_array("hess", hess)
                self.add_parameter_array("ihess", ihess)
                self.dashboard.update_content(
                    params=self.__rich_table(result.x, se),
                )
            self.dashboard.update_content(status="[green]optimization complete")
        except Exception as err:
            self.dashboard.update_content(status=f"[bright_red]ERROR [red]{err}")
            raise
        self._most_recent_estimation_result = result
        return result

    def jax_param_cov(self, params):
        hess = self.jax_d2_loglike(params)
        hess = jnp.nan_to_num(hess)
        holds = self.parameters["holdfast"] != 0
        frees = (self.parameters["holdfast"] == 0).astype(jnp.float32).values
        if holds.any():
            hess = hess * frees.reshape(-1, 1)
            hess = hess * frees.reshape(1, -1)
            ihess = jnp.linalg.pinv(hess)
            ihess = ihess * frees.reshape(-1, 1)
            ihess = ihess * frees.reshape(1, -1)
        else:
            ihess = jnp.linalg.inv(hess)
        se = jnp.sqrt(ihess.diagonal())
        self.pstderr = se
        return se, hess, ihess
