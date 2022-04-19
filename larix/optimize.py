from scipy.optimize import minimize, Bounds
from .compiled import compiledmethod, jitmethod
import jax
import jax.numpy as jnp
from abc import abstractmethod, ABC
from xarray import Dataset
import numpy as np
from rich.table import Table


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
                    " locked" if self.pholdfast[i] else f"{self._latest_gradient[i]: .8g}",
                    f"{best[i]: .8g}",
                )
        return rich_table

    def jax_neg_loglike(self, params):
        return -self.jax_loglike(params)

    def __neg_loglike(self, params):
        result = -self.jax_loglike(params)
        self._check_if_best(-result, params)
        if self.dashboard is not None:
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
        self._latest_gradient = -result
        return result

    def _check_if_best(self, computed_ll, pvalues=None):
        if self._cached_loglike_best is None or computed_ll > self._cached_loglike_best:
            self._cached_loglike_best = computed_ll
            if pvalues is None:
                self.add_parameter_array('best', self.pvals)
            else:
                self.add_parameter_array('best', pvalues)

    def jax_maximize_loglike(self, method='slsqp', stderr=False, **kwargs):
        from .model.dashboard import Dashboard
        self._latest_gradient = np.full_like(self.pvals, np.nan)
        self.dashboard = Dashboard(
            status="[yellow]compiling objective function...",
            params=self.__rich_table(self.pvals),
        )
        try:
            if method.lower() in {'nelder-mead', 'l-bfgs-b', 'tnc', 'slsqp', 'powell', 'trust-constr'}:
                kwargs['bounds'] = self.pbounds
            self.__neg_loglike(self.pvals)
            self.dashboard.update_content(status="[yellow]compiling gradient function...")
            self.__neg_d_loglike(self.pvals)
            self.dashboard.update_content(status="[yellow]optimizing parameters...")
            result = minimize(
                self.__neg_loglike,
                self.pvals,
                jac=self.__neg_d_loglike,
                method=method,
                **kwargs
            )
            self.pvals = result.x
            if 'fun' in result:
                result['loglike'] = -result['fun']
                del result['fun']
            if 'jac' in result:
                result['jac'] = -result['jac']
            if stderr:
                self.dashboard.update_content(status="[yellow]computing standard errors...")
                se, hess, ihess = self.jax_param_cov(result.x)
                result['stderr'] = se
                hess = np.asarray(hess).copy()
                hess[self.pholdfast.astype(bool),:] = 0
                hess[:,self.pholdfast.astype(bool)] = 0
                ihess = np.asarray(ihess).copy()
                ihess[self.pholdfast.astype(bool),:] = 0
                ihess[:,self.pholdfast.astype(bool)] = 0
                self.add_parameter_array('hess', hess)
                self.add_parameter_array('ihess', ihess)
                self.dashboard.update_content(
                    params=self.__rich_table(result.x, se),
                )
            self.dashboard.update_content(status="[green]optimization complete")
        except Exception as err:
            self.dashboard.update_content(status=f"[bright_red]ERROR [red]{err}")
            raise
        return result

    def jax_param_cov(self, params):
        hess = self.jax_d2_loglike(params)
        if self.parameters['holdfast'].sum():
            ihess = jnp.linalg.pinv(hess)
        else:
            ihess = jnp.linalg.inv(hess)
        se = jnp.sqrt(ihess.diagonal())
        self.pstderr = se
        return se, hess, ihess
