from scipy.optimize import minimize
from .compiled import compiledmethod, jitmethod
import jax
import jax.numpy as jnp
from abc import abstractmethod, ABC
from larch.numba import Dataset
import numpy as np

class BucketAccess(ABC):

    @abstractmethod
    def jax_loglike(self, params):
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


class OptimizeMixin(BucketAccess):

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

    def jax_neg_loglike(self, *args, **kwargs):
        return -self.jax_loglike(*args, **kwargs)

    def jax_neg_d_loglike(self, *args, **kwargs):
        return -self.jax_d_loglike(*args, **kwargs)

    def jax_maximize_loglike(self, method='slsqp', **kwargs):
        result = minimize(
            self.jax_neg_loglike,
            self.pvals,
            jac=self.jax_neg_d_loglike,
            method=method,
            **kwargs
        )
        self.pvals = result.x
        if 'fun' in result:
            result['loglike'] = -result['fun']
            del result['fun']
        if 'jac' in result:
            result['jac'] = -result['jac']
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
