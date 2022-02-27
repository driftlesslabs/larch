from scipy.optimize import minimize
from .compiled import compiledmethod
import jax
import jax.numpy as jnp
from abc import abstractmethod


class OptimizeMixin:

    @abstractmethod
    def jax_loglike(self, params):
        raise NotImplementedError

    @property
    @abstractmethod
    def pvals(self):
        raise NotImplementedError

    @pvals.setter
    @abstractmethod
    def pvals(self, x):
        raise NotImplementedError

    @compiledmethod
    def jax_d_loglike(self):
        return jax.grad(self.jax_loglike)

    @compiledmethod
    def jax_hvp_loglike(self):
        @jax.jit
        def hvp(params, vector):
            return jax.grad(lambda x: jnp.vdot(self.jax_d_loglike(x), vector))(params)
        return hvp

    @compiledmethod
    def jax_d2_loglike(self):
        return jax.jacfwd(jax.jacrev(self.jax_neg_loglike))

    @compiledmethod
    def jax_d2_loglike_(self):
        return jax.jacrev(jax.jacfwd(self.jax_neg_loglike))

    @compiledmethod
    def jax_invhess_loglike(self):
        @jax.jit
        def ihess(params):
            hess = self.jax_d2_loglike(params)
            return jnp.linalg.inv(hess)
        return ihess

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
        ihess = self.jax_invhess_loglike(params)
        se = jnp.sqrt(ihess.diagonal())
        return se
