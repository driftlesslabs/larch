from .param_core import ParameterBucket
from .compiled import compiledmethod, jitmethod, reset_compiled_methods
from .optimize import OptimizeMixin
import jax.numpy as jnp
import jax

class LatentClass(ParameterBucket, OptimizeMixin):

    def __init__(self, classmodel, choicemodels):
        super().__init__(choicemodels, classmodel=classmodel)
        assert sorted(classmodel.dataset.altids()) == sorted(choicemodels.keys())
        for choicemodel in choicemodels.values():
            self.dataset = choicemodel.dataset


    @jitmethod
    def jax_probability(self, params):
        classmodel = self['classmodel']

        pr_parts = []
        for n, k in enumerate(self['classmodel'].dataset.altids()):
            pr_parts.append(
                self._models[k].jax_probability(params) * jnp.expand_dims(classmodel.jax_probability(params)[:, n], -1)
            )
        return sum(pr_parts)

    @jitmethod
    def jax_loglike(self, params):
        n_alts = self.dataset.n_alts
        ch = jnp.asarray(self.dataset['ch'])
        pr = self.jax_probability(params)
        masked_pr = jnp.where(
            ch[...,:n_alts]>0,
            pr[..., :n_alts],
            1.0
        )
        log_pr = jnp.log(masked_pr)
        return (log_pr[...,:n_alts] * ch[...,:n_alts]).sum()

    @jitmethod
    def jax_d_loglike(self, params):
        return jax.grad(self.jax_loglike)(params)