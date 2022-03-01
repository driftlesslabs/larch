from .param_core import ParameterBucket
from .compiled import compiledmethod, jitmethod, reset_compiled_methods
from .optimize import OptimizeMixin
from .folding import fold_dataset
from .model import PanelMixin
import jax.numpy as jnp
import jax
from larch.numba import Dataset, DataTree
import numpy as np

class LatentClass(ParameterBucket, OptimizeMixin, PanelMixin):

    def __init__(self, classmodel, choicemodels, datatree=None, float_dtype=np.float32, **kwargs):
        PanelMixin.__init__(self, **kwargs)
        self._is_mangled = True
        self._dataset = None
        super().__init__(choicemodels, classmodel=classmodel)
        self.datatree = datatree
        assert sorted(classmodel.dataset.altids()) == sorted(choicemodels.keys())
        for choicemodel in choicemodels.values():
            self.dataset = choicemodel.dataset
        if float_dtype is not None:
            for v in self._models.values():
                v.float_dtype = float_dtype

    @jitmethod
    def jax_probability(self, params):
        classmodel = self['classmodel']
        pr_parts = []
        for n, k in enumerate(self['classmodel'].dataset.altids()):
            pr_parts.append(
                self._models[k].jax_probability(params)
                * jnp.expand_dims(classmodel.jax_probability(params)[..., n], -1)
            )
        return sum(pr_parts)

    @jitmethod
    def jax_loglike_casewise(self, params):
        n_alts = self.dataset.n_alts
        ch = jnp.asarray(self.dataset['ch'])
        if ch.ndim == 2:
            pr = self.jax_probability(params)
            masked_pr = jnp.where(
                ch[..., :n_alts]>0,
                pr[..., :n_alts],
                1.0
            )
            log_pr = jnp.log(masked_pr)
            return (log_pr[...,:n_alts] * ch[...,:n_alts]).sum()
        elif ch.ndim >= 3:
            classmodel = self['classmodel']
            likely_parts = []
            for n, k in enumerate(self['classmodel'].dataset.altids()):
                k_pr = self._models[k].jax_probability(params)
                masked_k_pr = jnp.where(
                    ch[..., :n_alts] > 0,
                    k_pr[..., :n_alts],
                    1.0
                )
                k_likely = jnp.power(masked_k_pr, ch[..., :n_alts]).prod([-2,-1])
                likely_parts.append(
                    k_likely
                    * classmodel.jax_probability(params)[..., 0, n]
                )
            return jnp.log(sum(likely_parts))

    @jitmethod
    def jax_loglike(self, params):
        return self.jax_loglike_casewise(params).sum()

    @jitmethod
    def jax_d_loglike(self, params):
        return jax.grad(self.jax_loglike)(params)

    def reflow_data_arrays(self):
        """
        Reload the internal data_arrays so they are consistent with the datatree.
        """

        datatree = self.datatree
        if datatree is None:
            raise ValueError("missing datatree")

        request = self._models['classmodel'].required_data()
        for kname, kmodel in self._models.items():
            if kname == 'classmodel':
                continue
            kreq = kmodel.required_data()
            for k, v in kreq.items():
                if k not in request:
                    request[k] = v
                else:
                    if isinstance(request[k], dict):
                        request[k].update(v)
                    elif isinstance(request[k], (list, tuple)):
                        request[k] = list(set(request[k]) | set(v))
                    else:
                        if request[k] != v:
                            raise ValueError("incompatible requests")

        if isinstance(self.groupid, str):
            request['group_co'] = self.groupid

        from larch.numba.data_arrays import prepare_data
        dataset, self.dataflows = prepare_data(
            datasource=datatree,
            request=request,
            float_dtype=np.float32,
            cache_dir=datatree.cache_dir,
            flows=getattr(self, 'dataflows', None),
        )
        if isinstance(self.groupid, str):
            dataset = fold_dataset(dataset, 'group')
        elif self.groupid is not None:
            dataset = fold_dataset(dataset, self.groupid)
        self.dataset = dataset

        for kname, kmodel in self._models.items():
            if kname == 'classmodel':
                continue
            kmodel.dataset = self.dataset
        self._models['classmodel'].dataset = self.dataset.drop_dims(
            self.dataset.ALTID
        ).set_altids(self._models['classmodel'].dataset.altids())

    def mangle(self):
        super().mangle()
        reset_compiled_methods(self)
        self._is_mangled = True

    def unmangle(self, *args, **kwargs):
        if self._is_mangled:
            super().unmangle(*args, **kwargs)
            self.reflow_data_arrays()
            self._is_mangled = False

    @property
    def dataset(self):
        """larch.Dataset : Data arrays as loaded for model computation."""
        super().unmangle()
        if self._dataset is None:
            self.reflow_data_arrays()
        try:
            return self._dataset
        except AttributeError:
            return None

    @dataset.setter
    def dataset(self, dataset):
        if dataset is self._dataset:
            return
        from xarray import Dataset as _Dataset
        if isinstance(dataset, Dataset):
            self._dataset = dataset
            self._data_arrays = None
            self.mangle()
        elif isinstance(dataset, _Dataset):
            self._dataset = Dataset(dataset)
            self._data_arrays = None
            self.mangle()
        else:
            raise TypeError(f"dataset must be Dataset not {type(dataset)}")

    @property
    def datatree(self):
        """DataTree : A source for data for the model"""
        try:
            return self._datatree
        except AttributeError:
            return None

    @datatree.setter
    def datatree(self, tree):
        if tree is self.datatree:
            return
        if isinstance(tree, DataTree) or tree is None:
            self._datatree = tree
            self.mangle()
        elif isinstance(tree, Dataset):
            self._datatree = tree.as_tree()
            self.mangle()
        else:
            try:
                self._datatree = DataTree(main=Dataset.construct(tree))
            except Exception as err:
                raise TypeError(f"datatree must be DataTree not {type(tree)}") from err
            else:
                self.mangle()

    @property
    def data_as_loaded(self):
        return self._dataset
