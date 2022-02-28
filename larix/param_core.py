import numpy as np
import xarray as xr

def new_name(existing_names):
    n = 1
    while f"model{n:05d}" in existing_names:
        n += 1
    return f"model{n:05d}"

def empty(dtype):
    return xr.DataArray(np.empty(0, dtype=dtype), dims='index')

class ParameterBucket():

    def __init__(self, *models, **kmodels):
        self._models = {}
        self._params = xr.Dataset({
            'value': empty(np.float32),
            'initvalue': empty(np.float32),
            'nullvalue': empty(np.float32),
            'minimum': empty(np.float32),
            'maximum': empty(np.float32),
            'holdfast': empty(np.int8),
        }, coords={'index':empty(np.object_)})
        self._fill_values = {
            'value': 0.0,
            'initvalue': 0.0,
            'nullvalue': 0.0,
            'minimum': -np.inf,
            'maximum': np.inf,
            'holdfast': 0,
        }
        for k, m in kmodels.items():
            self.attach_model(m, k, agg=False)
        for m in models:
            if isinstance(m, dict):
                for k_, m_ in m.items():
                    self.attach_model(m_, k_, agg=False)
            else:
                self.attach_model(m, agg=False)
        self._aggregate_parameters()

    def attach_model(self, model, name=None, agg=True):
        if name is None:
            name = new_name(self._models.keys())
        model.unmangle()
        self._models[name] = model
        if agg:
            self._aggregate_parameters()

    def _aggregate_parameters(self):
        should_mangle = False
        all_names = set()
        for m in self._models.values():
            all_names |= set(m.pnames)
        new_params = self._params.reindex({'index':sorted(all_names)}, fill_value=self._fill_values)
        if new_params['index'].size != self._params['index'].size or any(new_params['index'] != self._params['index']):
            should_mangle = True
        self._params = new_params
        if should_mangle:
            self.mangle()

    def mangle(self):
        for m in self._models.values():
            m.mangle()
            m._ensure_names(self.pnames)

    def unmangle(self, *args, **kwargs):
        for m in self._models.values():
            m.unmangle(*args, **kwargs)

    @property
    def parameters(self):
        return self._params

    @property
    def pvals(self):
        return self._params['value'].to_numpy()

    @pvals.setter
    def pvals(self, x):
        if isinstance(x, str) and x in self._params:
            x = self._params[x].to_numpy()
        self._params = self._params.assign({
            'value': xr.DataArray(x, dims=self._params['value'].dims)
        })

    @property
    def pnames(self):
        return self._params['index'].to_numpy()

    @property
    def pstderr(self):
        if "std_err" in self._params:
            return self._params['std_err'].to_numpy()
        else:
            return np.full_like(self.pvals, np.nan)

    @pstderr.setter
    def pstderr(self, x):
        self._params = self._params.assign({
            'std_err': xr.DataArray(x, dims=self._params['value'].dims)
        })

    def __getitem__(self, item):
        if item in self._models:
            return self._models[item]
        raise KeyError