from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..util import dictx
from .basemodel import BaseModel
from .constrainedmodel import ConstrainedModel

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class ModelGroup(ConstrainedModel, MutableSequence):
    def __init__(self, models, title=None):
        self._submodels = list()
        super().__init__(title=title)
        for model in models:
            if isinstance(model, ModelGroup):
                self._submodels.extend(model._submodels)
            else:
                self._submodels.append(model)
            self._parameter_bucket.attach_model(model)

    def __getitem__(self, x):
        return self._submodels[x]

    def __setitem__(self, i, value):
        assert isinstance(value, BaseModel)
        self._submodels[i] = value
        self._parameter_bucket.attach_model(value)

    def __delitem__(self, x):
        del self._submodels[x]

    def __len__(self):
        return len(self._submodels)

    def __contains__(self, x):
        return x in self._submodels

    def insert(self, i, value):
        assert isinstance(value, BaseModel)
        self._submodels.insert(i, value)
        self._parameter_bucket.attach_model(value)

    def total_weight(self) -> float:
        """
        Compute the total weight across all models in the group.

        Returns
        -------
        float
        """
        return sum([model.total_weight() for model in self._submodels])

    @property
    def n_cases(self):
        """Int : Total number of cases in all submodels."""
        return sum([model.n_cases for model in self._submodels])

    def logloss(
        self,
        x: ArrayLike | dict | None = None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        check_if_best=True,
    ) -> float:
        result = self.loglike(x, check_if_best=check_if_best)
        return -result / self.total_weight()

    def d_logloss(self, x=None, start_case=0, stop_case=-1, step_case=1, **kwargs):
        result = self.d_loglike(x, **kwargs)
        return -np.asarray(result) / self.total_weight()

    def loglike(
        self, x: ArrayLike | dict | None = None, *, check_if_best=True, **kwargs
    ) -> float:
        """
        Compute the log likelihood of the group of models.

        Note that unlike for a single model, the log likelihood cannot
        be computed for a slice of cases, but must be computed for all
        cases.

        Parameters
        ----------
        x : array-like or dict, optional
            New values to set for the parameters before evaluating
            the log likelihood.  If given as array-like, the array must
            be a vector with length equal to the length of the
            parameter frame, and the given vector will replace
            the current values.  If given as a dictionary,
            the dictionary is used to update the parameters.
        check_if_best : bool, default True
            If True, check if the current log likelihood is the best
            found so far, and if so, update the cached best log likelihood
            and cached best parameters.

        Returns
        -------
        float
        """
        result = sum(
            [
                model.loglike(x=x, check_if_best=False, **kwargs)
                for model in self._submodels
            ]
        )
        if check_if_best:
            self._check_if_best(result)
        return result

    def d_loglike(
        self,
        x=None,
        *,
        return_series=False,
        **kwargs,
    ):
        """
        Compute the gradient of the log likelihood of the group of models.

        Note that unlike for a single model, the gradient of the log
        likelihood cannot be computed for a slice of cases, but must be
        computed for all cases.

        Parameters
        ----------
        x : array-like or dict, optional
            New values to set for the parameters before evaluating
            the gradient of the log likelihood.  If given as array-like,
            the array must be a vector with length equal to the length of
            the parameter frame, and the given vector will replace the
            current values.  If given as a dictionary, the dictionary is
            used to update the parameters.

        Returns
        -------
        array-like
        """
        return sum(
            [
                model.d_loglike(x=x, return_series=return_series, **kwargs)
                for model in self._submodels
            ]
        )

    def loglike_casewise(
        self,
        x=None,
        **kwargs,
    ):
        """
        Compute the log likelihood case-by-case for the group of models.

        Parameters
        ----------
        x : array-like or dict, optional
            New values to set for the parameters before evaluating
            the log likelihood.  If given as array-like, the array must
            be a vector with length equal to the length of the
            parameter frame, and the given vector will replace
            the current values.  If given as a dictionary,
            the dictionary is used to update the parameters.

        Returns
        -------
        array-like
        """
        return np.concatenate(
            [model.loglike_casewise(x=x, **kwargs) for model in self._submodels]
        )

    def loglike_null(self, use_cache=True):
        """
        Compute the log likelihood at null values.

        Set all parameter values to the value indicated in the
        "nullvalue" column of the parameter frame, and compute
        the log likelihood with the currently loaded data.  Note
        that the null value for each parameter may not be zero
        (for example, the default null value for logsum parameters
        in a nested logit model is 1).

        Parameters
        ----------
        use_cache : bool, default True
            Use the cached value if available.  Set to -1 to
            raise an exception if there is no cached value.

        Returns
        -------
        float
        """
        return sum([model.loglike_null(use_cache) for model in self._submodels])

    def bhhh(
        self,
        x=None,
        *,
        return_dataframe=False,
        **kwargs,
    ):
        result = sum(
            [
                model.bhhh(x=x, return_dataframe=False, **kwargs)
                for model in self._submodels
            ]
        )
        if return_dataframe:
            result = pd.DataFrame(result, columns=self.pnames, index=self.pnames)
        return result

    def loglike2_bhhh(
        self,
        x=None,
        *,
        return_series=False,
        **kwargs,  # swallow any extra arguments
    ):
        result = dictx()
        for model in self._submodels:
            t = model.loglike2_bhhh(
                x=x,
                return_series=return_series,
                start_case=None,
                stop_case=None,
                step_case=None,
                persist=0,
            )
            for k, v in t.items():
                if k in result:
                    result[k] += v
                else:
                    result[k] = v
        return result

    def fit_bhhh(self, *args, **kwargs):
        from .numba_optimization import fit_bhhh

        return fit_bhhh(self, *args, **kwargs)

    def mangle(self, data=True, structure=True):
        for model in self._submodels:
            model.mangle(data, structure)

    def unmangle(self, force=False, structure_only=False):
        for model in self._submodels:
            model.unmangle(force, structure_only)

    def reflow_data_arrays(self) -> None:
        """Reload the internal data_arrays so they are consistent with the datatree."""
        for model in self._submodels:
            model.reflow_data_arrays()

    def doctor(self, **kwargs):
        diagnosis = []
        check_overspec = kwargs.pop("check_overspec", None)
        if check_overspec and len(kwargs) == 0:
            # when only check_overspec is passed, set verbose to 3 to signal
            # that no other checks are requested
            kwargs = {"verbose": 3}
        for model in self._submodels:
            diagnosis.append(model.doctor(**kwargs))
        from .troubleshooting import overspecification

        if check_overspec:
            self, overspec_diagnosis = overspecification(self, check_overspec)
            if overspec_diagnosis:
                diagnosis.append(overspec_diagnosis)
        return self, diagnosis

    def maximize_loglike(
        self,
        *args,
        **kwargs,
    ):
        """
        Maximize the log likelihood.

        Parameters
        ----------
        method : str, optional
            The optimization method to use.  See scipy.optimize for
            most possibilities, or use 'BHHH'. Defaults to SLSQP if
            there are any constraints or finite parameter bounds,
            otherwise defaults to BHHH.
        quiet : bool, default False
            Whether to suppress the dashboard.
        options : dict, optional
            These options are passed through to the `scipy.optimize.minimize`
            function.
        maxiter : int, optional
            Maximum number of iterations.  This argument is just added to
            `options` for most methods.

        Returns
        -------
        dictx
            A dictionary of results, including final log likelihood,
            elapsed time, and other statistics.  The exact items
            included in output will vary by estimation method.

        """
        from .optimization import maximize_loglike

        return maximize_loglike(self, *args, **kwargs)
