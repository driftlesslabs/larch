from __future__ import annotations

import warnings
from collections.abc import MutableSequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..util import dictx
from .basemodel import BaseModel
from .constrainedmodel import ConstrainedModel
from .numbamodel import _arr_inflate, _safe_sqrt
from .possible_overspec import (
    PossibleOverspecification,
    compute_possible_overspecification,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from larch.util import dictx
    from larch.util.excel import ExcelWriter


class ModelGroup(ConstrainedModel, MutableSequence):
    """
    A group of models that can be treated as a single model for estimation.

    This structure collects a group of models for simultaneous estimation.
    Each component model is a separate `Model` object, but they can share
    parameters. This sharing is controlled by parameter names, in that if
    two models in the group have a parameter with the same name, they will
    be treated as a common parameter. There is no requirement that the
    linked parameters be used in the same way in each model, but since they
    are treated as a single parameter, they will be estimated together, and
    must have the same bounds and constraints.

    The model group object acts like a list of Model objects (i.e., a
    MutableSequence), and can be indexed, sliced, and iterated over.

    Parameters
    ----------
    models : list of Model or ModelGroup
        A list of models to include in the group.  Each model can be a
        `Model` object or another `ModelGroup` object.  If a `ModelGroup`
        is included, its submodels are each included in the group, and the
        `ModelGroup` itself is not included.
    title : str, optional
        A title for the model group.  This is used for display purposes
        and does not affect the estimation.
    """

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

    def estimate(self, *args, **kwargs):
        """
        Maximize loglike, and then calculate parameter covariance.

        This convenience method runs the following methods in order:
        - maximize_loglike
        - calculate_parameter_covariance

        All arguments are passed through to maximize_loglike.

        Returns
        -------
        dictx
        """
        result = self.maximize_loglike(*args, **kwargs)
        self.calculate_parameter_covariance()
        return result

    def d2_loglike(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
    ):
        """
        Compute the Hessian matrix of the log likelihood.

        Parameters
        ----------
        x : array-like, optional
            The parameter values to use in the calculation.  If not
            given, the current parameter values are used.
        start_case : int, optional
            The first case to include in the calculation.
        stop_case : int, optional
            The last case to include in the calculation.
        step_case : int, optional
            The step size between cases to include in the calculation.

        Returns
        -------
        array
        """
        if x is None:
            x = self.pvals.copy()
        from ..util.math import approx_fprime

        return approx_fprime(
            x,
            lambda y: self.d_loglike(
                y,
                start_case=start_case,
                stop_case=stop_case,
                step_case=step_case,
            ),
        )

    def calculate_parameter_covariance(self, pvals=None, *, robust=False):
        """
        Calculate the parameter covariance matrix.

        Parameters
        ----------
        pvals : array-like, optional
            The parameter values to use in the calculation.  If not
            given, the current parameter values are used.
        robust : bool, default False
            Whether to calculate the robust covariance matrix.

        Returns
        -------
        se : array
            The standard errors of the parameter estimates.
        hess : array
            The Hessian matrix.
        ihess : array
            The inverse of the Hessian matrix.
        """
        if pvals is None:
            pvals = self.pvals
        locks = np.asarray(self.pholdfast.astype(bool))
        if self.compute_engine == "jax":
            se, hess, ihess = self.jax_param_cov(pvals)
        else:
            hess = -self.d2_loglike(pvals)
            if self.parameters["holdfast"].sum():
                free = self.pholdfast == 0
                hess_ = hess[free][:, free]
                try:
                    ihess_ = np.linalg.inv(hess_)
                except np.linalg.LinAlgError:
                    ihess_ = np.linalg.pinv(hess_)
                ihess = _arr_inflate(ihess_, locks)
            else:
                try:
                    ihess = np.linalg.inv(hess)
                except np.linalg.LinAlgError:
                    ihess = np.linalg.pinv(hess)
            se = _safe_sqrt(ihess.diagonal())
            self.pstderr = se
        hess = np.asarray(hess).copy()
        hess[locks, :] = 0
        hess[:, locks] = 0
        ihess = np.asarray(ihess).copy()
        ihess[locks, :] = 0
        ihess[:, locks] = 0
        self.add_parameter_array("hess", hess)
        self.add_parameter_array("ihess", ihess)

        overspec = compute_possible_overspecification(hess, self.pholdfast)
        if overspec:
            warnings.warn(
                "Model is possibly over-specified (hessian is nearly singular).",
                category=PossibleOverspecification,
                stacklevel=2,
            )
            possible_overspecification = []
            for eigval, ox, eigenvec in overspec:
                if eigval == "LinAlgError":
                    possible_overspecification.append((eigval, [ox], [""]))
                else:
                    paramset = list(np.asarray(self.pnames)[ox])
                    possible_overspecification.append((eigval, paramset, eigenvec[ox]))
            self._possible_overspecification = possible_overspecification

        # constrained covariance
        if self.constraints:
            constraints = list(self.constraints)
        else:
            constraints = []
        try:
            constraints.extend(self._get_bounds_constraints())
        except Exception:
            pass

        if constraints:
            binding_constraints = list()
            self.add_parameter_array("unconstrained_std_err", self.pstderr)
            self.add_parameter_array("unconstrained_covariance_matrix", ihess)

            s = np.asarray(ihess)
            pvals = self.pvals
            for c in constraints:
                if np.absolute(c.fun(pvals)) < c.binding_tol:
                    binding_constraints.append(c)
                    b = c.jac(self.pf.value)
                    den = b @ s @ b
                    if den != 0:
                        s = s - (1 / den) * s @ b.reshape(-1, 1) @ b.reshape(1, -1) @ s
            self.add_parameter_array("covariance_matrix", s)
            self.pstderr = _safe_sqrt(s.diagonal())

            # Fix numerical issues on some constraints, add constrained notes
            if binding_constraints or any(self.pholdfast != 0):
                notes = {}
                for c in binding_constraints:
                    pa = c.get_parameters()
                    for p in pa:
                        # if self.pf.loc[p, 't_stat'] > 1e5:
                        #     self.pf.loc[p, 't_stat'] = np.inf
                        #     self.pf.loc[p, 'std_err'] = np.nan
                        # if self.pf.loc[p, 't_stat'] < -1e5:
                        #     self.pf.loc[p, 't_stat'] = -np.inf
                        #     self.pf.loc[p, 'std_err'] = np.nan
                        n = notes.get(p, [])
                        n.append(c.get_binding_note(pvals))
                        notes[p] = n
                constrained_note = (
                    pd.Series({k: "\n".join(v) for k, v in notes.items()}, dtype=object)
                    .reindex(self.pnames)
                    .fillna("")
                )
                constrained_note[self.pholdfast != 0] = "fixed value"
                self.add_parameter_array("constrained", constrained_note)

        if robust:
            raise NotImplementedError(
                "Robust covariance not yet implemented for ModelGroup."
            )
            # self.robust_covariance()
            # se = self.parameters["robust_std_err"]

        return se, hess, ihess

    def to_xlsx(
        self,
        filename,
        save_now=True,
        data_statistics: bool = True,
        nesting: bool = True,
        embed_model: bool = True,
    ) -> ExcelWriter:
        """
        Write the estimation results to an Excel file.

        Parameters
        ----------
        filename : str
            The name of the file to write.
        save_now : bool, default True
            Whether to save the file immediately.  If False, the
            ExcelWriter object is returned.
        data_statistics : bool, default True
            Whether to include data statistics in the Excel file.
        nesting : bool, default True
            Whether to include nesting statistics in the Excel file.
        embed_model : bool, default True
            Whether to embed the model in the Excel file.

        Returns
        -------
        larch.util.excel.ExcelWriter or None
        """
        from larch.util.excel import _make_excel_writer

        result = _make_excel_writer(self, filename, save_now=False)
        result._post_init(
            filename,
            model=self,
            data_statistics=data_statistics,
            nesting=nesting,
            embed=embed_model,
        )
        if save_now:
            result.close()
        else:
            return result
