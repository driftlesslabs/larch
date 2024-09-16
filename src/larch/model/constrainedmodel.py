from __future__ import annotations

import numpy as np

from ..util.simple_attribute import SimpleAttribute
from .basemodel import BaseModel
from .constraints import ParametricConstraintList


class ConstrainedModel(BaseModel):
    constraint_intensity = SimpleAttribute(float)
    constraint_sharpness = SimpleAttribute(float)
    constraints = ParametricConstraintList()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_intensity = 1.0
        self.constraint_sharpness = 1.0

        self._constraint_funcs = None
        """Functions for calculating constraint values."""

    def mangle(self, data=True, structure=True) -> None:
        super().mangle(data, structure)
        if structure:
            self._constraint_funcs = None

    def unmangle(self, force=False, structure_only=False):
        if not self._mangled and not force:
            return
        marker = f"_currently_unmangling_{__file__}"
        if getattr(self, marker, False):
            return
        try:
            setattr(self, marker, True)
            super().unmangle(force=force)
            if not structure_only:
                if self._constraint_funcs is None:
                    self._constraint_funcs = [
                        c.as_soft_penalty() for c in self.constraints
                    ]
        finally:
            delattr(self, marker)

    def constraint_violation(
        self,
        on_violation="raise",
        intensity_check=False,
    ):
        """
        Check if constraints are currently violated.

        Parameters
        ----------
        on_violation : {'raise', 'return'}
            If set to 'raise', an exception is raised if any model constraint,
            including any bound constraint, is violated.  Otherwise, this method
            returns a message describing the first constraint violation found, or
            an empty string if no violation are found.
        intensity_check : bool, default False
            If True, when the model's `constraint_intensity` attribute is set
            to zero, this function always returns OK (empty string).

        Returns
        -------
        str
            If no exception is raised, this method returns a message describing
            the first constraint violation found, or an empty string if no
            violation are found.
        """
        OK = ""
        if intensity_check and self.constraint_intensity == 0:
            return OK
        over_max = self.pvals > self.pmaximum
        if np.any(over_max):
            failure = np.where(over_max)[0][0]
            failure_message = (
                f"{self.pnames[failure]} over maximum "
                f"({self.pvals[failure]} > {self.pmaximum[failure]})"
            )
            if on_violation != "raise":
                return failure_message
            raise ValueError(failure_message)
        under_min = self.pvals < self.pminimum
        if np.any(under_min):
            failure = np.where(under_min)[0][0]
            failure_message = (
                f"{self.pnames[failure]} under minimum "
                f"({self.pvals[failure]} < {self.pminimum[failure]})"
            )
            if on_violation != "raise":
                return failure_message
            raise ValueError(failure_message)
        for c in self.constraints:
            if c.fun(self.pvals) < 0:
                failure_message = str(c)
                if on_violation != "raise":
                    return failure_message
                raise ValueError(failure_message)
        return OK
