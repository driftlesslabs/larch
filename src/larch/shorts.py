from __future__ import annotations

from .model.linear import (
    DataRef,
    ParameterRef,
    Ref_Gen,
)

P = Ref_Gen(ParameterRef)
X = Ref_Gen(DataRef)


def PX(z):
    return P(z) * X(z)


__all__ = ["P", "X", "PX"]
