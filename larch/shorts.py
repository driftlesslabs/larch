from .model.linear import ParameterRef, DataRef, LinearFunction, \
    LinearComponent, DictOfLinearFunction, Ref_Gen

P = Ref_Gen(ParameterRef)
X = Ref_Gen(DataRef)

def PX(z):
    return P(z) * X(z)

__all__ = ['P', 'X', 'PX']
