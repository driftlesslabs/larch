import pandas as pd

from ..dataset import DataTree
from ..shorts import P, X
from .numbamodel import NumbaModel


def from_spec(cls, spec, datatree, title=None):
    """

    Parameters
    ----------
    cls : NumbaModel
    spec : pd.DataFrame
    datatree : DataTree
    title : str, optional

    Returns
    -------

    """
    m = cls(datatree=datatree)
    if title:
        m.title = title
    if len(spec.columns) == 1:
        # idca ActivitySim spec, all alternatives have same utility function
        raise NotImplementedError
    else:
        # idco ActivitySim spec, each alternative has its own utility function
        datatree.set_altnames(spec.columns)
        alts_name_to_id = datatree.alts_name_to_id()
        for row in spec.itertuples():
            expr = row[0]
            for i, v in zip(row._fields[1:], row[1:]):
                if not pd.isna(v):
                    m.utility_co[alts_name_to_id[i]] += P(v) * X(expr)
    m.should_preload_data(False)
    return m
