from __future__ import annotations

import numpy


class PossibleOverspecification(Warning):
    pass


def compute_possible_overspecification(a, holdfast_vector=None):
    """
    Compute the possible overspecification of a model from its hessian matrix.

    Parameters
    ----------
    a : matrix
        The hessian matrix
    holdfast_vector : vector
        A vector of indicators for which row/cols should be ignored as holdfast-ed

    Returns
    -------
    list of tuples
        A list of possible overspecification problems in the model.  Each problem
        is a tuple containing the eigenvalue, the indices of the non-zero elements
        in the eigenvector, and the eigenvector itself.
    """
    ret = []
    if holdfast_vector is None:
        holdfast_vector = numpy.zeros(a.shape[0], dtype=bool)
    else:
        holdfast_vector = holdfast_vector.astype(bool, copy=True)
    holdfast_vector |= (a == 0).all(0)
    a_packed = a[~holdfast_vector, :][:, ~holdfast_vector]
    try:
        eigenvalues_packed, eigenvectors_packed = numpy.linalg.eigh(a_packed)
    except numpy.linalg.linalg.LinAlgError as err:
        return [("LinAlgError", str(err), "")]
    for i in range(len(eigenvalues_packed)):
        if numpy.abs(eigenvalues_packed[i]) < 0.001:
            v = eigenvectors_packed[:, i]
            v = numpy.round(v, 7)
            v_unpacked = numpy.zeros(a.shape[0])
            v_unpacked[~holdfast_vector.astype(bool)] = v
            ret.append((eigenvalues_packed[i], numpy.where(v_unpacked)[0], v_unpacked))
    return ret
