.. currentmodule:: larch

=======
Dataset
=======

The :class:`larch.Dataset` class is the primary data structure in Larch. It is an
extension of :class:`xarray.Dataset`, with additional methods and attributes
specific to discrete choice modeling, collected under the `dc` accessor.


DC Constructors
---------------

.. autosummary::
    :toctree: generated/
    :template: autosummary/accessor_method.rst

    Dataset.dc.from_idca
    Dataset.dc.from_idce
    Dataset.dc.from_idco


DC Attributes
-------------

.. autosummary::
    :toctree: generated/
    :template: autosummary/accessor_attribute.rst

    Dataset.dc.n_cases
    Dataset.dc.n_alts
    Dataset.dc.CASEID
    Dataset.dc.ALTID
    Dataset.dc.alts_mapping

DC Methods
----------

.. autosummary::
    :toctree: generated/
    :template: autosummary/accessor_method.rst

    Dataset.dc.caseids
    Dataset.dc.dissolve_zero_variance
    Dataset.dc.query_cases
    Dataset.dc.set_altids
    Dataset.dc.set_altnames
    Dataset.dc.set_dtypes
    Dataset.dc.setup_flow
    Dataset.dc.get_expr
