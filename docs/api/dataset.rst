.. currentmodule:: larch

=======
Dataset
=======

Constructors
------------

.. autosummary::
    :toctree: generated/

    Dataset
    Dataset.construct.from_idca
    Dataset.construct.from_idce
    Dataset.construct.from_idco
    Dataset.construct
    Dataset.from_table
    Dataset.from_omx
    Dataset.from_omx_3d
    Dataset.from_zarr
    Dataset.from_named_objects


Attributes
----------

.. autosummary::
    :toctree: generated/

    Dataset.dc.n_cases
    Dataset.dc.n_alts
    Dataset.dc.CASEID
    Dataset.dc.ALTID
    Dataset.dc.alts_mapping
    Dataset.dims
    Dataset.sizes
    Dataset.data_vars
    Dataset.coords
    Dataset.attrs
    Dataset.encoding
    Dataset.indexes
    Dataset.chunks
    Dataset.chunksizes
    Dataset.nbytes

Methods
-------

.. autosummary::
    :toctree: generated/

    Dataset.caseids
    Dataset.dissolve_zero_variance
    Dataset.query_cases
    Dataset.set_altids
    Dataset.set_altnames
    Dataset.set_dtypes
    Dataset.setup_flow
    Dataset.get_expr
