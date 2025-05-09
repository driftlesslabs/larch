===========
ModelGroup
===========

.. currentmodule:: larch


.. autosummary::
    :toctree: generated/

    ModelGroup


Attributes
==========

Parameters
----------

.. autosummary::
    :toctree: generated/

    ModelGroup.pf
    ModelGroup.pvals
    ModelGroup.pnames
    ModelGroup.pholdfast
    ModelGroup.pnullvals
    ModelGroup.pmaximum
    ModelGroup.pminimum
    ModelGroup.pbounds
    ModelGroup.pstderr
    ModelGroup.mixtures

Estimation Results
------------------

.. autosummary::
    :toctree: generated/

    ModelGroup.most_recent_estimation_result
    ModelGroup.possible_overspecification


Methods
=======

Setting Parameters
------------------

.. autosummary::
    :toctree: generated/

    ModelGroup.set_values
    ModelGroup.lock_value
    ModelGroup.set_cap
    ModelGroup.remove_unused_parameters


Parameter Estimation
--------------------

.. autosummary::
    :toctree: generated/

    ModelGroup.estimate
    ModelGroup.maximize_loglike
    ModelGroup.calculate_parameter_covariance


Model Fitness
-------------

.. autosummary::
    :toctree: generated/

    ModelGroup.loglike_null

Reporting
---------

.. autosummary::
    :toctree: generated/

    ModelGroup.parameter_summary
    ModelGroup.to_xlsx


Ancillary Computation
---------------------

.. autosummary::
    :toctree: generated/

    ModelGroup.total_weight
    ModelGroup.loglike
    ModelGroup.logloss
    ModelGroup.d_loglike
    ModelGroup.d_logloss
    ModelGroup.loglike_casewise


Troubleshooting
---------------

.. autosummary::
    :toctree: generated/

    ModelGroup.doctor
