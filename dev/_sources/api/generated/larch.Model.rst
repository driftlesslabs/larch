larch.Model
===========

.. currentmodule:: larch

.. autoclass:: Model

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~Model.__init__
      ~Model.add_parameter_array
      ~Model.analyze_elasticity
      ~Model.analyze_predictions_co
      ~Model.analyze_predictions_co_figure
      ~Model.apply_random_draws
      ~Model.availability_def
      ~Model.bhhh
      ~Model.calculate_parameter_covariance
      ~Model.check_d_loglike
      ~Model.check_for_overspecification
      ~Model.check_random_draws
      ~Model.choice_avail_summary
      ~Model.choice_def
      ~Model.clear_cache
      ~Model.constraint_converge_tolerance
      ~Model.constraint_penalty
      ~Model.constraint_violation
      ~Model.copy
      ~Model.d2_loglike
      ~Model.d_loglike
      ~Model.d_loglike_casewise
      ~Model.d_logloss
      ~Model.distribution_on_idca_variable
      ~Model.distribution_on_idco_variable
      ~Model.doctor
      ~Model.dumps
      ~Model.estimate
      ~Model.estimation_statistics
      ~Model.estimation_statistics_raw
      ~Model.fit_bhhh
      ~Model.from_dict
      ~Model.get_param_loc
      ~Model.get_value
      ~Model.histogram_on_idca_variable
      ~Model.initialize_graph
      ~Model.is_mnl
      ~Model.jax_maximize_loglike
      ~Model.jax_neg_d_loglike
      ~Model.jax_neg_loglike
      ~Model.jax_param_cov
      ~Model.jumpstart_bhhh
      ~Model.load_data
      ~Model.lock_value
      ~Model.loglike
      ~Model.loglike2
      ~Model.loglike2_bhhh
      ~Model.loglike_casewise
      ~Model.loglike_null
      ~Model.loglike_problems
      ~Model.logloss
      ~Model.logsums
      ~Model.make_random_draws
      ~Model.mangle
      ~Model.maximize_loglike
      ~Model.mixture_density
      ~Model.mixture_summary
      ~Model.neg_d_loglike
      ~Model.neg_loglike
      ~Model.parameter_summary
      ~Model.plock
      ~Model.pretty_table
      ~Model.probability
      ~Model.quantity
      ~Model.reflow_data_arrays
      ~Model.release_memory
      ~Model.remove_unused_parameters
      ~Model.required_data
      ~Model.robust_covariance
      ~Model.save
      ~Model.set_cap
      ~Model.set_value
      ~Model.set_values
      ~Model.should_preload_data
      ~Model.swap_datatree
      ~Model.to_xlsx
      ~Model.total_weight
      ~Model.unmangle
      ~Model.update_parameters
      ~Model.utility
      ~Model.utility_breakdown
      ~Model.utility_functions
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Model.autoscale_weights
      ~Model.availability_any
      ~Model.availability_ca_var
      ~Model.availability_co_vars
      ~Model.availability_var
      ~Model.choice_any
      ~Model.choice_ca_var
      ~Model.choice_co_code
      ~Model.choice_co_vars
      ~Model.common_draws
      ~Model.compute_engine
      ~Model.constraint_intensity
      ~Model.constraint_sharpness
      ~Model.constraints
      ~Model.dashboard
      ~Model.data
      ~Model.data_as_loaded
      ~Model.data_as_possible
      ~Model.dataflows
      ~Model.dataset
      ~Model.datatree
      ~Model.float_dtype
      ~Model.graph
      ~Model.groupid
      ~Model.ident
      ~Model.is_mangled
      ~Model.jax_log_probability
      ~Model.jax_loglike
      ~Model.jax_loglike_casewise
      ~Model.jax_probability
      ~Model.jax_quantity
      ~Model.jax_random_params
      ~Model.jax_utility
      ~Model.jax_utility_include_nests
      ~Model.log_nans
      ~Model.logsum_parameter
      ~Model.mixtures
      ~Model.most_recent_estimation_result
      ~Model.n_cases
      ~Model.n_draws
      ~Model.n_params
      ~Model.ordering
      ~Model.parameters
      ~Model.pbounds
      ~Model.pf
      ~Model.pholdfast
      ~Model.pinitvals
      ~Model.pmaximum
      ~Model.pminimum
      ~Model.pnames
      ~Model.pnullvals
      ~Model.possible_overspecification
      ~Model.prerolled_draws
      ~Model.pstderr
      ~Model.pvals
      ~Model.quantity_ca
      ~Model.quantity_scale
      ~Model.rename_parameters
      ~Model.seed
      ~Model.streaming
      ~Model.title
      ~Model.use_streaming
      ~Model.utility_ca
      ~Model.utility_co
      ~Model.utility_for_nests
      ~Model.weight_co_var
      ~Model.weight_normalization
      ~Model.work_arrays
   
   