import numpy as np
import pandas as pd
import xarray as xr
import logging

from .linear import DictOfLinearFunction, LinearFunction, DictOfAlts
from .tree import NestingTree
from .constraints import ParametricConstraintList
from .single_parameter import SingleParameter
from ..roles import ParameterRef
from .param_core import ParameterBucket
from .mixtures import MixtureList

logger = logging.getLogger("larix.model")

class BaseModel:

    _parameter_bucket = ParameterBucket()

    utility_co = DictOfLinearFunction()
    """DictOfLinearFunction : The portion of the utility function computed from |idco| data.

    The keys of this mapping are alternative codes for the applicable elemental
    alteratives, and the values are linear functions to compute for the indicated
    alternative.  Each alternative that has any idco utility components must have
    a unique linear function given.

    Examples
    --------

    >>> from larch import Model, P, X
    >>> m = Model()
    >>> m.utility_co = {
    ... 	1: P.ParamA1 * X.DataA,
    ... 	2: P.ParamA2 * X.DataA + P.ParamASC2,
    ... }
    >>> print(m.utility_co)
    DictOfLinearFunction({1: P.ParamA1 * X.DataA, 2: P.ParamA2 * X.DataA + P.ParamASC2})
    """

    utility_ca = LinearFunction()
    """LinearFunction : The portion of the utility function computed from |idca| data.

    Data expressions in this utility function can actually reference both |idca|
    and |idco| format variables. Except in unusual model designs, every complete data
    expression should have at least one |idca| component.

    Examples
    --------

    >>> from larch import Model, P, X
    >>> m = Model()
    >>> m.utility_ca = P.Param1 * X.Data1 + P.Param2 * X.Data2
    >>> print(m.utility_ca)
    P.Param1 * X.Data1 + P.Param2 * X.Data2
    >>> m.utility_ca += P.Param3 * X.Data3 / X.DataCO4
    >>> print(m.utility_ca)
    P.Param1 * X.Data1 + P.Param2 * X.Data2 + P.Param3 * X('Data3/DataCO4')
    """

    quantity_ca = LinearFunction()
    """LinearFunction : The portion of the quantity function computed from |idca| data.

    Data expressions in this utility function can actually reference both |idca|
    and |idco| format variables. Except in unusual model designs, every complete data
    expression should have at least one |idca| component.

    Note that for the quantity function, the actual computed linear function
    uses the exponential of the parameter value(s), not the raw values. Thus,
    if the quantity function is given as `P.Param1 * X.Data1 + P.Param2 * X.Data2`,
    the computed values will actually be `exp(P.Param1) * X.Data1 + exp(P.Param2) * X.Data2`.
    This transformation ensures that the outcome from the quantity function is
    always positive, so long as at all of the data terms in the function are
    positive.  The `LinearFunction` class itself is not intrinsically aware
    of this implementation detail, but the `Model.utility_functions()` method is,
    and will render the complete utility function in a mathematically correct form.

    Examples
    --------

    >>> from larch import Model, P, X
    >>> m = Model()
    >>> m.quantity_ca = P.Param1 * X.Data1 + P.Param2 * X.Data2
    >>> print(m.quantity_ca)
    P.Param1 * X.Data1 + P.Param2 * X.Data2
    >>> m.quantity_ca += P.Param3 * X.Data3 / X.DataCO4
    >>> print(m.quantity_ca)
    P.Param1 * X.Data1 + P.Param2 * X.Data2 + P.Param3 * X('Data3/DataCO4')
    """

    _graph = NestingTree()

    constraints = ParametricConstraintList()

    _availability_co_vars = DictOfAlts()
    _choice_co_vars = DictOfAlts()

    mixtures = MixtureList()

    def __init__(self, *, title=None, datatree=None, compute_engine=None):
        self._mangled = True
        self._datatree = None
        self.title = title
        self.rename_parameters = {}
        self._parameter_bucket = ParameterBucket()
        self.datatree = datatree
        self._cached_loglike_best = None

        self._choice_ca_var = None
        self._choice_co_code = None
        self._choice_co_vars = None
        self._choice_any = None

        self._weight_co_var = None

        self._availability_var = None
        self._availability_co_vars = None
        self._availability_any = True

        self._compute_engine = compute_engine

    @property
    def compute_engine(self):
        return self._compute_engine

    @compute_engine.setter
    def compute_engine(self, engine):
        if engine not in {'numba', 'jax', None}:
            raise ValueError('invalid compute engine')
        self._compute_engine = engine

    @property
    def datatree(self):
        """DataTree : A source for data for the model"""
        return self._datatree

    @datatree.setter
    def datatree(self, tree):
        from ..dataset import DataTree
        if tree is self.datatree:
            return
        if isinstance(tree, DataTree) or tree is None:
            self._datatree = tree
            self.mangle()
        elif isinstance(tree, xr.Dataset):
            self._datatree = tree.dc.as_tree()
            self.mangle()
        else:
            try:
                self._datatree = DataTree(main=xr.Dataset.construct(tree))
            except Exception as err:
                raise TypeError(f"datatree must be DataTree not {type(tree)}") from err
            else:
                self.mangle()

    @property
    def parameters(self):
        self.unmangle()
        return self._parameter_bucket.parameters

    def update_parameters(self, x):
        self.unmangle()
        return self._parameter_bucket.update_parameters(x)

    def add_parameter_array(self, name, values):
        return self._parameter_bucket.add_array(name, values)

    @property
    def n_params(self):
        self.unmangle()
        return self._parameter_bucket.n_params

    @property
    def pvals(self):
        self.unmangle()
        return self._parameter_bucket.pvals

    @pvals.setter
    def pvals(self, x):
        self.unmangle()
        self._parameter_bucket.pvals = x

    @property
    def pnames(self):
        self.unmangle()
        return self._parameter_bucket.pnames

    @property
    def pholdfast(self):
        self.unmangle()
        return self._parameter_bucket.pholdfast

    @pholdfast.setter
    def pholdfast(self, x):
        self.unmangle()
        self._parameter_bucket.pholdfast = x

    @property
    def pnullvals(self):
        self.unmangle()
        return self._parameter_bucket.pnullvals

    @property
    def pmaximum(self):
        self.unmangle()
        return self._parameter_bucket.pmaximum

    @pmaximum.setter
    def pmaximum(self, x):
        self.unmangle()
        self._parameter_bucket.pmaximum = x

    @property
    def pminimum(self):
        self.unmangle()
        return self._parameter_bucket.pminimum

    @pminimum.setter
    def pminimum(self, x):
        self.unmangle()
        self._parameter_bucket.pminimum = x

    @property
    def pbounds(self):
        """scipy.optimize.Bounds : A copy of the current min-max bounds of the parameters."""
        self.unmangle()
        from scipy.optimize import Bounds
        return Bounds(
            self._parameter_bucket.pminimum,
            self._parameter_bucket.pmaximum,
        )

    def set_cap(self, cap=25):
        """
        Set limiting values for one or more parameters.

        Parameters
        ----------
        cap : numeric, default 25.0
            Set a global limit on parameters.  The maximum has a ceiling
            at this value, and the minimum a floor at the negative of this, unless
            the existing bounds are entirely outside this range.
        """
        self.unmangle()
        return self._parameter_bucket.set_cap(cap=cap)

    def plock(self, values=None, **kwargs):
        self.unmangle()
        self._parameter_bucket.lock(values, **kwargs)

    def get_param_loc(self, name):
        self.unmangle()
        return self._parameter_bucket.get_param_loc(name)

    @property
    def pf(self):
        self.unmangle()
        cols = ['value', 'best', 'initvalue', 'minimum', 'maximum', 'nullvalue']
        cols = [i for i in cols if i in self._parameter_bucket._params]
        return self._parameter_bucket._params[cols].to_dataframe()

    def pretty_table(self):
        self.unmangle()
        return self._parameter_bucket.pretty_table()

    def __p_rename(self, x):
        return str(self.rename_parameters.get(str(x), str(x)))

    def __repr__(self):
        s = "<larch."
        s += self.__class__.__name__
        if self.is_mnl():
            s += " (MNL)"
        else:
            s += " (GEV)"
        if self.title != "Untitled":
            s += f' "{self.title}"'
        s += ">"
        return s


    def initialize_graph(self, alternative_codes=None, alternative_names=None, root_id=0):
        """
        Write a nesting tree graph for a MNL model.

        Parameters
        ----------
        alternative_codes : array-like, optional
            Explicitly give alternative codes. Ignored if `dataframes` is given
            or if the model has dataframes or a dataservice already set.
        alternative_names : array-like, optional
            Explicitly give alternative names. Ignored if `dataframes` is given
            or if the model has dataframes or a dataservice already set.
        root_id : int, default 0
            The id code of the root node.

        Raises
        ------
        ValueError
            The model is unable to infer the alternative codes to use.  This can
            be avoided by giving alternative codes explicitly or having previously
            set dataframes or a dataservice that will give the alternative codes.
        """
        if self.datatree is not None:

            def get_coords_array(*names):
                for name in names:
                    if name in self.datatree.root_dataset.coords:
                        return self.datatree.root_dataset.coords[name].values

            if alternative_codes is None:
                alternative_codes = get_coords_array(
                    self.datatree.ALTID,
                    '_altid_', 'altid', 'alt_id', 'alt_ids',
                    'alternative_id', 'alternative_ids',
                )
            if alternative_names is None:
                alternative_names = get_coords_array(
                    'altname', 'altnames', 'alt_name', 'alt_names',
                    'alternative_name', 'alternative_names',
                )

        if alternative_codes is None:
            return

        from .tree import NestingTree
        g = NestingTree(root_id=root_id)
        if alternative_names is None:
            for a in alternative_codes:
                g.add_node(a)
        else:
            for a, name in zip(alternative_codes, alternative_names):
                g.add_node(a, name=name)
        self.graph = g

    @property
    def graph(self):
        if self._graph is None:
            try:
                self.initialize_graph()
            except ValueError:
                import warnings
                warnings.warn('cannot initialize graph, must define alternatives somehow')
                raise RuntimeError('cannot initialize graph, must define alternatives somehow')
        return self._graph

    @graph.setter
    def graph(self, x):
        self._graph = x


    def utility_functions(self, subset=None, resolve_parameters=False):
        """
        Generate an XHTML output of the utility function(s).

        Parameters
        ----------
        subset : Collection, optional
            A collection of alternative codes to include. This only has effect if
            there are separate utility_co functions set by alternative. It is
            recommended to use this parameter if there are a very large number of
            alternatives, and the utility functions of most (or all) of them
            can be effectively communicated by showing only a few.
        resolve_parameters : bool, default False
            Whether to resolve the parameters to the current (estimated) value
            in the output.

        Returns
        -------
        xmle.Elem
        """
        self.unmangle()
        from xmle import Elem
        x = Elem('div')
        t = x.elem('table', style="margin-top:1px;", attrib={'class': 'floatinghead'})
        if len(self.utility_co):
            # t.elem('caption', text=f"Utility Functions",
            # 	   style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;"
            # 			 "font-style:normal;font-size:100%;padding:0px;color:black;")

            # iterate over all alternatives if a dataframes is attached and lists the alternatives
            try:
                if self.dataservice is not None:
                    alts = self.dataservice.alternative_codes()
                elif self.dataframes is not None:
                    alts = self.dataframes.alternative_codes()
                else:
                    alts = self.utility_co.keys()
            except:
                alts = self.utility_co.keys()
            t_head = t.elem('thead')
            tr = t_head.elem('tr')
            tr.elem('th', text="alt")
            tr.elem('th', text='formula', attrib={'style': 'text-align:left;'})
            t_body = t.elem('tbody')
            for j in alts:
                if subset is None or j in subset:
                    tr = t_body.elem('tr')
                    tr.elem('td', text=str(j))
                    utilitycell = tr.elem('td', attrib={'style': 'text-align:left;'})
                    utilitycell.elem('div')
                    anything = False
                    if len(self.utility_ca):
                        utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
                        utilitycell << list(self.utility_ca.__xml__(linebreaks=True, resolve_parameters=self,
                                                                    value_in_tooltips=not resolve_parameters))
                        anything = True
                    if j in self.utility_co:
                        v = self.utility_co[j]
                        if len(v):
                            if anything:
                                utilitycell << Elem('br')
                            utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
                            utilitycell << list(v.__xml__(linebreaks=True, resolve_parameters=self,
                                                          value_in_tooltips=not resolve_parameters))
                            anything = True
                    if len(self.quantity_ca):
                        if anything:
                            utilitycell << Elem('br')
                        if self.quantity_scale:
                            utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
                            from .linear import ParameterRef_C
                            utilitycell << list(ParameterRef_C(self.quantity_scale).__xml__(resolve_parameters=self,
                                                                                            value_in_tooltips=not resolve_parameters))
                            utilitycell[-1].tail = (utilitycell[-1].tail or "") + " * log("
                        else:
                            utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + log("
                        content = self.quantity_ca.__xml__(linebreaks=True, lineprefix="  ",
                                                           exponentiate_parameters=True, resolve_parameters=self,
                                                           value_in_tooltips=not resolve_parameters)
                        utilitycell << list(content)
                        utilitycell.elem('br', tail=")")
        else:
            # there is no differentiation by alternatives, just give one formula
            # t.elem('caption', text=f"Utility Function",
            # 	   style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;"
            # 			 "font-style:normal;font-size:100%;padding:0px;color:black;")
            tr = t.elem('tr')
            utilitycell = tr.elem('td', attrib={'style': 'text-align:left;'})
            utilitycell.elem('div')
            anything = False
            if len(self.utility_ca):
                utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
                utilitycell << list(self.utility_ca.__xml__(linebreaks=True, resolve_parameters=self,
                                                            value_in_tooltips=not resolve_parameters))
                anything = True
            if len(self.quantity_ca):
                if anything:
                    utilitycell << Elem('br')
                if self.quantity_scale:
                    utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
                    from .linear import ParameterRef_C
                    utilitycell << list(ParameterRef_C(self.quantity_scale).__xml__(resolve_parameters=self,
                                                                                    value_in_tooltips=not resolve_parameters))
                    utilitycell[-1].tail = (utilitycell[-1].tail or "") + " * log("
                else:
                    utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + log("
                content = self.quantity_ca.__xml__(linebreaks=True, lineprefix="  ", exponentiate_parameters=True,
                                                   resolve_parameters=self, value_in_tooltips=not resolve_parameters)
                utilitycell << list(content)
                utilitycell.elem('br', tail=")")
        return x


    def _utility_functions_as_frame(self, subset=None, resolve_parameters=False):
        """
        Generate a tabular output of the utility function(s).

        Parameters
        ----------
        subset : Collection, optional
            A collection of alternative codes to include. This only has effect if
            there are separate utility_co functions set by alternative. It is
            recommended to use this parameter if there are a very large number of
            alternatives, and the utility functions of most (or all) of them
            can be effectively communicated by showing only a few.
        resolve_parameters : bool, default False
            Whether to resolve the parameters to the current (estimated) value
            in the output.  Not implemented.

        Returns
        -------
        xmle.Elem
        """
        self.unmangle()

        tf = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=['Alt', 'Line']),
            columns=['Formula'],
        )

        if len(self.utility_co):
            # iterate over all alternatives if a dataframes is attached and lists the alternatives
            try:
                if self.dataservice is not None:
                    alts = self.dataservice.alternative_codes()
                elif self.dataframes is not None:
                    alts = self.dataframes.alternative_codes()
                else:
                    alts = self.utility_co.keys()
            except:
                alts = self.utility_co.keys()

            for a in alts:
                if not (subset is None or a in subset):
                    continue
                line = 1
                op = ' '
                for i in self.utility_ca:
                    tf.loc[(a, line), :] = f"{op} {str(i)}"
                    op = '+'
                    line += 1
                if a in self.utility_co:
                    for i in self.utility_co[a]:
                        tf.loc[(a, line), :] = f"{op} {str(i)}"
                        op = '+'
                        line += 1
                if len(self.quantity_ca):
                    if self.quantity_scale:
                        from .linear import ParameterRef_C
                        q = ParameterRef_C(self.quantity_scale)
                        tf.loc[(a, line), :] = f"{op} {str(q)} * log("
                    else:
                        tf.loc[(a, line), :] = f"{op} log("
                    op = ' '
                    line += 1
                    for i in self.quantity_ca:
                        tf.loc[(a, line), :] = f"    {op} {i._str_exponentiate()}"
                        op = '+'
                        line += 1
                    tf.loc[(a, line), :] = f")"
                    op = '+'
                    line += 1

        else:
            # there is no differentiation by alternatives, just give one formula
            a = '*'
            line = 1
            op = ' '
            for i in self.utility_ca:
                tf.loc[(a, line), :] = f"{op} {str(i)}"
                op = '+'
                line += 1
            if len(self.quantity_ca):
                if self.quantity_scale:
                    from .linear import ParameterRef_C
                    q = ParameterRef_C(self.quantity_scale)
                    tf.loc[(a, line), :] = f"{op} {str(q)} * log("
                else:
                    tf.loc[(a, line), :] = f"{op} log("
                op = ' '
                line += 1
                for i in self.quantity_ca:
                    tf.loc[(a, line), :] = f"    {op} {i._str_exponentiate()}"
                    op = '+'
                    line += 1
                tf.loc[(a, line), :] = f")"
                op = '+'
                line += 1

        return tf


    def required_data(self):
        """
        What data is required in DataFrames for this model to be used.

        Returns
        -------
        dictx
        """
        try:
            from ..util import dictx
            req_data = dictx()

            if self.utility_ca is not None and len(self.utility_ca):
                if 'ca' not in req_data:
                    req_data.ca = set()
                for i in self.utility_ca:
                    req_data.ca.add(str(i.data))

            if self.quantity_ca is not None and len(self.quantity_ca):
                if 'ca' not in req_data:
                    req_data.ca = set()
                for i in self.quantity_ca:
                    req_data.ca.add(str(i.data))

            if self.utility_co is not None and len(self.utility_co):
                if 'co' not in req_data:
                    req_data.co = set()
                for alt, func in self.utility_co.items():
                    for i in func:
                        if str(i.data) != '1':
                            req_data.co.add(str(i.data))

            if 'ca' in req_data:
                req_data.ca = list(sorted(req_data.ca))
            if 'co' in req_data:
                req_data.co = list(sorted(req_data.co))

            if self.choice_ca_var:
                req_data.choice_ca = self.choice_ca_var
            elif self.choice_co_vars:
                req_data.choice_co = self.choice_co_vars
            elif self.choice_co_code:
                req_data.choice_co_code = self.choice_co_code
            elif self.choice_any:
                req_data.choice_any = True

            if self.weight_co_var:
                req_data.weight_co = self.weight_co_var

            if self.availability_var:
                req_data.avail_ca = self.availability_var
            elif self.availability_co_vars:
                req_data.avail_co = self.availability_co_vars
            elif self.availability_any:
                req_data.avail_any = True

            return req_data
        except:
            logger.exception("error in required_data")


    def __contains__(self, item):
        return (item in self.pf.index)  # or (item in self.rename_parameters)

    @property
    def is_mangled(self):
        return self._mangled

    def mangle(self):
        self._mangled = True

    def unmangle(self, force=False):
        if not self._mangled and not force:
            return
        marker = f"_currently_unmangling_{__file__}"
        if getattr(self, marker, False):
            return
        try:
            setattr(self, marker, True)
            if force:
                self.mangle()
            if self._mangled or force:
                self._scan_all_ensure_names()
                self._mangled = False
        finally:
            delattr(self, marker)


    def _scan_all_ensure_names(self):
        self._scan_utility_ensure_names()
        self._scan_quantity_ensure_names()
        self._scan_logsums_ensure_names()
        self._scan_mixtures_ensure_names()

    def _scan_utility_ensure_names(self):
        """
        Scan the utility functions and ensure all named parameters appear in the parameter frame.

        Any named parameters that do not appear in the parameter frame are added.
        """
        nameset = set()
        u_co_dataset = set()
        for altcode, linear_function in self.utility_co.items():
            for component in linear_function:
                nameset.add(self.__p_rename(component.param))
                try:
                    u_co_dataset.add(str(component.data))
                except:
                    import warnings
                    warnings.warn(f'bad data in altcode {altcode}')
                    raise
        linear_function_ca = self.utility_ca
        for component in linear_function_ca:
            nameset.add(self.__p_rename(component.param))
        self._ensure_names(nameset)

    def _scan_quantity_ensure_names(self):
        if self.quantity_ca is not None:
            nameset = set()
            for component in self.quantity_ca:
                nameset.add(self.__p_rename(component.param))
            self._ensure_names(nameset)

    def _scan_logsums_ensure_names(self):
        nameset = set()
        if self._graph is not None:
            for nodecode in self._graph.topological_sorted_no_elementals:
                if nodecode != self._graph._root_id:
                    param_name = str(self._graph.nodes[nodecode]['parameter'])
                    nameset.add(self.__p_rename(param_name))
        if self.quantity_ca is not None and len(self.quantity_ca) > 0:
            if self.quantity_scale is not None:
                nameset.add(self.__p_rename(self.quantity_scale))
        if self.logsum_parameter is not None:
            nameset.add(self.__p_rename(self.logsum_parameter))
        self._ensure_names(nameset, nullvalue=1, initvalue=1, minimum=0.001, maximum=1)

    def _scan_mixtures_ensure_names(self):
        for i in self.mixtures:
            for name, default_value in i.param_names().items():
                self._ensure_names([name], value=default_value, initvalue=default_value)

    def _ensure_names(self, names, **kwargs):
        if self._parameter_bucket is not None:
            self._parameter_bucket.add_parameters(names, fill_values=kwargs)

    quantity_scale = SingleParameter()
    logsum_parameter = SingleParameter()

    def _check_if_best(self, computed_ll, pvalues=None):
        if self._cached_loglike_best is None or computed_ll > self._cached_loglike_best:
            self._cached_loglike_best = computed_ll
            if pvalues is None:
                self._parameter_bucket._params = self._parameter_bucket._params.assign(
                    best=self._parameter_bucket._params['value']
                )
            else:
                self._parameter_bucket._params = self._parameter_bucket._params.assign(
                    best=xr.DataArray(pvalues, dims=self._parameter_bucket.index_name)
                )

    @property
    def choice_ca_var(self):
        """str : An |idca| variable giving the choices as indicator values."""
        return self._choice_ca_var

    @choice_ca_var.setter
    def choice_ca_var(self, x):
        if x is not None:
            x = str(x)
        if self._choice_ca_var != x:
            self.mangle()
        self._choice_ca_var = x
        if x is not None:
            self._choice_co_vars = None
            self._choice_co_code = None
            self._choice_any = False

    @property
    def choice_co_vars(self):
        """Dict[int,str] : A mapping giving |idco| expressions that evaluate to indicator values.

        Each key represents an alternative code number, and the associated expression
        gives the name of an |idco| variable or some function of |idco| variables that
        indicates whether that alternative was chosen.
        """
        if self._choice_co_vars:
            return self._choice_co_vars
        else:
            return None

    @choice_co_vars.setter
    def choice_co_vars(self, x):
        if isinstance(x, dict):
            if self._choice_co_vars != x:
                self.mangle()
            self._choice_co_vars = x
            self._choice_ca_var = None
            self._choice_co_code = None
            self._choice_any = False
        elif x is None:
            if self._choice_co_vars != x:
                self.mangle()
            self._choice_co_vars = x
        else:
            raise TypeError('choice_co_vars must be a dictionary')

    @choice_co_vars.deleter
    def choice_co_vars(self):
        self._choice_co_vars = None

    @property
    def choice_co_code(self):
        """str : An |idco| variable giving the choices as alternative id's."""
        if self._choice_co_code:
            return self._choice_co_code
        else:
            return None

    @choice_co_code.setter
    def choice_co_code(self, x):
        if isinstance(x, str):
            if self._choice_co_code != x:
                self.mangle()
            self._choice_co_code = x
            self._choice_co_vars = None
            self._choice_ca_var = None
            self._choice_any = False
        elif x is None:
            if self._choice_co_code != x:
                self.mangle()
            self._choice_co_code = x
        else:
            raise TypeError('choice_co_vars must be a str')

    @choice_co_code.deleter
    def choice_co_code(self):
        if self._choice_co_code is not None:
            self.mangle()
        self._choice_co_code = None

    @property
    def choice_any(self):
        if self._choice_any:
            return True
        else:
            return False

    @choice_any.setter
    def choice_any(self, x):
        if x:
            self._choice_any = True
            self._choice_co_code = None
            self._choice_co_vars = None
            self._choice_ca_var = None
        else:
            self._choice_any = False

    @choice_any.deleter
    def choice_any(self):
        self._choice_any = False

    @property
    def weight_co_var(self):
        return self._weight_co_var

    @weight_co_var.setter
    def weight_co_var(self, x):
        self._weight_co_var = x

    @property
    def availability_ca_var(self):
        """str : An |idca| variable or expression indicating if alternatives are available."""
        return self._availability_var

    @availability_ca_var.setter
    def availability_ca_var(self, x):
        if x is not None:
            x = str(x)
        if self._availability_var != x:
            self.mangle()
        self._availability_var = x
        self._availability_co_vars = None
        self._availability_any = False

    @property
    def availability_var(self):
        """str : An |idca| variable or expression indicating if alternatives are available.

        Deprecated, prefer `availability_ca_var` for clarity.
        """
        return self._availability_var

    @availability_var.setter
    def availability_var(self, x):
        if x is not None:
            x = str(x)
        if self._availability_var != x:
            self.mangle()
        self._availability_var = x
        self._availability_co_vars = None
        self._availability_any = False

    @property
    def availability_co_vars(self):
        """Dict[int,str] : A mapping giving |idco| expressions that evaluate to availability indicators.

        Each key represents an alternative code number, and the associated expression
        gives the name of an |idco| variable or some function of |idco| variables that
        indicates whether that alternative is available.
        """
        return self._availability_co_vars

    @availability_co_vars.setter
    def availability_co_vars(self, x):
        from typing import Mapping
        if not isinstance(x, Mapping):
            raise TypeError(f'availability_co_vars must be dict not {type(x)}')
        if self._availability_co_vars != x:
            self.mangle()
        self._availability_co_vars = x
        self._availability_var = None
        self._availability_any = False

    @property
    def availability_any(self):
        """bool : A flag indicating whether availability should be inferred from the data.

        This only applies to DataFrames-based models, as the Dataset interface does
        not include a mechanism for the data to self-describe an availability feature.
        """
        return self._availability_any

    @availability_any.setter
    def availability_any(self, x):
        self._availability_any = True
        self._availability_co_vars = None
        self._availability_var = None

