from __future__ import annotations

import os
import time
import warnings
from collections.abc import Mapping

import numpy
import pandas
import tables as _tb

from .util import Dict
from .util.text_manip import truncate_path_for_display

if os.environ.get("READTHEDOCS", None) == "True":
    _omx_base_class = object
else:
    _omx_base_class = _tb.file.File


class OMX_Error(Exception):
    pass


class OMXBadFormat(OMX_Error):
    pass


class OMXIncompatibleShape(OMXBadFormat):
    pass


class OMXNonUniqueLookup(OMX_Error):
    pass


class OMX(_omx_base_class):
    """A subclass of the :class:`tables.File` class, adding an interface for openmatrix files.

    As suggested in the openmatrix documentation, the default when creating an OMX file
    is to use zlib compression level 1, although this can be overridden.
    """

    def __repr__(self):
        from .util.text_manip import max_len

        s = f"<larch.OMX> {truncate_path_for_display(self.filename)}"
        s += f"\n |  shape:{self.shape}"
        if len(self.data._v_children):
            s += "\n |  data:"
        just = max_len(self.data._v_children.keys())
        for i in sorted(self.data._v_children.keys()):
            s += "\n |    {0:{2}s} ({1})".format(
                i, self.data._v_children[i].dtype, just
            )
        if len(self.lookup._v_children):
            s += "\n |  lookup:"
        just = max_len(self.lookup._v_children.keys())
        for i in sorted(self.lookup._v_children.keys()):
            s += "\n |    {0:{3}s} ({1} {2})".format(
                i,
                self.lookup._v_children[i].shape[0],
                self.lookup._v_children[i].dtype,
                just,
            )
        return s

    def __init__(self, *arg, complevel=1, complib="zlib", **kwarg):
        if len(arg) > 0 and isinstance(arg[0], str) and arg[0][-3:] == ".gz":
            from .util.temporaryfile import TemporaryGzipInflation

            self._temp_inflated_filename = TemporaryGzipInflation(arg[0])
            arg = (self._temp_inflated_filename,) + arg[1:]
            if "mode" in kwarg and kwarg["mode"] != "r":
                raise TypeError("cannot open .gz with other than readonly mode")
            if len(arg) > 1 and arg[1] != "r":
                raise TypeError("cannot open .gz with other than readonly mode")

        if len(arg) == 0:
            # Make a temporary file
            from .util.temporaryfile import TemporaryDirectory

            self._temporary_dir = TemporaryDirectory()
            kwarg["driver"] = "H5FD_CORE"
            kwarg["driver_core_backing_store"] = 0
            n = 1
            while os.path.exists(os.path.join(self._temporary_dir, f"TEMP{n}.omx")):
                n += 1
            arg = (os.path.join(self._temporary_dir, f"TEMP{n}.omx"), "w")

        if "filters" in kwarg:
            super().__init__(*arg, **kwarg)
        else:
            super().__init__(
                *arg, filters=_tb.Filters(complib=complib, complevel=complevel), **kwarg
            )
        try:
            self.data = self._get_or_create_path("/data", True)
        except _tb.exceptions.FileModeError:
            raise OMXBadFormat(
                "the '/data' node does not exist and cannot be created"
            ) from None
        try:
            self.lookup = self._get_or_create_path("/lookup", True)
        except _tb.exceptions.FileModeError:
            raise OMXBadFormat(
                "the '/lookup' node does not exist and cannot be created"
            ) from None
        if "OMX_VERSION" not in self.root._v_attrs:
            try:
                self.root._v_attrs.OMX_VERSION = b"0.2"
            except _tb.exceptions.FileModeError:
                raise OMXBadFormat(
                    "the root OMX_VERSION attribute does not exist and cannot be created"
                ) from None
        if "SHAPE" not in self.root._v_attrs:
            try:
                self.root._v_attrs.SHAPE = numpy.zeros(2, dtype=int)
            except _tb.exceptions.FileModeError:
                raise OMXBadFormat(
                    "the root SHAPE attribute does not exist and cannot be created"
                ) from None
        self.rlookup = Dict()
        self.rlookup._helper = self.get_reverse_lookup

    @property
    def version(self):
        """The open matrix format version for this file."""
        if "OMX_VERSION" not in self.root._v_attrs:
            raise OMXBadFormat("the root OMX_VERSION attribute does not exist")
        return self.root._v_attrs.OMX_VERSION

    def change_mode(self, mode, **kwarg):
        """
        Change the file mode of the underlying HDF5 file.

        Can be used to change from read-only to read-write.

        Parameters
        ----------
        mode : {'r','a'}
            The file mode to set. Note that the 'w' mode, which would
            open a blank HDF5 overwriting an existing file, is not allowed
            here.

        Returns
        -------
        self
        """
        try:
            if mode == self.mode:
                return
            if mode == "w":
                raise TypeError("cannot change_mode to w, close the file and delete it")
        except RecursionError:
            pass  # the file was probably closed elsewhere, reopen it
        filename = self.filename
        self.close()
        self.__init__(filename, mode, **kwarg)
        return self

    def keys(self, kind="all"):
        """
        Get the names of data and/or lookup stored in this file.

        Parameters
        ----------
        kind : {'all', 'data', 'lookup'}

        Returns
        -------
        list
        """
        if kind == "data":
            return list(self.data._v_children.keys())
        elif kind == "lookup":
            return list(self.lookup._v_children.keys())
        else:
            return list(self.data._v_children.keys()) + list(
                self.lookup._v_children.keys()
            )

    @property
    def shape(self):
        """The shape of the OMX file.

        As required by the standard, all OMX files must have a two dimensional shape. This
        attribute accesses or alters that shape. Note that attempting to change the
        shape of an existing file that already has data tables that would be incompatible
        with the new shape will raise an OMXIncompatibleShape exception.
        """
        sh = self.root._v_attrs.SHAPE[:]
        proposal = (sh[0], sh[1])
        if proposal == (0, 0) and self.data._v_nchildren > 0:
            first_child_name = next(iter(self.data._v_children.keys()))
            proposal = self.data._v_children[first_child_name].shape
            shp = numpy.empty(2, dtype=int)
            shp[0] = proposal[0]
            shp[1] = proposal[1]
            self.root._v_attrs.SHAPE = shp
        return proposal

    @shape.setter
    def shape(self, x):
        if self.data._v_nchildren > 0:
            if x[0] != self.shape[0] and x[1] != self.shape[1]:
                raise OMXIncompatibleShape(
                    f"this omx has shape {self.shape!s} but you want to set {x!s}"
                )
        if self.data._v_nchildren == 0:
            shp = numpy.empty(2, dtype=int)
            shp[0] = x[0]
            shp[1] = x[1]
            self.root._v_attrs.SHAPE = shp

    def crop(self, shape0, shape1=None, *, complevel=1, complib="zlib"):
        if isinstance(shape0, (tuple, list)) and len(shape0) == 2 and shape1 is None:
            shape0, shape1 = shape0[0], shape0[1]
        elif isinstance(shape0, (int)) and shape1 is None:
            shape1 = shape0
        start0, start1 = self.shape[0], self.shape[1]
        if start0 == shape0 and start1 == shape1:
            return
        if shape0 > start0:
            raise TypeError(
                f"crop must shrink the shape, but {shape0} > {start0} in first dimension"
            )
        if shape1 > start1:
            raise TypeError(
                f"crop must shrink the shape, but {shape1} > {start1} in first dimension"
            )
        if shape0 != shape1 and start0 == start1:
            raise TypeError(
                "do not automatically crop a square array into a non-square shape, "
                "results may be ambiguous"
            )
        for i in self.data._v_children:
            self.add_matrix(
                i,
                self.data._v_children[i][:shape0, :shape1],
                overwrite=True,
                complevel=complevel,
                complib=complib,
                ignore_shape=True,
            )
        for i in self.lookup._v_children:
            if self.lookup[i].shape[0] == start0:
                self.add_lookup(
                    i,
                    self.lookup._v_children[i][:shape0],
                    overwrite=True,
                    complevel=complevel,
                    complib=complib,
                    ignore_shape=True,
                )
            elif self.lookup[i].shape[1] == start1:
                self.add_lookup(
                    i,
                    self.lookup._v_children[i][:shape1],
                    overwrite=True,
                    complevel=complevel,
                    complib=complib,
                    ignore_shape=True,
                )
        shp = numpy.empty(2, dtype=int)
        shp[0] = shape0
        shp[1] = shape1
        self.root._v_attrs.SHAPE = shp

    def add_blank_lookup(
        self, name, atom=None, shape=None, complevel=1, complib="zlib", **kwargs
    ):
        """
        Add a blank lookup to the OMX file.

        Parameters
        ----------
        name : str
            The name of the matrix to add.
        atom : tables.Atom, optional
            The atomic data type for the new matrix. If not given, Float64 is assumed.
        shape : int, optional
            The length of the lookup to add. Must match one of the dimensions of an existing
            shape.  If there is no existing shape, the shape will be initialized as a square
            with this size.
        complevel : int, default 1
            Compression level.
        complib : str, default 'zlib'
            Compression library.

        Returns
        -------
        tables.CArray

        Note
        ----
        This method allows you to add a blank matrix of 3 or more dimemsions,
        only the first 2 must match the OMX.  Adding a matrix of more than
        two dimensions may violate compatability with other open matrix tools,
        do so at your own risk.
        """
        if name in self.lookup._v_children:
            return self.lookup._v_children[name]
        if atom is None:
            atom = _tb.Float64Atom()
        if self.shape == (0, 0) and shape is None:
            raise OMXBadFormat("must set a nonzero shape first or give a shape")
        if shape is not None:
            if shape != self.shape[0] and shape != self.shape[1]:
                if self.shape[0] == 0 and self.shape[1] == 0:
                    self.shape = (shape, shape)
                else:
                    raise OMXIncompatibleShape(
                        f"this omx has shape {self.shape!s} but you want to set {shape!s}"
                    )
        else:
            if self.shape[0] != self.shape[1]:
                raise OMXIncompatibleShape(
                    f"this omx has shape {self.shape!s} but you did not pick one"
                )
            shape = self.shape[0]
        return self.create_carray(
            self.lookup,
            name,
            atom=atom,
            shape=numpy.atleast_1d(shape),
            filters=_tb.Filters(complib=complib, complevel=complevel),
            **kwargs,
        )

    def add_blank_matrix(
        self, name, atom=None, shape=None, complevel=1, complib="zlib", **kwargs
    ):
        """
        Add a blank matrix to the OMX file.

        Parameters
        ----------
        name : str
            The name of the matrix to add.
        atom : tables.Atom, optional
            The atomic data type for the new matrix. If not given, Float64 is assumed.
        shape : tuple, optional
            The shape of the matrix to add.  The first two dimensions must match the
            shape of any existing matrices; giving a shape is mostly used on initialization.
        complevel : int, default 1
            Compression level.
        complib : str, default 'zlib'
            Compression library.

        Returns
        -------
        tables.CArray

        Note
        ----
        This method allows you to add a blank matrix of 3 or more dimemsions,
        only the first 2 must match the OMX.  Adding a matrix of more than
        two dimensions may violate compatability with other open matrix tools,
        do so at your own risk.
        """
        if name in self.data._v_children:
            return self.data._v_children[name]
        if atom is None:
            atom = _tb.Float64Atom()
        if self.shape == (0, 0):
            raise OMXBadFormat("must set a nonzero shape first")
        if shape is not None:
            if shape[:2] != self.shape:
                raise OMXIncompatibleShape(
                    f"this omx has shape {self.shape!s} but you want to set {shape[:2]!s}"
                )
        else:
            shape = self.shape
        return self.create_carray(
            self.data,
            name,
            atom=atom,
            shape=shape,
            filters=_tb.Filters(complib=complib, complevel=complevel),
            **kwargs,
        )

    def add_matrix(
        self,
        name,
        obj,
        *,
        overwrite=False,
        complevel=1,
        complib="zlib",
        ignore_shape=False,
        **kwargs,
    ):
        """
        Add a matrix to the OMX file.

        Parameters
        ----------
        name : str
            The name of the matrix to add.
        obj : array-like
            The data for the new matrix.
        overwrite : bool, default False
            Whether to overwrite an existing matrix with the same name.
        complevel : int, default 1
            Compression level.
        complib : str, default 'zlib'
            Compression library.
        ignore_shape : bool, default False
            Whether to ignore all checks on the shape of the matrix being added.
            Setting this to True allows adding a matrix with a different shape
            than any other existing matrices, which may result in an invalid OMX
            file.

        Returns
        -------
        tables.CArray
        """
        obj = numpy.asanyarray(obj)
        if not ignore_shape:
            if len(obj.shape) != 2:
                raise OMXIncompatibleShape(
                    "all omx arrays must have 2 dimensional shape"
                )
            if self.data._v_nchildren > 0:
                if obj.shape != self.shape:
                    raise OMXIncompatibleShape(
                        f"this omx has shape {self.shape!s} but you want to add {obj.shape!s}"
                    )
        if self.data._v_nchildren == 0:
            shp = numpy.empty(2, dtype=int)
            shp[0] = obj.shape[0]
            shp[1] = obj.shape[1]
            self.root._v_attrs.SHAPE = shp
        if name in self.data._v_children and not overwrite:
            raise TypeError(f"{name} exists")
        if name in self.data._v_children:
            self.remove_node(self.data, name)
        return self.create_carray(
            self.data,
            name,
            obj=obj,
            filters=_tb.Filters(complib=complib, complevel=complevel),
            **kwargs,
        )

    def add_lookup(
        self,
        name,
        obj,
        *,
        overwrite=False,
        complevel=1,
        complib="zlib",
        ignore_shape=False,
        **kwargs,
    ):
        """
        Add a lookup mapping to the OMX file.

        Parameters
        ----------
        name : str
            The name of the matrix to add.
        obj : array-like
            The data for the new matrix.
        overwrite : bool, default False
            Whether to overwrite an existing matrix with the same name.
        complevel : int, default 1
            Compression level.
        complib : str, default 'zlib'
            Compression library.
        ignore_shape : bool, default False
            Whether to ignore all checks on the shape of the lookup being added.
            Setting this to True allows adding a lookup with a different length
            than either of the two shape dimensions, which may result in an invalid
            OMX file.

        Returns
        -------
        tables.CArray
        """
        obj = numpy.asanyarray(obj)
        if not ignore_shape:
            if len(obj.shape) != 1:
                raise OMXIncompatibleShape(
                    "all omx lookups must have 1 dimensional shape"
                )
            if self.data._v_nchildren > 0:
                if obj.shape[0] not in self.shape:
                    raise OMXIncompatibleShape(
                        f"this omx has shape {self.shape!s} "
                        f"but you want to add a lookup with {obj.shape!s}"
                    )
        if self.data._v_nchildren == 0 and self.shape == (0, 0):
            raise OMXIncompatibleShape(
                "don't add lookup to omx with no data and no shape"
            )
        if name in self.lookup._v_children and not overwrite:
            raise TypeError(f"{name} exists")
        if name in self.lookup._v_children:
            self.remove_node(self.lookup, name)
        return self.create_carray(
            self.lookup,
            name,
            obj=obj,
            filters=_tb.Filters(complib=complib, complevel=complevel),
            **kwargs,
        )

    def change_all_atoms_of_type(self, oldatom, newatom):
        for name in self.data._v_children:
            if self.data._v_children[name].dtype == oldatom:
                print(f"changing matrix {name} from {oldatom} to {newatom}")
                self.change_atom_type(name, newatom, matrix=True, lookup=False)
        for name in self.lookup._v_children:
            if self.lookup._v_children[name].dtype == oldatom:
                print(f"changing lookup {name} from {oldatom} to {newatom}")
                self.change_atom_type(name, newatom, matrix=False, lookup=True)

    def change_atom_type(
        self, name, atom, matrix=True, lookup=True, require_smaller=True
    ):
        if isinstance(atom, numpy.dtype):
            atom = _tb.Atom.from_dtype(atom)
        elif not isinstance(atom, _tb.atom.Atom):
            atom = _tb.Atom.from_type(atom)
        if matrix:
            if name in self.data._v_children:
                orig = self.data._v_children[name]
                neww = self.add_blank_matrix(name + "_temp_atom", atom=atom)
                for i in range(self.shape[0]):
                    neww[i] = orig[i]
                if require_smaller and neww.size_on_disk >= orig.size_on_disk:
                    warnings.warn(
                        f"abort change_atom_type on {name}, {neww.size_on_disk} > {orig.size_on_disk}",
                        stacklevel=2,
                    )
                    neww._f_remove()
                else:
                    neww._f_rename(name, overwrite=True)
        if lookup:
            if name in self.lookup._v_children:
                orig = self.lookup._v_children[name]
                neww = self.add_blank_lookup(name + "_temp_atom", atom=atom)
                for i in range(self.shape[0]):
                    neww[i] = orig[i]
                if require_smaller and neww.size_on_disk >= orig.size_on_disk:
                    warnings.warn(
                        f"abort change_atom_type on {name}, {neww.size_on_disk} > {orig.size_on_disk}",
                        stacklevel=2,
                    )
                    neww._f_remove()
                else:
                    neww._f_rename(name, overwrite=True)

    def get_reverse_lookup(self, name):
        labels = self.lookup._v_children[name][:]
        label_to_i = dict(enumerate(labels))
        self.rlookup[name] = label_to_i
        return label_to_i

    def lookup_to_index(self, lookupname, arr):
        """
        Convert an array of lookup-able values into indexes.

        If you have an array of lookup-able values (e.g., TAZ identifiers) and you
        want to convert them to 0-based indexes for use in accessing matrix data,
        this is the function for you.

        Parameters
        ----------
        lookupname : str
            The name of the lookup in the open matrix file. This lookup must already
            exist and have a set of unique lookup values.
        arr : array-like
            An array of values that appear in the lookup. This method uses
            numpy.digitize to process values, so any target value that appears in `arr` but does
            not appear in the lookup will be assigned the index of the smallest lookup
            value that is greater than the target, or the maximum lookup value if no lookup value
            is greater than the target.

        Returns
        -------
        array
            An array of index (int) values, with the same shape as `arr`.

        Raises
        ------
        OMXNonUniqueLookup
            When the lookup does not contain a set of unique values, this tool is not appropriate.

        """
        from .util.arraytools import is_sorted_and_unique

        labels = self.lookup._v_children[lookupname][:]
        if is_sorted_and_unique(labels):
            return numpy.digitize(arr, labels, right=True)
        uniq_labels, uniq_indexes = numpy.unique(labels, return_inverse=True)
        if len(uniq_labels) != len(labels):
            raise OMXNonUniqueLookup(
                f"lookup '{lookupname}' does not have unique labels for each item"
            )
        index_malordered = numpy.digitize(arr, uniq_labels, right=True)
        return uniq_indexes[index_malordered]

    def import_datatable(
        self,
        filepath,
        one_based=True,
        chunksize=10000,
        column_map=None,
        default_atom="float32",
        log=None,
    ):
        """Import a table in r,c,x,x,x... format into the matrix.

        The r and c columns need to be either 0-based or 1-based index values
        (this may be relaxed in the future). The matrix must already be set up
        with the correct size before importing the datatable.

        Parameters
        ----------
        filepath : str or buffer
            This argument will be fed directly to the :func:`pandas.read_csv` function.
        one_based : bool
            If True (the default) it is assumed that zones are indexed sequentially
            starting with 1 (as is typical for travel demand forecasting applications).
            Otherwise, it is assumed that zones are indexed sequentially starting
            with 0 (typical for other c and python applications).
        chunksize : int
            The number of rows of the source file to read as a chunk.  Reading a
            giant file in moderate sized chunks can be much faster and less memory
            intensive than reading the entire file.
        column_map : dict or None
            If given, this dict maps columns of the input file to OMX tables, with
            the keys as the columns in the input and the values as the tables in
            the output.
        default_atom : str or dtype
            The default atomic type for imported data when the table does not already
            exist in this openmatrix.
        """
        if log is not None:
            log("START import_datatable")
        from .util.smartread import SmartFileReader

        sfr = SmartFileReader(filepath)
        reader = pandas.read_csv(sfr, chunksize=chunksize)
        chunk0 = next(pandas.read_csv(filepath, chunksize=10))
        offset = 1 if one_based else 0

        if column_map is None:
            column_map = {i: i for i in chunk0.columns}

        if isinstance(default_atom, str):
            default_atom = _tb.Atom.from_dtype(numpy.dtype(default_atom))
        elif isinstance(default_atom, numpy.dtype):
            default_atom = _tb.Atom.from_dtype(default_atom)

        for t in column_map.values():
            self.add_blank_matrix(t, atom=default_atom)

        for n, chunk in enumerate(reader):
            if log is not None:
                log(f"PROCESSING CHUNK {n} [{sfr.progress()}]")
            r = chunk.iloc[:, 0].values - offset
            c = chunk.iloc[:, 1].values - offset

            if not numpy.issubdtype(r.dtype, numpy.integer):
                r = r.astype(int)
            if not numpy.issubdtype(c.dtype, numpy.integer):
                c = c.astype(int)

            for col in chunk.columns:
                if col in column_map:
                    self.data._v_children[column_map[col]][r, c] = chunk[col].values
                    self.data._v_children[column_map[col]].flush()
            if log is not None:
                log(f"finished processing chunk {n} [{sfr.progress()}]")

            self.flush()

        if log is not None:
            log(f"import_datatable({filepath}) complete")

    def import_datatable_3d(
        self,
        filepath,
        one_based=True,
        chunksize=10000,
        default_atom="float32",
        log=None,
    ):
        """Import a table in r,c,x,x,x... format into the matrix.

        The r and c columns need to be either 0-based or 1-based index values
        (this may be relaxed in the future). The matrix must already be set up
        with the correct size before importing the datatable.

        This method is functionally the same as :meth:`import_datatable` but uses
        a different implementation. It is much more memory intensive but also much
        faster than the non-3d version.

        Parameters
        ----------
        filepath : str or buffer
            This argument will be fed directly to the :func:`pandas.read_csv` function.
        one_based : bool
            If True (the default) it is assumed that zones are indexed sequentially
            starting with 1 (as is typical for travel demand forecasting applications).
            Otherwise, it is assumed that zones are indexed sequentially starting
            with 0 (typical for other c and python applications).
        chunksize : int
            The number of rows of the source file to read as a chunk.  Reading a
            giant file in moderate sized chunks can be much faster and less memory
            intensive than reading the entire file.
        default_atom : str or dtype
            The default atomic type for imported data when the table does not already
            exist in this openmatrix.
        """
        if log is not None:
            log("START import_datatable")
        from .util.smartread import SmartFileReader

        sfr = SmartFileReader(filepath)
        reader = pandas.read_csv(sfr, chunksize=chunksize)
        chunk0 = next(pandas.read_csv(filepath, chunksize=10))
        offset = 1 if one_based else 0

        if isinstance(default_atom, str):
            default_dtype = numpy.dtype(default_atom)
            default_atom = _tb.Atom.from_dtype(numpy.dtype(default_atom))
        elif isinstance(default_atom, numpy.dtype):
            default_dtype = default_atom
            default_atom = _tb.Atom.from_dtype(default_atom)
        else:
            default_dtype = default_atom.dtype

        # self.add_blank_matrix(matrixname, atom=default_atom, shape=self.shape+(len(chunk0.columns)-2,))

        temp_slug = numpy.zeros(
            self.shape + (len(chunk0.columns) - 2,), dtype=default_dtype
        )

        for n, chunk in enumerate(reader):
            if log is not None:
                log(f"PROCESSING CHUNK {n} [{sfr.progress()}]")
            r = chunk.iloc[:, 0].values - offset
            c = chunk.iloc[:, 1].values - offset

            if not numpy.issubdtype(r.dtype, numpy.integer):
                r = r.astype(int)
            if not numpy.issubdtype(c.dtype, numpy.integer):
                c = c.astype(int)

            temp_slug[r, c] = chunk.values[:, 2:]
            if log is not None:
                log(f"finished processing chunk {n} [{sfr.progress()}]")

            self.flush()

        for cn, colname in enumerate(chunk0.columns[2:]):
            self.add_blank_matrix(colname, atom=default_atom, shape=self.shape)
            self.data._v_children[colname][:] = temp_slug[:, :, cn]

        if log is not None:
            log(f"import_datatable({filepath}) complete")

    @classmethod
    def import_dbf(cls, dbffile, omxfile, shape, o, d, cols, smallest_zone_number=1):
        try:
            from simpledbf import Dbf5
        except ImportError:
            raise
        dbf = Dbf5(dbffile, codec="utf-8")
        tempstore = {c: numpy.zeros(shape, dtype=numpy.float32) for c in cols}
        for df in dbf.to_dataframe(chunksize=shape[1]):
            oz = df[o].values.astype(int) - smallest_zone_number
            dz = df[d].values.astype(int) - smallest_zone_number
            for c in cols:
                tempstore[c][oz, dz] = df[c].values
        omx = cls(omxfile, mode="a")
        for c in cols:
            omx.add_matrix(c, tempstore[c])
        omx.flush()
        return omx

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.data._v_children:
                if key in self.lookup._v_children:
                    raise KeyError(f"key {key} found in both data and lookup")
                else:
                    return self.data._v_children[key]
            if key in self.lookup._v_children:
                return self.lookup._v_children[key]
            raise KeyError(f"matrix named {key} not found")
        raise TypeError("OMX matrix access must be by name (str)")

    def __setitem__(self, key, value):
        try:
            value_shape = value.shape
        except Exception as err:
            raise TypeError("value must array-like with one or two dimensions") from err
        if len(value_shape) == 1:
            if value_shape[0] == self.shape[0] or value_shape[0] == self.shape[1]:
                self.add_lookup(key, numpy.asarray(value))
            else:
                raise OMXIncompatibleShape(
                    f"cannot add vector[{value_shape[0]}] to OMX with shape {self.shape}"
                )
        elif len(value_shape) == 2:
            if value_shape[0] == self.shape[0] and value_shape[1] == self.shape[1]:
                self.add_matrix(key, numpy.asarray(value))
            else:
                raise OMXIncompatibleShape(
                    f"cannot add matrix[{value_shape}] to OMX with shape {self.shape}"
                )
        else:
            raise OMXIncompatibleShape(
                f"cannot add matrix[{value_shape}] which has more than 3 dimnensions to OMX"
            )

    def __getattr__(self, key):
        if key in self.data._v_children:
            if key not in self.lookup._v_children:
                return self.data._v_children[key]
            else:
                raise AttributeError(f"key {key} found in both data and lookup")
        if key in self.lookup._v_children:
            return self.lookup._v_children[key]
        raise AttributeError(f"key {key} not found")

    def get_dataframe(self, matrix, index=None, columns=None):
        """
        Get a matrix array as a pandas.DataFrame.

        This will return the entire content from a single matrix
        array as a pandas.DataFrame, using index or column labels
        from lookup vectors contained in the OMX file, or as
        provided directly.

        Parameters
        ----------
        matrix : str
            The name of the matrix to extract
        index : str or array-like, optional
            The name of the lookup to use as the index, or an array
            of length equal to the first shape dimension.
            If not given, a RangeIndex is used.
        columns : str, optional
            The name of the lookup to use as the columns, or an array
            of length equal to the second shape dimension.
            If not given, a RangeIndex is used.

        Returns
        -------
        pandas.DataFrame
        """
        if isinstance(index, str):
            index = self.lookup._v_children[index]
        if isinstance(columns, str):
            columns = self.lookup._v_children[columns]
        if matrix in self.data._v_children:
            return pandas.DataFrame(
                data=self.data._v_children[matrix],
                index=index,
                columns=columns,
            )
        elif matrix in self.lookup._v_children:
            return pandas.DataFrame(
                data=self.lookup._v_children[matrix],
                index=index,
                columns=[matrix],
            )
        else:
            raise KeyError(matrix)

    def all_matrix_at(self, r, c):
        """
        Get the value from all matrices at a coordinate.

        Parameters
        ----------
        r,c : int
            Coordinate to extract.  This is the zero-based index.

        Returns
        -------
        Dict
            The keys are the names of the matrices, and the value are
            taken from the given coordinates.
        """
        from .util import Dict

        result = Dict()
        for name, vals in self.data._v_children.items():
            result[name] = vals[r, c]
        return result

    def get_rc_dataframe(self, row_indexes, col_indexes, mat_names=None, index=None):
        """
        Build a DataFrame containing values pulled from this OMX.

        Parameters
        ----------
        row_indexes : array-like or int
            The row index within the matrix for each output row.
        col_indexes : array-like or int
            The column index within the matrix for each output row.
            Must have the same shape as `row_indexes`, unless one of
            these is just an integer, in which case that value is
            broadcast to the shape of the other.
        mat_names : Sequence or Mapping, optional
            A sequence of matrix names to draw values from.  Each
            name should be a matrix table that exists in the OMX
            file. If not given, all matrix arrays from the `data`
            node in the OMX file will be used. If given as a mapping,
            the keys are used to identify the columns to draw values
            from, and the mapping is then used to rename the columns.
        index : array-like, or 'rc', optional
            An array to use as the index on the returned DataFrame.
            Set to 'rc' to get a row-and-column MultiIndex.

        Returns
        -------
        pandas.DataFrame
        """
        if mat_names is None:
            mat_names = list(self.data._v_children.keys())
        if isinstance(mat_names, Mapping):
            _mat_names = mat_names.keys()
        elif isinstance(mat_names[0], (tuple, list)):
            _mat_names = [i[0] for i in mat_names]
        else:
            _mat_names = mat_names

        if isinstance(row_indexes, int):
            row_indexes = numpy.full_like(col_indexes, row_indexes)
        elif isinstance(col_indexes, int):
            col_indexes = numpy.full_like(row_indexes, col_indexes)

        data = {mat: self[mat][row_indexes, col_indexes] for mat in _mat_names}

        if index is None:
            try:
                index = row_indexes.index
            except AttributeError:
                pass

        if index is None:
            try:
                index = col_indexes.index
            except AttributeError:
                pass

        if index is None:
            try:
                rows_name = row_indexes.name
            except AttributeError:
                rows_name = "i"
                while rows_name in mat_names:
                    rows_name = rows_name + "_"
            data[rows_name] = row_indexes

            try:
                cols_name = col_indexes.name
            except AttributeError:
                cols_name = "j"
                while cols_name in mat_names:
                    cols_name = cols_name + "_"
            data[cols_name] = col_indexes

        result = pandas.DataFrame.from_dict(data)
        if index is None or (isinstance(index, str) and index == "rc"):
            result = result.set_index([rows_name, cols_name])
        else:
            result.index = index
        if isinstance(mat_names, Mapping):
            result = result.rename(columns=mat_names)
        return result

    def join_rc_dataframe(
        self,
        df,
        rowidx,
        colidx,
        mat_names=None,
        prefix="",
    ):
        """
        Join RC data pulled from this OMX with an existing DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The existing dataframe to join to.
        rowidx, colidx : str or array-like
            The columns to use for the rowindexes and colindexes,
            respectively. Give as a string to name columns in `df`,
            or as an eval-capable instruction, or give an array
            explicitly.
        mat_names : Sequence or Mapping, optional
            A sequence of matrix names to draw values from.  Each
            name should be a matrix table that exists in the OMX
            file. If not given, all matrix arrays from the `data`
            node in the OMX file will be used. If given as a mapping,
            the keys are used to identify the columns to draw values
            from, and the mapping is then used to rename the columns.
        prefix : str, optional
            Add this prefix to every matrix name used.

        Returns
        -------
        pandas.DataFrame
        """
        if isinstance(rowidx, str):
            _row = df.eval(rowidx)
        else:
            _row = rowidx
        if isinstance(colidx, str):
            _col = df.eval(colidx)
        else:
            _col = colidx
        data = self.get_rc_dataframe(
            _row,
            _col,
            mat_names,
            index=df.index,
        )
        if prefix:
            data = data.add_prefix(prefix)
        return df.join(data)

    def import_omx(self, otherfile, tablenames, rowslicer=None, colslicer=None):
        oth = OMX(otherfile, mode="r")
        if tablenames == "*":
            tablenames = oth.data._v_children.keys()
        for tab in tablenames:
            if rowslicer is None and colslicer is None:
                self.add_matrix(tab, oth.data._v_children[tab][:])
            else:
                self.add_matrix(tab, oth.data._v_children[tab][rowslicer, colslicer])

    @classmethod
    def change_cols(
        cls,
        newfile,
        otherfile,
        datanames,
        lookuprownames,
        lookupcolnames,
        newshape,
        newslicecols=None,
        oldslicecols=None,
    ):
        """
        Read an existing OMX into a new file, remapping columns as appropriate.

        The new matrix is filled with blanks, and then the values are set by
        slicing the arrays and assigning the old array to the sliced new.
        (Sorry if this is confusing.)

        Parameters
        ----------
        otherfile
        datanames
        lookuprownames
        lookupcolnames
        newshape
        slicecols
        """
        if oldslicecols is None and newslicecols is None:
            raise TypeError(
                "there is no reason to change cols if you don't slice something"
            )
        if oldslicecols is None:
            oldslicecols = slice(None)
        if newslicecols is None:
            newslicecols = slice(None)

        self = cls(newfile, mode="a")
        self.shape = newshape
        if isinstance(otherfile, str):
            otherfile = OMX(otherfile)
        for name in datanames:
            n = self.add_blank_matrix(
                name,
                atom=otherfile.data._v_children[name].atom,
                shape=newshape,
                complevel=1,
                complib="zlib",
            )
            n[:, newslicecols] = otherfile.data._v_children[name][:, oldslicecols]
        for name in lookuprownames:
            n = self.add_lookup(name, obj=otherfile.lookup._v_children[name][:])
        for name in lookupcolnames:
            n = self.add_blank_lookup(
                name, atom=otherfile.lookup._v_children[name].atom, shape=newshape[1]
            )
            n[newslicecols] = otherfile.lookup._v_children[name][oldslicecols]
        self.flush()
        return self

    # def info(self):
    #     from .model_reporter.art import ART
    #
    #     ## Header
    #     a = ART(
    #         columns=("TABLE", "DTYPE"),
    #         n_head_rows=1,
    #         title="OMX ({},{}) @ {}".format(
    #             self.shape[0], self.shape[1], self.filename
    #         ),
    #         short_title=None,
    #     )
    #     a.addrow_kwd_strings(TABLE="Table", DTYPE="dtype")
    #     ## Content
    #     if len(self.data._v_children):
    #         a.add_blank_row()
    #         a.set_lastrow_iloc(
    #             0, a.encode_cell_value("DATA"), {"class": "parameter_category"}
    #         )
    #     for v_name in sorted(self.data._v_children):
    #         a.addrow_kwd_strings(
    #             TABLE=v_name, DTYPE=self.data._v_children[v_name].dtype
    #         )
    #     if len(self.lookup._v_children):
    #         a.add_blank_row()
    #         a.set_lastrow_iloc(
    #             0, a.encode_cell_value("LOOKUP"), {"class": "parameter_category"}
    #         )
    #     for v_name in sorted(self.lookup._v_children):
    #         a.addrow_kwd_strings(
    #             TABLE=v_name, DTYPE=self.lookup._v_children[v_name].dtype
    #         )
    #     return a

    @classmethod
    def FromDataFrame(cls, frame, *arg, **kwarg):
        """
        Create a new OMX file from a `pandas.DataFrame`.

        Will create a new OMX file containing only lookups (no matrix data).
        The shape of the OMX will be set as square, with size equal to the
        number of rows in the frame.

        Parameters
        ----------
        frame : pandas.DataFrame
            Import this data.
        """
        self = cls(*arg, **kwarg)
        self.shape = (len(frame), len(frame))
        for columnname in frame.columns:
            self.add_lookup(columnname, frame[columnname].values)
        return self

    @classmethod
    def FromCSV(cls, filename, *arg, csv_kwarg=None, **kwarg):
        """
        Create a new OMX file from a csv file.

        This is a convenience function that wraps reading the CSV into a
        `pandas.DataFrame`, and then writing to a new OMX with `FromDataFrame`.

        Parameters
        ----------
        filename : str
            The CSV filename.
        """
        if csv_kwarg is None:
            csv_kwarg = {}
        self = cls.FromDataFrame(pandas.read_csv(filename, **csv_kwarg), *arg, **kwarg)
        return self

    @classmethod
    def FromXLSX(cls, filename, *arg, excel_kwarg=None, **kwarg):
        """
        Create a new OMX file from an excel file.

        This is a convenience function that wraps reading the XLSX into a
        `pandas.DataFrame`, and then writing to a new OMX with `FromDataFrame`.

        Parameters
        ----------
        filename : str
            The CSV filename.
        """
        if excel_kwarg is None:
            excel_kwarg = {}
        if "engine" not in excel_kwarg:
            excel_kwarg["engine"] = "openpyxl"
        self = cls.FromDataFrame(
            pandas.read_excel(filename, **excel_kwarg), *arg, **kwarg
        )
        return self

    def _remake_command(self, cmd, selector=None, receiver=None):
        from tokenize import NAME, OP, tokenize, untokenize

        DOT = (OP, ".")
        COLON = (OP, ":")
        COMMA = (OP, ",")
        OBRAC = (OP, "[")
        CBRAC = (OP, "]")
        from io import BytesIO

        recommand = []

        if receiver:
            recommand += [
                (NAME, receiver),
                OBRAC,
                COLON,
                CBRAC,
                (OP, "="),
            ]

        try:
            cmd_encode = cmd.encode("utf-8")
        except AttributeError:
            cmd_encode = str(cmd).encode("utf-8")
        g = tokenize(BytesIO(cmd_encode).readline)
        if selector is None:
            screen_tokens = [
                COLON,
            ]
        else:
            screen_tokens = [
                (NAME, "selector"),
            ]
        for toknum, tokval, _, _, _ in g:
            if toknum == NAME and tokval in self.data:
                # replace NAME tokens
                partial = [
                    (NAME, "self"),
                    DOT,
                    (NAME, "data"),
                    DOT,
                    (NAME, tokval),
                    OBRAC,
                ]
                partial += screen_tokens
                if len(self._groupnode._v_children[tokval].shape) > 1:
                    partial += [
                        COMMA,
                        COLON,
                    ]
                if len(self._groupnode._v_children[tokval].shape) > 2:
                    partial += [
                        COMMA,
                        COLON,
                    ]
                if len(self._groupnode._v_children[tokval].shape) > 3:
                    partial += [
                        COMMA,
                        COLON,
                    ]
                partial += [
                    CBRAC,
                ]
                recommand.extend(partial)
            elif toknum == NAME and tokval in self.lookup:
                # replace NAME tokens
                partial = [
                    (NAME, "self"),
                    DOT,
                    (NAME, "lookup"),
                    DOT,
                    (NAME, tokval),
                    OBRAC,
                ]
                partial += screen_tokens
                if len(self._groupnode._v_children[tokval].shape) > 1:
                    partial += [
                        COMMA,
                        COLON,
                    ]
                if len(self._groupnode._v_children[tokval].shape) > 2:
                    partial += [
                        COMMA,
                        COLON,
                    ]
                if len(self._groupnode._v_children[tokval].shape) > 3:
                    partial += [
                        COMMA,
                        COLON,
                    ]
                partial += [
                    CBRAC,
                ]
                recommand.extend(partial)
            else:
                recommand.append((toknum, tokval))
        # print("<recommand>")
        # print(recommand)
        # print("</recommand>")
        ret = untokenize(recommand).decode("utf-8")
        from .util.aster import asterize

        return asterize(ret, mode="exec" if receiver is not None else "eval"), ret

    def _evaluate_single_item(self, cmd, selector=None, receiver=None):
        j, j_plain = self._remake_command(
            cmd,
            selector=selector,
            receiver="receiver" if receiver is not None else None,
        )
        # important globals

        try:
            if receiver is not None:
                exec(j)
            else:
                return eval(j)
        except Exception as exc:
            args = exc.args
            if not args:
                arg0 = ""
            else:
                arg0 = args[0]
            arg0 = arg0 + f'\nwithin parsed command: "{cmd!s}"'
            arg0 = arg0 + f'\nwithin re-parsed command: "{j_plain!s}"'
            if selector is not None:
                arg0 = arg0 + f'\nwith selector: "{selector!s}"'
            if "max" in cmd:
                arg0 = (
                    arg0
                    + '\n(note to get the maximum of arrays use "fmax" not "max")'.format()
                )
            if "min" in cmd:
                arg0 = (
                    arg0
                    + '\n(note to get the minimum of arrays use "fmin" not "min")'.format()
                )
            if isinstance(exc, NameError):
                badname = str(exc).split("'")[1]
                goodnames = dir()
                from .util.text_manip import case_insensitive_close_matches

                did_you_mean_list = case_insensitive_close_matches(
                    badname, goodnames, n=3, cutoff=0.1, excpt=None
                )
                if len(did_you_mean_list) > 0:
                    arg0 = (
                        arg0
                        + "\n"
                        + "did you mean {}?".format(
                            " or ".join(f"'{s}'" for s in did_you_mean_list)
                        )
                    )
            exc.args = (arg0,) + args[1:]
            raise

    def to_dataset(self, zone_id=None, dim_names=("otaz", "dtaz")):
        import numpy as np
        import xarray as xr

        z = xr.Dataset()
        if zone_id is None:
            z = z.assign_coords(
                {
                    dim_names[0]: np.arange(1, self.shape[0] + 1),
                    dim_names[1]: np.arange(1, self.shape[1] + 1),
                }
            )
        else:
            z = z.assign_coords(
                {
                    dim_names[0]: self[zone_id],
                    dim_names[1]: self[zone_id],
                }
            )
        for k in self.data._v_children:
            z = z.assign({k: ((dim_names[0], dim_names[1]), self[k])})
        for k in self.lookup._v_children:
            if k == zone_id:
                continue
            k_ = self[k][:]
            if k_.size == self.shape[0]:
                z = z.assign({k: (dim_names[0], self[k])})
            elif k_.size == self.shape[1]:
                z = z.assign({k: (dim_names[1], self[k])})
            else:
                raise OMXIncompatibleShape
        return z


def _shrink_bitwidth(v, dtype_shrink=32):
    if dtype_shrink <= 32:
        if v.dtype == "float64":
            v = v.astype("float32")
        elif (
            v.dtype == "int64" and v.max() < 2_147_483_648 and v.min() > -2_147_483_648
        ):
            v = v.astype("int32")
    return v


def convert_omx(
    existing_filename,
    new_filename,
    complevel=3,
    complib="blosc2:zstd",
    dtype_shrink=32,
    part=0,
    n_part=1,
) -> float:
    """
    Convert an existing OMX file using different data types and filters.

    This function will read an existing OMX file and write it back out
    in a new format.  This is useful for updating old OMX files to
    the new format, which may be more efficient.

    Parameters
    ----------
    existing_filename : str
        The filename of the existing OMX file to convert.
    new_filename : str
        The filename of the new OMX file to write.
    complevel : int, default 1
        Compression level.
    complib : str, default 'blosc2:zstd'
        Compression library.
    dtype_shrink : int, default 32
        The maximum bitwidth to use for integer and float types.
    part : int, default 0
        The part of the file to start with.
    n_part : int, default 1
        The number of partitions to create.

    Returns
    -------
    float
        The time in seconds it took to convert the file.
    """
    start = time.time()
    with OMX(existing_filename, mode="r") as old_omx:
        if part != 0 or n_part != 1:
            newbase, nextext = os.path.splitext(new_filename)
            new_filename = newbase + f".part{part:03d}" + nextext
        with OMX(new_filename, mode="w") as new_omx:
            keys = sorted(old_omx.data._v_children)
            for name in keys[part::n_part]:
                v = old_omx.data._v_children[name][:]
                v = _shrink_bitwidth(v, dtype_shrink)
                new_omx.add_matrix(
                    name,
                    v,
                    complevel=complevel,
                    complib=complib,
                )
            keys = sorted(old_omx.lookup._v_children)
            for name in keys[part::n_part]:
                v = old_omx.lookup._v_children[name][:]
                v = _shrink_bitwidth(v, dtype_shrink)
                new_omx.add_lookup(
                    name,
                    v,
                    complevel=complevel,
                    complib=complib,
                )
    return time.time() - start


def convert_omx_parallel(
    existing_filename,
    new_filename,
    complevel=3,
    complib="blosc2:zstd",
    dtype_shrink=32,
    n_processes=4,
) -> float:
    """
    Convert an existing OMX file using different data types and filters.

    Parameters
    ----------
    existing_filename : str
        The filename of the existing OMX file to convert.
    new_filename : str
        The filename of the new OMX file to write.
    complevel : int, default 1
        Compression level.
    complib : str, default 'blosc2:zstd'
        Compression library.
    dtype_shrink : int, default 32
        The maximum bitwidth to use for integer and float types.
    n_processes : int, default 4
        The number of processes to use for parallel conversion.

    Returns
    -------
    float
        The time in seconds it took to convert the file.
    """
    start = time.time()
    from multiprocessing import Pool

    # start 4 worker processes
    with Pool(processes=n_processes) as pool:
        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [
            pool.apply_async(
                convert_omx,
                (
                    existing_filename,
                    new_filename,
                    complevel,
                    complib,
                    dtype_shrink,
                    i,
                    n_processes,
                ),
            )
            for i in range(n_processes)
        ]
        for res in multiple_results:
            res.get(timeout=600)

    return time.time() - start


def _old_convert_multiple_omx(
    glob_pattern,
    complevel=3,
    complib="blosc2:zstd",
    dtype_shrink=32,
    n_processes=4,
    out_dir=None,
):
    """
    Convert multiple OMX files using different data types and filters.

    Parameters
    ----------
    glob_pattern : str
    complevel : int, default 1
        Compression level.
    complib : str, default 'blosc2:zstd'
        Compression library.
    dtype_shrink : int, default 32
        The maximum bitwidth to use for integer and float types.
    n_processes : int, default 4
        The number of processes to use for parallel conversion.
    out_dir : str, default None
        The directory to write the new OMX files to.  If None, the new files
        will be written to the same directory as the existing files.
    """
    import glob

    for f in glob.glob(glob_pattern):
        new_filename = f
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            new_filename = os.path.join(out_dir, os.path.basename(f))
        print(f"converting {f} to {new_filename}...")
        t = convert_omx_parallel(
            f, new_filename, complevel, complib, dtype_shrink, n_processes
        )
        print(f"\rconverted {f} to {new_filename} in {t:.2f} seconds.")


def convert_multiple_omx(
    glob_pattern,
    complevel=3,
    complib="blosc2:zstd",
    dtype_shrink=32,
    n_processes=8,
    out_dir=None,
) -> float:
    """
    Convert multiple OMX files using different data types and filters.

    Parameters
    ----------
    glob_pattern : str
    complevel : int, default 1
        Compression level.
    complib : str, default 'blosc2:zstd'
        Compression library.
    dtype_shrink : int, default 32
        The maximum bitwidth to use for integer and float types.
    n_processes : int, default 4
        The number of processes to use for parallel conversion.
    out_dir : str, default None
        The directory to write the new OMX files to.  If None, the new files
        will be written to the same directory as the existing files.

    Returns
    -------
    float
        The wall clock time it took to convert all the files, in seconds.
    """
    start = time.time()
    import glob
    from multiprocessing import Pool

    # start worker processes
    with Pool(processes=n_processes) as pool:
        multiple_results = []
        results_outstanding = 0

        for f in glob.glob(glob_pattern):
            new_filename = f
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                new_filename = os.path.join(out_dir, os.path.basename(f))
            print(f"converting {f} to {new_filename}...")
            r = pool.apply_async(
                convert_omx,
                (f, new_filename, complevel, complib, dtype_shrink),
            )
            multiple_results.append((r, f, new_filename))
            results_outstanding += 1

        while results_outstanding > 0:
            for i, (res, f, new_filename) in enumerate(multiple_results):
                if res.ready():
                    t = res.get(timeout=60)
                    print(f"\rconverted {f} to {new_filename} in {t:.2f} seconds.")
                    results_outstanding -= 1
                    del multiple_results[i]
                    break
            time.sleep(1)

    wall_time = time.time() - start
    print(f"converted all files in {wall_time:.2f} seconds.")
    return wall_time
