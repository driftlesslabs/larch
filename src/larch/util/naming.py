from __future__ import annotations

import keyword
import re
import warnings


class NotAPythonIdentifier(Warning):
    pass


def make_valid_identifier(x, suppress_warnings=False):
    x = str(x)
    if keyword.iskeyword(x):
        y = "_" + x
        if not suppress_warnings:
            warnings.warn(
                f"name {x} is a python keyword, converting to {y}",
                stacklevel=2,
            )
    else:
        y = x
    replacer = re.compile(r"(\W+)")
    y = replacer.sub("_", y.strip())
    if not y.isidentifier():
        y = "_" + y
    if y != x:
        if not suppress_warnings:
            warnings.warn(
                f"name {x} is not a valid python identifier, converting to {y}",
                stacklevel=2,
                category=NotAPythonIdentifier,
            )
    return y


def valid_identifier_or_parenthized_string(x, leading_dot=True):
    x = str(x)
    if x.isidentifier():
        if leading_dot:
            return "." + x
        return x
    else:
        if "'" in x and '"' not in x:
            return f'("{x}")'
        if "'" not in x and '"' in x:
            return f"('{x}')"
        raise NotImplementedError("cannot handle strings with both quote types")


def parenthize(x, signs_qualify=False):
    """Wrap a string in parentheses if needed for unambiguous clarity.

    Parameters
    ----------
    x : str
            The string to wrap
    signs_qualify : bool
            If True, a leading + or - on a number triggers the parenthesis (defaults False)

    """
    x = str(x).strip()
    replacer = re.compile(r"(\W+)")
    if replacer.search(x):
        if signs_qualify:
            numeric = re.compile(
                r"^(([1-9][0-9]*\.?[0-9]*)|(\.[0-9]+))([Ee][+-]?[0-9]+)?\Z"
            )
        else:
            numeric = re.compile(
                r"^[+-]?(([1-9][0-9]*\.?[0-9]*)|(\.[0-9]+))([Ee][+-]?[0-9]+)?\Z"
            )
        if numeric.search(x):
            return x
        return f"({x})"
    return x
