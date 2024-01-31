from __future__ import annotations

import pytest

import larch as lx


def test_bad_avail_declaration():
    d = lx.examples.MTC(format="dataset")
    m = lx.Model(d)
    with pytest.raises(TypeError):
        m.unknown_attr = 123
    with pytest.raises(TypeError):
        m.availability_var = "avail"
