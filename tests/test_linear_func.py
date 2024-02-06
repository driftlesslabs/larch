from __future__ import annotations

from larch import PX, P, X


def test_P():
    assert P("one") == "one"


def test_X():
    # Add test cases for X function here
    assert X(2) == "2"
    assert X(3.14) == "3.14"
    assert X("hello") == "hello"


def test_PX():
    # Add test cases for PX function here
    assert PX("hello") == P.hello * X.hello


def test_linear_components():
    # Add test cases for LinearFunction class here
    f = P.Aaa * X.Bbb + P.Ccc * X.Ddd
    assert f[0].param == "Aaa"
    assert f[0].data == "Bbb"
    assert f[1].param == "Ccc"
    assert f[1].data == "Ddd"
    assert len(f) == 2


def test_linear_component_math():
    # Add test cases for LinearFunction class here
    f = P.Aaa * X.Bbb * 3
    assert f.param == "Aaa"
    assert f.data == "Bbb"
    assert f.scale == 3

    f = P.Aaa * (X.Bbb * 3)
    assert f.param == "Aaa"
    assert f.data == "Bbb"
    assert f.scale == 3

    f = P.Aaa * (3 * X.Bbb)
    assert f.param == "Aaa"
    assert f.data == "Bbb"
    assert f.scale == 3

    f = P.Aaa * (X.Bbb * 3.1)
    assert f.param == "Aaa"
    assert f.data == "Bbb"
    assert f.scale == 3.1

    f = P.Aaa * (3.1 * X.Bbb)
    assert f.param == "Aaa"
    assert f.data == "Bbb"
    assert f.scale == 3.1
