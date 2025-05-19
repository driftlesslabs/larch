from __future__ import annotations

import pytest

import larch
from larch import PX, P, X


def test_old_commands():
    d = larch.examples.MTC()

    with pytest.raises(NotImplementedError):
        m = larch.Model(dataservice=d)

    m = larch.Model(datatree=d)

    m.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
    m.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
    m.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
    m.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
    m.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
    m.utility_ca = PX("tottime") + PX("totcost")

    with pytest.raises(NotImplementedError):
        m.availability_var = "_avail_"

    m.availability_ca_var = "_avail_"
    m.choice_ca_var = "_choice_"
    m.title = "MTC Example 1 (Simple MNL)"

    with pytest.warns(DeprecationWarning):
        m.load_data()

    with pytest.raises(ValueError):
        m.maximize_loglike()

    m.choice_ca_var = "chose"
    result = m.maximize_loglike()
    assert result.loglike == pytest.approx(-3626.18625551293)
