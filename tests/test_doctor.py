from __future__ import annotations

import pytest
from pytest import approx

import larch as lx


@pytest.fixture(params=["numba", "jax"])
def ref_model(request):
    m = lx.example(1)
    m.compute_engine = request.param
    m.graph.new_node(parameter="mu_motor", children=[1, 2, 3, 4], name="Motorized")
    m.graph.new_node(parameter="mu_nonmotor", children=[5, 6], name="Nonmotorized")
    m.unmangle()
    return m


def test_chosen_but_not_available_query(ref_model: lx.Model):
    m = ref_model

    # change first 3 observations to WALK
    m.dataset["ch"][:3, :] = 0
    m.dataset["ch"][:3, -1] = 1

    # test just checking
    m, problems = m.doctor(repair_ch_av="?")
    assert len(problems) == 1
    assert "chosen_but_not_available" in problems
    ll = m.loglike_casewise()
    assert ll[0] < -9e9
    assert ll[1] < -9e9
    assert ll[2] < -9e9

    # test raising error
    with pytest.raises(ValueError):
        m, problems = m.doctor(repair_ch_av="!")


def test_chosen_but_not_available_plus(ref_model: lx.Model):
    m = ref_model

    # change first 3 observations to WALK
    m.dataset["ch"][:3, :] = 0
    m.dataset["ch"][:3, -1] = 1

    m, problems = m.doctor(repair_ch_av="+")
    assert (m.dataset["av"][:3, -1] == 1).all()
    ll = m.loglike_casewise()
    assert ll[:3] == approx([-1.79175947, -1.79175947, -1.60943791])


def test_chosen_but_not_available_minus(ref_model: lx.Model):
    m = ref_model

    # change first 3 observations to WALK
    m.dataset["ch"][:3, :] = 0
    m.dataset["ch"][:3, -1] = 1

    m, problems = m.doctor(repair_ch_av="-")
    assert (m.dataset["av"][:3, -1] == 0).all()
    assert (m.dataset["ch"][:3, -1] == 0).all()
    ll = m.loglike_casewise()
    assert ll[:3] == approx([0, 0, 0])
    assert m.loglike() == approx(-7304.995801563645)
