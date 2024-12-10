from __future__ import annotations

import altair as alt
import pandas as pd
from pytest import fixture

import larch as lx


@fixture(scope="module")
def model():
    m = lx.example(1)
    m.loglike(
        {
            "ASC_BIKE": -2.3776321824746316,
            "ASC_SR2": -2.1764779675270782,
            "ASC_SR3P": -3.723708490793479,
            "ASC_TRAN": -0.6671048298163328,
            "ASC_WALK": -0.2012241299742951,
            "hhinc#2": -0.0021788304992503785,
            "hhinc#3": 0.00035610619543557237,
            "hhinc#4": -0.005300245144846563,
            "hhinc#5": -0.012728753373485677,
            "hhinc#6": -0.009701442798519953,
            "totcost": -0.004917728615263069,
            "tottime": -0.051427250886585466,
        }
    )
    return m


def test_analyze_age(model, dataframe_regression):
    df = model.analyze_predictions_co("age")
    assert isinstance(df, pd.io.formats.style.Styler)
    df = df.data
    dataframe_regression.check(df)


def test_analyze_age_fig(model):
    fig = model.analyze_predictions_co_figure("age")
    assert isinstance(fig, alt.TopLevelMixin)


def test_analyze_age_binned(model, dataframe_regression):
    df = model.analyze_predictions_co("age", bins=[0, 25, 45, 65, 99])
    assert isinstance(df, pd.io.formats.style.Styler)
    df = df.data
    dataframe_regression.check(df)


def test_analyze_age_bool(model, dataframe_regression):
    df = model.analyze_predictions_co("age > 35")
    assert isinstance(df, pd.io.formats.style.Styler)
    df = df.data
    dataframe_regression.check(df)


def test_analyze_age_bool_fig(model):
    fig = model.analyze_predictions_co_figure("age > 35")
    assert isinstance(fig, alt.TopLevelMixin)


def test_analyze_few_values(model, dataframe_regression):
    df = model.analyze_predictions_co("age // 70")
    assert isinstance(df, pd.io.formats.style.Styler)
    df = df.data
    dataframe_regression.check(df)


def test_analyze_few_values_fig(model):
    fig = model.analyze_predictions_co_figure("age // 70")
    assert isinstance(fig, alt.TopLevelMixin)
