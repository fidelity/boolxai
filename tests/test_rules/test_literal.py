# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from boolxai import Literal

try:
    import networkx as nx
    import IPython.display
except ImportError:
    plot = None


@pytest.fixture
def x0():
    return Literal(0)


@pytest.fixture
def x1():
    return Literal(1)


@pytest.fixture
def not_x1():
    return Literal(1, negated=True)


@pytest.fixture
def X0():
    return np.array([[True, False], [False, True]]).T


@pytest.fixture
def X1():
    return np.array([[False, True], [True, False]]).T


def test_init_index(x0, not_x1):
    assert x0.index == 0
    assert not_x1.index == 1


def test_init_negated(x0, not_x1):
    assert x0.negated is False
    assert not_x1.negated is True


def test_init_parent(x0, not_x1):
    assert x0.parent is None
    assert not_x1.parent is None


def test_evaluate(x0, X0, X1):
    assert np.array_equal(x0.evaluate(X0), np.array([True, False]))
    assert np.array_equal(x0.evaluate(X1), np.array([False, True]))


@pytest.mark.parametrize("invalid_element", [2, -1, "bla"])
def test_evaluate_invalid_data_raises(x0, invalid_element):
    X = np.array([[invalid_element, 1]])
    with pytest.raises(ValueError):
        x0.evaluate(X)


def test_evaluate_negated(not_x1, X0, X1):
    assert np.array_equal(not_x1.evaluate(X0), np.array([True, False]))
    assert np.array_equal(not_x1.evaluate(X1), np.array([False, True]))


def test_flatten(x0, not_x1):
    assert x0.flatten() == [x0]
    assert not_x1.flatten() == [not_x1]


def test_to_str_no_feature_names(x0, not_x1):
    assert x0.to_str() == "0"
    assert not_x1.to_str() == "~1"


def test_to_str_feature_names_dict(x0, not_x1):
    assert x0.to_str({0: "bla"}) == "bla"
    assert not_x1.to_str({0: "a", 1: "b"}) == "~b"


def test_to_str_feature_names_list(x0, not_x1):
    assert x0.to_str(["bla"]) == "bla"
    assert not_x1.to_str(["a", "b"]) == "~b"


def test_to_str_no_feature_name_raises(x0):
    with pytest.raises(KeyError):
        x0.to_str({1: "bla"})


@pytest.mark.parametrize("literal", ["x0", "not_x1"])
@pytest.mark.parametrize("feature_names", [None, {0: "fdas"}])
def test_to_dict_raises(literal, feature_names, request):
    # When parametrizing over a fixture need to:
    # 1. Pass in the name (string) of the fixture
    # 2. Pull out the value from request
    literal = request.getfixturevalue(literal)
    with pytest.raises(NotImplementedError):
        literal.to_dict(feature_names)


@pytest.mark.parametrize("filename", [None, "test_plot.pdf"])
@pytest.mark.parametrize("literal", ["x0", "not_x1"])
@pytest.mark.skipif(plot is None, reason="Plotting requires ipython, networkx, and pygraphviz")
def test_plot(filename, literal, request):
    # When parametrizing over a fixture need to:
    # 1. Pass in the name (string) of the fixture
    # 2. Pull out the value from request
    literal = request.getfixturevalue(literal)

    literal.plot(filename=filename)

    if filename is None:
        assert True  # An explicit assert statement is needed to tell pytest to cleanup
    else:
        filename_ = Path(filename)
        assert filename_.exists() and filename_.is_file()
        filename_.unlink()  # Clean up


def test_depth(x0, not_x1):
    assert x0.depth() == 0
    assert not_x1.depth() == 0


def test_complexity(x0, not_x1):
    assert x0.complexity() == 1
    assert not_x1.complexity() == 1


def test_len(x0, not_x1):
    assert len(x0) == 1
    assert len(not_x1) == 1


def test_deepcopy(x0, not_x1):
    x0.parent = not_x1
    x0_copy = deepcopy(x0)
    assert x0_copy.index == x0.index
    assert x0_copy.negated == x0.negated
    assert x0_copy.parent == x0.parent

    # Changing the copy shouldn't change the original
    x0_copy.index = 162
    assert x0.index == 0
    x0_copy.negated = True
    assert x0.negated is False


def test_eq(x0, not_x1):
    assert x0 == x0
    assert not_x1 == not_x1
    assert x0 != not_x1
    assert not_x1 != x0


def test_neq(x1, not_x1):
    assert x1 != not_x1
    assert not_x1 != x1


def test_invert(x1, not_x1):
    assert ~x1 == not_x1
    assert ~not_x1 == x1
