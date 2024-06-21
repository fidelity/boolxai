# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from boolxai.rules.trivials import One, Wildcard, Zero

try:
    import networkx as nx
    import IPython.display
except ImportError:
    plot = None

trivials = [Zero(), One(), Wildcard()]


@pytest.mark.parametrize("trivial", trivials)
def test_init_subrules(trivial):
    assert trivial.subrules == []


@pytest.mark.parametrize("trivial", trivials)
def test_init_negated(trivial):
    assert trivial.negated is False


@pytest.mark.parametrize("trivial", trivials)
def test_init_parent(trivial):
    assert trivial.parent is None


@pytest.fixture
def X0():
    return np.array([[True, False], [False, True]]).T


@pytest.mark.parametrize("trivial,value", [(Zero(), 0), (One(), 1)])
def test_evaluate(X0, trivial, value):
    assert np.array_equal(trivial.evaluate(X0), np.array([value, value]))


@pytest.mark.parametrize("trivial", trivials)
def test_flatten(trivial):
    assert trivial.flatten() == [trivial]


@pytest.mark.parametrize("trivial", trivials)
def test_trivial_rule_len(trivial):
    assert len(trivial) == 0


@pytest.mark.parametrize("trivial", trivials)
@pytest.mark.parametrize("feature_names", [None, {0: "fdas"}])
def test_to_str(trivial, feature_names):
    if not isinstance(trivial, Wildcard):
        assert trivial.to_str(feature_names) == type(trivial).__name__
    else:
        assert trivial.to_str(feature_names) == "*"


@pytest.mark.parametrize("trivial", trivials)
@pytest.mark.parametrize("feature_names", [None, {0: "fdas"}])
def test_to_dict_raises(trivial, feature_names):
    with pytest.raises(NotImplementedError):
        trivial.to_dict(feature_names)


@pytest.mark.parametrize("trivial", trivials)
@pytest.mark.skipif(plot is None, reason="Plotting requires ipython, networkx, and pygraphviz")
def test_to_graph(trivial):
    G = trivial.to_graph()

    assert isinstance(G, nx.DiGraph)
    assert nx.is_weakly_connected(G)
    assert G.number_of_nodes() == 1
    assert G.number_of_edges() == 0


@pytest.mark.parametrize("filename", [None, "test_plot.pdf"])
@pytest.mark.parametrize("trivial", trivials)
@pytest.mark.skipif(plot is None, reason="Plotting requires ipython, networkx, and pygraphviz")
def test_plot(filename, trivial):
    trivial.plot(filename=filename)

    if filename is None:
        assert True  # An explicit assert statement is needed to tell pytest to cleanup
    else:
        filename_ = Path(filename)
        assert filename_.exists() and filename_.is_file()
        filename_.unlink()  # Clean up


@pytest.mark.parametrize("trivial", trivials)
def test_deepcopy(trivial):
    trivial_copy = deepcopy(trivial)
    assert trivial_copy.negated == trivial.negated
    assert trivial_copy.subrules == trivial.subrules
    assert trivial_copy.parent == trivial.parent

    # Changing the copy shouldn't change the original
    trivial_copy.negated = True
    assert trivial.negated is False


@pytest.mark.parametrize("trivial", trivials)
def test_depth(trivial):
    assert trivial.depth() == 0


@pytest.mark.parametrize("trivial", trivials)
def test_complexity(trivial):
    assert trivial.complexity() == 0


@pytest.mark.parametrize("trivial", trivials)
def test_len(trivial):
    assert len(trivial) == 0


def test_eq():
    assert Zero() == Zero()
    assert One() == One()


def test_neq():
    assert Zero() != One()
    assert One() != Zero()


@pytest.mark.parametrize("trivial,not_trivial", [(Zero(), One()), (One(), Zero())])
def test_invert(trivial, not_trivial):
    assert ~trivial == not_trivial
    assert ~not_trivial == trivial
