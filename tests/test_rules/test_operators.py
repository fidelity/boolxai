# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from boolxai import (
    Literal,
    Operator,
    all_operator_classes,
    parametrized_operator_classes,
    unparametrized_operator_classes,
)

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
def x2():
    return Literal(2)


@pytest.fixture
def x3():
    return Literal(3)


@pytest.fixture
def X2():
    return np.array([[True, True, False, False], [True, False, True, False]]).T


@pytest.fixture
def X3():
    return np.array(
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
        ]
    ).T


@pytest.fixture
def X4():
    return np.array(
        [
            [True, True, True, False],
            [False, True, False, True],
            [True, True, True, True],
            [False, False, True, True],
        ]
    ).T


def get_rule(operator_class, subrules, negated):
    """A helper function for more compact tests below.

    Returns a rule with operator_class as the parent, subrules below, and possibly
    negated. If the operator is parametrized, sets param=1.
    """
    if operator_class in unparametrized_operator_classes:
        return operator_class(subrules, negated=negated)
    else:
        return operator_class(subrules, param=1, negated=negated)


@pytest.mark.parametrize("negated", {False, True})
@pytest.mark.parametrize("operator_class", all_operator_classes)
def test_init(x0, not_x1, negated, operator_class):
    f = get_rule(operator_class, [x0, not_x1], negated=negated)

    assert f.subrules == [x0, not_x1]
    for subrule in f.subrules:
        assert subrule.parent == f
    assert f.negated is negated


@pytest.mark.parametrize("negated", {False, True})
@pytest.mark.parametrize("operator_class", all_operator_classes)
@pytest.mark.parametrize("subrules", [[], [Literal(0)]])
def test_evaluate_not_enough_subrules_raises(negated, operator_class, subrules):
    with pytest.raises(ValueError):
        get_rule(operator_class, subrules, negated=negated)


@pytest.mark.parametrize("operator_class", all_operator_classes)
@pytest.mark.parametrize("invalid_element", [2, -1, "bla"])
def test_evaluate_invalid_data_raises(x0, not_x1, operator_class, invalid_element):
    f = get_rule(operator_class, [x0, not_x1], negated=False)
    X = np.array([[invalid_element, 1]])
    with pytest.raises(ValueError):
        f.evaluate(X)


@pytest.mark.parametrize("operator_class", all_operator_classes)
@pytest.mark.parametrize(
    "X",
    [
        np.array([[0, 1], [1, 0], [0, 0], [1, 1]]),  # int
        np.array([[False, True], [True, False], [False, False], [True, True]]),  # bool
    ],
)
def test_evaluate_type(x0, not_x1, operator_class, X):
    # Rule
    f = get_rule(operator_class, [x0, not_x1], negated=False)

    y_pred = f.evaluate(X)

    # Output type should be ndarray always
    assert isinstance(y_pred, np.ndarray)

    # Output dtype should be bool always
    assert y_pred.dtype == bool

    # Unique values are True and False
    assert np.array_equal(np.unique(y_pred), np.array([False, True]))


def test_evaluate_two_literals_and(x0, not_x1, X2):
    formula = Operator.And([x0, not_x1])
    assert np.array_equal(formula.evaluate(X2), np.array([False, True, False, False]))


def test_evaluate_two_literals_negated_and(x0, not_x1, X2):
    formula = Operator.And([x0, not_x1], negated=True)
    assert np.array_equal(formula.evaluate(X2), np.array([True, False, True, True]))


def test_evaluate_two_literals_or(x0, not_x1, X2):
    formula = Operator.Or([x0, not_x1])
    assert np.array_equal(formula.evaluate(X2), np.array([True, True, False, True]))


def test_evaluate_two_literals_atmost(x0, not_x1, X2):
    formula = Operator.AtMost([x0, not_x1], 1)
    assert np.array_equal(formula.evaluate(X2), np.array([True, False, True, True]))


def test_evaluate_two_literals_atleast(x0, not_x1, X2):
    formula = Operator.AtLeast([x0, not_x1], 1)
    assert np.array_equal(formula.evaluate(X2), np.array([True, True, False, True]))


def test_evaluate_two_literals_choose(x0, not_x1, X2):
    formula = Operator.Choose([x0, not_x1], 1)
    assert np.array_equal(formula.evaluate(X2), np.array([True, False, False, True]))


def test_evaluate_nesting1(x0, not_x1, x2, x3, X3):
    formula = Operator.And([Operator.Or([x0, not_x1]), Operator.And([x2, x3])])
    assert np.array_equal(formula.evaluate(X3), np.array([True, False, False, False]))


def test_evaluate_nesting2(x0, not_x1, x2, x3, X4):
    formula = Operator.And(
        [Operator.AtMost([x0, Operator.And([x2, x3])], 1), Operator.And([not_x1, x0])]
    )
    assert np.array_equal(formula.evaluate(X4), np.array([True, False, False, False]))


@pytest.mark.parametrize("negated", {False, True})
@pytest.mark.parametrize("operator_class", all_operator_classes)
def test_flatten(x0, not_x1, x2, negated, operator_class):
    f = get_rule(operator_class, [x0, not_x1], negated=negated)
    assert f.flatten() == [f, x0, not_x1]

    g = get_rule(operator_class, [f, x2], negated=negated)
    assert g.flatten() == [g, f, x0, not_x1, x2]


@pytest.mark.parametrize("negated", {False, True})
@pytest.mark.parametrize("operator_class", all_operator_classes)
def test_depth(x0, not_x1, negated, operator_class):
    f = get_rule(operator_class, [x0, not_x1], negated=negated)
    assert f.depth() == 1

    g = get_rule(operator_class, [Operator.And([x0, not_x1]), x0], negated=negated)
    assert g.depth() == 2


@pytest.mark.parametrize("negated", {False, True})
@pytest.mark.parametrize("operator_class", all_operator_classes)
def test_complexity(x0, not_x1, negated, operator_class):
    f = get_rule(operator_class, [x0, not_x1], negated=negated)
    assert f.complexity() == 3
    assert len(f) == 3

    g = get_rule(operator_class, [x0, not_x1, x0], negated=negated)
    assert g.complexity() == 4
    assert len(g) == 4


@pytest.mark.parametrize("negated", {False, True})
@pytest.mark.parametrize("operator_class", all_operator_classes)
@pytest.mark.parametrize(
    "feature_names", [None, {0: "a", 1: "b", 2: "c"}, ["a", "b", "c"]]
)
def test_to_str(x0, not_x1, x2, negated, operator_class, feature_names):
    param_label = "" if operator_class in unparametrized_operator_classes else "1"
    prefix_label = "~" if negated is True else ""
    label = f"{prefix_label}{operator_class.__name__}{param_label}"

    f = get_rule(
        operator_class,
        [Operator.Or([x0, not_x1], negated=True), Operator.Choose([not_x1, x2, x0], 1)],
        negated=negated,
    )

    if feature_names is None:
        expected_str = f"{label}(~Or(0, ~1), Choose1(~1, 2, 0))"
    else:
        expected_str = f"{label}(~Or(a, ~b), Choose1(~b, c, a))"
    assert f.to_str(feature_names) == expected_str


@pytest.mark.parametrize("negated", {False, True})
@pytest.mark.parametrize("operator_class", all_operator_classes)
@pytest.mark.parametrize(
    "feature_names", [None, {0: "a", 1: "b", 2: "c"}, ["a", "b", "c"]]
)
def test_to_dict(x0, not_x1, x2, negated, operator_class, feature_names):
    param_label = "" if operator_class in unparametrized_operator_classes else "1"
    prefix_label = "~" if negated is True else ""
    label = f"{prefix_label}{operator_class.__name__}{param_label}"

    f = get_rule(
        operator_class,
        [Operator.Or([x0, not_x1], negated=True), Operator.Choose([not_x1, x2, x0], 1)],
        negated=negated,
    )

    if feature_names is None:
        expected_dict = {label: [{"~Or": ["0", "~1"]}, {"Choose1": ["~1", "2", "0"]}]}
    else:
        expected_dict = {label: [{"~Or": ["a", "~b"]}, {"Choose1": ["~b", "c", "a"]}]}

    print(f.to_dict(feature_names), expected_dict)


@pytest.mark.skipif(plot is None, reason="Plotting requires ipython, networkx, and pygraphviz")
def test_to_graph(x0, not_x1, x2):
    f = Operator.AtMost(
        [Operator.Or([x0, not_x1], negated=True), Operator.Choose([not_x1, x2, x0], 1)],
        param=2,
    )
    G = f.to_graph()

    assert isinstance(G, nx.DiGraph)
    assert nx.is_weakly_connected(G)
    assert G.number_of_nodes() == 8
    assert G.number_of_edges() == 7


@pytest.mark.parametrize("filename", [None, "test_plot.pdf"])
@pytest.mark.parametrize("operator_class", all_operator_classes)
@pytest.mark.skipif(plot is None, reason="Plotting requires ipython, networkx, and pygraphviz")
def test_plot(x0, not_x1, filename, operator_class):
    if operator_class in parametrized_operator_classes:
        rule = operator_class([x0, not_x1], param=1)
    else:
        rule = operator_class([x0, not_x1])

    rule.plot(filename=filename)

    if filename is None:
        assert True  # An explicit assert statement is needed to tell pytest to cleanup
    else:
        filename_ = Path(filename)
        assert filename_.exists() and filename_.is_file()
        filename_.unlink()  # Clean up


@pytest.mark.parametrize("negated", {False, True})
@pytest.mark.parametrize("operator_class", all_operator_classes)
def test_deepcopy(x0, not_x1, negated, operator_class):
    f = get_rule(operator_class, [x0, not_x1], negated=negated)
    f_copy = deepcopy(f)

    assert f_copy.subrules == f.subrules

    if operator_class in parametrized_operator_classes:
        assert f_copy.param == f.param

    # Changing the copy shouldn't change the original
    f_copy.subrules = [x0]
    assert f.subrules[0] == x0
    assert f.subrules[1] == not_x1

    f_copy.negated = not negated
    assert f.negated == negated


@pytest.mark.parametrize("negated", {False, True})
@pytest.mark.parametrize("operator_class", all_operator_classes)
def test_eq_neq(x0, x1, not_x1, negated, operator_class):
    f = get_rule(operator_class, [x0, not_x1], negated=negated)
    g = get_rule(operator_class, [x0, x1], negated=negated)

    # Make h the same as g but with a different operator_class
    if operator_class is not Operator.Or:
        h = Operator.Or([x0, x1], negated=negated)
    else:
        h = Operator.And([x0, x1], negated=negated)

    assert f == f
    assert g == g
    assert h == h

    assert f != g
    assert g != f

    assert g != h
    assert h != g

    assert f != h
    assert h != f


@pytest.mark.parametrize("operator_class", all_operator_classes)
def test_invert(x0, not_x1, operator_class):
    f = get_rule(operator_class, [x0, not_x1], negated=False)
    not_f = get_rule(operator_class, [x0, not_x1], negated=True)

    assert ~f == not_f
    assert ~not_f == f
