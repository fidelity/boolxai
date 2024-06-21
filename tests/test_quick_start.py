# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from boolxai import BoolXAI, Operator

try:
    import networkx as nx
    import IPython.display
except ImportError:
    plot = None


def test_quick_start():
    # Test that the example from ../README.md runs through

    # Create random toy data for binary classification. X and y must be binary!
    rng = np.random.default_rng(seed=42)
    X = rng.choice([0, 1], size=(100, 10))
    y = rng.choice([0, 1], size=100)

    # Rule classifier with maximum depth, complexity, possible operators
    rule_classifier = BoolXAI.RuleClassifier(max_depth=3,
                                             max_complexity=6,
                                             operators=[Operator.And, Operator.Or, Operator.Choose, Operator.AtMost,
                                                        Operator.AtLeast],
                                             random_state=42)

    # Learn the best rule
    rule_classifier.fit(X, y)

    # Best rule and best score
    best_rule = rule_classifier.best_rule_
    best_score = rule_classifier.best_score_
    print(f"{best_rule=} {best_score=:.2f}")

    # The depth of a rule is the number of edges in the longest path from the root to any leaf/literal.
    # The complexity of a rule is the total number of operators and literals.
    print(f"depth={best_rule.depth()} complexity={best_rule.complexity()}")

    # Predict and score
    y_pred = rule_classifier.predict(X)
    score = balanced_accuracy_score(y, y_pred)
    print(f"{score=:.2f}")

    # pytest relies on assert statements to deal with teardown, so we put a trivial
    # one here for that purpose. Any errors in the above would still cause the test to
    # fail.
    assert True


@pytest.mark.skipif(plot is None, reason="Plotting requires ipython, networkx, and pygraphviz")
def test_quick_start_plot():
    # Test that the example from ../README.md runs through

    # Create random toy data for binary classification. X and y must be binary!
    rng = np.random.default_rng(seed=42)
    X = rng.choice([0, 1], size=(100, 10))
    y = rng.choice([0, 1], size=100)

    # Rule classifier with maximum depth, complexity, possible operators
    rule_classifier = BoolXAI.RuleClassifier(max_depth=3,
                                             max_complexity=6,
                                             operators=[Operator.And, Operator.Or, Operator.Choose, Operator.AtMost,
                                                        Operator.AtLeast],
                                             random_state=42)

    # Learn the best rule
    rule_classifier.fit(X, y)

    # Best rule and best score
    best_rule = rule_classifier.best_rule_
    best_score = rule_classifier.best_score_
    print(f"{best_rule=} {best_score=:.2f}")

    # The depth of a rule is the number of edges in the longest path from the root to any leaf/literal.
    # The complexity of a rule is the total number of operators and literals.
    print(f"depth={best_rule.depth()} complexity={best_rule.complexity()}")

    # Predict and score
    y_pred = rule_classifier.predict(X)
    score = balanced_accuracy_score(y, y_pred)
    print(f"{score=:.2f}")

    # It is also possible to plot the best rule
    best_rule.plot()

    # or get a networkx.DiGraph representation of the rule
    G = best_rule.to_graph()
    print(G)

    # pytest relies on assert statements to deal with teardown, so we put a trivial
    # one here for that purpose. Any errors in the above would still cause the test to
    # fail.
    assert True
