# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np

from boolxai.util import check_binary


class Rule(ABC):
    """An expressive Boolean formula, can be negated."""

    @abstractmethod
    def __init__(self):
        # Must initialize/populate the attributes: parent, subrules, and negated
        pass  # pragma: no cover

    @abstractmethod
    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluates this rule as if negated=False."""
        pass  # pragma: no cover

    @abstractmethod
    def _get_label(
        self, feature_names: Optional[Union[Dict[int, str], List[str]]] = None
    ) -> str:
        """Returns the label for this rule, for example "~AtMost1"."""
        pass  # pragma: no cover

    def evaluate(self, X: np.ndarray, check: bool = True) -> np.ndarray:
        """Evaluates this rule on data X, returns a one-dimensional Boolean ndarray.

        Args:
            X (np.ndarray): Input data - each row is a sample. Expected to be
                binary (i.e., 0/1/False/True).
            check (bool, optional): If True, check validity of inputs. Defaults to True.

        Returns:
            np.ndarray: A one-dimensional Boolean array. Each value corresponds to the
            evaluation of the rule on the respective data row.
        """
        if check:
            check_binary(np.unique(X))

        if self.negated:
            return np.logical_not(self._evaluate(X))
        else:
            return self._evaluate(X)

    def flatten(self) -> list:
        """Returns a flattened list including this rule and all child rules."""
        return [self] + [el for subrule in self.subrules for el in subrule.flatten()]

    def to_str(
        self, feature_names: Optional[Union[Dict[int, str], List[str]]] = None
    ) -> str:
        """Returns string rule, optionally replacing indices with feature names.

        Example:
            >>> from boolxai import Operator, Literal
            >>> rule = Operator.AtMost([Literal(0), Literal(1, negated=True)], param=1)
            >>> rule.to_str()
            "AtMost1(0, ~1)"
            >>> rule.to_str({0:"a", 1:"b"})
            "AtMost1(a, ~b)"

        Args:
            feature_names (dict or list, optional): Feature names for each index. Can
                be provided as a dict, in which case the key is the index and the value
                is the feature name. Alternatively, a list of feature names can be
                passed in, such that the index in the list is the respective index of
                each feature. If None is passed in, the indices are used in the rule.

        Returns:
            str: String representation of the rule.
        """
        label = self._get_label(feature_names)

        if len(self.subrules) == 0:
            return label
        else:
            return (
                label
                + "("
                + ", ".join(subrule.to_str(feature_names) for subrule in self.subrules)
                + ")"
            )

    def to_dict(self, feature_names: Optional[Dict[str, int]] = None) -> dict:
        """Returns dict rule, optionally replacing indices with feature names.

        Operators are represented as a dictionary from the name of the operator
        to a list of subrules. Literals are represented as a string. These definitions
        are used recursively to return a dict representing the full rule.

        Note: `to_dict()` is undefined, and hence not implemented, for literals and
        trivial rules.

        Example:
            >>> from boolxai import Operator, Literal
            >>> x0 = Literal(0)
            >>> x1 = Literal(1)
            >>> x2 = Literal(2)
            >>> rule = Operator.AtMost([Operator.And([x0, x1]), ~Operator.Or([x0, ~x2])], param=1)
            >>> rule.to_dict()
            {'AtMost1': [{'And': ['0', '1']}, {'~Or': ['0', '~2']}]}
            >>> rule.to_dict({0:"a", 1:"b", 2:"c"})
            {'AtMost1': [{'And': ['a', 'b']}, {'~Or': ['a', '~c']}]}

        Args:
            feature_names (dict or list, optional): Feature names for each index. Can
                be provided as a dict, in which case the key is the index and the value
                is the feature name. Alternatively, a list of feature names can be
                passed in, such that the index in the list is the respective index of
                each feature. If None is passed in, the indices are used in the rule.

        Returns:
            dict: A dict representation of the rule.
        """
        label = self._get_label(feature_names)

        subrules = [
            subrule.to_dict(feature_names)
            if len(subrule.subrules) > 1
            else subrule.to_str(feature_names)
            for subrule in self.subrules
        ]

        return {label: subrules}

    def to_graph(self, feature_names: Optional[Dict[str, int]] = None):
        """Returns a NetworkX directed graph representing this rule.

        Args:
            feature_names (dict or list, optional): Feature names for each index. Can
                be provided as a dict, in which case the key is the index and the value
                is the feature name. Alternatively, a list of feature names can be
                passed in, such that the index in the list is the respective index of
                each feature. If None is passed in, the indices are used in the rule.

        Returns:
            nx.DiGraph: A NetworkX directed graph representing this rule.
        """
        import networkx as nx

        G = nx.DiGraph()
        self._add_to_graph(G, feature_names)
        return G

    def _add_to_graph(
        self, G, feature_names: Optional[Dict[str, int]] = None
    ):
        """Adds this rule to the directional graph G.

        This is a helper function for `to_graph()`. Users call `rule.to_graph()` which
        calls `_add_to_graph(G)` recursively.
        """

        G.add_node(
            id(self),
            label=self._get_label(feature_names),
            num_children=len(self.subrules),
        )

        if self.parent is not None:
            G.add_edge(id(self), id(self.parent))

        for subrule in self.subrules:
            subrule._add_to_graph(G, feature_names)

    def plot(
        self,
        feature_names: Optional[Union[Dict[int, str], List[str]]] = None,
        filename: Optional[str] = None,
        graph_attr: Optional[Dict] = None,
        node_attr: Optional[Dict] = None,
        edge_attr: Optional[Dict] = None,
    ):
        """Plots this rule.

        Args:
            feature_names (dict or list, optional): Feature names for each index. Can
                be provided as a dict, in which case the key is the index and the value
                is the feature name. Alternatively, a list of feature names can be
                passed in, such that the index in the list is the respective index of
                each feature. If None is passed in, the indices are used in the rule.
            filename (str): Path where the plot should be saved. The format will be
                inferred from the extension. Pass in None to plot the figure on the
                screen instead of to a file. The latter currently only supports plotting
                within a Jupyter notebook. Defaults to None.
            graph_attr (dict, optional): Mapping from attribute name to value, which
                will update the default values of `AGraph().graph_attr`. If None is
                passed in, the default values are used.
            node_attr (dict, optional): Mapping from attribute name to value, which
                will update the default values of `AGraph().node_attr`. If None is
                passed in, the default values are used.
            edge_attr (dict, optional): Mapping from attribute name to value, which
                will update the default values of `AGraph().edge_attr`. If None is
                passed in, the default values are used.
        """
        import networkx as nx
        from IPython.display import Image, display

        # Get a NetworkX DiGraph object
        G = self.to_graph(feature_names)
        # Convert it to a graphviz AGraph object
        A = nx.nx_agraph.to_agraph(G)
        # We're going to fill in the nodes, so we have to set this flag
        A.node_attr["style"] = "filled"

        # Apply default styles
        default_graph_attr = {
            "fontcolor": "#1c2b39",
            "bgcolor": "white",
            "rankdir": "BT",  # Bottom to top gives the look we want
            "dpi": 300,
            "size": "10,10",
        }
        default_node_attr = {
            "fontcolor": "#1c2b39",
            "fontname": "Brandon",
            "style": "bold",
            "color": "#1c2b39",
            "shape": "rect",
        }
        default_edge_attr = {
            "fontcolor": "#1c2b39",
            "fontname": "Brandon",
            "style": "bold",
            "arrowhead": "open",
        }
        A.graph_attr.update(default_graph_attr)
        A.node_attr.update(default_node_attr)
        A.edge_attr.update(default_edge_attr)

        # Apply passed attribute dicts (if not None)
        A.graph_attr.update(graph_attr or {})
        A.node_attr.update(node_attr or {})
        A.edge_attr.update(edge_attr or {})

        # Apply appropriate fill colour to each node
        for node in A.nodes():
            num_children = int(node.attr["num_children"])
            if num_children == 0:
                # Literals
                node.attr["fillcolor"] = "#d6e8d4"
            else:
                # Operators
                node.attr["fillcolor"] = "#dae8fc"

        if filename is not None:
            A.draw(prog="dot", path=filename)
        else:
            # Make a layout and draw the graph - this will likely only work in
            # Jupyter notebooks, but that's sufficient for now (and there doesn't seem
            # to be a straightforward alternative).
            img_bytes = A.draw(prog="dot", path=None, format="png")
            img = Image(img_bytes)
            display(img)

    def depth(self) -> int:
        """Calculates and returns the depth of this rule."""
        if len(self.subrules) == 0:
            return 0

        return 1 + max(subrule.depth() for subrule in self.subrules)

    def complexity(self) -> int:
        """Calculates and returns the complexity of this rule."""
        return 1 + sum(len(subrule) for subrule in self.subrules)

    def __len__(self):
        return self.complexity()  # For convenience, len() returns the complexity

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return self.to_str()

    def __invert__(self):
        # It's safer to return a fresh copy since otherwise there could be unintended
        # consequences. For example, if x0 = Literal(0) and someone uses ~x0 somewhere,
        # then from then on x0 will be negated, which is totally non-intuitive!
        new = deepcopy(self)
        new.negated = not self.negated
        return new

    @abstractmethod
    def __deepcopy__(self, memo):
        # Must define how to deep copy rules
        pass  # pragma: no cover

    @abstractmethod
    def __eq__(self, other):
        # Must define how to compare rules
        pass  # pragma: no cover
