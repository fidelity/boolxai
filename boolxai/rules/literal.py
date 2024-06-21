# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import numpy as np

from .rule import Rule


class Literal(Rule):
    """A Literal is a feature, possibly negated."""

    def __init__(self, index: int, negated: bool = False):
        """Initializes a literal.

        Args:
            index (int): The feature index that this literal corresponds to.
            negated (bool, optional): True if this literal is negated. Defaults to
                False.
        """
        if not np.issubdtype(type(index), np.integer):
            raise TypeError(f"index must be of type int but {index=} was passed")

        self.index = index
        self.negated = negated

        # No parent until modified
        self.parent = None

        # Literals never have any subrules
        self.subrules = []

    def to_dict(self, feature_names=None) -> dict:
        """Returns dict rule, not implemented for literals."""
        raise NotImplementedError("Literal.to_dict() should not be called")

    def _evaluate(self, X: np.ndarray):
        """Evaluates this literal - returns the feature given by index."""
        return X[:, self.index]

    def _get_label(self, feature_names=None):
        """Returns string representation of literal, for example "~2"."""
        prefix = "~" if self.negated else ""
        name = str(self.index) if feature_names is None else feature_names[self.index]
        label = prefix + name
        return label

    def __deepcopy__(self, memo):
        new_obj = Literal(index=self.index, negated=self.negated)
        new_obj.parent = self.parent
        return new_obj

    def __eq__(self, other):
        if not isinstance(other, Literal):
            return False

        # Does not check parent on purpose! Equality does not require that the parent be
        # the same.
        if self.index != other.index or self.negated != other.negated:
            return False

        return True
