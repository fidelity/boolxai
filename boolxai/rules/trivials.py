# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from typing import Dict, Optional

import numpy as np

from .rule import Rule


class Trivial(Rule):
    """A Trivial rule evaluates to a constant."""

    def __init__(self):
        self.subrules = []
        self.negated = False

        # No parent until modified
        self.parent = None

    def to_dict(self, feature_names: Optional[Dict[str, int]] = None) -> dict:
        raise NotImplementedError("Literal.to_dict() should not be called")

    def complexity(self):
        # Note: the Rule class' __len__() method calls complexity(), so it's enough
        # to override complexity() - no need to override __len__()
        return 0

    def __deepcopy__(self, memo):
        return self.__class__()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        # Does not check parent on purpose! Equality does not require that the parent be
        # the same.
        return True


class Zero(Trivial):
    """A Zero rule evaluates to zero always."""

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def _get_label(self, feature_names=None):
        return "Zero"

    def __invert__(self):
        return One()


class One(Trivial):
    """A One rule evaluates to one always."""

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X))

    def _get_label(self, feature_names=None):
        return "One"

    def __invert__(self):
        return Zero()


class Wildcard(Trivial):
    """A Wildcard rule cannot be evaluated - it is used for marking purposes only."""

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Wildcard rule cannot be evaluated")

    def _get_label(self, feature_names=None):
        return "*"

    def __invert__(self):
        return Wildcard()
