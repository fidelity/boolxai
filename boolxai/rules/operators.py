# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from copy import deepcopy
from typing import Dict, List, Optional, Union, NewType

import numpy as np

from .literal import Literal
from .rule import Rule


class UnparametrizedOperator(Rule):
    """An UnparametrizedOperator operates on subrules, can be negated."""

    def __init__(
        self,
        subrules: List[Union["Operator", Literal]],
        negated: bool = False,
    ):
        """Initializes the operator.

        Args:
            subrules (list): A list of rules that will be children of this new operator.
            negated (bool, optional): True if this operator is negated. Defaults to
                False.
        """
        if not isinstance(subrules, list):
            raise TypeError("Subrules should be list")
        if len(subrules) < 2 or subrules is None:
            raise ValueError("Subrules should be of length two or more")

        self.subrules = deepcopy(subrules)
        for subrule in self.subrules:
            subrule.parent = self

        self.negated = negated

        # No parent until modified
        self.parent = None

    def _get_label(
        self, feature_names: Optional[Union[Dict[int, str], List[str]]] = None
    ) -> str:
        """Returns the label for this operator, for example "~AtMost1"."""
        prefix = "~" if self.negated else ""
        name = type(self).__name__
        label = prefix + name
        return label

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls(subrules=deepcopy(self.subrules), negated=self.negated)
        new_obj.parent = self.parent
        return new_obj

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        # Note: Currently the order matters for equality, so that And(a,b) != And(b,a)
        if self.subrules != other.subrules:
            return False
        if self.negated != other.negated:
            return False

        # Does not check parent on purpose! Equality does not require that the parent be
        # the same.

        return True


class ParametrizedOperator(Rule):
    """A ParametrizedOperator operates on subrules, can be negated, has a parameter."""

    def __init__(
        self,
        subrules: List[Union["Operator", Literal]],
        param: int,
        negated: bool = False,
    ):
        """Initializes the operator.

        Args:
            subrules (list): A list of rules that will be children of this new operator.
            param (int): An additional integer parameter for this operator.
            negated (bool, optional): True if this operator is negated. Defaults to
                False.
        """
        if not isinstance(subrules, list):
            raise TypeError("Subrules should be list")
        if len(subrules) < 2 or subrules is None:
            raise ValueError("Subrules should be of length two or more")

        self.subrules = deepcopy(subrules)
        for subrule in self.subrules:
            subrule.parent = self

        self.param = param
        self.negated = negated

        # No parent until modified
        self.parent = None

    def _get_label(
        self, feature_names: Optional[Union[Dict[int, str], List[str]]] = None
    ) -> str:
        """Returns the label for this operator, for example "~AtMost1"."""
        prefix = "~" if self.negated else ""
        name = type(self).__name__
        param = str(self.param) if self.param is not None else ""
        label = prefix + name + param
        return label

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls(
            param=self.param, subrules=deepcopy(self.subrules), negated=self.negated
        )
        new_obj.parent = self.parent
        return new_obj

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        # Note: Currently the order matters for equality, so that And(a,b) != And(b,a)
        if self.subrules != other.subrules:
            return False
        if self.param != other.param:
            return False
        if self.negated != other.negated:
            return False

        # Does not check parent on purpose! Equality does not require that the parent be
        # the same.

        return True


class Operator:
    """An Operator is a rule with two or more subrules and an optional integer param.

    Note: can be negated.
    """

    class And(UnparametrizedOperator):
        """Returns True if all subrules are True."""

        def _evaluate(self, state):
            return np.all(
                np.column_stack([subrule.evaluate(state) for subrule in self.subrules]),
                axis=1,
            )

    class Or(UnparametrizedOperator):
        """Returns True if at least one subrule is True."""

        def _evaluate(self, state):
            return np.any(
                np.column_stack([subrule.evaluate(state) for subrule in self.subrules]),
                axis=1,
            )

    class AtMost(ParametrizedOperator):
        """Returns True if at most param subrules are True."""

        def _evaluate(self, state):
            return (
                np.sum(
                    np.column_stack(
                        [subrule.evaluate(state) for subrule in self.subrules]
                    ),
                    axis=1,
                )
                <= self.param
            )

    class AtLeast(ParametrizedOperator):
        """Returns True if at least param subrules are True."""

        def _evaluate(self, state):
            return (
                np.sum(
                    np.column_stack(
                        [subrule.evaluate(state) for subrule in self.subrules]
                    ),
                    axis=1,
                )
                >= self.param
            )

    class Choose(ParametrizedOperator):
        """Returns True if exactly param subrules are True."""

        def _evaluate(self, state):
            return (
                np.sum(
                    np.column_stack(
                        [subrule.evaluate(state) for subrule in self.subrules]
                    ),
                    axis=1,
                )
                == self.param
            )


unparametrized_operator_classes = [Operator.And, Operator.Or]
parametrized_operator_classes = [Operator.AtMost, Operator.AtLeast, Operator.Choose]
all_operator_classes = unparametrized_operator_classes + parametrized_operator_classes

# OperatorType = NewType("OperatorType",
#                        Union[Operator.And, Operator.Or, Operator.AtMost, Operator.AtLeast, Operator.Choose])
