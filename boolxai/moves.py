# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from copy import deepcopy

from boolxai import Literal
from boolxai.rules import ParametrizedOperator


class _Move:
    @staticmethod
    def remove_literal(literal):
        """Removes literal but only if no operators end up with less than two subrules.

        Note: Adjusts param of parent if needed.
        """
        if not isinstance(literal, Literal):
            raise TypeError("literal must be Literal")

        parent = literal.parent
        if parent is None:
            raise NoParentError("Cannot remove literal if it has no parent")

        if len(parent.subrules) <= 2:
            raise NotEnoughSubrulesError(
                "Cannot remove literal from rule with two or less subrules"
            )

        parent.subrules.remove(literal)

        if hasattr(parent, "param") and parent.param >= len(parent.subrules):
            # Adjust down to maximum allowed value
            parent.param = len(parent.subrules) - 1

        assert len(parent.subrules) >= 2

    @staticmethod
    def swap_literal(literal, index, negated):
        """Swaps a literal for a new literal defined by index and negated."""
        literal.index = index
        literal.negated = negated

    @staticmethod
    def expand_literal_to_operator(
        literal, sibling_literal, operator_class, operator_param=None
    ):
        """Expands a literal to an operator, adding a sibling literal to that operator.

        Note: Adjusts the param of the new operator if needed.
        """
        if len(literal.parent.subrules) <= 2:
            raise NotEnoughSubrulesError()

        # Create a copy of literal and place it under the new operator
        new_literal = deepcopy(literal)
        if operator_param is None:
            new_operator = operator_class(subrules=[new_literal, sibling_literal])
        else:
            new_operator = operator_class(
                param=operator_param, subrules=[new_literal, sibling_literal]
            )
        new_operator.parent = literal.parent
        # Remove literal and sibling_literal from the parent
        new_operator.parent.subrules.remove(sibling_literal)
        new_operator.parent.subrules.remove(literal)
        new_operator.parent.subrules.append(new_operator)

        parent = new_operator.parent
        if hasattr(parent, "param") and parent.param >= len(parent.subrules):
            # Adjust down to maximum allowed value
            parent.param = len(parent.subrules) - 1

    @staticmethod
    def remove_operator(operator):
        """Removes an operator, promoting any literals and operators under it."""
        parent = operator.parent
        if parent is None:
            raise NoParentError("Cannot remove operator if it has no parent")

        # If a literal under operator already exists under the parent, we don't promote
        # it - in order to avoid literals appearing twice (regardless of negation) under
        # the same operator.
        existing_literal_indices = {
            subrule.index for subrule in parent.subrules if isinstance(subrule, Literal)
        }

        # Update the parent and its subrules
        parent.subrules.remove(operator)
        for subrule in operator.subrules:
            if (
                isinstance(subrule, Literal)
                and subrule.index in existing_literal_indices
            ):
                continue
            subrule.parent = parent
            parent.subrules.append(subrule)

        assert len(parent.subrules) >= 2

    @staticmethod
    def add_literal(literal, operator):
        """Adds a literal to operator."""
        literal_copy = deepcopy(literal)
        literal_copy.parent = operator
        operator.subrules.append(literal_copy)

    @staticmethod
    def swap_operator(operator, new_operator_class, new_operator_param=None):
        """Swaps an operator for a new operator class name and parameter."""
        operator.__class__ = new_operator_class
        if new_operator_param is not None:
            operator.param = new_operator_param
        elif hasattr(operator, "param"):
            del operator.param

    @staticmethod
    def replace(target, new_target):
        """Replaces target with new_target."""
        target.__class__ = new_target.__class__
        if isinstance(new_target, ParametrizedOperator):
            target.param = new_target.param
        else:
            try:
                del target.param
            except AttributeError:
                pass
        if isinstance(new_target, Literal):
            target.index = new_target.index
        else:
            try:
                del target.index
            except AttributeError:
                pass

        target.subrules = deepcopy(new_target.subrules)
        target.negated = new_target.negated

    @staticmethod
    def find_and_replace(rule, target, new_target):
        """Finds target in rule and replaces with new_target. Returns a copy.

        Note: raises ValuError if target appears more than once in rule.
        """
        rule_new = deepcopy(rule)
        target_in_copy = None
        count = 0
        for possible_target in rule_new.flatten():
            if possible_target == target:
                count += 1
                target_in_copy = possible_target
        if target_in_copy is None:
            raise ValueError("target not found in rule")
        if count > 1:
            raise ValueError("target found in multiple places in rule")
        _Move.replace(target_in_copy, new_target)
        return rule_new


class NotEnoughSubrulesError(Exception):
    """A rule doesn't have enough subrules for this operation."""


class NoParentError(Exception):
    """A rule doesn't have a parent."""


class NoSiblingLiteralsError(Exception):
    """A rule doesn't have any sibling literals."""
