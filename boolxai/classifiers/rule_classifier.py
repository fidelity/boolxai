# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import logging
from copy import deepcopy
from itertools import cycle, starmap
from math import exp
from typing import Callable, Optional, Union, NewType

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)

from boolxai.moves import NoParentError, NotEnoughSubrulesError, _Move
from boolxai.rules import Literal, One, Operator, Wildcard, Zero, all_operator_classes
from boolxai.util import check_binary
from .baseline_classifier import BaselineClassifier
from .postprocessing import remove_nested_and, remove_nested_or
from .util import (
    formulate_effective_problem,
    generate_random_rule,
    get_random_indices,
    operator_factory,
)

logger = logging.getLogger(__name__)

AllOperatorType = NewType("AllOperatorType",
                          Union[Operator.And, Operator.Or, Operator.AtMost, Operator.AtLeast, Operator.Choose])


class RuleClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-style Simulated Annealing native local classifier."""

    def __init__(
        self,
        max_complexity: Optional[int] = 6,
        max_depth: Optional[int] = 3,
        operators: AllOperatorType = all_operator_classes,
        num_iterations: int = 500,
        num_starts: int = 10,
        max_samples: Optional[int] = 2000,
        temp_high: float = 0.2,
        temp_low: float = 0.000001,
        regularization: float = 0.0001,
        metric: Callable = balanced_accuracy_score,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        num_jobs: int = -1,
        baseline: bool = True,
        base_rule: Optional[Operator] = None,
        postprocess: bool = True,
        on_start: Optional[Callable] = None,
        on_iteration: Optional[Callable] = None,
    ):
        """A simulated annealing native local classifier.

        Local moves change only a very localized part of the current rule -
        typically only one or two subrules.

        Notes:

        1. In order to see intermediate logged information set the logging level to
           `logger.DEBUG`, for example by setting up the root logger like so:
           ``logging.basicConfig(level=logging.DEBUG)``. However, note that logging
           can significantly slow the code, and that most logging messages will only
           appear if `num_jobs=1`.

        2. The computational effort is largely controlled by the product of `num_starts`
           and `num_iterations` divided by `num_jobs`. For large datasets, `max_samples`
           can reduce the computational effort (via a form of bagging).

        Args:
            max_complexity (int, optional): Maximum allowed complexity - must be 3
                or more. Defaults to 6. To impose no maximum complexity, pass in
                None.
            max_depth (int, optional): Maximum allowed depth - must be 1
                or more. Defaults to 3. To impose no maximum depth, pass in None.
            operators (list, optional): List of operator classes to include when
                generating solutions or moves. Defaults to `all_operator_classes`.
            num_iterations (int, optional): Number of iterations = moves to propose.
                Defaults to 500.
            num_starts (int, optional): Number of independent starts (runs).
                Defaults to 10.
            max_samples (int, optional): Maximum number of samples from the input
                data X passed to `fit()` to use in each start. If None or if it is
                smaller than the number of samples, all samples will be used.
                Otherwise, will populate `best_rule_` with the rule with the highest
                regularized score based on all samples, not just the smaller sample
                used in training, and will similarly populate `scores_` with the
                (unregularized) scores calculated on all samples.
            temp_high (float, optional): High temperature (exploration) in geometric
                temperature schedule. Defaults to 0.2.
            temp_low (float, optional): Low temperature (exploitation) in geometric
                temperature schedule. Defaults to 0.000001.
            regularization (float, optional): Controls the strength of the
                complexity term in the objective function = metric - regularization
                * complexity. Defaults to 0.0001. To turn off regularization pass in
                0.
            metric (callable, optional): An sklearn-style metric function, see
                `sklearn.metrics`, used internally to score rules during training.
                Note: make sure to use the same metric for any external evaluation
                of the model, for example in cross-validation. Defaults to
                `balanced_accuracy_score`.
            random_state (int or RandomState, optional). The seed of the pseudo
                random number generator that selects a random sample. Pass an int
                for reproducible output across multiple function calls. Defaults to
                None which uses the RandomState singleton used by np.random.
            num_jobs (int, optional): Number of simultaneous jobs to use for
                multiprocessing over starts. Pass in -1 to use the number of CPUs.
                Defaults to -1.
            baseline (bool, optional): Whether to include the best trivial or
                single-feature rules. If this baseline rule has a higher regularized
                score than the best rule found by the solver, the former will be
                returned for each start. Defaults to True.
            base_rule (Operator, optional): Used to optimize part of a rule. Pass in the
                (user-defined) rule that will be held fixed as base_rule. The base_rule
                should include exactly one Wildcard rule, which cannot be at the root.
                The classifier will then optimize the subtree under the Wildcard, while
                keeping the rest of the base_rule fixed. Note that if base_rule is
                passed in, max_complexity and max_depth still constrain the full
                resulting rules (and not just the part being optimized over).
            postprocess (bool, optional): Whether to post-process the rules found by
                the solver to simplify them. Defaults to True.
            on_start (callable, optional): Callback function that is executed at the
                beginning of each start, after initialization. Called within the
                code as `on_start(locals(), globals())`. Defaults to None.
            on_iteration (callable, optional): Callback function that is executed at
                the end of each iteration. Can return True to exit the iteration
                loop for the current start. Called within the code as
                `on_iteration(locals(), globals())`. Defaults to None.
        """
        self.max_complexity = max_complexity
        self.max_depth = max_depth
        self.operators = operators
        self.num_iterations = num_iterations
        self.num_starts = num_starts
        self.max_samples = max_samples
        self.temp_high = temp_high
        self.temp_low = temp_low
        self.regularization = regularization
        self.metric = metric
        self.random_state = random_state
        self.num_jobs = num_jobs
        self.baseline = baseline
        self.base_rule = base_rule
        self.postprocess = postprocess
        self.on_start = on_start
        self.on_iteration = on_iteration

        self.rules_ = None
        self.scores_ = None
        self.best_rule_ = None
        self.best_score_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.random_state_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "RuleClassifier":
        """Finds the best rule for training data X and labels y.

        Note: populates the `best_rule_` and `best_score_` attributes, as well as the
        `rules_` and `scores_` attributes, which contain a rule and score (respectively)
        for each start.

        Args:
            X (np.ndarray): Input data - each row is a sample. Expected to be
                binary (i.e., 0/1/False/True).
            y (np.ndarray): A label for each sample in the input data. Expected
                to be binary (i.e., 0/1/False/True).
            sample_weight (np.ndarray, optional): A weight for each sample or
                None for equal sample weights. Defaults to None.

        Returns:
            RuleClassifier: Returns the classifier object - useful for
            chaining.
        """
        logger.debug("-" * 80)
        logger.debug("-" * 80)
        logger.debug(
            f"fit() called with {self.max_complexity=}, {self.max_depth=}, {self.num_starts=}, {self.num_iterations=}"
        )

        # Initialization
        self.random_state_ = check_random_state(self.random_state)

        # Check that X, y, and sample_weight have the correct shape
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        # Check the validity of the labels and data
        check_binary(self.classes_)
        check_binary(np.unique(X))

        # Optimize the subtree under the wildcard node in base_rule
        if self.base_rule is not None:
            # Find wildcard node
            wildcard_count = 0
            for node in self.base_rule.flatten():
                if isinstance(node, Wildcard):
                    if node.parent is None:
                        raise ValueError(
                            "wildcard node in base_rule cannot be the root node"
                        )
                    wildcard_node = node
                    wildcard_count += 1
            if wildcard_count != 1:
                raise ValueError("base_rule must contain exactly one wildcard node")

            (
                X_,
                y_,
                sample_weight_,
            ) = formulate_effective_problem(
                self.base_rule, wildcard_node, X, y, sample_weight
            )

            if self.max_complexity is not None:
                # The maximum complexity that the new subtree under target could have is
                # the total complexity minus the fixed part of the complexity
                max_complexity_ = self.max_complexity - self.base_rule.complexity()
            else:
                max_complexity_ = None

            if self.max_depth is not None:
                # The maximum depth that the new subtree under target could have is
                # the total depth minus the fixed part of the depth
                max_depth_ = self.max_depth - self.base_rule.depth()
            else:
                max_depth_ = None
        else:
            X_, y_, sample_weight_ = X, y, sample_weight
            max_complexity_ = self.max_complexity
            max_depth_ = self.max_depth

        # This helper generator yields the arguments that are common to all starts
        def yield_args():
            args = (
                max_complexity_,
                max_depth_,
                self.operators,
                self.num_iterations,
                self.temp_high,
                self.temp_low,
                self.regularization,
                self.metric,
                self.baseline,
                self.on_start,
                self.on_iteration,
            )

            # Each start has its own seed for reproducibility
            seeds = self.random_state_.randint(99999999, size=self.num_starts)

            for num_start in range(self.num_starts):
                seed = seeds[num_start]

                # Add the arguments that are different for each start
                if self.max_samples is None or self.max_samples > X_.shape[0]:
                    # Use the full data
                    yield X_, y_, sample_weight_, seed, *args
                else:
                    # Generate a random array of indices and slice X, y, and
                    # sample_weight. We use a fresh random number generator seeded with
                    # seed so that we can recreate the indices used if we need them
                    indices = get_random_indices(len(X_), self.max_samples, seed)
                    sample_weight__ = (
                        sample_weight_[indices] if sample_weight_ is not None else None
                    )
                    yield X_[indices, :], y_[indices], sample_weight__, seed, *args

        # Run solver - single core or multi-core
        if self.num_jobs == 1:
            results = list(starmap(RuleClassifier._single_start, yield_args()))

        else:
            results = Parallel(n_jobs=self.num_jobs, prefer="processes")(
                delayed(RuleClassifier._single_start)(*args) for args in yield_args()
            )

        # Find best rule and score
        self.best_score_ = None
        self.best_rule_ = None
        best_regularized_score = -1
        self.scores_ = []
        self.rules_ = []
        logger.debug("-" * 80)
        logger.debug("Score and rule for each start:")
        # Having seed in the results allows us to identify (and reproduce) the result
        # for each start, if needed
        for rule, score, seed in results:
            if self.base_rule is not None or (
                self.max_samples is not None and self.max_samples < X.shape[0]
            ):
                if self.base_rule is not None:
                    # Find target in base_rule and replace it with the optimized
                    # subrule ("rule") - returns a copy.
                    rule = _Move.find_and_replace(self.base_rule, wildcard_node, rule)

                # score reflects partial data only, so recalculate on the full data
                score = RuleClassifier._score_rule(
                    rule,
                    X,
                    y,
                    self.metric,
                    sample_weight,
                    check=False,
                )

            if self.postprocess:
                self._postprocess_rule(rule)

            # Regularization penalizes rule complexity
            regularized_score = score - self.regularization * rule.complexity()
            if regularized_score > best_regularized_score:
                self.best_score_ = score
                self.best_rule_ = rule
                best_regularized_score = regularized_score

            logger.debug(f"    {score=:.3f}, {rule=}, {regularized_score:.3f}")

            self.scores_.append(score)
            self.rules_.append(rule)

        logger.debug(
            f"best_score={self.best_score_:.3f}, best_rule={self.best_rule_}, {best_regularized_score=}"
        )

        # Return the classifier
        return self

    @staticmethod
    def _single_start(
        X,
        y,
        sample_weight,
        seed,
        max_complexity,
        max_depth,
        operators,
        num_iterations,
        temp_high,
        temp_low,
        regularization,
        metric,
        baseline,
        on_start,
        on_iteration,
    ):
        # Edge case - just one class present, return a trivial rule
        # Note: this must be done here and not in fit() (over the whole data)
        # since if max_samples is lower than num_samples, then each start will have
        # a different sample.
        classes = unique_labels(y)
        if len(classes) == 1:
            label = classes[0]
            if label == 0:
                best_rule = Zero()
            elif label == 1:
                best_rule = One()
            else:
                raise ValueError("Single class but unidentified")
            best_score = RuleClassifier._score_rule(
                best_rule,
                X,
                y,
                metric,
                sample_weight,
                check=False,
            )
            return best_rule, best_score, seed

        # Initialization
        literal_move_types = cycle(
            ("remove_literal", "expand_literal_to_operator", "swap_literal")
        )
        operator_move_types = cycle(("remove_operator", "add_literal", "swap_operator"))
        num_features = X.shape[1]
        literal_indices = list(range(num_features))
        random_state = np.random.RandomState(seed)

        temperatures = np.geomspace(temp_high, temp_low, num_iterations)

        # Keep track of the current rule and score, start with a random rule
        # (below we have -1 since the operator is counted as well in the complexity)
        num_iteration = 0
        max_num_literals = max_complexity - 1 if max_complexity is not None else 5
        current_rule = generate_random_rule(
            literal_indices, max_num_literals, operators, random_state
        )

        current_score = RuleClassifier._score_rule(
            current_rule,
            X,
            y,
            metric,
            sample_weight,
            check=False,
        )
        # Regularization penalizes rule complexity
        regularized_current_rule_score = (
            current_score - regularization * current_rule.complexity()
        )

        # Keep track of the best rule and score, potentially starting from the baseline
        # NOTE: for the case max_samples > num_samples (or None), we're doing an
        # identical computation in each start. This might seem wasteful, but it's fairly
        # fast for small datasets. For large datasets, we expect
        # max_samples < num_samples, in which case we do actually need to do it in each
        # start separately, since each start sees a different chunk of data.
        if baseline:
            # Initially, the best rule and score are equal to the best of the
            # initial rule and score and the baseline rule and score
            baseline_classifier = BaselineClassifier(metric=metric)
            baseline_classifier.fit(X, y, sample_weight)
            baseline_rule = baseline_classifier.best_rule_
            baseline_score = baseline_classifier.best_score_

            if (
                current_score - regularization * current_rule.complexity()
                > baseline_score - regularization * baseline_rule.complexity()
            ):
                best_rule = deepcopy(current_rule)
                best_score = current_score
            else:
                best_rule = deepcopy(baseline_rule)
                best_score = baseline_score
        else:
            # Initially, the best rule and score are equal to the initial rule and
            # score
            best_rule = deepcopy(current_rule)
            best_score = current_score

        logger.debug(f"{num_iteration=}, {best_score=:.3f}, {best_rule=}")
        logger.debug(
            f"{current_score=:.3f}, {regularized_current_rule_score=:.3f}, {current_rule=}"
        )

        # Optional on-start callback
        if on_start is not None:
            on_start(locals(), globals())

        # Each iteration consists of proposing and possibly accepting a move
        # The first iteration is iteration 1 since we consider the initial solution
        # as being iteration 0
        for num_iteration, T in enumerate(temperatures, start=1):
            logger.debug("-" * 80)
            logger.debug(f"{num_iteration=}, {best_score=:.3f}, {best_rule=}")
            logger.debug(
                f"{current_score=:.3f}, {regularized_current_rule_score=:.3f}, {current_rule=}"
            )

            proposed_rule = RuleClassifier._propose_local_move(
                current_rule,
                literal_indices,
                max_complexity,
                max_depth,
                literal_move_types,
                operator_move_types,
                operators,
                random_state,
            )
            proposed_rule_score = RuleClassifier._score_rule(
                proposed_rule,
                X,
                y,
                metric,
                sample_weight,
                check=False,
            )

            logger.debug(f"{proposed_rule_score=:.3f}, {proposed_rule=}")

            # Metropolis criterion
            # Regularization penalizes rule complexity
            regularized_proposed_rule_score = (
                proposed_rule_score - regularization * proposed_rule.complexity()
            )
            dE = regularized_proposed_rule_score - regularized_current_rule_score
            accept = dE >= 0 or random_state.random() < exp(dE / T)
            if accept:
                logger.debug(
                    f"Accepted {proposed_rule_score=:.3f} over {current_score=:.3f} "
                    f"({regularized_proposed_rule_score=:.3f} over {regularized_current_rule_score=:.3f})"
                )

                # Update current solution
                current_rule = proposed_rule
                current_score = proposed_rule_score
                regularized_current_rule_score = regularized_proposed_rule_score

                # If we found a new best solution (based on the regularized scores),
                # update best_rule and best_score
                regularized_best_rule_score = (
                    best_score - regularization * best_rule.complexity()
                )
                improved_best_score = (
                    regularized_current_rule_score > regularized_best_rule_score
                )
                if improved_best_score:
                    logger.debug(
                        f"Updating best_score from {best_score=:.3f} to {current_score=:.3f} "
                        f"({regularized_best_rule_score=:.3f} to {regularized_current_rule_score=:.3f})"
                    )

                    best_rule = deepcopy(current_rule)
                    best_score = current_score

            # Optional on-iteration callback
            if on_iteration is not None:
                if on_iteration(locals(), globals()):
                    break

        return best_rule, best_score, seed

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions for X with `self.best_rule_`.

        Args:
            X (np.ndarray): Input data - each row is a sample. Expected to be
                binary (i.e., 0/1/False/True).

        Returns:
            np.ndarray: A one-dimensional Boolean ndarray. Each value corresponds to
            the evaluation of the rule on the respective data row.
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.best_rule_.evaluate(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Returns a "soft" score for each sample in X.

        Note: in our case, we simply return the same result as when calling
        `predict()`. We only implement this method since it is required for some
        higher-level ensemble models in sklearn.
        """
        return self.predict(X)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Generates predictions for X with `best_rule_` and returns their score vs. y.

        Note: the metric used to calculate the score is the one in the `metric`
        attribute.

        Args:
            X (np.ndarray): Input data - each row is a sample. Expected to be
                binary (i.e., 0/1/False/True).
            y (np.ndarray): A label for each sample in the input data. Expected
                to be binary (i.e., 0/1/False/True).
            sample_weight (np.ndarray, optional): A weight for each sample or
                None for equal sample weights. Defaults to None.

        Returns:
            float: The score of the rule in the `best_rule_` attribute, given the
            metric in the `metric` attribute.
        """
        # Input validation
        X, y = check_X_y(X, y)

        return RuleClassifier._score_rule(
            self.best_rule_,
            X,
            y,
            self.metric,
            sample_weight,
            check=True,
        )

    @staticmethod
    def _propose_local_move(
        current_rule,
        literal_indices,
        max_complexity,
        max_depth,
        literal_move_types,
        operator_move_types,
        operators,
        random_state,
    ):
        """Proposes local move for current_rule. Returns proposed rule (a deep copy)."""
        proposed_rule = deepcopy(current_rule)

        # Find a list of all operators and literals
        all_operators_and_literals = proposed_rule.flatten()

        found_move = False
        while not found_move:
            target = random_state.choice(all_operators_and_literals)

            if isinstance(target, Literal):
                literal_move_type = next(literal_move_types)
                logger.debug(f"{literal_move_type=}")

                if literal_move_type == "remove_literal":
                    try:
                        _Move.remove_literal(target)
                    except NoParentError:
                        logger.debug(f"  Skipping, target {target} has no parent")
                    except NotEnoughSubrulesError:
                        logger.debug(
                            f"  Skipping, not enough subrules to remove {target}"
                        )

                    else:
                        found_move = True

                elif literal_move_type == "swap_literal":
                    chosen_literal_index = random_state.choice(literal_indices)

                    # Only swap to this literal if it doesn't already exist in this
                    # operator. If we chose the same literal, negate it
                    if chosen_literal_index == target.index:
                        negated = not target.negated
                        _Move.swap_literal(target, chosen_literal_index, negated)
                        found_move = True
                    else:
                        existing_literal_indices = {
                            subrule.index
                            for subrule in target.parent.subrules
                            if isinstance(subrule, Literal)
                        }
                        if chosen_literal_index not in existing_literal_indices:
                            negated = random_state.choice([True, False])
                            _Move.swap_literal(target, chosen_literal_index, negated)
                            found_move = True
                        else:
                            logger.debug(
                                f"  {chosen_literal_index=} is already in rule"
                            )

                elif literal_move_type == "expand_literal_to_operator":
                    # Must have two siblings or more to be able to expand. This is also
                    # checked inside expand_literal_to_operator() but we might as well
                    # catch it early
                    if len(target.parent.subrules) < 3:
                        logger.debug(
                            "  Literal does not have enough siblings to expand"
                        )
                        continue

                    if max_complexity is not None:
                        current_rule_complexity = current_rule.complexity()
                        if current_rule_complexity == max_complexity:
                            # This move increases the complexity by 1, so it would take
                            # us over the cutoff, so skip it
                            logger.debug("  Skipping, at max_complexity")
                            continue

                        if current_rule_complexity > max_complexity:
                            logger.error(
                                "Complexity of current rule exceeds the complexity cutoff"
                            )
                            raise AssertionError(
                                "Complexity of current rule exceeds the complexity cutoff"
                            )

                    if max_depth is not None:
                        current_rule_depth = current_rule.depth()
                        if current_rule_depth == max_depth:
                            # This move increases the depth by 1, so it would take us
                            # over the cutoff, so skip it
                            logger.debug("  Skipping, at max_depth")
                            continue

                        if current_rule_depth > max_depth:
                            logger.error(
                                "Depth of current rule exceeds the depth cutoff"
                            )
                            raise AssertionError(
                                "Depth of current rule exceeds the depth cutoff"
                            )

                    # Find random sibling literal with different index
                    sibling_literals = []
                    for subrule in target.parent.subrules:
                        if (
                            isinstance(subrule, Literal)
                            and subrule.index != target.index
                        ):
                            sibling_literals.append(subrule)
                    if not sibling_literals:
                        logger.debug(f"  Skipping, {target} has no sibling literals")
                        continue
                    sibling_literal = random_state.choice(sibling_literals)

                    # Find a random operator and param
                    (
                        new_operator_class,
                        new_operator_param,
                    ) = operator_factory(
                        operators, num_subrules=2, random_state=random_state
                    )

                    try:
                        _Move.expand_literal_to_operator(
                            target,
                            sibling_literal,
                            new_operator_class,
                            new_operator_param,
                        )
                    except NotEnoughSubrulesError:
                        logger.debug(
                            f"  Skipping, not enough subrules to expand {target}"
                        )

                    else:
                        found_move = True

            else:
                operator_move_type = next(operator_move_types)
                logger.debug(f"{operator_move_type=}")

                if operator_move_type == "remove_operator":
                    try:
                        _Move.remove_operator(target)
                    except NoParentError:
                        logger.debug(f"  Skipping, target {target} has no parent")
                    else:
                        found_move = True

                elif operator_move_type == "add_literal":
                    if max_complexity is not None:
                        current_rule_complexity = current_rule.complexity()
                        if current_rule_complexity == max_complexity:
                            # This move increases the complexity by 1, so it would take
                            # us over the cutoff, so skip it
                            logger.debug("  Skipping, at max_complexity")
                            continue

                        if current_rule_complexity > max_complexity:
                            logger.error(
                                "Complexity of current rule exceeds the complexity cutoff"
                            )
                            raise AssertionError(
                                "Complexity of current rule exceeds the complexity cutoff"
                            )

                    chosen_literal_index = random_state.choice(literal_indices)

                    # Only add this literal if it doesn't already exist in this
                    # operator (even negated)
                    existing_literal_indices = [
                        subrule.index
                        for subrule in target.subrules
                        if isinstance(subrule, Literal)
                    ]
                    if chosen_literal_index not in existing_literal_indices:
                        literal = Literal(
                            index=chosen_literal_index,
                            negated=random_state.choice([True, False]),
                        )
                        _Move.add_literal(literal, target)
                        found_move = True

                elif operator_move_type == "swap_operator":
                    (
                        new_operator_class,
                        new_operator_param,
                    ) = operator_factory(operators, len(target.subrules), random_state)
                    same_operator = isinstance(target, new_operator_class)
                    if not same_operator or (
                        same_operator
                        and hasattr(target, "param")
                        and target.param != new_operator_param
                    ):
                        _Move.swap_operator(
                            target, new_operator_class, new_operator_param
                        )
                        found_move = True
                    else:
                        logger.debug(
                            f"  same operator {new_operator_class} and same param {new_operator_param}"
                        )

        return proposed_rule

    @staticmethod
    def _score_rule(rule, X, y, metric, sample_weight, check):
        """Returns the score (using metric) of the given rule on X and y."""
        # Note the difference from score() - the latter returns the score of
        # best_rule, whereas this method returns the score of a given rule.

        # We allow the checking of inputs to be switched off here. The reason is
        # that in the context of a solver, it's enough to check the input just
        # once - re-checking each evaluation would be very wasteful.
        predictions = rule.evaluate(X, check)
        score = metric(y_true=y, y_pred=predictions, sample_weight=sample_weight)
        return score

    def _more_tags(self):
        return {"binary_only": True}

    @staticmethod
    def _postprocess_rule(rule):
        remove_nested_or(rule)
        remove_nested_and(rule)
