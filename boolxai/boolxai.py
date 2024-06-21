# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from .classifiers import BaselineClassifier, RuleClassifier


class BoolXAI:
    class BaselineClassifier(BaselineClassifier):
        pass

    class RuleClassifier(RuleClassifier):
        pass
