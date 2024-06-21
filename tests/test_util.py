# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import numpy as np
import pytest

from boolxai.util import check_binary


@pytest.mark.parametrize("X", [[0, 1], [False, True, False], [0], [1], [True], [False]])
def test_check_binary_typical_positive(X):
    assert check_binary(X) is True
    assert check_binary(np.array(X)) is True
    assert check_binary(np.array(X).astype("O")) is True


@pytest.mark.parametrize("X", [[0, 2], [False, True, -1], ["Apple"], [0, "orange"]])
def test_check_binary_typical_negative(X):
    assert check_binary(X, error=False) is False
    assert check_binary(np.array(X), error=False) is False
    assert check_binary(np.array(X).astype("O"), error=False) is False


@pytest.mark.parametrize("X", [[0, 2], [False, True, -1], ["Apple"], [0, "orange"]])
@pytest.mark.parametrize(
    "func", [lambda x: x, lambda x: np.array(x), lambda x: np.array(x).astype("O")]
)
def test_check_binary_typical_negative_raises(X, func):
    with pytest.raises(ValueError):
        check_binary(func(X), error=True)
