# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import numpy as np


def check_binary(data: np.ndarray, error: bool = True) -> bool:
    """Checks if all elements in data are binary.

    Note: it's usually preferred to pass in only the unique elements, for example
    via a call to np.unique().
    """
    binary = True
    for element in data:
        if element not in [0, 1, False, True]:
            binary = False
            break

    # Only raise once, for the first invalid element seen
    if error and not binary:
        raise ValueError(
            "Non-binary element detected - expected only 0/1 or False/True"
        )

    return binary
