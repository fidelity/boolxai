# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import warnings

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class BoolXAIKBinsDiscretizer(KBinsDiscretizer):
    """Patched KBinsDiscretizer - provides more legible feature names.

    Note: for most use cases we expect there to be a better binarization than sklearn's
    KBinsDiscretizer - this code is provided for pedagogical reasons only, rather than
    as a recommended way of binarizing data.
    """

    def get_feature_names_out(self, input_features=None):
        """Get output feature names including the respective bin edges."""
        if input_features is None:
            input_features = self.feature_names_in_

        output_features = []
        for input_feature, edges, n_bins in zip(
            input_features, self.bin_edges_, self.n_bins_
        ):
            # There is a known bug in which the effective number of bins can be one
            # (due to the collapsing of quantiles that are close together, for example),
            # resulting in a single trivial bin, see the twin issues:
            # https://github.com/scikit-learn/scikit-learn/issues/25594
            # https://github.com/scikit-learn/scikit-learn/issues/19433
            if n_bins == 1:
                warnings.warn(
                    f"The binarization of feature {input_feature} has only one (trivial) bin. "
                    f"Consider using different parameters or a different binarization scheme."
                )
                output_features.append(f"[{input_feature}]")
                continue

            # Hard-coded for this example code only
            edges = np.round(edges, 4)

            # Note: bins are inclusive of the lower bound but not the upper bound,
            # as explained in the docs for KBinsDiscretizer.
            # First feature stretches to minus infinity (first edge ignored)
            output_features.append(f"[{input_feature}<{edges[1]}]")
            # In between features stretch to their respective bin edges
            for i in range(2, len(edges) - 1):
                output_features.append(f"[{edges[i-1]}<={input_feature}<{edges[i]}]")
            # Last feature stretches to plus infinity (last edge ignored)
            output_features.append(f"[{input_feature}>={edges[-2]}]")

        return output_features
