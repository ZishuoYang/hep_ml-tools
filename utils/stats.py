#!/usr/bin/env python

import numpy

from hep_ml.metrics_utils import ks_2samp_weighted


def print_statistics(names, original, target,
                     original_weights=None, target_weights=None):
    # Assume weights to be equal if there are not provided
    original_weights = numpy.ones(len(original), dtype=float) if \
        original_weights is None else original_weights
    target_weights = numpy.ones(len(target)) if \
        target_weights is None else target_weights

    for n in names:
        print('KS over %s = %s') % (
            n,
            ks_2samp_weighted(
                original[n], target[n],
                weights1=original_weights,
                weights2=target_weights
            ))

    print('========')
