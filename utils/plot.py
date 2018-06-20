#!/usr/bin/env python

import numpy
import matplotlib

from hep_ml.metrics_utils import ks_2samp_weighted
from matplotlib import pyplot as plt

#######################
# Statistics printout #
#######################


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


########
# Plot #
########

# Plot style
plt.style.use('bmh')

# Font family
# available families: ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
matplotlib.rcParams.update({'font.family': 'monospace'})
# matplotlib.rcParams.update({'font.serif': ['Dejavu Sans']})


def draw_distributions(filename, names, original, target,
                       original_weights=None, target_weights=None,
                       nrows=2, ncols=3,
                       hist_settings={'bins': 20, 'density': True, 'alpha': 0.7}
                       ):
    # Assume weights to be equal if there are not provided
    original_weights = numpy.ones(len(original), dtype=float) if \
        original_weights is None else original_weights
    target_weights = numpy.ones(len(target)) if \
        target_weights is None else target_weights

    # Main figure
    figure = plt.figure()

    for idx, n in enumerate(names, 1):
        plot_range = numpy.percentile(numpy.hstack([target[n]]), [0.01, 99.99])

        # subplot(nrows, ncols, idx)
        subfigure = figure.add_subplot(nrows, ncols, idx)
        subfigure.set_title(n)

        # Actually draw histograms to this subfigure
        subfigure.hist(original[n], weights=original_weights,
                       range=plot_range, **hist_settings)
        subfigure.hist(target[n], weights=target_weights,
                       range=plot_range, **hist_settings)

    # Minimize overlapping
    figure.tight_layout()

    figure.savefig(filename, dpi=300)
