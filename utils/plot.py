#!/usr/bin/env python

import numpy
import matplotlib
matplotlib.use('agg')  # for png output

from matplotlib import pyplot as plt

# Plot style
plt.style.use('bmh')

# Font family
# available families: ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
matplotlib.rcParams.update({'font.family': 'monospace'})

# Font size
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams.update({'figure.titlesize': 10})


def draw_distributions(filename, names, original, target,
                       original_weights=None, target_weights=None,
                       filename_as_title=False,
                       nrows=2, ncols=3,
                       xlim=None, ylim=None,
                       hist_settings={'bins': 20, 'density': True, 'alpha': 0.7}
                       ):
    # Assume weights to be equal if there are not provided
    original_weights = numpy.ones(len(original), dtype=float) if \
        original_weights is None else original_weights
    target_weights = numpy.ones(len(target)) if \
        target_weights is None else target_weights

    figure = plt.figure()

    for idx, n in enumerate(names, 1):
        plot_range = numpy.percentile(numpy.hstack([target[n]]), [0.01, 99.99])

        # subplot(nrows, ncols, idx)
        subfigure = figure.add_subplot(nrows, ncols, idx)
        subfigure.set_title(n)

        # If provided, set axes limit for each subplot
        if ylim is not None:
            subfigure.set_ylim(ylim[idx-1])
        if xlim is not None:
            subfigure.set_xlim(xlim[idx-1])

        # Actually draw histograms to this subfigure
        subfigure.hist(original[n], weights=original_weights,
                       range=plot_range, **hist_settings)
        subfigure.hist(target[n], weights=target_weights,
                       range=plot_range, **hist_settings)

    # Minimize overlapping
    if filename_as_title:
        figure.suptitle('.'.join(filename.split('.')[:-1]), fontsize=14)
        figure.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        figure.tight_layout()

    figure.savefig(filename, dpi=300)
