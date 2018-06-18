#!/usr/bin/env python

import root_numpy
import pandas
import numpy

from hep_ml import reweight
from sklearn.cross_validation import train_test_split
from hep_ml.metrics_utils import ks_2samp_weighted
from matplotlib.pyplot import subplot, hist, title


def draw_distributions(original, target, new_original_weights):
    for id, column in enumerate(columns, 1):
        xlim = numpy.percentile(numpy.hstack([target[column]]), [0.01, 99.99])
        subplot(2, 3, id)
        hist(original[column], weights=new_original_weights, range=xlim,
             **hist_settings)
        hist(target[column], range=xlim, **hist_settings)
        title(column)
        print('KS over %s = %s' % (
            column, ks_2samp_weighted(
                original[column], target[column],
                weights1=new_original_weights,
                weights2=numpy.ones(len(target), dtype=float)
            )
        ))


# Import data
columns = ['hSPD', 'pt_b', 'pt_phi', 'vchi2_b', 'mu_pt_sum']
original = root_numpy.root2array('MC_distribution.root', branches=columns)
target = root_numpy.root2array('RD_distribution.root', branches=columns)
original = pandas.DataFrame(original)
target = pandas.DataFrame(target)
original_weights = numpy.ones(len(original))

# Prepare train and test samples
# Divide original samples into training ant test parts
original_train, original_test = train_test_split(original)

# Divide target samples into training ant test parts
target_train, target_test = train_test_split(target)
original_weights_train = numpy.ones(len(original_train))
original_weights_test = numpy.ones(len(original_test))
hist_settings = {'bins': 100, 'normed': True, 'alpha': 0.7}

# Pay attention, actually we have very few data
len(original), len(target)
draw_distributions(original, target, original_weights)

# Train part of original distribution
draw_distributions(original_train, target_train, original_weights_train)

# Test part of target distribution
draw_distributions(original_test, target_test, original_weights_test)

# Gradient boosted Reweighter
reweighter = reweight.GBReweighter(n_estimators=50, learning_rate=0.1,
                                   max_depth=3, min_samples_leaf=1000,
                                   gb_args={'subsample': 0.4})
reweighter.fit(original_train, target_train)
gb_weights_test = reweighter.predict_weights(original_test)

# Validate reweighting rule on the test part comparing 1d projections
draw_distributions(original_test, target_test, gb_weights_test)
