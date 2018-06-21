#!/usr/bin/env python

import sys
sys.path.insert(0, '..')

import root_numpy
import pandas
import numpy

from hep_ml import reweight
from sklearn.cross_validation import train_test_split

from utils.plot import draw_distributions
from utils.stats import print_statistics

###############
# Import data #
###############

columns = ['hSPD', 'pt_b', 'pt_phi', 'vchi2_b', 'mu_pt_sum']

original = root_numpy.root2array('MC_distribution.root', branches=columns)
original = pandas.DataFrame(original)
target = root_numpy.root2array('RD_distribution.root', branches=columns)
target = pandas.DataFrame(target)

original_weights = numpy.ones(len(original))


##################################
# Prepare train and test samples #
##################################

# Divide original samples into training ant test parts
original_train, original_test = train_test_split(original)

# Divide target samples into training ant test parts
target_train, target_test = train_test_split(target)

original_weights_train = numpy.ones(len(original_train))
original_weights_test = numpy.ones(len(original_test))

# Pay attention, actually we have very few data
print('Length of original data: %s\nLength of target data: %s' % (
    len(original), len(target)
))


###################################################
# Print the unmodified original and test data set #
###################################################

draw_distributions('initial.png',
                   columns, original, target, original_weights)
print_statistics(columns, original, target, original_weights)

# Train part of original distribution
draw_distributions('initial_train.png',
                   columns, original_train, target_train, original_weights_train)

# Test part of target distribution
draw_distributions('initial_test.png',
                   columns, original_test, target_test, original_weights_test)


###############################
# Gradient boosted Reweighter #
###############################

reweighter = reweight.GBReweighter(n_estimators=50, learning_rate=0.1,
                                   max_depth=3, min_samples_leaf=1000,
                                   gb_args={'subsample': 0.4})
reweighter.fit(original_train, target_train)
gb_weights_test = reweighter.predict_weights(original_test)

# Validate reweighting rule on the test part comparing 1d projections
draw_distributions('gb_weights_test.png',
                   columns, original_test, target_test, gb_weights_test)
print_statistics(columns, original_test, target_test, original_weights_test)


######################
# Folding Reweighter #
######################

# Define base reweighter
reweighter_base = reweight.GBReweighter(n_estimators=50,
                                        learning_rate=0.1, max_depth=3,
                                        min_samples_leaf=1000,
                                        gb_args={'subsample': 0.4})
reweighter = reweight.FoldingReweighter(reweighter_base, n_folds=2)

# Not need to divide data into train/test parts
reweighter.fit(original, target)
folding_weights = reweighter.predict_weights(original)

draw_distributions('folding_weights.png',
                   columns, original, target, folding_weights)
print_statistics(columns, original, target, folding_weights)
