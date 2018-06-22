#!/usr/bin/env python

import sys
sys.path.insert(0, '../..')

import root_numpy
import pandas
import numpy

from hep_ml import reweight
from sklearn.cross_validation import train_test_split

from utils.plot import draw_distributions

###############
# Import data #
###############

columns = ['Bu_LOKI_CONE2MULT', 'Bu_LOKI_modNumVertsOneTrack']

original = root_numpy.root2array('MC_K+Pi-.root', branches=columns)
original = pandas.DataFrame(original)
target = root_numpy.root2array('RD_K+Pi-_withSW.root', branches=columns)
target = pandas.DataFrame(target)

original_weights = numpy.ones(len(original))
target_sWeights = root_numpy.root2array('RD_K+Pi-_withSW.root',
                                        branches=['Nsig_sw'])

# We'll be applying our reweight to this dataset
toReweight = root_numpy.root2array('DVntuple_MC16_forGBR.root',
                                   branches=['Bu_CONE3MULT', 'Bu_VTXISONUMVTX'])
toReweight = pandas.DataFrame(toReweight)


##################################
# Prepare train and test samples #
##################################

# Divide datasets into train and test parts
original_train, original_test = train_test_split(original)
target_train, target_test = train_test_split(target)

# Initial weighting---uniform
original_weights_train = numpy.ones(len(original_train))
original_weights_test = numpy.ones(len(original_test))

# Plot settings
hist_settings = {'bins': 50, 'density': True, 'alpha': 0.5}


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
                   columns, original_test, target_test, gb_weights_test,
                   filename_as_title=True,
                   xlim=((0, 50), (0, 12)),
                   ylim=((0, 0.07), (0, 1.7)),
                   nrows=1, ncols=2, hist_settings=hist_settings)


######################
# Folding Reweighter #
######################

# Define base reweighter
reweighter_base = reweight.GBReweighter(n_estimators=50,
                                        learning_rate=0.1, max_depth=2,
                                        min_samples_leaf=1000,
                                        gb_args={'subsample': 0.4})

reweighter = reweight.FoldingReweighter(reweighter_base, n_folds=2)

# Not need to divide data into train/test parts
reweighter.fit(original, target, target_weight=target_sWeights)

# Prediect weights for the input file
folding_weights = reweighter.predict_weights(toReweight)

# Cast the sWeight array into float
cast_target_sWeights = target_sWeights.astype(float)

draw_distributions('GBReweight_before_after.png',
                   columns, toReweight, toReweight, folding_weights,
                   filename_as_title=True,
                   xlim=((0, 50), (0, 12)),
                   ylim=((0, 0.07), (0, 1.7)),
                   nrows=1, ncols=2, hist_settings=hist_settings)

# Need to provide a column name
folding_weights.dtype = [('weight', numpy.float64)]

# Write the new weights to the same root tree
array2root(folding_weights,
           'DVntuple_MC16_forGBR.root', 'DecayTree', mode='recreate')
