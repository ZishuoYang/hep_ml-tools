#!/usr/bin/env python

import sys
sys.path.insert(0, '../..')

import root_numpy
import pandas
import numpy

from hep_ml import reweight
from sklearn.cross_validation import train_test_split
from pathlib import Path
from shutil import copyfile

from utils.plot import draw_distributions
from utils.io import array2root
from root_numpy import array2root

###############
# Import data #
###############

columns = ['Bu_P', 'Bu_PT',
           'Ku_P', 'Ku_PT',]

original = root_numpy.root2array('BjpsiK_skim.root', branches=columns)
original = pandas.DataFrame(original)
target = root_numpy.root2array('BKPi_MC_skim.root', branches=columns)
target = pandas.DataFrame(target)

original_weights = numpy.ones(len(original))
target_weights = numpy.ones(len(target))

# We'll be applying our reweight to this dataset
toReweight = original

##################################
# Prepare train and test samples #
##################################

# Divide datasets into train and test parts
# original_train, original_test = train_test_split(original)
# target_train, target_test = train_test_split(target)

# Initial weighting---uniform
# original_weights_train = numpy.ones(len(original_train))
# original_weights_test = numpy.ones(len(original_test))

# Plot settings
hist_settings = {'bins': 60, 'density': True, 'alpha': 0.65}


###############################
# Gradient boosted Reweighter #
###############################

# reweighter = reweight.GBReweighter(n_estimators=100, learning_rate=0.1,
#                                    max_depth=3, min_samples_leaf=100,
#                                    )
# reweighter.fit(original_train, target_train)
# gb_weights_test = reweighter.predict_weights(original_test)
#
# # Validate reweighting rule on the test part comparing 1d projections
# draw_distributions('n100gbr4_validate_BuP.png',
#                    [columns[0],], original_test, target_test, gb_weights_test,
#                    filename_as_title=True,
#                    nrows=1, ncols=1, hist_settings=hist_settings)
#
# draw_distributions('n100gbr4_validate_BuPT.png',
#                    [columns[1]], original_test, target_test, gb_weights_test,
#                    filename_as_title=True,
#                    nrows=1, ncols=1, hist_settings=hist_settings)
#
# draw_distributions('n100gbr4_validate_KuP.png',
#                    [columns[2]], original_test, target_test, gb_weights_test,
#                    filename_as_title=True,
#                    nrows=1, ncols=1, hist_settings=hist_settings)
#
# draw_distributions('n100gbr4_validate_KuPT.png',
#                    [columns[3]], original_test, target_test, gb_weights_test,
#                    filename_as_title=True,
#                    # yscale=('log',),
#                    nrows=1, ncols=1, hist_settings=hist_settings)

######################
# Folding Reweighter #
######################

# Define base reweighter
reweighter_base = reweight.GBReweighter(n_estimators=120,
                                        learning_rate=0.1, max_depth=3,
                                        min_samples_leaf=5000,
                                       )

reweighter = reweight.FoldingReweighter(reweighter_base, n_folds=2)

# Not need to divide data into train/test parts
reweighter.fit(original, target, target_weight=target_weights)

# Prediect weights for the input file
folding_weights = reweighter.predict_weights(toReweight)

draw_distributions('GBR4_validate.png',
                   columns, toReweight, target, folding_weights,
                   filename_as_title=True,
                   # yscale=('log',),
                   nrows=2, ncols=2, hist_settings=hist_settings)

draw_distributions('GBR4_before_after.png',
                   columns, toReweight, toReweight, None, folding_weights,
                   filename_as_title=True,
                   yscale=('linear','linear','linear','linear'),
                   nrows=2, ncols=2, hist_settings=hist_settings)

# Need to provide a column name
folding_weights.dtype = [('weight_ml', numpy.float64)]

# Create a copy for consistency
copyfile(Path('BjpsiK_skim.root'),
         Path('BjpsiK_skim_4dGBReweighted.root'))

# Write the new weights to the same root tree
array2root(folding_weights,
           'BjpsiK_skim_4dGBReweighted.root', 'DecayTree', mode='update')
