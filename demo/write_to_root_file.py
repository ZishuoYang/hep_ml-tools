#!/usr/bin/env python

import sys
sys.path.insert(0, '..')

import numpy
from utils.io import array2root

# folding_weights.dtype = [('weight', numpy.float64)]
test_tree = numpy.array([(1, 2.5, 3.4), (4, 5, 6.8)],
                        dtype=[('a', int), ('b', float), ('c', float)])

# Write the new weights to a root tree
array2root(test_tree, 'test_tree.root', 'test_tree', mode='recreate')
