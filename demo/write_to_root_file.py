#!/usr/bin/env python

import root_numpy
import numpy

# folding_weights.dtype = [('weight', numpy.float64)]
test_tree = numpy.array([(1, 2.5, 3.4), (4, 5, 6.8)],
                        dtype=[('a', int), ('b', float), ('c', float)])

# Write the new weights to a root tree
root_numpy.array2root(test_tree,
                      'test_tree.root', 'test_tree', mode='recreate')
