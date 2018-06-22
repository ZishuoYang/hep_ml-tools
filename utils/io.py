#!/usr/bin/env python

import ROOT
from root_numpy import array2tree


def array2root(array, filename,
               treename='tree',
               mode='update', compression='zlib'):
    if compression == 'zlib':
        file = ROOT.TFile(filename, mode, "", 101)
    else:
        file = ROOT.TFile(filename, mode)

    tree = array2tree(array, name=treename)

    file.Write()
    file.Close()
