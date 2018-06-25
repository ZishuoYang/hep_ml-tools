#!/usr/bin/env python

import ROOT
from root_numpy import array2tree
from pathlib import Path


def array2root(array, filename,
               treename='tree', mode='update',
               compression='zlib'):
    # Stolen from 'root_numpy', in:
    #   root_numpy/src/tree.pyx

    # First, if the file is yet to exist, forcing 'recreate mode'
    target_file = Path(filename)
    if not target_file.is_file():
        mode = 'recreate'

    if compression == 'zlib':
        rfile = ROOT.TFile.Open(filename, mode, "", 101)
    else:
        rfile = ROOT.TFile.Open(filename, mode)

    if mode == 'update':
        tree = rfile.Get(treename)
        datatree = array2tree(array, name=treename, tree=tree)
    else:
        datatree = array2tree(array, name=treename)

    # Possible alternative write modes:
    #   ROOT.TObject.kWriteDelete, ROOT.TObject.kOverwrite
    rfile.Write("", ROOT.TObject.kOverwrite)
    # rfile.Write()

    rfile.Close()

    del datatree
    del rfile
