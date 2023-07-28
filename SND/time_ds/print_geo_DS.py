#python print_geo_DS.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root

#!/usr/bin/env python
import ROOT, array
ROOT.gROOT.SetBatch(True)
import os,sys,subprocess
import ctypes
from array import array
import rootUtils as ut
from scipy import stats
import math
import numpy as np
from tqdm import tqdm
#import pandas as pd
import matplotlib.pyplot as plt

import shipunit as u
c_light = u.speedOfLight


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--inputFile", dest="inputFile", help="single input file", required=False, default="sndLHC.Ntuple-TGeant4_dig.root")
parser.add_argument("-d", "--inputDigiFile", dest="inputDigiFile", help="single  digi input file", required=False, default="sndLHC.Ntuple-TGeant4_dig.root")
parser.add_argument("-mc", "--inputMCFile", dest="inputMCFile", help="single Monte Carlo input file", required=False, default="/eos/user/g/golovati/SND/data/time_track/MC_8Dec.root")
parser.add_argument("-p", "--partitions", dest="nParts", help="number of partitions", default=0)
parser.add_argument("-n", "--nEvents", dest="nEvents", help="number of events to process", default=2000001)
parser.add_argument("-g", "--geoFile", dest="geoFile", help="geofile", required=True)

options = parser.parse_args()
import SndlhcGeo
geo = SndlhcGeo.GeoInterface(options.geoFile)
lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
lsOfGlobals.Add(geo.modules['Scifi'])
lsOfGlobals.Add(geo.modules['MuFilter'])
Scifi = geo.modules['Scifi']
Mufi = geo.modules['MuFilter']
nav = ROOT.gGeoManager.GetCurrentNavigator()








## Processing the geometry
print('-- Getting detector coordinates --')
A,B=ROOT.TVector3(),ROOT.TVector3()

det_base = 30000
det_ids = list(map(int,np.ravel([np.arange(det_base+i*1000, det_base+i*1000+120, dtype=int) for i in range(3)])))
DS_coords, det_z = {}, {}
#for det_id in range(30000,30120):
for det_id in det_ids:
    Mufi.GetPosition(det_id,A,B)
    DS_coords[det_id] = {
    'Lx':A[0],
    'Ly':A[1],
    'Lz':A[2],
    'Rx':B[0],
    'Ry':B[1],
    'Rz':B[2],
    'Z_c':(B[2]+A[2])/2,
    'HV':abs(B[0]-A[0])>3
    }
# print(np.unique(np.floor(list(det_z.values())), return_counts=True))

## Printing detector bar coordinates
for det_id, feats in DS_coords.items():
    print('det_id:',det_id)
    for fname in feats.keys():
        if 'L' in fname or 'R' in fname:
            if 'x' in fname: print(end='\t')
            print(fname+':', np.around(feats[fname], decimals=1), end=' ')
            if 'z' in fname:
                print()
        elif fname=='HV':
            print('\t', fname, 'horiz' if feats[fname] else 'vert')
        else: print('\t', fname, np.around(feats[fname], decimals=1))
