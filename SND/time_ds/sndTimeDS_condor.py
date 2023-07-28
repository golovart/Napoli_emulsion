## digi inside
# python sndTimeDS_Thomas.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root -f /afs/cern.ch/work/s/sii/public/forArtem/run4705_v2/sndsw_raw-100000_6_4705_muonReco.root -p 1

## all partitions
# python sndTimeDS_Thomas.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root -f /eos/user/s/sii/run4705/Trks3/sndsw_raw-5_4705_muonReco.root -p all

# !/usr/bin/env python
import ROOT, array

ROOT.gROOT.SetBatch(True)
import os, sys, subprocess
import ctypes
from array import array
# import rootUtils as ut
import scipy
import math
import numpy as np
from tqdm import tqdm
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from scipy.optimize import curve_fit
import scipy.stats #.norm
# import h5py

import shipunit as u

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--inputFile", dest="inputFile", help="single input file", required=False,
                    default="sndLHC.Ntuple-TGeant4_dig.root")
# parser.add_argument("-d", "--inputDigiFile", dest="inputDigiFile", help="single  digi input file", required=False,
#                     default="sndLHC.Ntuple-TGeant4_dig.root")
# parser.add_argument("-mc", "--inputMCFile", dest="inputMCFile", help="single Monte Carlo input file", required=False,
#                     default="/eos/user/g/golovati/SND/data/time_track/MC_8Dec.root")
parser.add_argument("-p", "--partition", dest="nPart", help="number of partition", default=0)
parser.add_argument("-n", "--nEvents", dest="nEvents", help="number of events to process", default=3000001)
parser.add_argument("-g", "--geoFile", dest="geoFile", help="geofile", required=True)
parser.add_argument("-o", "--outFile", dest="outFile", help="output directory", required=False,
                    default="/afs/cern.ch/user/g/golovati/private/track_extr/condor_time_4705/numpy_out/ds_detid_time_p5_4705.npy")

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
# print('-- Getting detector coordinates --')
A, B = ROOT.TVector3(), ROOT.TVector3()
proc_start_time = datetime.now()

# Set mandatory items for genfit::Extrapolate* methods
# No magnetic field and assuming no (negligible) multiple scattering
geoMat = ROOT.genfit.TGeoMaterialInterface()
bfield = ROOT.genfit.ConstField(0, 0, 0)  # constant field of zero
fM = ROOT.genfit.FieldManager.getInstance()
fM.init(bfield)
ROOT.genfit.MaterialEffects.getInstance().init(geoMat)
ROOT.genfit.MaterialEffects.getInstance().setNoEffects()


# def re_trackDir(aTrack):
#    pos = aTrack.getStart()
#    mom = aTrack.getTrackMom()
#    lam = (pos_zero_scifi - pos.z())/mom.z()
#    # nominal first position
#    pos1 = ROOT.TVector3(pos.x()+lam*mom.x(), pos.y()+lam*mom.y(), pos_zero_scifi)
    
    

### calculation constants
c_light = u.speedOfLight
TDC2ns = 1E9 / 160.316E6
ds_speed = lsOfGlobals.FindObject('MuFilter').GetConfParF("MuFilter/DsPropSpeed")
pos_zero_scifi = 300*u.cm
slope = 0.083
offset_corrected = False # os.path.exists('gaus_offsets_ds.txt')
    


# Get input files
# For the moment track files contain only track information
# tchain = ROOT.TChain("cbmsim")
# tchain.Add(options.inputMCFile)
tchain = ROOT.TChain("rawConv")
# tree_digi = ROOT.TChain("rawConv")
if options.nPart=='all':
    file_dir = options.inputFile.split('sndsw_raw')[0]
    for fname in os.listdir(file_dir):
        if not fname.endswith('.root'): continue
        tchain.Add(file_dir+fname)        
else:
    p = options.nPart
    inFileName = options.inputFile+'/sndsw_raw-'+p+'_4705_muonReco.root'
    tchain.Add(inFileName)
    # tchain.SetBranchStatus("EventHeader",0)
    # inDigiFileName = options.inputDigiFile.replace("0000", str(p).zfill(4))
    # tree_digi.Add(inDigiFileName)
    # tree_digi.SetAlias("SNDEventHeader","EventHeader")
# tchain.AddFriend(tree_digi)
# tchain.SetBranchStatus("EventHeader",1)

# x = options.inputDigiFile
# filename = x[x.rfind('run_') + 4:x.rfind('run_') + 10]
# outFileName = 'recTrk_' + filename + '.root'
# file = ROOT.TFile(outFileName, 'recreate')

t_delta_horiz, t_delta_vert = [], []
fullDS_ids = set()
allDS_ids = set()
nhits_extra_list, nhits_record_list = [], []

mean_offsets = None
if offset_corrected:
    mean_offsets = np.loadtxt('gaus_offsets_ds.txt')
    
def line_c(x, A, x_c, y_c):
    return A*(x-x_c)+y_c

# croissant_times = [[[[] for k in range(120)] for j in range(2)] for i in range(3)]

data_header = ['detID', 'id_side', 'delta_t', 'isB1', 'isB2', 'isB1only', 'isB2notB1']
np_times = np.zeros((0,7))
#croissant_times = [[{'bar': [], 'time': []} for j in range(2)] for i in range(4)]
for i_event, event in tqdm(enumerate(tchain), total=tchain.GetEntries()):

    # if i_event > int(options.nEvents) - 1: break
    # if i_event < 250000: continue
    # if i_event > 200000: break
    # if not event.EventHeader.isB1(): continue

    nhits_extra = 0
    nhits_record = len([1 for aHit in event.Digi_MuFilterHits if aHit.isValid() and aHit.GetDetectorID() // 10000 == 3])

    for aTrack in event.Reco_MuonTracks:
        ### checking that track is from SciFi for extrapolation
        if not aTrack.getTrackFlag() or not (
                aTrack.getTrackType() == 1): continue  # or aTrack.getTrackType() == 11): continue  #
        # allDS_ids.add(i_event)
        posTrack = aTrack.getStart()
        mom = aTrack.getTrackMom()
        pos_zero = 300*u.cm # aTrack.getTrackPoints()[0]
        lam = (pos_zero - posTrack.z())/mom.z()
        pos1 = ROOT.TVector3(posTrack.x() + lam*mom.x(), posTrack.y() + lam*mom.y(), pos_zero)
        # print(pos_zero-posTrack[2])

        # trackTimes = aTrack.getCorrTimes()
        #print('\nEvent ID:\t', i_event, '\nTrack type:', aTrack.getTrackType(), '\nTrack Corr Times:\t', trackTimes,
        #      '\n', type(trackTimes))
        # Tlast = trackTimes[-1]  # last SciFi track point time
        rc = aTrack.trackDir()  # following Thomas notation on first SciFi time
        Tfirst = rc[2]

        # Zlast = aTrack.getStop()[2]
        pos_last = aTrack.getStop()
        # mom = aTrack.getFittedState().getMom()
        # TDC2ns = 1E9 / 160.316E6
        # ds_speed = lsOfGlobals.FindObject('MuFilter').GetConfParF("MuFilter/DsPropSpeed")

        # flag_fullDS = True

        for aHit in event.Digi_MuFilterHits:
            if not aHit.isValid(): continue
            detID = aHit.GetDetectorID()
            if detID // 10000 != 3: continue  # checking it's a DS hit
            Mufi.GetPosition(detID, A, B)  # A=Left, B=Right for horiz
            zEx = (A.Z() + B.Z()) / 2.  # A=Top, B=Bottom for vert
            lam = (zEx - pos_last.z()) / mom.z()  # Following Thomas notations
            xEx, yEx = pos_last.x() + lam * mom.x(), pos_last.y() + lam * mom.y()  # instead of extrapolate

            posEx = ROOT.TVector3(xEx, yEx, zEx)  # aTrack.extrapolateToPlaneAtZ(zEx)
            uCrossv = (B - A).Cross(mom)
            doca = (A - posEx).Dot(uCrossv) / uCrossv.Mag()
            if np.abs(doca) > 1.0*u.cm: continue  # track far from the hit-bar, should be 1 or 2.5?
            nhits_extra += 1

            # t_ds = Tlast + (posEx - pos_last).Mag() / c_light  # np.abs(posEx[2] - pos_last[2])
            t_ds = Tfirst + (posEx - pos1).Mag()/c_light # np.abs(posEx[2] - posTrack[2]) / c_light  # Using just Z difference from the first SciFi hit

            for i_side in range(2 - aHit.isVertical()):
                # side_coord = A if (aHit.isVertical or not i_side) else B
                t_signal_prop = (posEx - (B if i_side else A)).Mag() / ds_speed
                t_ds_hit_read = aHit.GetTime(i_side) * TDC2ns
                if t_ds_hit_read<0: continue
                t_ds_delta = t_ds_hit_read - t_ds - t_signal_prop

                # print('\tdetID:', detID)
                # print('extra:', t_ds + t_signal_prop, '\thit:', t_ds_hit_read)
                #if aHit.isVertical():
                #    t_delta_vert.append(t_ds_delta)
                #else:
                #    t_delta_horiz.append(t_ds_delta)
                id0 = (detID - 30000) // 1000
                id1 = ((detID - 30000) % 1000) // 60
                id2 = ((detID - 30000) % 1000) % 60
                id_half = (((detID - 30000) % 1000) % 60) // 30
                id_off = ((id0*3 if id0<3 else 7) + (id1*2) + i_side)*2 + id_half
                
                if not (mean_offsets is None):
                    time_off = mean_offsets[id_off]
                    t_ds_delta -= line_c(id2%30, slope*(-1 if not id_half else 1), x_c=(12 if id_half else 15), y_c=time_off)
                
                # croissant_times[id0][id1][id2 * 2 + i_side].append(t_ds_delta)
                # croissant_times[id0][id1]['bar'].append(id2 * 2 + i_side)
                # croissant_times[id0][id1]['time'].append(t_ds_delta)
                np_times = np.vstack((np_times, [detID, id2 * 2 + i_side, t_ds_delta, event.EventHeader.isB1(), event.EventHeader.isB2(), event.EventHeader.isB1Only(), event.EventHeader.isB2noB1()]))

    # nhits_extra_list.append(nhits_extra)
    # nhits_record_list.append(nhits_record)
    # print('\n')

# Pyplot interactive mode
# plt.ion()

np.save(options.outFile, np_times)

print('\nProcessing time:', datetime.now() - proc_start_time,'\n')



print('Done')

