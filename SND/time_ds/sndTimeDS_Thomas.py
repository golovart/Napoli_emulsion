# python sndRecoTrack_Ana.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root -f /afs/cern.ch/user/s/sii/public/for_Artem/sndsw_raw-0000_004612_muonReco.root -d /eos/experiment/sndlhc/convertedData/commissioning/TI18/run_004612/sndsw_raw-0000.root -p 1

# python sndTimeDS_Thomas.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root -f /afs/cern.ch/work/s/sii/public/forArtem/run4705/sndsw_raw-0000_4705_muonReco-1.root -d /eos/experiment/sndlhc/convertedData/physics/2022/run_004705/sndsw_raw-0000.root -p 1

## digi inside
# python sndTimeDS_Thomas.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root -f /afs/cern.ch/work/s/sii/public/forArtem/run4705_v2/sndsw_raw-100000_6_4705_muonReco.root -p 1

## all partitions
# python sndTimeDS_Thomas.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root -f /eos/user/s/sii/run4705/Trks2/sndsw_raw-700000_45_4705_muonReco.root -p all

# python sndRecoTrack_Ana.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root -mc /eos/user/g/golovati/SND/data/time_track/MC_8Dec.root -p 1

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

import shipunit as u

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--inputFile", dest="inputFile", help="single input file", required=False,
                    default="sndLHC.Ntuple-TGeant4_dig.root")
parser.add_argument("-d", "--inputDigiFile", dest="inputDigiFile", help="single  digi input file", required=False,
                    default="sndLHC.Ntuple-TGeant4_dig.root")
parser.add_argument("-mc", "--inputMCFile", dest="inputMCFile", help="single Monte Carlo input file", required=False,
                    default="/eos/user/g/golovati/SND/data/time_track/MC_8Dec.root")
parser.add_argument("-p", "--partitions", dest="nParts", help="number of partitions", default=0)
parser.add_argument("-n", "--nEvents", dest="nEvents", help="number of events to process", default=3000001)
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
if options.nParts=='all':
    file_dir = options.inputFile.split('sndsw_raw')[0]
    for fname in os.listdir(file_dir):
        if not fname.endswith('.root'): continue
        tchain.Add(file_dir+fname)        
else:
    for p in range(0, int(options.nParts)):
        # inFileName= options.inputFile.replace("0000", str(p).zfill(4))
        inFileName = options.inputFile.replace("100000", str(p)+('0'*5 if p else ''))
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
croissant_times = [[{'bar': [], 'time': []} for j in range(2)] for i in range(4)]
for i_event, event in tqdm(enumerate(tchain), total=tchain.GetEntries()):

    if i_event > int(options.nEvents) - 1: break
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
                croissant_times[id0][id1]['bar'].append(id2 * 2 + i_side)
                croissant_times[id0][id1]['time'].append(t_ds_delta)

    nhits_extra_list.append(nhits_extra)
    nhits_record_list.append(nhits_record)
    # print('\n')

# Pyplot interactive mode
# plt.ion()

print('\nProcessing time:', datetime.now() - proc_start_time,'\n')

#fig, ax = plt.subplots(1, 2, figsize=(10, 5))

mean_times = []
bar_steps = [0]
for i in range(4):
    for j in range(2):
        if i==3 and j==0: continue
        for b in range(0,120,2):
            bar_mask = np.array(croissant_times[i][j]['bar'])==b
            if bar_mask.any():
                mu_gaus, std_gaus = scipy.stats.norm.fit(np.array(croissant_times[i][j]['time'])[bar_mask])
                mean_times.append(mu_gaus)
                # mean_times.append(np.mean(np.array(croissant_times[i][j]['time'])[bar_mask]))
            else:
                mean_times.append(mean_times[-1] if mean_times[-1]>-7 else -5)
        bar_steps.append(len(mean_times))
        if j: continue
        for b in range(1,120,2):
            bar_mask = np.array(croissant_times[i][j]['bar'])==b
            if bar_mask.any():
                mu_gaus, std_gaus = scipy.stats.norm.fit(np.array(croissant_times[i][j]['time'])[bar_mask])
                mean_times.append(mu_gaus)
                # mean_times.append(np.mean(np.array(croissant_times[i][j]['time'])[bar_mask]))
            else:
                mean_times.append(mean_times[-1] if mean_times[-1]>-7 else -5)
        bar_steps.append(len(mean_times))
chan_times = np.arange(len(mean_times))

print('bar steps: ',bar_steps, '\n'*5)

def line_c(x, A, x_c, y_c):
    return A*(x-x_c)+y_c


corr_times = np.array(mean_times).copy()

plt.figure(figsize=(11,4))
if not offset_corrected:
    time_slopes, median_times = [], []
    corr_off = []
    for i in range(len(bar_steps)-1):
        if (bar_steps[i+1] - bar_steps[i])<2: continue
        for left, right in [(bar_steps[i], (bar_steps[i+1] + bar_steps[i])//2), ((bar_steps[i+1] + bar_steps[i])//2, bar_steps[i+1]-6)]:
            chan_slice = chan_times[left:right]
            chan_x_slice = chan_slice%left if left>0 else chan_slice
            time_slice = mean_times[left:right]
            mean_slice = np.median(time_slice)
            x_c = (right-left)/2 # chan_x_slice[-1]/2
            saw_line = lambda x, A: line_c(x, A, x_c, mean_slice)
            # print('x_c:', chan_x_slice[-1]/2)
            par_opt, par_cov = curve_fit(saw_line, chan_x_slice, time_slice)
            time_slopes.append(par_opt[0])# if (bar_steps[i+1] - bar_steps[i])<100 else par_opt[0]*2)
            median_times.append(mean_slice)
            corr_times[left:right] = corr_times[left:right] - saw_line(chan_x_slice,np.sign(par_opt[0])*0.083)
            corr_off.append(np.median(corr_times[left:right]))
            
            plt.plot(chan_slice, saw_line(chan_x_slice,par_opt[0]), 'r')
            plt.plot(chan_slice, saw_line(chan_x_slice,np.sign(par_opt[0])*0.083), 'g')
            plt.plot(chan_slice, np.ones(len(chan_slice))*mean_slice, color='magenta')
            print(left, right,':\t', np.around(par_opt[0], decimals=3), np.around(mean_slice, decimals=3),'\n')
        
    print()
    print('Slopes: ', np.around(time_slopes, decimals=3))
    print('Mean_abs slope: ', np.around(np.abs(time_slopes).mean(), decimals=4))
    print('Offsets: ', np.around(median_times, decimals=3))

    np.savetxt('gaus_offsets_ds.txt', median_times, fmt='%.3f')   

plt.step(chan_times, mean_times, alpha=0.5)
plt.grid()
plt.minorticks_on()
if offset_corrected:
    plt.ylim(-2,2)
else:
    plt.ylim(-8,-3)
plt.title('Median times')
plt.savefig('gaus_time_align_DS.png')
plt.show()

print('\nCorrected offsets: ', np.around(corr_off, decimals=3))
print('Corrected times spread:', np.around(np.std(corr_times[np.abs(corr_times)<1]), decimals=3))

plt.step(chan_times, corr_times, alpha=0.5)
plt.grid()
plt.minorticks_on()
plt.ylim(-2,2)
plt.title('Corrected times')
plt.savefig('gaus_corr_time_align_DS.png')
plt.show()

'''
median_times = []
for i in range(len(bar_steps)-1):
    if (bar_steps[i+1] - bar_steps[i])<2: continue
    left = bar_steps[i]
    right = (bar_steps[i+1] + bar_steps[i])//2
    chan_slice = chan_times[left:right]
    chan_x_slice = chan_slice%left if left>0 else chan_slice
    time_slice = np.array(mean_times[left:right])
    # print(chan_slice)
    # print(time_slice)
    par_opt, par_cov = curve_fit(line, chan_x_slice, time_slice)
    #time_slopes.append(par_opt[0] if (bar_steps[i+1] - bar_steps[i])<100 else par_opt[0]*2)
    mean_slice = np.median(time_slice)
    median_times.append(mean_slice)
    plt.step(chan_slice, time_slice - line(chan_x_slice,par_opt[0],par_opt[1]) + mean_slice, color='blue', alpha=0.5)
    plt.plot(chan_slice, np.ones(len(chan_slice))*mean_slice, 'r')
    
    left = (bar_steps[i+1] + bar_steps[i])//2
    right = bar_steps[i+1]-10
    chan_slice = chan_times[left:right]
    chan_x_slice = chan_slice%left if left>0 else chan_slice
    time_slice = np.array(mean_times[left:right])
    # print(chan_slice)
    # print(time_slice)
    par_opt, par_cov = curve_fit(line, chan_x_slice, time_slice)
    #time_slopes.append(par_opt[0] if (bar_steps[i+1] - bar_steps[i])<100 else par_opt[0]*2)
    mean_slice = np.median(time_slice)
    median_times.append(mean_slice)
    plt.step(chan_slice, time_slice - line(chan_x_slice,par_opt[0],par_opt[1]) + mean_slice, color='blue', alpha=0.5)
    plt.plot(chan_slice, np.ones(len(chan_slice))*mean_slice, 'r')


# print('Mean_abs slope: ', np.around(np.abs(time_slopes).mean(), decimals=3))

plt.grid()
plt.minorticks_on()
plt.ylim(-7,-4)
plt.title('Median times corrected')
plt.savefig('mean_time_corr_align_DS.png')
plt.show()
'''

## Croissant plots
# croissant_hists = [
#     [np.array([np.histogram(croissant_times[i][j][k], bins=20, range=(hist_low, hist_up))[0] for k in range(120)]) for j
#      in range(2)] for i in range(3)]

# hist_bins = max(int(np.abs(hist_up-hist_low)/2 + 1), 30)
if not offset_corrected:
    hist_low, hist_up, hist_bins = -10., 0., 40
else:
    hist_low, hist_up, hist_bins = -3., 3., 24
croissant_hists = [
    [np.histogram2d(croissant_times[i][j]['time'], croissant_times[i][j]['bar'], bins=[hist_bins, 120],
                    range=[[hist_low, hist_up], [0, 120]])
     for j in range(2)]
    for i in range(4)]

# print('\n\n', croissant_hists[1][1], '\n\n')

vmin = np.min([[np.min(croissant_hists[i][j][0]) for j in range(2)] for i in range(4)])
vmax = np.max([[np.max(croissant_hists[i][j][0]) for j in range(2)] for i in range(4)])
print('Croissant plots. Min:', vmin, 'Max:', vmax)
fig, ax = plt.subplots(4, 2, figsize=(5, 18), sharex=True)
for id0 in range(4):
    for id1 in range(2):
        X, Y = np.meshgrid(croissant_hists[id0][id1][1], croissant_hists[id0][id1][2])
        # ax[id0, id1].pcolormesh(X, Y, croissant_hists[id0][id1][0].T, norm=mpl.colors.SymLogNorm(linthresh=1e-7*vmax, linscale=0.01, vmin=vmin, vmax=vmax))
        ax[id0, id1].pcolormesh(X, Y, croissant_hists[id0][id1][0].T, vmin=vmin, vmax=vmax)
        # ax[id0, id1].set_xlabel('t, bins')
        ax[id0, id1].set_ylabel('bar')
        ax[id0, id1].set_title(('Vert ' if id1 else 'Horiz ') + str(id0))
# plt.suptitle('['+str(np.around(hist_low, decimals=1))+','+str(np.around(hist_up, decimals=1))+']')
plt.savefig('log_time_croissant_DS.png')
plt.show()

# plt.ioff()
# plt.show()


print('Done')

