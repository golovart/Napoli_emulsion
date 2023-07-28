## run example
# python -i hit_sandbox.py -mc /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000 -p 137 -t neutrino

## neutrino path: /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/137/sndLHC.Genie-TGeant4_digCPP.root

## neutron path: /eos/experiment/sndlhc/users/marssnd/PGsim/neutrons/neu_20_30_double/Ntuples/137/sndLHC.PG_2112-TGeant4_digCPP.root

## event metadata format: [Startz, MCTrack[0].GetPdgCode(), MCTrack[1].GetPdgCode()]


import ROOT
import os,sys,subprocess,atexit
import rootUtils as ut
from array import array
import shipunit as u
import SndlhcMuonReco
import json
from rootpyPickler import Unpickler
import time
from XRootD import client

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl


def pyExit():
       "unfortunately need as bypassing an issue related to use xrootd"
       os.system('kill '+str(os.getpid()))
atexit.register(pyExit)


A,B = ROOT.TVector3(),ROOT.TVector3()
freq      =  160.316E6

h={}
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-mc", "--inputMCDir", dest="inputMCDir", help="Monte Carlo input directory", required=False, default="/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volMuFilter_20fb-1_SNDG18_02a_01_000/")
parser.add_argument("-p", "--partition", dest="part", help="number of starting partition", default=0)
# parset.add_argument("-np", "--npartitions", dest="nParts", help="number of partitions to process", default=1)
parser.add_argument("-ne", "--nEvents", dest="nEvents", help="number of events to process", default=3000001)
parser.add_argument("-o", "--outPath", dest="outPath", help="output directory", required=False,
                    default="/afs/cern.ch/user/g/golovati/work/neutrino_ml/MC_explore/data/")
parser.add_argument("-t", "--type", dest="etype", help="event type to select", default='neutrino')

options = parser.parse_args()
import SndlhcGeo

# find geofile in the MC dir
MCDir = options.inputMCDir+'/'+options.part # '/'.join(options.inputMCFile.split('/')[:-1])
geo_path = None
for name in os.listdir(MCDir):
    if 'geofile' in name:
        geo_path = MCDir+'/'+name
if geo_path is None:
    raise RuntimeError("no geofile found in the MC directory")

geo = SndlhcGeo.GeoInterface(geo_path)
lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
lsOfGlobals.Add(geo.modules['Scifi'])
lsOfGlobals.Add(geo.modules['MuFilter'])
Scifi = geo.modules['Scifi']
Mufi = geo.modules['MuFilter']
nav = ROOT.gGeoManager.GetCurrentNavigator()

## Processing the geometry
detSize = {}
si = geo.snd_geo.Scifi
detSize[0] =[si.channel_width, si.channel_width, si.scifimat_z ]
mi = geo.snd_geo.MuFilter
detSize[1] =[mi.VetoBarX/2,                   mi.VetoBarY/2,            mi.VetoBarZ/2]
detSize[2] =[mi.UpstreamBarX/2,           mi.UpstreamBarY/2,    mi.UpstreamBarZ/2]
detSize[3] =[mi.DownstreamBarX_ver/2,mi.DownstreamBarY/2,mi.DownstreamBarZ/2]


print('\n\n-- Getting detector sizes --')
print(detSize)
A, B = ROOT.TVector3(), ROOT.TVector3()
proc_start_time = datetime.now()

## For MonteCarlo INPUT FILE
tchain = ROOT.TChain("cbmsim")

MCFile_path = None
for name in os.listdir(MCDir):
    if name.endswith('digCPP.root'):
        MCFile_path = MCDir+'/'+name
if MCFile_path is None:
    raise RuntimeError("no MC digi file found in the MC directory")
tchain.Add(MCFile_path)

## OUTPUT FILE
out_path = options.outPath
out_path += '/'+options.etype+'/'+'/'.join(MCDir.split('/')[-2:])+'/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
    

nchan = {'scifi':1536, 'us':10, 'ds':60}
nplane = {'scifi':5, 'us':5, 'ds':4}

N_events = tchain.GetEntries()
print("N events:", N_events)
event_meta = np.zeros((N_events, 3))
hitmap = {}
hitmap['scifi'] = np.zeros((N_events, 2, nplane['scifi'], nchan['scifi']), dtype=bool) # use uint8?
hitmap['us'] = np.zeros((N_events, 2, nplane['us'], nchan['us']), dtype=bool)
hitmap['ds'] = np.zeros((N_events, 2, nplane['ds'], nchan['ds']), dtype=bool)

def scifi_array_id(detID):
    #/* STMRFFF
    # First digit S: 		station # within the sub-detector
    # Second digit T: 		type of the plane: 0-horizontal fiber plane, 1-vertical fiber plane
    # Third digit M: 		determines the mat number 0-2
    # Fourth digit S: 		SiPM number  0-3
    # Last three digits F: 	local SiPM channel number in one mat  0-127
    #*/
    # print('detID: ', detID)
    n_plane = (detID // 1000000) % 10
    n_vert = (detID // 100000) % 10
    n_chan = detID % 1000
    n_chan += 128 * ( (detID // 1000) % 10)
    n_chan += 128 * 4 * ( (detID // 10000) % 10)
    # print('n_vert: {}, n_plane: {}, n_chan: {}'.format(n_vert, n_plane, n_chan))
    return n_vert, n_plane-1, n_chan

def mufi_array_id(detID, vert=True):
    #int subsystem     = floor(fDetectorID/10000);
    #int plane             = floor(fDetectorID/1000) - 10*subsystem;
    #int bar_number   = fDetectorID%1000;
    # print('detID: ', detID)
    n_sys = detID // 10000
    n_plane = (detID // 1000) % 10
    n_chan = detID % 1000
    if n_sys==3 and vert:
        n_chan -= 60
    if n_sys==2 and vert:
        print('\n\t2 VERT\n')
    n_vert = int(vert)
    # print('n_sys: {}, n_vert: {}, n_plane: {}, n_chan: {}'.format(n_sys, n_vert, n_plane, n_chan))
    return n_sys, n_vert, n_plane, n_chan

scifi_depth = []
for i_event, event in tqdm(enumerate(tchain), total=N_events):
    event_pdg0 = event.MCTrack[0].GetPdgCode()
    if options.etype=='neutrino':
        if not ((np.abs(event_pdg0)//10)==1 and (np.abs(event_pdg0)%2)==0): continue
        #print('event ', i_event,' track0 type: ', event.MCTrack[0].GetPdgCode())
    if options.etype=='neutron':
        if not (event.MCTrack[0].GetPdgCode()==2112): continue
        #print('event ', i_event,' track0 type: ', event.MCTrack[0].GetPdgCode())
    
    event_meta[i_event] = (event.MCTrack[0].GetStartZ(), event.MCTrack[0].GetPdgCode(), event.MCTrack[1].GetPdgCode())
    
    for aHit in event.Digi_ScifiHits: # digi_hits:
        # if not aHit.isValid(): continue
        detID = aHit.GetDetectorID()
        vert = aHit.isVertical()
        geo.modules['Scifi'].GetSiPMPosition(detID, A, B)
        scifi_depth = np.append(scifi_depth, A.z())
        n_vert, n_plane, n_chan = scifi_array_id(detID)
        hitmap['scifi'][i_event, n_vert, n_plane, n_chan] = 1
        
    for aHit in event.Digi_MuFilterHits: # digi_hits:
        # if not aHit.isValid(): continue
        detID = aHit.GetDetectorID()
        vert = aHit.isVertical()
        n_sys, n_vert, n_plane, n_chan = mufi_array_id(detID, vert)
        if n_sys==2:
            hitmap['us'][i_event, n_vert, n_plane, n_chan] = 1
        if n_sys==3:
            hitmap['ds'][i_event, n_vert, n_plane, n_chan] = 1
    # if i_event>10: break


for name, hits in hitmap.items():
    np.save(out_path+name+'.npy', hits)
np.save(out_path+'event_metadata.npy', event_meta)

