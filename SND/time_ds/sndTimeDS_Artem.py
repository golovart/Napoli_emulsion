#python sndRecoTrack_Ana.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root -f /afs/cern.ch/user/s/sii/public/for_Artem/sndsw_raw-0000_004612_muonReco.root -d /eos/experiment/sndlhc/convertedData/commissioning/TI18/run_004612/sndsw_raw-0000.root -p 1

# python sndTimeDS_Thomas.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root -f /afs/cern.ch/work/s/sii/public/forArtem/run4705/sndsw_raw-0000_4705_muonReco-1.root -d /eos/experiment/sndlhc/convertedData/commissioning/TI18/run_004705/sndsw_raw-0000.root -p 1

#python sndRecoTrack_Ana.py -g /eos/experiment/sndlhc/convertedData/commissioning/TI18/geofile_sndlhc_TI18_V5_14August2022.root -mc /eos/user/g/golovati/SND/data/time_track/MC_8Dec.root -p 1

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

## main Z-values: [492., 494., 516., 517., 543., 544.]
DS_zlist = [492.6, 494.1, 516.3, 517.8, 543.4, 544.9]


## groud det_ids by the closest common Z and H/V
DS_groupZid = {'H':{}, 'V':{}}
for hv in DS_groupZid.keys():
    for det_z in (DS_zlist[::2] if hv=='H' else DS_zlist[1::2]):
        DS_groupZid[hv][det_z] = []
        for det_id, feats in DS_coords.items():
            if abs(det_z-feats['Z_c'])<5 and feats['HV']==(hv=='H'): DS_groupZid[hv][det_z].append(det_id)

## Printing DS groups by Z
for hv, groupZhv in DS_groupZid.items():
    print(hv)
    for det_z, group_ids in groupZhv.items():
        print('\nZ:',det_z)
        print('\t', group_ids)
        print('\tmean Z:', np.around(np.mean([DS_coords[z_id]['Z_c'] for z_id in group_ids]), decimals=1))

# print(DS_zlist_1dig)


def inside_box(D, C, CL, CU, CB, horiz=True):
    '''
    D - datapoint; L - left; B - back; U - upper; C - center (front-down-right)
    '''
    CD = np.array(D) - C
    # CD, CL, CB, CU = np.array(CD), np.array(CL), np.array(CB), np.array(CU)
    # CB = B - C # orthogonal to Front plane
    # CU = U - C # orthogonal to Down plane
    # CL = L - C # orthogonal to Right plane
    # CD = D - C # vector towards data point
    
    # dist_front = np.dot(CB,CD)/np.linalg.norm(CB)
    dist_down = np.dot(CU,CD)/np.linalg.norm(CU)
    dist_right = np.dot(CL,CD)/np.linalg.norm(CL)
    ## Disabling "front plane" check (over Z), as track point use extrapolated coordinates from specific Z plane
    # inside_f = (dist_front>=0) and (dist_front<np.linalg.norm(CB))
    inside_f = True
    if horiz:
        inside_r = (dist_right>=0) and (dist_right<np.linalg.norm(CL))
        inside_d = (dist_down >= -np.linalg.norm(CU)/2) and (dist_down < np.linalg.norm(CU)/2)
    else:
        inside_r = (dist_right>=-np.linalg.norm(CL)/2) and (dist_right<np.linalg.norm(CL)/2)
        inside_d = (dist_down >= 0) and (dist_down < np.linalg.norm(CU))
    return inside_f and inside_d and inside_r

def get_club_coord(det_l, det_r, det_next, next=True, horiz=True):
    '''
    det_next -> det_r;
    for Vert - det_l = up, det_r = bot;
    out:  C, CL, CU, CB
    '''
    C = np.array(det_r)
    if horiz:
        CL = np.array(det_l) - C
        CU = np.array(det_next) - C
        if not next: CU = -CU
        width = np.linalg.norm(CU)
    else:
        CL = - np.array(det_next) + C
        if not next: CL = -CL
        CU = np.array(det_l) - C
        width = np.linalg.norm(CL)
    CB = np.cross(CL, CU)/np.linalg.norm(CL)/np.linalg.norm(CU)*width #*np.sign(CL[0])
    return C, CL, CU, CB, horiz


def track_cross_detid(pos, hv, DS_coords, DS_groupZid):
    """
    pos -- 3-vector of extrapolated track intersection for a specified Z.
    DS_coords -- dictionary (dataframe) with coords for each DS detector id.
    DS_groupZid -- dictionary with DS ids grouped over Z and horiz/vert types to loop only over nearby plates.
    """
    det_zgroup = pos[2]
    for det_z in DS_groupZid[hv].keys():
        if abs(det_z-det_zgroup)<5: det_zgroup = det_z
    z_idgroup = DS_groupZid[hv][det_zgroup]
    
    for z_id in z_idgroup:
        # print('\t', z_id)
        bar_coords = DS_coords[z_id]
        if z_id+1 in DS_coords.keys():
            next_coords = DS_coords[z_id+1] if DS_coords[z_id+1]['HV']==bar_coords['HV'] else DS_coords[z_id-1]
            next = DS_coords[z_id+1]['HV']==bar_coords['HV']
        else:
            next_coords = DS_coords[z_id-1]
            next = False
        det_l = [bar_coords['Lx'], bar_coords['Ly'], bar_coords['Lz']]
        det_r = [bar_coords['Rx'], bar_coords['Ry'], bar_coords['Rz']]
        #if bar_coords['HV']:
        det_next = [next_coords['Rx'], next_coords['Ry'], next_coords['Rz']]
        #else:
        #    det_next = [next_coords['Lx'], next_coords['Ly'], next_coords['Lz']]
        det_found = inside_box(pos, *get_club_coord(det_l, det_r, det_next, next=next, horiz=bar_coords['HV']))
        
        if det_found:
            return z_id
    print('-- for extrapolated position:',np.around(pos, decimals=1),'no DS intersaction found --')
    return -1

print('\n\n')
# 30054
z_id = track_cross_detid([-0.5, 63.7, 492.9], hv='H', DS_coords=DS_coords, DS_groupZid=DS_groupZid)
print('for [-0.5, 63.7, 492.9]  z_id:', z_id)
if z_id>0: print('horiz' if DS_coords[z_id]['HV'] else 'vert')
if z_id>0: print('HV', DS_coords[z_id]['HV'])
# 32032
z_id = track_cross_detid([-3.5, 40.7, 543.8], hv='H', DS_coords=DS_coords, DS_groupZid=DS_groupZid)
print('for [-3.5, 40.7, 543.8]  z_id:', z_id)
if z_id>0: print('horiz' if DS_coords[z_id]['HV'] else 'vert')
# 30099
z_id = track_cross_detid([-36.7, 14.5, 494.4], hv='V', DS_coords=DS_coords, DS_groupZid=DS_groupZid)
print('for [-36.7, 14.5, 494.4]  z_id:', z_id)
if z_id>0: print('horiz' if DS_coords[z_id]['HV'] else 'vert')
# 30100
z_id = track_cross_detid([-38.2, 43.6, 494.4], hv='V', DS_coords=DS_coords, DS_groupZid=DS_groupZid)
print('for [-38.2, 43.6, 494.4]  z_id:', z_id)
if z_id>0: print('horiz' if DS_coords[z_id]['HV'] else 'vert')









#Set mandatory items for genfit::Extrapolate* methods
#No magnetic field and assuming no (negligible) multiple scattering  
geoMat =  ROOT.genfit.TGeoMaterialInterface()
bfield     = ROOT.genfit.ConstField(0,0,0)   # constant field of zero
fM = ROOT.genfit.FieldManager.getInstance()
fM.init(bfield)
ROOT.genfit.MaterialEffects.getInstance().init(geoMat)
ROOT.genfit.MaterialEffects.getInstance().setNoEffects()

h={}

ut.bookHist(h,'corrTimes','Hit times after signal propagation correction; corr_times [ns]',100, -1, 26)
ut.bookHist(h,'slope_zx_forward','Direction Scifi-> DS: Rec. angle in ZX; tan#theta_{xz}', 2200,-1.1,1.1)
ut.bookHist(h,'slope_zx_backward','Direction DS -> Scifi: Rec. angle in ZX; tan#theta_{xz}', 2200,-1.1,1.1)
ut.bookHist(h,'slope_zx_posVel','Direction Scifi-> DS based on velocity sign: Rec. angle in ZX; tan#theta_{xz}', 2200,-1.1,1.1)
ut.bookHist(h,'slope_zx_Sf','SciFi tracks: Rec. angle in ZX; tan#theta_{xz}', 2200,-1.1,1.1)
ut.bookHist(h,'slope_zy_Sf','SciFi tracks: Rec. angle in ZX; tan#theta_{yz}', 2200,-1.1,1.1)
ut.bookHist(h,'slopes_DS','DS tracks: 2D reconstructed angles; tan#theta_{xz}; tan#theta_{yz}', 2200,-1.1,1.1, 2200,-1.1,1.1)

# Get input files
# For the moment track files contain only track information
tchain = ROOT.TChain("cbmsim")
#tchain.Add(options.inputMCFile)
tchain = ROOT.TChain("rawConv")
tree_digi = ROOT.TChain("rawConv")
for p in range(0,int(options.nParts)):
    # inFileName= options.inputFile.replace("0000", str(p).zfill(4))
    inFileName= options.inputFile.replace("0000", str(p))
    tchain.Add(inFileName)
    inDigiFileName= options.inputDigiFile.replace("0000", str(p).zfill(4))
    tree_digi.Add(inDigiFileName)    
tchain.AddFriend(tree_digi)

x = options.inputDigiFile
filename = x[x.rfind('run_')+4:x.rfind('run_')+10]
outFileName = 'recTrk_'+filename+'.root'
file = ROOT.TFile(outFileName, 'recreate')


t_delta_horiz, t_delta_vert = [],[] 
fullDS_ids = set()
allDS_ids = set()
nhits_extra_list, nhits_record_list = [], []

croissant_times = [[[[] for k in range(120)] for j in range(2)] for i in range(3)]
for i_event, event in tqdm(enumerate(tchain), total=tchain.GetEntries()):

    if i_event > int(options.nEvents)-1: break
    # if i_event: break
    # if i_event < 250000: continue
    if i_event > 2000: break
    # if i_event != 9929: continue
    # if i_event > 200000 : break
    # if i_event%100000==0 : print(i_event)
    # print('\n'*10)
    
    ### checking that event has at least some DS track
    #noDSflag = True
    #for aTrack in event.Reco_MuonTracks:
    #    if aTrack.getTrackType() == 3 or aTrack.getTrackType() == 13:
    #        noDSflag = False
    #if noDSflag: continue
    
    
    # example how to read hit data
    #for aHit in event.Digi_MuFilterHits:
    #    # aHit.Print()
    #    ### DS hits are MuFilter system 3
    #    if not aHit.GetSystem()==3: continue
    #    ### only horizontal bars have readout on both sides
    #    if not aHit.isVertical(): 
    #        time = aHit.GetImpactT()
    #        # print('\thit time:',time,'\n')
            

    for aTrack in event.Reco_MuonTracks:
        ### checking that track is from SciFi for extrapolation
        if not aTrack.getTrackFlag() or not (aTrack.getTrackType() == 1 or aTrack.getTrackType() == 11): continue #
        allDS_ids.add(i_event)
        posTrack = aTrack.getStart()
        mom = aTrack.getTrackMom()
        slopeX= aTrack.getSlopeXZ()
        slopeY= aTrack.getSlopeYZ()
        
        ### how to call extrapolate method
        # If this method is used, one has to add the detector geometries
        # to the list of globals and set fields and effects for genfit::Extrapolate*, as done above.
        # x_extrap = aTrack.extrapolateToPlaneAtZ(350.).X()

        ### an example reading class member objects of type vector - these all have the same size!
        # trackTimes = aTrack.getCorrTimes()
        # for i in range( len(aTrack.getTrackPoints()) ):
        #    h['corrTimes'].Fill(trackTimes[i])

        trackTimes = aTrack.getCorrTimes()
        print('\nEvent ID:\t',i_event, '\nTrack type:', aTrack.getTrackType(),'\nTrack Corr Times:\t', trackTimes,'\n',type(trackTimes))
        Tlast = trackTimes[-1] # last SciFi track point time
        #Zlast = aTrack.getStop()[2]
        pos_last = np.array(list(aTrack.getStop()))
        
        flag_fullDS = True
        nhits_extra = 0
        nhits_record = 0
        for DS_z in DS_zlist:
            hv = 'H' if DS_z in DS_groupZid['H'].keys() else 'V'                
            
            #t_ds = Tlast + abs(DS_z - Zlast)/c_light
            
            DS_track_pos = np.array(list(aTrack.extrapolateToPlaneAtZ(DS_z)))
            t_ds = Tlast + np.linalg.norm(DS_track_pos - pos_last)/c_light
            
            z_id = track_cross_detid(DS_track_pos, hv=hv, DS_coords=DS_coords, DS_groupZid=DS_groupZid)
            if z_id<0:
                flag_fullDS = False
                continue
            nhits_extra += 1
            print('for ',np.around(DS_track_pos, decimals=1),'  z_id:', z_id)
            if z_id>0: print('horiz' if DS_coords[z_id]['HV'] else 'vert')
            
            for i_side in range(1+DS_coords[z_id]['HV']):
                t_ds_hit_read = None
                for aHit in event.Digi_MuFilterHits:
                    #if aHit.GetDetectorID() in [z_id-1, z_id, z_id+1]:
                    #    if not DS_coords[z_id]['HV']==DS_coords[aHit.GetDetectorID()]['HV']: continue
                    if aHit.GetDetectorID()==z_id:
                        #t_ds_hit_read = aHit.GetAllTimes()[0]*6.25
                        t_ds_hit_read = aHit.GetTime(i_side)*6.25
                        #z_id = aHit.GetDetectorID()
                        print('\thit det ID:',aHit.GetDetectorID())
                if t_ds_hit_read is None: continue
                nhits_record += 1
                ds_speed = lsOfGlobals.FindObject('MuFilter').GetConfParF("MuFilter/DsPropSpeed")
                if (not DS_coords[z_id]['HV']) or (not i_side):
                    DS_coord_det = np.array([DS_coords[z_id]['Lx'],DS_coords[z_id]['Ly'],DS_coords[z_id]['Lz']])
                else:
                    DS_coord_det = np.array([DS_coords[z_id]['Rx'],DS_coords[z_id]['Ry'],DS_coords[z_id]['Rz']])
                
                t_ds_prop = t_ds + np.linalg.norm(DS_track_pos - DS_coord_det)/ds_speed
                t_ds_delta = t_ds_prop - t_ds_hit_read # time difference of expected and recorded hit
                print('extra:', t_ds_prop, '\thit:', t_ds_hit_read)
                if DS_coords[z_id]['HV']: t_delta_horiz.append(t_ds_delta)
                else: t_delta_vert.append(t_ds_delta)
                id0 = (z_id-30000)//1000
                id1 = ((z_id-30000)%1000)//60
                id2 = ((z_id-30000)%1000)%60
                croissant_times[id0][id1][id2*2+i_side].append(t_ds_delta)
                
                
            
            
            
            
            
            
            '''
            if DS_coords[z_id]['HV']:
                t_ds_hit = None
                for aHit in event.Digi_MuFilterHits:
                    if aHit.GetDetectorID() in [z_id-1, z_id, z_id+1]: 
                        if not DS_coords[z_id]['HV']==DS_coords[aHit.GetDetectorID()]['HV']: continue
                        t_ds_hit = aHit.GetImpactT()
                        print('\thit det ID:',aHit.GetDetectorID())
                if t_ds_hit is None: continue
                nhits_record += 1
                t_ds_delta = t_ds - t_ds_hit # time difference of expected and recorded hit
                print('extra:', t_ds, '\thit:', t_ds_hit)
                t_delta_horiz.append(t_ds_delta)
                id0 = (z_id-30000)//1000
                id1 = ((z_id-30000)%1000)//60
                id2 = ((z_id-30000)%1000)%60
                croissant_times[id0][id1][id2].append(t_ds_delta)
            else:
                t_ds_hit_read = None
                for aHit in event.Digi_MuFilterHits:
                    if aHit.GetDetectorID() in [z_id-1, z_id, z_id+1]:
                        if not DS_coords[z_id]['HV']==DS_coords[aHit.GetDetectorID()]['HV']: continue
                        #t_ds_hit_read = aHit.GetAllTimes()[0]*6.25
                        t_ds_hit_read = aHit.GetTime(0)*6.25
                        # z_id = aHit.GetDetectorID()
                        print('\thit det ID:',aHit.GetDetectorID())
                if t_ds_hit_read is None: continue
                nhits_record += 1
                #t_ds_prop = t_ds + abs(DS_track_pos[1]-DS_coords[z_id]['Ry'])/lsOfGlobals.FindObject('MuFilter').GetConfParF("MuFilter/DsPropSpeed") ## Mufi.DsPropSpeed
                ds_speed = lsOfGlobals.FindObject('MuFilter').GetConfParF("MuFilter/DsPropSpeed")
                #DS_coord_det = np.array([DS_coords[z_id]['Rx'],DS_coords[z_id]['Ry'],DS_coords[z_id]['Rz']])
                DS_coord_det = np.array([DS_coords[z_id]['Lx'],DS_coords[z_id]['Ly'],DS_coords[z_id]['Lz']])
                t_ds_prop = t_ds + np.linalg.norm(DS_track_pos - DS_coord_det)/ds_speed
                t_ds_delta = t_ds_prop - t_ds_hit_read # time difference of expected and recorded hit
                print('extra:', t_ds_prop, '\thit:', t_ds_hit_read)
                t_delta_vert.append(t_ds_delta)
                id0 = (z_id-30000)//1000
                id1 = ((z_id-30000)%1000)//60
                id2 = ((z_id-30000)%1000)%60
                croissant_times[id0][id1][id2].append(t_ds_delta)
            '''
        # break
        if flag_fullDS:
            fullDS_ids.add(i_event)
        nhits_extra_list.append(nhits_extra); nhits_record_list.append(nhits_record)
        print('\n')
        
        # usage of the track direction method that is identical to the one in SndlhcTracking
        # SL = aTrack.trackDir()
        # if abs(SL[0]) < 0.03:  
        #    h['slope_zx_forward'].Fill(slopeX)
        # elif SL[0] < -0.07:
        #    h['slope_zx_backward'].Fill(slopeX)

        # there is also a 'velocity' estimator - the method returns a pair of values (velocity, error)
        # here it is used to determine direction
        # if aTrack.Velocity()[0] >= 0: 
        #   h['slope_zx_posVel'].Fill(slopeX)

        # if track is a Scifi track
        # if aTrack.getTrackType() == 11: 
        #    h['slope_zx_Sf'].Fill(slopeX)
        #    h['slope_zy_Sf'].Fill(slopeY)

        # a snippet for DS tracks 
        # elif aTrack.getTrackType() == 13:
        #    h['slopes_DS'].Fill(slopeX, slopeY)


#print('\n\nEventIDs not passing all DS bars', allDS_ids - fullDS_ids)
print('N_events passing all DS bars', len(fullDS_ids))
print('N_events passing at least 1 DS bar', len(allDS_ids))
print('Part of events passing all DS bars:', np.around(len(fullDS_ids)/len(allDS_ids), decimals=3))

# Pyplot interactive mode
# plt.ion()

fig, ax = plt.subplots(1,2, figsize=(10,5))

hist_up, hist_low = 0, 20
if t_delta_horiz and t_delta_vert:
    t_delta_horiz = np.array(t_delta_horiz)
    t_delta_horiz = t_delta_horiz[np.abs(t_delta_horiz)<100]
    t_delta_vert = np.array(t_delta_vert)
    t_delta_vert = t_delta_vert[np.abs(t_delta_vert)<100]
    print('horiz.\n median:{:.2f},\t 0.1quant:{:.2f},\t 0.9quant:{:.2f}'.format(np.median(t_delta_horiz), np.quantile(t_delta_horiz, 0.1), np.quantile(t_delta_horiz, 0.9)))
    print('vert.\n median:{:.2f},\t 0.1quant:{:.2f},\t 0.9quant:{:.2f}'.format(np.median(t_delta_vert), np.quantile(t_delta_vert, 0.1), np.quantile(t_delta_vert, 0.9)))
    hist_up = np.max([np.quantile(t_delta_horiz, 0.98), np.quantile(t_delta_vert, 0.98)])
    hist_low = np.min([np.quantile(t_delta_horiz, 0.02), np.quantile(t_delta_vert, 0.02)])
    ax[0].hist(t_delta_horiz, range=(hist_low, hist_up), bins=30, alpha=0.8, label='horiz', density=True)
    ax[0].hist(t_delta_vert, range=(hist_low, hist_up), bins=30, alpha=0.8, label='vert', density=True)
    ax[0].set_xlabel('t, ns')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title('Time corrections')
    # plt.show()
    '''
    t_delta_horiz_980 = np.array(t_delta_horiz)-985
    t_delta_horiz_980 = t_delta_horiz_980[np.abs(t_delta_horiz_980)<100]
    t_delta_horiz_1600 = np.array(t_delta_horiz)-1600
    t_delta_horiz_1600 = t_delta_horiz_1600[np.abs(t_delta_horiz_1600)<100]
    t_delta_vert_980 = np.array(t_delta_vert)-985
    t_delta_vert_980 = t_delta_vert_980[np.abs(t_delta_vert_980)<100]
    t_delta_vert_1600 = np.array(t_delta_vert)-1600
    t_delta_vert_1600 = t_delta_vert_1600[np.abs(t_delta_vert_1600)<100]
    print('horiz-980.\n median:{:.2f},\t 0.1quant:{:.2f},\t 0.9quant:{:.2f}'.format(np.median(t_delta_horiz_980), np.quantile(t_delta_horiz_980, 0.1), np.quantile(t_delta_horiz_980, 0.9)))
    print('vert-980.\n median:{:.2f},\t 0.1quant:{:.2f},\t 0.9quant:{:.2f}'.format(np.median(t_delta_vert_980), np.quantile(t_delta_vert_980, 0.1), np.quantile(t_delta_vert_980, 0.9)))
    print('horiz-1600.\n median:{:.2f},\t 0.1quant:{:.2f},\t 0.9quant:{:.2f}'.format(np.median(t_delta_horiz_1600), np.quantile(t_delta_horiz_1600, 0.1), np.quantile(t_delta_horiz_1600, 0.9)))
    print('vert-1600.\n median:{:.2f},\t 0.1quant:{:.2f},\t 0.9quant:{:.2f}'.format(np.median(t_delta_vert_1600), np.quantile(t_delta_vert_1600, 0.1), np.quantile(t_delta_vert_1600, 0.9)))

    fig, ax = plt.subplots(1,2)
    hist_up = np.max([np.quantile(t_delta_horiz_980, 0.98), np.quantile(t_delta_vert_980, 0.98)])
    hist_low = np.min([np.quantile(t_delta_horiz_980, 0.02), np.quantile(t_delta_vert_980, 0.02)])
    ax[0].hist(t_delta_horiz_980, range=(hist_low, hist_up), bins=30, alpha=0.8, label='horiz', density=True)
    ax[0].hist(t_delta_vert_980, range=(hist_low, hist_up), bins=30, alpha=0.8, label='vert', density=True)
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title('985 cluster')
    hist_up = np.max([np.quantile(t_delta_horiz_1600, 0.98), np.quantile(t_delta_vert_1600, 0.98)])
    hist_low = np.min([np.quantile(t_delta_horiz_1600, 0.02), np.quantile(t_delta_vert_1600, 0.02)])
    ax[1].hist(t_delta_horiz_1600, range=(hist_low, hist_up), bins=30, alpha=0.8, label='horiz', density=True)
    ax[1].hist(t_delta_vert_1600, range=(hist_low, hist_up), bins=30, alpha=0.8, label='vert', density=True)
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title('1600 cluster')
    plt.show()
    '''

nhits_extra_list, nhits_record_list = np.array(nhits_extra_list), np.array(nhits_record_list)
hit_mask = nhits_extra_list>0
print('Overall extra record eff:', np.around(nhits_record_list.sum()/nhits_extra_list.sum(), decimals=3))

ax[1].hist(nhits_record_list[hit_mask]/nhits_extra_list[hit_mask], bins=20, range=(0,1), label='efficiency')
ax[1].grid()
ax[1].legend()
ax[1].set_title('Recorded extrapolated hits')
plt.savefig('time_align_DS.png')
plt.show()

## Croissant plots
croissant_hists = [[np.array([np.histogram(croissant_times[i][j][k], bins=20, range=(hist_low, hist_up))[0] for k in range(120)]) for j in range(2)] for i in range(3)]
print('\n\n', croissant_hists[1][1],'\n\n')

vmin, vmax = np.min(croissant_hists), np.max(croissant_hists)
print('Croissant plots. Min:', vmin, 'Max:', vmax)
fig, ax = plt.subplots(3,2, figsize=(5,18))
for id0 in range(3):
    for id1 in range(2):
        ax[id0, id1].imshow(croissant_hists[id0][id1], vmin=vmin, vmax=vmax)
        ax[id0, id1].set_xlabel('t, bins')
        ax[id0, id1].set_ylabel('bar')
        ax[id0, id1].set_title(('Vert ' if id1 else 'Horiz ')+str(id0))
#plt.suptitle('['+str(np.around(hist_low, decimals=1))+','+str(np.around(hist_up, decimals=1))+']')
plt.savefig('time_croissant_DS.png')
plt.show()

# plt.ioff()
# plt.show()

#for item in h:
#   h[item].Write()

print('Done')

