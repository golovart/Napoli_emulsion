from argparse import ArgumentParser
import os, gc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("-mc", "--MCfile", dest="MCfile", help="Monte Carlo data directory", required=False, default="sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000")
parser.add_argument("-d", "--dataPath", dest="dataPath", help="path to directory with npy data", default="/afs/cern.ch/user/g/golovati/work/neutrino_ml/MC_explore/data")
parser.add_argument("-t", "--type", dest="etype", help="event type to select", default='neutrino')
parser.add_argument("-o", "--outPath", dest="outPath", help="path to directory to store joint npy data", default="/eos/user/g/golovati/SND/data/neutrino_ml/MC_explore")

options = parser.parse_args()

nchan = {'scifi':1536, 'us':10, 'ds':60}
nplane = {'scifi':5, 'us':5, 'ds':4}

event_meta = np.zeros((0,3))
hitmap = {}
hitmap['scifi'] = np.zeros((0, 2, nplane['scifi'], nchan['scifi']), dtype=bool) # use uint8?
hitmap['us'] = np.zeros((0, 2, nplane['us'], nchan['us']), dtype=bool)
hitmap['ds'] = np.zeros((0, 2, nplane['ds'], nchan['ds']), dtype=bool)

MCpath = options.dataPath+'/'+options.etype+'/'+options.MCfile
for p in tqdm(os.listdir(MCpath)):
    # if int(p[0])>1: break
    dfiles = os.listdir(MCpath+'/'+p)
    if not ('scifi.npy' in dfiles) or not ('ds.npy' in dfiles) or not ('us.npy' in dfiles) or not ('event_metadata.npy' in dfiles):
        print('not all out files in partition:',p)
        continue
    for name in hitmap.keys():
        hitmap[name] = np.vstack((hitmap[name], np.load(MCpath+'/'+p+'/'+name+'.npy')))
    event_meta = np.vstack((event_meta, np.load(MCpath+'/'+p+'/event_metadata.npy')))
    gc.collect()
gc.collect()

out_path = options.outPath+'/'+options.etype+'/'+options.MCfile+'/'
for name, hits in hitmap.items():
    np.save(out_path+name+'.npy', hits)
np.save(out_path+'event_metadata.npy', event_meta)


