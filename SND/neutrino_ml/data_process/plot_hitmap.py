from argparse import ArgumentParser
import os, gc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("-mc", "--MCfile", dest="MCfile", help="Monte Carlo data directory", required=False, default="sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000")
parser.add_argument("-d", "--dataPath", dest="dataPath", help="path to directory with npy data", default="/eos/user/g/golovati/SND/data/neutrino_ml/MC_explore")
parser.add_argument("-t", "--type", dest="etype", help="event type to select", default='neutrino')

options = parser.parse_args()

nchan = {'scifi':1536, 'us':10, 'ds':60}
nplane = {'scifi':5, 'us':5, 'ds':4}

MCpath = options.dataPath+'/'+options.etype+'/'+options.MCfile+'/'
event_meta = np.load(MCpath+'event_metadata.npy')
hitmap = {}
for name in ['scifi', 'us', 'ds']:
    hitmap[name] = np.load(MCpath+name+'.npy')

event_map = (np.abs(event_meta[:,1])==14) * (np.abs(event_meta[:,2])!=14)

wall_coords = [0, 300.+1, 313.+1, 326.+1, 339.+1, 352.+1]

hit_tot = [{},{},{},{},{}]
for i_wall in range(5):
    event_map_w = event_map * (event_meta[:,0]>wall_coords[i_wall]) * (event_meta[:,0]<wall_coords[i_wall+1])
    hit_tot[i_wall]['scifi'] = np.sum(hitmap['scifi'][event_map_w], axis=0, dtype=int)
    tmp_scifi = np.zeros((2, nplane['scifi'], 12))
    for i in range(12):
        tmp_scifi[:, :, i] = hit_tot[i_wall]['scifi'][..., 128*i:128*(i+1)].sum(axis=-1)
    hit_tot[i_wall]['scifi'] = tmp_scifi
    hit_tot[i_wall]['us'] = np.sum(hitmap['us'][event_map_w], axis=0, dtype=int)
    hit_tot[i_wall]['ds'] = np.sum(hitmap['ds'][event_map_w], axis=0, dtype=int)

imextent = {
    'scifi_h':(0, 5, 15, 54),
    'scifi_v':(0, 5, -46, -7),
    'us_h':(0, 5, 12, 66),
    'us_v':(0, 5, 4, -79),
    'ds_h':(0, 4, 7, 71),
    'ds_v':(0, 4, 2, -79),
}

fig, ax = plt.subplots(5, 3, figsize=(7,20))
for i_wall in range(5):
    i_ax = 0
    for name, hitmap in hit_tot[i_wall].items():
        if name=='ds': continue
        for vert in range(2):
            if name=='us' and vert: continue
            ax[i_wall, i_ax].imshow(hitmap[vert].T, origin='lower')#, extent=imextent[name+'_'+('v' if vert else 'h')])
            ax[i_wall, i_ax].set_title('Wall '+str(i_wall+1)+' '+name+' '+('vert' if vert else 'horiz'))
            i_ax += 1
savepath = 'data/neutrino/wall_plot_vm_cc.png'
plt.savefig(savepath)
# print('saved fig: '+savepath)
plt.show()
