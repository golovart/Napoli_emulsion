#/usr/local/lib/python3.6/dist-packages/srim
## problem in RANGE.txt for 1 ion generation with lines like
## Total Ions calculated =.91
## pysrim expects number before dot
### cp /opt/pysrim/srim_fix.py /usr/local/lib/python3.6/dist-packages/srim/srim.py
# pip3 install joblib
#sh -c "xvfb-run -a python3.6 /opt/pysrim/wimp.py"
'''
cp /opt/pysrim/srim_fix.py /usr/local/lib/python3.6/dist-packages/srim/srim.py
pip3 install joblib scipy
sh -c "xvfb-run -a python3.6 /opt/pysrim/wimp.py"
'''

import os, shutil, gc
import srim

import numpy as np
from joblib import Parallel, delayed
from distutils.dir_util import copy_tree
import scipy as sp
import scipy.optimize
# import pandas as pd

### Important constants for calculations
n_ion = 7000 # total n_wimps to be recoiled
E_ion = 2.9e4 # eV
theta_ion = 0 # degrees
# n_bins = {'E':50, 'theta':30} # N_ex2N_theta binning of E-theta spectrum
# M_w = 100 # GeV
track_3d = True # look at 3D coordinates or XY-projection
output_dir = os.path.abspath('/tmp/output') # store the output files


def dist(a,b):
    return np.sum((a-b)**2)

def find_track(coords):
    ### Find furthest points and track length
    start = coords[0]; end = coords[-1]
    max_dist = dist(start,end)
    while True:
        changed = False
        for i in range(len(coords)//2):
            if dist(coords[i],end)>max_dist:
                start = coords[i]; max_dist = dist(coords[i],end)
                changed = True
            elif dist(start,coords[-i])>max_dist:
                end = coords[-i]; max_dist = dist(start,coords[-i])
                changed = True
        if not changed: break
    return np.hstack((np.sqrt(max_dist)/10, (end-start)[0]/np.sqrt(max_dist), start, end))

def proc_exyz(exyz_dir='/tmp/srim/SRIM Outputs/', track_3d=False, E_in=0, theta_in=0):
    with open(exyz_dir+'EXYZ.txt','r') as f:
        exyz = f.read()
        #print('\n'.join(mopa.split('\n')[:15]))
        # drop the backscatter
        exyz = np.array(exyz.split('\n'))[15:-1]
        mask = np.ones(len(exyz), dtype=bool)
        for i,line in enumerate(exyz):
            if '- ' in line: mask[i] = False
        exyz = exyz[mask]
        #
        exyz = np.array([s.split() for s in exyz], dtype=float)[:,:5]
    ions = np.unique(exyz[:,0], return_counts=True)
    in_step = 0
    tracks = np.zeros((0,10)) if track_3d else np.zeros((0,8))
    for i, steps in zip(ions[0],ions[1]):
        coords = exyz[in_step:in_step+steps, 2:] if track_3d else exyz[in_step:in_step+steps, 2:-1]
        tracks = np.vstack((tracks,np.hstack((E_in,theta_in,find_track(coords)))))
        in_step += steps
    return tracks


###
### Defining SRIM target (emulsion)
###
layer = srim.Layer({
    'C': {
        'stoich': 0.21400,
        'E_d': 28.0, # Displacement Energy
        'lattice': 3.0,
        'surface': 7.41
    },
    'H': {
        'stoich': 0.4100,
        'E_d': 10.0, # Displacement Energy
        'lattice': 3.0,
        'surface': 2.0
    },
    'N': {
        'stoich': 0.04900,
        'E_d': 28.0, # Displacement Energy
        'lattice': 3.0,
        'surface': 2.0
    },
    'O': {
        'stoich': 0.11700,
        'E_d': 28.0, # Displacement Energy
        'lattice': 3.0,
        'surface': 2.0
    },
    # 'S': {
    #     'stoich': 0.0,
    #     'E_d': 25.0, # Displacement Energy
    #     'lattice': 3.0,
    #     'surface': 2.88
    # },
    # 'K': {
    #     'stoich': 0.0,
    #     'E_d': 25.0, # Displacement Energy
    #     'lattice': 3.0,
    #     'surface': 0.93
    # },
    'Ag': {
        'stoich': 0.10500,
        'E_d': 25.0, # Displacement Energy
        'lattice': 3.0,
        'surface': 2.97
    },
    'Br': {
        'stoich': 0.101000,
        'E_d': 25.0, # Displacement Energy
        'lattice': 3.0,
        'surface': 2.0
    },
    'I': {
        'stoich': 0.004000,
        'E_d': 25.0, # Displacement Energy
        'lattice': 3.0,
        'surface': 2.0
    }
}, density=3.2, width=1.0e5, phase=0, name="NIT3.2")

layer_zn = srim.Layer({
    'W': {
        'stoich': 0.16667,
        'E_d': 25.0, # Displacement Energy
        'lattice': 3.0,
        'surface': 8.68
    },
    'O': {
        'stoich': 0.66666,
        'E_d': 28.0, # Displacement Energy
        'lattice': 3.0,
        'surface': 2.0
    },
    'Zn': {
        'stoich': 0.16667,
        'E_d': 25.0, # Displacement Energy
        'lattice': 3.0,
        'surface': 1.35
    }
}, density=7.87, width=1.0e5, phase=0, name="ZnWO4")


srim_dir = os.path.abspath('/tmp/srim')



###
### Calculating spectrum
###



###
### Simulating tracks
###

def parallel_tracks(n_ion=1000, E_ion=1e5, theta_ion=0, layer=None, srim_dir=None, output_dir=None, track_3d=None):
    srim_tmp = srim_dir+'_tmp'+str(os.getpid())
    if not os.path.exists(srim_tmp):
        os.makedirs(srim_tmp)#, exist_ok=True)
        copy_tree(srim_dir,srim_tmp)
    output_tmp = output_dir+'/tmp'+str(os.getpid())+'/';
    if not os.path.exists(output_tmp): os.makedirs(output_tmp)#, exist_ok=True)

    ion = srim.Ion('C', energy=E_ion) #eV
    target = srim.Target([layer])
    TRIM_settings = {'calculation': 1, 'autosave':1, 'exyz':1, 'plot_xmax':100000, 'angle_ions': theta_ion}
    #print(n_ion)
    trim = srim.TRIM(target, ion, number_ions=n_ion, **TRIM_settings)
    trim.run(srim_tmp)
    #os.makedirs(output_dir, exist_ok=True)
    shutil.copy(srim_tmp+'/TRIM.IN', output_tmp)
    shutil.copy(srim_tmp+'/SRIM Outputs/EXYZ.txt', output_tmp)
    shutil.copy(srim_tmp+'/RANGE.txt', output_tmp)
    del trim, target, ion; gc.collect();#
    return proc_exyz(srim_tmp+'/SRIM Outputs/', track_3d)

            #
            # SRIM simulation
            #
#tracks_ion = np.zeros((0,10)) if track_3d else np.zeros((0,8))
#jobs = 5; paral_params = dict(E_ion=E_ion, theta_ion=theta_ion, layer=layer, srim_dir=srim_dir, output_dir=output_dir, track_3d=track_3d)
#tracks_for_N = Parallel(n_jobs=jobs, verbose=1)(delayed(parallel_tracks)(n, **paral_params) for n in [N_ion//jobs for i in range(jobs) ] )
#for tr in tracks_for_N: tracks_ion = np.vstack((tracks_ion, tr))

srim_tmp = srim_dir#+'_tmp'+str(os.getpid())
if not os.path.exists(srim_tmp):
    os.makedirs(srim_tmp)#, exist_ok=True)
    copy_tree(srim_dir,srim_tmp)
output_tmp = output_dir#+'/tmp'+str(os.getpid())+'/';
if not os.path.exists(output_tmp): os.makedirs(output_tmp)#, exist_ok=True)
ion = srim.Ion('O', energy=E_ion) #eV
target = srim.Target([layer_zn])
TRIM_settings = {'calculation': 1, 'autosave':1, 'exyz':1, 'plot_xmax':100000, 'angle_ions': theta_ion}
#print(n_ion)
trim = srim.TRIM(target, ion, number_ions=n_ion, **TRIM_settings)
trim.run(srim_tmp)
#os.makedirs(output_dir, exist_ok=True)
shutil.copy(srim_tmp+'/TRIM.IN', output_tmp)
shutil.copy(srim_tmp+'/SRIM Outputs/EXYZ.txt', output_tmp)
shutil.copy(srim_tmp+'/RANGE.txt', output_tmp)
del trim, target, ion; gc.collect();#
tracks_ion = proc_exyz(srim_tmp+'/SRIM Outputs/', track_3d, E_in=E_ion, theta_in=theta_ion)



# ion = srim.Ion('C', energy=E_ion) #eV
# target = srim.Target([layer])
# TRIM_settings = {'calculation': 1, 'autosave':1, 'exyz':1, 'plot_xmax':100000, 'angle_ions': theta_ion}
# print(N_ion)
# trim = srim.TRIM(target, ion, number_ions=N_ion, **TRIM_settings)
# trim.run(srim_dir)
# os.makedirs(output_dir, exist_ok=True)
# shutil.copy(srim_dir+'/TRIM.IN', output_dir)
# shutil.copy(srim_dir+'/SRIM Outputs/EXYZ.txt', output_dir)
# shutil.copy(srim_dir+'/RANGE.txt', output_dir)
# #
# tracks_ion = proc_exyz(srim_dir+'/SRIM Outputs/')
# del trim, target, ion; gc.collect()
np.savetxt(output_dir+'/tracks_O'+str(int(E_ion/1000))+'.csv', tracks_ion, delimiter=',')

for out_name in os.listdir(output_dir):
    if 'tmp' in out_name: shutil.rmtree(output_dir+'/'+out_name)
#srim.TRIM.copy_output_files(srim_dir, output_dir)
#shutil.copytree(srim_dir+'/SRIM Outputs', output_dir)
