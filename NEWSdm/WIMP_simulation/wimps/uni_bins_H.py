
#/usr/local/lib/python3.6/dist-packages/srim
## problem in RANGE.txt for 1 ion generation with lines like
## Total Ions calculated =.91
## pysrim expects number before dot
### cp /opt/pysrim/srim_fix.py /usr/local/lib/python3.6/dist-packages/srim/srim.py
# pip3 install joblib
# sh -c "xvfb-run -a python3.6 /opt/pysrim/wimp.py"
# sh -c "cp /opt/pysrim/srim_fix.py /usr/local/lib/python3.6/dist-packages/srim/srim.py; pip3 install joblib; xvfb-run -a python3.6 /opt/pysrim/wimp.py"
'''
time docker run -v $PWD/input:/opt/pysrim/  -v $PWD/output/tmp/wimp:/tmp/output  -it costrouc/pysrim sh -c "cp /opt/pysrim/srim_fix.py /usr/local/lib/python3.6/dist-packages/srim/srim.py; pip3 install numpy joblib scipy; xvfb-run -a python3.6 /opt/pysrim/wimp/wimp.py"
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
N_wimps = 100000 # total n_wimps to be recoiled
n_ions = 1000 # number of ions per bin
n_bins = {'E':100, 'theta':10} # N_ex2N_theta binning of E-theta spectrum
M_w = 1000 # GeV # maximum WIMP mass
n_jobs = 10
verb = 10 # verbosity of output during computation
track_3d = True # look at 3D coordinates or XY-projection
output_dir = os.path.abspath('/tmp/output') # store the output files

### Functions for simulating wimp-nuclei recoils
def mu(m1,m2):
    return m1*m2/(m1+m2)

def E_max(m_n, M_w=100, v_esc=544, v_E=232):
    m_n *= 0.9315
    return 2*1e-4*(mu(m_n,M_w)**2)*((v_esc+v_E)**2)/m_n/2.99792**2

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
### Defining emulsion parameters for spectrum simulation
###
elems = {
    'Ag':{
        'st_n':0.105,
        'Z_n':47,
        'm_n':107.868
    },
    'Br':{
        'st_n':0.101,
        'Z_n':35,
        'm_n':79.904
    },
    'I':{
        'st_n':0.004,
        'Z_n':53,
        'm_n':126.9
    },
    'O':{
        'st_n':0.117,
        'Z_n':8,
        'm_n':15.999
    },
    'N':{
        'st_n':0.049,
        'Z_n':7,
        'm_n':14.007
    },
    'C':{
        'st_n':0.214,
        'Z_n':6,
        'm_n':12.011
    },
    'H':{
        'st_n':0.41,
        'Z_n':1,
        'm_n':1.008
    }
}
for name,el in elems.items():
    elems[name]['c_n'] = np.round(el['st_n']*el['m_n']/np.sum([el['st_n']*el['m_n'] for el in elems.values()]),3)

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
}, density=3.44, width=1.0e5, phase=0, name="NIT3.44")

srim_dir = os.path.abspath('/tmp/srim')


###
### Simulating tracks
###
def uni_energy_tracks(name, E_ion, theta_range=None, N_ion=None, layer=None, srim_dir=None, output_dir=None, track_3d=None, track_dir=None):
    #tracks_theta = np.zeros((0,10)) if track_3d else np.zeros((0,8))
    srim_tmp = srim_dir+'_tmp'+str(os.getpid())
    if not os.path.exists(srim_tmp):
        os.makedirs(srim_tmp)#, exist_ok=True)
        copy_tree(srim_dir,srim_tmp)
    output_tmp = output_dir+'/tmp'+str(os.getpid())+'/';
    if not os.path.exists(output_tmp): os.makedirs(output_tmp)#, exist_ok=True)

    E_ion *= 1e3
    for theta_ion in theta_range:
        theta_ion *= 180/np.pi
        #
        # SRIM simulation
        #
        ion = srim.Ion(name, energy=E_ion) #eV
        target = srim.Target([layer])
        TRIM_settings = {'calculation': 1, 'autosave':1, 'exyz':1, 'plot_xmax':100000, 'angle_ions': theta_ion}
        #print(N_ion)
        trim = srim.TRIM(target, ion, number_ions=N_ion, **TRIM_settings)
        trim.run(srim_tmp)
        #os.makedirs(output_dir, exist_ok=True)
        shutil.copy(srim_tmp+'/TRIM.IN', output_tmp)
        shutil.copy(srim_tmp+'/SRIM Outputs/EXYZ.txt', output_tmp+'/EXYZ_'+name+'_E'+str(np.around(E_ion/1000,decimals=2))+'keV_th'+str(np.around(theta_ion,decimals=1))+'.txt')
        shutil.copy(srim_tmp+'/RANGE.txt', output_tmp)
        del trim, target, ion; gc.collect();
        #print('\n\n\n',tracks_theta.shape)
        #print('\n\n\n',proc_exyz(srim_tmp+'/SRIM Outputs/', track_3d, E_ion, theta_ion).shape,'\n\n\n')
        tracks_theta = proc_exyz(srim_tmp+'/SRIM Outputs/', track_3d, E_ion, theta_ion)
        np.savetxt(track_dir+'/tracks_'+name+'_E'+str(np.around(E_ion/1000,decimals=2))+'keV_th'+str(np.around(theta_ion,decimals=1))+'.csv', tracks_theta, delimiter=',')
        #tracks_theta = np.vstack((tracks_theta, proc_exyz(srim_tmp+'/SRIM Outputs/', track_3d, E_ion, theta_ion)))
    return True #tracks_theta


theta_range = np.linspace(0, np.pi/2, num=n_bins['theta'], endpoint=False)
paral_params = dict(theta_range=theta_range, N_ion=n_ions, layer=layer, srim_dir=srim_dir, output_dir=output_dir, track_3d=track_3d)

for name,el in elems.items():
    if not 'H' in name: continue
    if not 'tracks_'+name in os.listdir(output_dir): os.mkdir(output_dir+'/tracks_'+name)
    #spectrum[name] = np.zeros((n_bins['E'],n_bins['theta']))
    #spec_Etheta[name] = []
    nucl_par = {i:el[i] for i in ['c_n', 'm_n']}

    # Sampling energy points from uniform distribution
    E_range = np.linspace(0, E_max(m_n=el['m_n'], M_w=M_w), num=n_bins['E']+1, endpoint=False)[1:]

    print('\n',name,'\nmax Energy: {:.2f} keV'.format(E_range[-1]),'\n ions per bin: {:d}'.format(n_ions))
    #tracks_ion = np.zeros((0,10)) if track_3d else np.zeros((0,8))
    os.makedirs(output_dir, exist_ok=True)
    #for i_E in range(n_bins['E']): os.makedirs(output_dir+'/iter'+str(i_E)+'/', exist_ok=True)

    success_for_E = Parallel(n_jobs=n_jobs, verbose=verb)(delayed(uni_energy_tracks)(name, E_ion, track_dir=output_dir+'/tracks_'+name, **paral_params) for E_ion in E_range )
    #for tr in tracks_for_E: tracks_ion = np.vstack((tracks_ion, tr))
    #success_ion = [E_range, success_for_E]

    #np.savetxt(output_dir+'/simulated_'+name+'.csv', success_ion, delimiter=',')
    # df_columns = ['len','start_x','start_y','start_z','end_x','end_y','end_z'] if track_3d else ['len','start_x','start_y','end_x','end_y']
    # df_tracks_ion = pd.DataFrame(tracks_ion,columns=df_columns)
    # df_tracks_ion.to_csv(output_dir+'/tracks_'+name+'.csv')
if not 'exyz' in os.listdir(output_dir): os.mkdir(output_dir+'/exyz')
for out_name in os.listdir(output_dir):
    if 'tmp' in out_name:
        for tmp_name in os.listdir(output_dir+'/'+out_name):
            if 'EXYZ_' in tmp_name: shutil.copy(output_dir+'/'+out_name+'/'+tmp_name, output_dir+'/exyz/')
        shutil.rmtree(output_dir+'/'+out_name)


#srim.TRIM.copy_output_files(srim_dir, output_dir)
#shutil.copytree(srim_dir+'/SRIM Outputs', output_dir)
