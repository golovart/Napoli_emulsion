#/usr/local/lib/python3.6/dist-packages/srim
## problem in RANGE.txt for 1 ion generation with lines like
## Total Ions calculated =.91
## pysrim expects number before dot
### cp /opt/pysrim/srim_fix.py /usr/local/lib/python3.6/dist-packages/srim/srim.py
# pip3 install joblib
# sh -c "xvfb-run -a python3.6 /opt/pysrim/wimp.py"
# sh -c "cp /opt/pysrim/srim_fix.py /usr/local/lib/python3.6/dist-packages/srim/srim.py; pip3 install joblib; xvfb-run -a python3.6 /opt/pysrim/wimp.py"
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
N_wimps = 1000000 # total n_wimps to be recoiled
n_bins = {'E':200, 'theta':50} # N_ex2N_theta binning of E-theta spectrum
M_w = 500 # GeV
n_jobs = 15
verb = 10 # verbosity of output during computation
track_3d = True # look at 3D coordinates or XY-projection
output_dir = os.path.abspath('/tmp/output') # store the output files

### Functions for simulating wimp-nuclei recoils
def mu(m1,m2):
    return m1*m2/(m1+m2)

def R_etheta_rel(E, theta, c_n, A_n, m_n, M_w=100, v_0=220, v_esc=544, sig_v=220/np.sqrt(2)):
    m_n *= 0.9315 #atomic mass to GeV
    w_n = np.sqrt(m_n*E*1e4*(2.99792**2)/(2*mu(m_n,M_w)**2))
    #print(w_n)
    #print(w_n-v_0*np.cos(theta))
    return c_n*(A_n**2)*( np.exp(-(w_n-v_0*np.cos(theta))**2/(2*sig_v**2)) - np.exp(-v_esc**2/(2*sig_v**2)) )

def R_etheta_abs(E, theta, c_n, A_n, m_n, M_w=100, sig_p=1e-46, rho=0.3, v_0=220, v_esc=544, sig_v=220/np.sqrt(2)):
    N_esc = sp.special.erf(v_esc/(np.sqrt(2)*sig_v)) - np.sqrt(2/sp.pi)*(v_esc/sig_v)*np.exp(-v_esc**2/(2*sig_v**2))
    #N_esc = 0.9933607713839784 # for v_esc=544, sig_v=220/np.sqrt(2)
    sig_p *= 1e44
    m_p = 0.93827 #GeV
    return 1.595*rho*sig_p/(2*M_w*N_esc*np.sqrt(2*np.pi*sig_v**2)*mu(m_p,M_w)**2)*R_etheta_rel(E, theta, c_n, A_n, m_n, M_w, v_0=v_0, v_esc=v_esc, sig_v=sig_v)

def E_max(m_n, M_w=100, v_esc=544, v_0=220):
    m_n *= 0.9315
    return 2*1e-4*(mu(m_n,M_w)**2)*((v_esc+v_0)**2)/m_n/2.99792**2

def find_beta_ab(e_p, e_v=None):
    if e_v==None: e_v = e_p**2

    def beta_eq(beta, ep, ev):
        return (beta*(1-2*ep+beta*ep)*(1-ep)**2)/((1-2*ep+beta)**2 *(2-3*ep+beta))-ev
    def alpha_from_beta(b, ep):
        return (1-2*ep+b*ep)/(1-ep)

    b = sp.optimize.fsolve(beta_eq, 20, args=(e_p, e_v))
    return alpha_from_beta(b, e_p), b

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

def proc_exyz(exyz_dir='/tmp/srim/SRIM Outputs/', track_3d=False):
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
    tracks = np.zeros((0,8)) if track_3d else np.zeros((0,6))
    for i, steps in zip(ions[0],ions[1]):
        coords = exyz[in_step:in_step+steps, 2:] if track_3d else exyz[in_step:in_step+steps, 2:-1]
        tracks = np.vstack((tracks,find_track(coords)))
        in_step += steps
    return tracks

###
### Defining emulsion parameters for spectrum simulation
###
elems = {
    'Ag':{
        'st_n':0.105,
        'A_n':47,
        'm_n':107.868
    },
    'Br':{
        'st_n':0.101,
        'A_n':35,
        'm_n':79.904
    },
    'I':{
        'st_n':0.004,
        'A_n':53,
        'm_n':126.9
    },
    'O':{
        'st_n':0.117,
        'A_n':8,
        'm_n':15.999
    },
    'N':{
        'st_n':0.049,
        'A_n':7,
        'm_n':14.007
    },
    'C':{
        'st_n':0.214,
        'A_n':6,
        'm_n':12.011
    },
    'H':{
        'st_n':0.41,
        'A_n':1,
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
### Calculating spectrum
###
spectrum, tot_spec, spec_Etheta = {},{},{}
for name,el in elems.items():
    spectrum[name] = np.zeros((n_bins['E'],2*n_bins['theta']))
    spec_Etheta[name] = []
    nucl_par = {i:el[i] for i in ['c_n', 'A_n', 'm_n']}
    # Sampling from beta distribution. Peak = E_peak, varience = E_peak**2
    # E_range = np.linspace(0,1, n_bins['E'], endpoint=False)
    # E_peak = E_max(m_n=el['m_n'], M_w=M_w, v_esc=0)/E_max(m_n=el['m_n'], M_w=M_w)
    # a,b = find_beta_ab(E_peak, (E_peak*1.5)**2); E_range = sp.special.betaincinv(a, b, E_range)
    # E_range = E_max(m_n=el['m_n'], M_w=M_w)*(E_range+E_range[1]/6) # Scaling energy range + shifting a bit so start is not exact 0

    # Sampling energy points from uniform distribution
    E_range = np.linspace(0, E_max(m_n=el['m_n'], M_w=M_w), num=n_bins['E']+1, endpoint=False)[1:]

    for i, E in enumerate(E_range):
        spec_Etheta[name].append([])
        for j, theta in enumerate(np.linspace(0, np.pi, num=2*n_bins['theta'], endpoint=False)):
            spectrum[name][i,j] = R_etheta_rel(E, theta, M_w=M_w, **nucl_par)
            spectrum[name][i,j] = max(0, spectrum[name][i,j])
            spec_Etheta[name][i].append((E,theta))
    tot_spec[name] = np.sum(spectrum[name])
    # Mirroring ions with angle > pi/2
    for i in range(n_bins['E']):
        for j in range(n_bins['theta']):
            spectrum[name][i,n_bins['theta']-1-j] += spectrum[name][i,n_bins['theta']-1+j]
whole_spec = np.sum([tot for tot in tot_spec.values()])

int_spectrum = {}
for name, spec in spectrum.items():
    int_spectrum[name] = np.array(spec[:,:n_bins['theta']]/tot_spec[name]*(N_wimps*tot_spec[name]/whole_spec), dtype=int)
effic_bin = np.sum([sp_count.sum() for sp_count in int_spectrum.values()])/N_wimps
# part of tracks not lost due to integer bins and forward direction
print('effic_bin:\t',effic_bin)


###
### Simulating tracks
###
def energy_tracks(i_E, name, int_spec, n_bins=None, spec_Etheta=None, layer=None, srim_dir=None, output_dir=None, track_3d=None):
    tracks_theta = np.zeros((0,8)) if track_3d else np.zeros((0,6))
    srim_tmp = srim_dir+'_tmp'+str(os.getpid())
    if not os.path.exists(srim_tmp):
        os.makedirs(srim_tmp)#, exist_ok=True)
        copy_tree(srim_dir,srim_tmp)
    output_tmp = output_dir+'/tmp'+str(os.getpid())+'/';
    if not os.path.exists(output_tmp): os.makedirs(output_tmp)#, exist_ok=True)

    for j in range(n_bins['theta']):
        if not int_spec[i_E,j]: continue
        E_ion = spec_Etheta[name][i_E][j][0]*1e3
        theta_ion = spec_Etheta[name][i_E][j][1]*180/np.pi
        N_ion = int_spec[i_E,j]
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
        shutil.copy(srim_tmp+'/SRIM Outputs/EXYZ.txt', output_tmp)
        shutil.copy(srim_tmp+'/RANGE.txt', output_tmp)
        del trim, target, ion; gc.collect();
        tracks_theta = np.vstack((tracks_theta, proc_exyz(srim_tmp+'/SRIM Outputs/', track_3d)))
    return tracks_theta

paral_params = dict(n_bins=n_bins, spec_Etheta=spec_Etheta, layer=layer, srim_dir=srim_dir, output_dir=output_dir, track_3d=track_3d)

for name, int_spec in int_spectrum.items():
    tracks_ion = np.zeros((0,8)) if track_3d else np.zeros((0,6))
    os.makedirs(output_dir, exist_ok=True)
    #for i_E in range(n_bins['E']): os.makedirs(output_dir+'/iter'+str(i_E)+'/', exist_ok=True)

    tracks_for_E = Parallel(n_jobs=n_jobs, verbose=verb)(delayed(energy_tracks)(i_E, name, int_spec, **paral_params) for i_E in range(n_bins['E']) )
    for tr in tracks_for_E: tracks_ion = np.vstack((tracks_ion, tr))
    #for i_E in range(n_bins['E']): shutil.rmtree(output_dir+'/iter'+str(i_E)+'/')

    # for i in range(n_bins['E']):
    #     for j in range(n_bins['theta']):
    #         if not int_spec[i,j]: continue
    #         E_ion = spec_Etheta[name][i][j][0]*1e3
    #         theta_ion = spec_Etheta[name][i][j][1]*180/np.pi
    #         N_ion = int_spec[i,j]
    #         #
    #         # SRIM simulation
    #         #
    #         ion = srim.Ion(name, energy=E_ion) #eV
    #         target = srim.Target([layer])
    #         TRIM_settings = {'calculation': 1, 'autosave':1, 'exyz':1, 'plot_xmax':100000, 'angle_ions': theta_ion}
    #         print(N_ion)
    #         trim = srim.TRIM(target, ion, number_ions=N_ion, **TRIM_settings)
    #         trim.run(srim_dir)
    #         os.makedirs(output_dir, exist_ok=True)
    #         shutil.copy(srim_dir+'/TRIM.IN', output_dir)
    #         shutil.copy(srim_dir+'/SRIM Outputs/EXYZ.txt', output_dir)
    #         shutil.copy(srim_dir+'/RANGE.txt', output_dir)
    #         #
    #         tracks_ion = np.vstack((tracks_ion, proc_exyz(srim_dir+'/SRIM Outputs/', track_3d)))
    #         del trim, target, ion; gc.collect()
    np.savetxt(output_dir+'/tracks_'+name+'.csv', tracks_ion, delimiter=',')
    # df_columns = ['len','start_x','start_y','start_z','end_x','end_y','end_z'] if track_3d else ['len','start_x','start_y','end_x','end_y']
    # df_tracks_ion = pd.DataFrame(tracks_ion,columns=df_columns)
    # df_tracks_ion.to_csv(output_dir+'/tracks_'+name+'.csv')
for out_name in os.listdir(output_dir):
    if 'tmp' in out_name: shutil.rmtree(output_dir+'/'+out_name)


#srim.TRIM.copy_output_files(srim_dir, output_dir)
#shutil.copytree(srim_dir+'/SRIM Outputs', output_dir)
