
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
#import srim

import numpy as np
from joblib import Parallel, delayed
from distutils.dir_util import copy_tree
import scipy as sp
import scipy.optimize
import pandas as pd

### Important constants for calculations
N_wimps = 100000 # total n_wimps to be recoiled
n_bins = {'E':50, 'theta':10} # N_ex2N_theta binning of E-theta spectrum
M_w_max = 1000 # GeV
n_jobs = 10
verb = 10 # verbosity of output during computation
track_3d = True # look at 3D coordinates or XY-projection
#output_dir = os.path.abspath('/tmp/output') # store the output files

### Functions for simulating wimp-nuclei recoils
def mu(m1,m2):
    return m1*m2/(m1+m2)

def R_etheta_go(E, theta, m_n, c_n=1.0, M_w=100, v_0=220, sig_p=1e-41, rho=0.3, v_esc=544, v_E=232):
    N_esc = sp.special.erf(v_esc/v_0) - 2/np.sqrt(sp.pi)*(v_esc/v_0)*np.exp(-v_esc**2/(v_0**2))
    #m_p = 0.93827; A = m_n; m_n *= 0.9315 #atomic mass to GeV
    m_p = 0.9315; A = m_n; m_n *= 0.9315 #atomic mass to GeV
    sig_0 = A**2*mu(m_n,M_w)**2/mu(m_p,M_w)**2 *(sig_p*1e36)
    rho /= 0.3
    R_0 = 361/(m_n*M_w)*sig_0*rho*(v_0/220)
    E_0r = (M_w/100)*(v_0/220)**2 *(4*m_n*M_w/(m_n+M_w)**2)*26.9
    #w_n = np.sqrt(m_n*E*1e4*(2.99792**2)/(2*mu(m_n,M_w)**2))
    w_e = v_0*np.sqrt(E/E_0r)
    #return max(R_0/(N_esc*E_0r)*(np.sqrt(np.pi)*v_0/(4*v_E)*(sp.special.erf((w_n+v_E)/v_0)-sp.special.erf((w_n-v_E)/v_0))-np.exp(-v_esc**2/v_0**2) ), 0)
    return c_n*max(R_0/(2*N_esc*E_0r)*(np.exp(-(v_E*np.cos(theta)-w_e)**2/v_0**2)-np.exp(-v_esc**2/v_0**2) ), 0)

def form_fact(E, m_n):
    A = m_n; m_n *= 0.9315
    c = 1.23*A**(1/3)-0.6; a = 0.52; s = 0.9
    r = np.sqrt(c**2+7/3*(np.pi*a)**2-5*s**2) # fm
    q = np.sqrt(2*m_n*E) * 5.06*1e-3 # fm^-1
    #qr = q*r; qs = q*s
    qr = 6.92*1e-3 * np.sqrt(A*E) * r # L-S
    #return 3*(np.sin(qr)-qr*np.cos(qr))*np.exp(-qs**2/2)/qr**3
    return 3*(np.sin(qr)-qr*np.cos(qr))*np.exp(-(q*s)**2/2)/qr**3

def R_etheta_formgo(E, theta, m_n, c_n=1.0, M_w=100, v_0=220, sig_p=1e-41, rho=0.3, v_esc=544, v_E=232):
    return R_etheta_go(E, theta, m_n, c_n, M_w, v_0, sig_p, rho, v_esc, v_E)*form_fact(E, m_n)**2

def E_max(m_n, M_w=100, v_esc=544, v_E=232):
    m_n *= 0.9315
    return 2*1e-4*(mu(m_n,M_w)**2)*((v_esc+v_E)**2)/m_n/2.99792**2

###
### Defining emulsion parameters for spectrum simulation
###
elems = {
    'C':{
        'st_n':0.204633,
        'Z_n':6,
        'm_n':12.011
    },
    'H':{
        'st_n':0.092664,
        'Z_n':1,
        'm_n':1.008
    },
    'F':{
        'st_n':0.702703,
        'Z_n':9,
        'm_n':18.998
    }
}
for name,el in elems.items():
    elems[name]['c_n'] = np.round(el['st_n']*el['m_n']/np.sum([el['st_n']*el['m_n'] for el in elems.values()]),3)


###
### Calculating spectrum
###
E_range = {}
theta_range = np.linspace(0, np.pi/2, num=n_bins['theta'], endpoint=False)
Mw_range = np.logspace(1,3, num=11, base=10)

for name,el in elems.items():
    E_range[name] = np.linspace(0, E_max(m_n=el['m_n'], M_w=M_w_max), num=n_bins['E']+1, endpoint=False)[1:]
    print('max Energy',name, E_range[name][-1])
with open('ranges.txt','w',newline='') as frange:
    frange.write('theta range (rad):\n')
    frange.write(','.join([str(th) for th in theta_range])+'\n\n')
    frange.write('Energy range (keV):\n')
    E_range_ions = pd.DataFrame(E_range)
    E_range_ions.index.name = 'id'
    E_range_ions.to_csv(frange)

spectrum, tot_spec, whole_spec = {},{},{}
for M_w in Mw_range:
    spectrum[M_w], tot_spec[M_w] = {},{}
    for name,el in elems.items():
        nucl_par = {i:el[i] for i in ['c_n', 'm_n']}

        spectrum[M_w][name] = np.zeros((n_bins['E'],n_bins['theta']))
        for i, E in enumerate(E_range[name]):
            for j, theta in enumerate(theta_range):
                spectrum[M_w][name][i,j] = R_etheta_formgo(E, theta, M_w=M_w, **nucl_par) + R_etheta_formgo(E, np.pi-theta, M_w=M_w, **nucl_par)
        tot_spec[M_w][name] = np.sum(spectrum[M_w][name])
    whole_spec[M_w] = np.sum(list(tot_spec[M_w].values()))

    Mw_dir = 'Mw_{:d}GeV'.format(int(M_w))
    os.makedirs(Mw_dir, exist_ok=True)
    with open(Mw_dir+'/R_tot.txt','a') as f_tot:
        f_tot.write('R_tot (kg*d): {:e}'.format(whole_spec[M_w]))
    for name,el in elems.items():
        np.savetxt(Mw_dir+'/spec_'+name+'.txt', spectrum[M_w][name], delimiter=',')
