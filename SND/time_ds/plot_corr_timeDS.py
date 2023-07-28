import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

plt.rcParams.update({'font.size': 16})

data_header = ['detID', 'id_side', 'delta_t', 'isB1', 'isB2', 'isB1only', 'isB2notB1']
header2id = {'detID':0, 'id_side':1, 'delta_t':2, 'isB1':3, 'isB2':4, 'isB1only':5, 'isB2notB1':6}

time_data = np.load('corrected_timeDS.npy')
'''
time_data = np.ones((0,7))
# time_data = np.ones((0,3))
names_condor = os.listdir('numpy_out')
for name_p in tqdm(names_condor, desc='numpy partitions loading:'):
    name_p = 'numpy_out/'+name_p
    time_data = np.vstack((time_data, np.load(name_p)))
'''
def this_bar(barray, side_array, id0, id1, bar_id):
    assert(len(barray)==len(side_array))
    mask_id0 = ((barray - 30000) // 1000) == id0
    mask_id1 = (((barray - 30000) % 1000) // 60) == id1
    mask_side = side_array == bar_id
    return mask_id0 * mask_id1 * mask_side
    
def this_hist(barray, id0, id1):
    mask_id0 = ((barray - 30000) // 1000) == id0
    mask_id1 = (((barray - 30000) % 1000) // 60) == id1
    return mask_id0 * mask_id1


## Full bananas
hist_low, hist_up, hist_bins = -10., 5., 100

croissant_hists = [ [() for j in range(2)] for i in range(4)]
for i in range(4):
    for j in range(2):
        mask_hist = this_hist(time_data[:, 0], i, j)
        croissant_hists[i][j] = np.histogram2d(time_data[mask_hist, 2], time_data[mask_hist, 1], bins=[hist_bins, 120], range=[[hist_low, hist_up], [0, 120]])

vmin = np.min([[np.min(croissant_hists[i][j][0]) for j in range(2)] for i in range(4)])
vmax = np.max([[np.max(croissant_hists[i][j][0]) for j in range(2)] for i in range(4)])
print('Croissant plots. Min:', vmin, 'Max:', vmax)
fig, ax = plt.subplots(4, 2, figsize=(12, 20), sharex=True)
for id0 in range(4):
    for id1 in range(2):
        X, Y = np.meshgrid(croissant_hists[id0][id1][1], croissant_hists[id0][id1][2])
        # ax[id0, id1].pcolormesh(X, Y, croissant_hists[id0][id1][0].T, norm=mpl.colors.SymLogNorm(linthresh=1e-7*vmax, linscale=0.01, vmin=vmin, vmax=vmax))
        alpha_zero = (croissant_hists[id0][id1][0].T>10)
        ax[id0, id1].pcolormesh(X, Y, croissant_hists[id0][id1][0].T, vmin=vmin, vmax=vmax, alpha=alpha_zero)
        if id0==3: ax[id0, id1].set_xlabel(r'$\Delta$t [ns]')
        ax[id0, id1].set_ylabel('bar')
        ax[id0, id1].set_title(('Vert ' if id1 else 'Horiz ') + str(id0))
        ax[id0, id1].minorticks_on()
plt.suptitle('Corrected run 4705')
plt.savefig('/afs/cern.ch/user/g/golovati/public/large_corr_time_4705.png')
plt.show()

'''
## B1 bananas
croissant_hists = [ [() for j in range(2)] for i in range(4)]
for i in range(4):
    for j in range(2):
        mask_hist = this_hist(time_data[:, 0], i, j) * (time_data[:, 5] == 1)
        croissant_hists[i][j] = np.histogram2d(time_data[mask_hist, 2], time_data[mask_hist, 1], bins=[hist_bins, 120], range=[[hist_low, hist_up], [0, 120]])

vmin = np.min([[np.min(croissant_hists[i][j][0]) for j in range(2)] for i in range(4)])
vmax = np.max([[np.max(croissant_hists[i][j][0]) for j in range(2)] for i in range(4)])
print('Croissant plots. Min:', vmin, 'Max:', vmax)
fig, ax = plt.subplots(4, 2, figsize=(12, 18), sharex=True)
for id0 in range(4):
    for id1 in range(2):
        X, Y = np.meshgrid(croissant_hists[id0][id1][1], croissant_hists[id0][id1][2])
        # ax[id0, id1].pcolormesh(X, Y, croissant_hists[id0][id1][0].T, norm=mpl.colors.SymLogNorm(linthresh=1e-7*vmax, linscale=0.01, vmin=vmin, vmax=vmax))
        ax[id0, id1].pcolormesh(X, Y, croissant_hists[id0][id1][0].T, vmin=vmin, vmax=vmax)
        # ax[id0, id1].set_xlabel('t, bins')
        ax[id0, id1].set_ylabel('bar')
        ax[id0, id1].set_title(('Vert ' if id1 else 'Horiz ') + str(id0))
        ax[id0, id1].minorticks_on()
plt.suptitle('B1only run 4705')
# plt.savefig('log_time_croissant_DS.png')
plt.show()


## B2 bananas
croissant_hists = [ [() for j in range(2)] for i in range(4)]
for i in range(4):
    for j in range(2):
        mask_hist = this_hist(time_data[:, 0], i, j) * (time_data[:, 6] == 1)
        croissant_hists[i][j] = np.histogram2d(time_data[mask_hist, 2], time_data[mask_hist, 1], bins=[hist_bins, 120], range=[[hist_low, hist_up], [0, 120]])

vmin = np.min([[np.min(croissant_hists[i][j][0]) for j in range(2)] for i in range(4)])
vmax = np.max([[np.max(croissant_hists[i][j][0]) for j in range(2)] for i in range(4)])
print('Croissant plots. Min:', vmin, 'Max:', vmax)
fig, ax = plt.subplots(4, 2, figsize=(12, 18), sharex=True)
for id0 in range(4):
    for id1 in range(2):
        X, Y = np.meshgrid(croissant_hists[id0][id1][1], croissant_hists[id0][id1][2])
        # ax[id0, id1].pcolormesh(X, Y, croissant_hists[id0][id1][0].T, norm=mpl.colors.SymLogNorm(linthresh=1e-7*vmax, linscale=0.01, vmin=vmin, vmax=vmax))
        ax[id0, id1].pcolormesh(X, Y, croissant_hists[id0][id1][0].T, vmin=vmin, vmax=vmax)
        # ax[id0, id1].set_xlabel('t, bins')
        ax[id0, id1].set_ylabel('bar')
        ax[id0, id1].set_title(('Vert ' if id1 else 'Horiz ') + str(id0))
        ax[id0, id1].minorticks_on()
plt.suptitle('B2noB1 run 4705')
# plt.savefig('log_time_croissant_DS.png')
plt.show()
'''

for id0 in range(3):
    mask_hist = this_hist(time_data[:,0], id0, 0) + this_hist(time_data[:,0], id0, 1)
    plt.hist(time_data[mask_hist, 2], bins=100, range=(-25., 5.), histtype='step', label='DS '+str(id0))
plt.grid()
plt.legend()
plt.minorticks_on()
plt.yscale('log')
plt.xlabel('dt, ns')
plt.title('dT per DS stations')
plt.show()



