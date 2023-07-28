import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import matplotlib as mpl
from datetime import datetime
from scipy.optimize import curve_fit


data_header = ['detID', 'id_side', 'delta_t', 'isB1', 'isB2', 'isB1only', 'isB2notB1']
header2id = {'detID':0, 'id_side':1, 'delta_t':2, 'isB1':3, 'isB2':4, 'isB1only':5, 'isB2notB1':6}

'''
time_data = np.ones((0,7))
# time_data = np.ones((0,3))
names_condor = os.listdir('numpy_out')
for name_p in tqdm(names_condor, desc='numpy partitions loading:'):
    name_p = 'numpy_out/'+name_p
    time_data = np.vstack((time_data, np.load(name_p)))

def this_bar(barray, side_array, id0, id1, bar_id):
    assert(len(barray)==len(side_array))
    mask_id0 = ((barray - 30000) // 1000) == id0
    mask_id1 = (((barray - 30000) % 1000) // 60) == id1
    mask_side = side_array == bar_id
    print(id0, id1, bar_id, np.sum(mask_id0 * mask_id1 * mask_side))
    return mask_id0 * mask_id1 * mask_side
    
def this_hist(barray, id0, id1):
    mask_id0 = ((barray - 30000) // 1000) == id0
    mask_id1 = (((barray - 30000) % 1000) // 60) == id1
    return mask_id0 * mask_id1

median_file = open("median_timeDS.txt", "a")

mean_times = []
bar_steps = [0]
print('bar steps: ', end='\t')
for id0 in range(4):
    for id1 in range(2):
        if id0==3 and id1==0: continue
        for b in range(0,120,2):
            bar_mask = this_bar(time_data[:,0], time_data[:,1], id0, id1, b)
            if bar_mask.any():
                med_tmp = np.median(time_data[bar_mask, 2])
                mean_times.append(med_tmp)
                median_file.write(str(id0)+' '+str(id1)+' '+str(b)+' '+str(med_tmp))
                median_file.write('\n')
                # mean_times.append(np.mean(np.array(croissant_times[i][j]['time'])[bar_mask]))
            else:
                mean_times.append(mean_times[-1] if mean_times[-1]>-7 else -5)
        bar_steps.append(len(mean_times))
        print(bar_steps[-1], end='\t')
        if id1: continue
        for b in range(1,120,2):
            bar_mask = this_bar(time_data[:,0], time_data[:,1], id0, id1, b)
            if bar_mask.any():
                med_tmp = np.median(time_data[bar_mask, 2])
                mean_times.append(med_tmp)
                median_file.write(str(id0)+' '+str(id1)+' '+str(b)+' '+str(med_tmp))
                median_file.write('\n')
                # mean_times.append(np.mean(np.array(croissant_times[i][j]['time'])[bar_mask]))
            else:
                mean_times.append(mean_times[-1] if mean_times[-1]>-7 else -5)
        bar_steps.append(len(mean_times))
        print(bar_steps[-1], end='\t')
chan_times = np.arange(len(mean_times))

median_file.close()
print('\n'*5)
'''

median_data = np.loadtxt('median_timeDS.txt')
bar_chan = 0
bar_steps = [0]
for i,j,b,m in median_data:
    bar_chan += 1
    if b>117:
        bar_steps.append(bar_chan)
        print(bar_chan)
mean_times = median_data[:,-1]
chan_times = np.arange(len(mean_times))

def line_c(x, A, y_c, x_c):
    return A*(x-x_c)+y_c


corr_times = np.array(mean_times).copy()

plt.figure(figsize=(11,4))
time_slopes, median_times = [], []
corr_off = []
for i in range(len(bar_steps)-1):
    if (bar_steps[i+1] - bar_steps[i])<2: continue
    for left, right in [(bar_steps[i], (bar_steps[i+1] + bar_steps[i])//2), ((bar_steps[i+1] + bar_steps[i])//2, bar_steps[i+1]-6)]:
        chan_slice = chan_times[left:right]
        chan_x_slice = chan_slice%left if left>0 else chan_slice
        time_slice = mean_times[left:right]
        mean_slice = np.median(time_slice)
        x_c = 15 #(right-left)/2 # chan_x_slice[-1]/2
        saw_line = lambda x, A, y_c: line_c(x, A, y_c, x_c)
        # print('x_c:', chan_x_slice[-1]/2)
        par_opt, par_cov = curve_fit(saw_line, chan_x_slice, time_slice)
        time_slopes.append(par_opt[0])# if (bar_steps[i+1] - bar_steps[i])<100 else par_opt[0]*2)
        mean_slice = par_opt[1]
        median_times.append(mean_slice)
        corr_times[left:right] = corr_times[left:right] - saw_line(chan_x_slice,np.sign(par_opt[0])*0.0858, mean_slice)
        corr_off.append(np.median(corr_times[left:right]))
        
        #plt.plot(chan_slice, saw_line(chan_x_slice,par_opt[0]), 'r')
        plt.plot(chan_slice, saw_line(chan_x_slice,np.sign(par_opt[0])*0.0858, mean_slice), 'g')
        plt.plot(chan_slice, np.ones(len(chan_slice))*mean_slice, color='magenta')
        print(left, right,':\t', np.around(par_opt[0], decimals=3), np.around(mean_slice, decimals=3),'\n')
    
print()
print('Slopes: ', np.around(time_slopes, decimals=3))
print('Mean_abs slope: ', np.around(np.abs(time_slopes).mean(), decimals=4))
print('Offsets: ', np.around(median_times, decimals=3))

    # np.savetxt('gaus_offsets_ds.txt', median_times, fmt='%.3f')   

plt.step(chan_times, mean_times, alpha=0.5)
plt.grid()
plt.minorticks_on()
plt.ylim(-8,-3)
plt.title('Median times')
# plt.savefig('gaus_time_align_DS.png')
plt.show()

corr_times[np.abs(corr_times)>0.5] = 0
print('\nCorrected offsets: ', np.around(corr_off, decimals=3))
print('Corrected times spread:', np.around(np.std(corr_times), decimals=3))


plt.figure(figsize=(11,4))
plt.step(chan_times, corr_times, alpha=0.5)
plt.grid()
plt.minorticks_on()
plt.ylim(-1,1)
plt.title('Corrected times')
# plt.savefig('gaus_corr_time_align_DS.png')
plt.show()


