import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


data_header = ['detID', 'id_side', 'delta_t', 'isB1', 'isB2', 'isB1only', 'isB2notB1']
header2id = {'detID':0, 'id_side':1, 'delta_t':2, 'isB1':3, 'isB2':4, 'isB1only':5, 'isB2notB1':6}

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
