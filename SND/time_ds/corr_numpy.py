import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import matplotlib as mpl
from datetime import datetime
from scipy.optimize import curve_fit

start = datetime.now()
data_header = ['detID', 'id_side', 'delta_t', 'isB1', 'isB2', 'isB1only', 'isB2notB1']
header2id = {'detID':0, 'id_side':1, 'delta_t':2, 'isB1':3, 'isB2':4, 'isB1only':5, 'isB2notB1':6}

time_data = np.ones((0,7))
# time_data = np.ones((0,3))
names_condor = os.listdir('numpy_out')
counter = 0
for name_p in tqdm(names_condor, desc='numpy partitions loading:'):
    name_p = 'numpy_out/'+name_p
    time_data = np.vstack((time_data, np.load(name_p)))
    counter += 1
    # if counter>2: break

mean_offsets = np.loadtxt('mean_offsets.txt')
slope = 0.0858

def line_c(x, A, y_c, x_c):
    return A*(x-x_c)+y_c

print('Data shape:', time_data.shape)

corr_data = np.copy(time_data[:,:3])
#for time_line in tqdm(time_data, desc='correcting time'):
detID = time_data[:,0].astype(int)
id0 = (detID - 30000) // 1000
id1 = ((detID - 30000) % 1000) // 60
id2 = ((detID - 30000) % 1000) % 60
id_half = (((detID - 30000) % 1000) % 60) // 30
i_side = time_data[:,1].astype(int)%2
id_off = id0.copy()*3
id_off[id0>=3] = 7
id_off = (id_off + (id1*2) + i_side)*2 + id_half
slope_array = np.ones(len(detID))*slope
slope_array[id_half==0] = -1*slope

#print(id0.shape, id1.shape, id2.shape, slope_array.shape, id_off.shape)

#plt.hist(id2%30, range=(0,30))
#plt.show()
#plt.hist(slope_array, range=(-0.1, 0.1))
#plt.show()

corr_data[:,2] = time_data[:,2] - slope_array*(id2%30 - 15) - mean_offsets[id_off]
#corr_data[:,2] = time_data[:,2] - line_c(id2%30, slope_array, mean_offsets[id_off], x_c=15)
#corr_data = np.vstack((corr_data, [time_line[0], time_line[1], corr_time]))
'''
for i in range(10):
    print('\n',i)
    print(time_data[i,:3])
    print(id0[i], id1[i], id2[i], slope_array[i], mean_offsets[id_off[i]], id_half[i])
    print(corr_data[i])
'''

np.save('corrected_timeDS.npy', corr_data)
print('Correction time:', datetime.now()-start)
print('Done')



