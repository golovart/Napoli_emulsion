import ROOT
import os, re, gc, h5py, sys, io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from contextlib import contextmanager
import ctypes
import tempfile


ROOT.gErrorIgnoreLevel = ROOT.kError

# parameters
scan_name = 'ADDA/col/NP40' # 'fog/Scan1'
log_dir = 'logs/'+scan_name+'/'
out_data_dir = '/home/scanner-ml/ML-drive/Artem/Python/NEWS/ADDA/article/experim_data/'
part = 1000
dr = 40; n_col = 3


### OUTPUT REDIRECTOR
###
libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)

### END OF OUTPUT REDIRECTOR
###

def name_path(name, root=True):
    #'/mnt/NEWS/Artem/Colour_70nm/Oct2019_Feb2020_N123gf/21.02.20_C100keV/45degr/Scan1'
    #'/mnt/ML-drive/Artem/ADDA/col/NP40'
    return '/mnt/ML-drive/Artem/'+name
    path = '/mnt/ML-drive/Artem/'
    if 'ADDA' in name:
        if root:
            return '/mnt/NEWS/Artem/10g_run/06.08.21_Run2/'+name.split('/')[2]
        else:
            return '/mnt/ML-drive/Artem/'+name
    if name=='fog': 
        name = 'ref'
    dirs = os.listdir(path)
    for n in dirs:
        if name in n.lower():
            path += n+'/scan'
            return path
    return path
#     if not 'fog' in name:
#         path += ([n for n in dirs if name.split('/')[1] in n])[0]+'/'
#         if not 'test' in name:
#             path += name.split('/')[-1]+'/Scan1'
#         else:
#             path += '0degr/Scan2'
#     else:
#         path += ([n for n in dirs if 'Fog' in n])[0]+'/flim/'+name.split('/')[-1]
#     return path



def load_root_col_images(imcheck, root_run, dr=20, n_col=3, return_pols=False):
    """
    Loads images into numpy array of shape (N_im,2*dr,2*dr,n_pol+1)
    Empty pol images are filled with ZEROS

    Arguments:
    imcheck -- DataFrame with samples details: HeaderID, ViewID, GrainID, cluser_ids, micro-track_flag, num_polarizations
    root_run -- DMRRun from the ROOT file with data (after InitFramesMap)
    dr -- radius of the image to load
    n_cols -- number of colours of each image

    Returns:
    images -- numpy array with N_im cluster images
    """

    pols = []
    imgs = np.zeros((imcheck.shape[0],dr*2,dr*2,n_col+1), dtype=np.uint8)
    err_vid = set()
#     caput = StringIO()
#     sys.stdout = caput
    #for ind, row in tqdm(imcheck.iterrows(), total=imcheck.shape[0], desc='loading event data'):
    for ind, row in imcheck.iterrows():
        ind = ind % imcheck.shape[0]
        clust_im = np.zeros((dr*2,dr*2,n_col+1), dtype=np.uint8)
        skip = False
        for i_col in range(n_col+1):
            ### GetGRIM(iv, igr, ipol, icol, r)
            im = root_run.GetGRIM(int(row['HeaderID']), int(row['GrainID']), 0, i_col, dr) #arun.GetCLIMBFC(int(row['HeaderID']), int(row['pol'+str(i_pol)]), i_pol, dr, int(x0), int(y0) )
            ROOT.SetOwnership(im, True)
            try:
                h = im.GetHist2()
                ROOT.SetOwnership(h, True)
            except:
                if not int(row['HeaderID']) in err_vid: err_vid |= {int(row['HeaderID'])}
                skip = True
                del im
                #sys.stdout = sys.__stdout__
                continue
            for i,y_ind in enumerate(range(h.GetNbinsY(),1,-1)):
                for j,x_ind in enumerate(range(2,h.GetNbinsX()+1)):
                    clust_im[i,j,i_col] = h.GetBinContent(x_ind,y_ind)
            del h,im
            gc.collect()
            #if not clust_im[...,i_col].all(): skip = True
            #print(i_col); plt.imshow(clust_im[...,i_col]); plt.colorbar(); plt.show()
        #print(skip)
        if skip: 
            pols.append(-1*np.ones_like(row.values))
            continue
        imgs[ind] = clust_im[...]
        pols.append(row.values)
#     sys.stdout = sys.__stdout__
    #imgs = np.array(imgs, dtype=np.uint8)
    pols = np.array(pols)
#     del caput
    gc.collect()
    #print('fake views:', len(err_vid),'\n\n',err_vid,'\n')
    if return_pols: return imgs, pols, err_vid
    return imgs, err_vid



# path_dir = name_path(scan_name)
col_vgr = np.loadtxt(name_path(scan_name, root=False)+'/imcheck_bfcl.txt', delimiter=',')
#col_iso = col_vgr[:,-1]==-1
#col_vgr = (col_vgr[col_iso])[:,[-3,-1]]
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


print(scan_name,'\n')

start = datetime.now()
#params = dict(path=path_dir, name=name, name_dict=name_dict, dr=40, n_pol=8, return_pols=True)
#fold = datetime.now()
#print(name_dict[name])
col_ids = pd.DataFrame(col_vgr, columns=['HeaderID','ViewID','GrainID','isolated'])
# col_ids = pd.DataFrame(col_vgr, columns=['HeaderID','GrainID'])
num_parts = col_ids.shape[0]//part if col_ids.shape[0]/part==col_ids.shape[0]//part else (col_ids.shape[0]//part+1)
print('\n Total parts:', num_parts, '\t part size:', part, '\n')
err_hid = set()
root_file = name_path(scan_name, root=True)+'/dm_tracks_cl.dm.root'
f = ROOT.TFile.Open(root_file,'read')
ROOT.gSystem.Load('libDMRoot')
arun = ROOT.DMRRun(root_file)
ROOT.SetOwnership(arun, True)
#arun.GetHeader().Print()
arun.InitFramesMap()

for id_part in tqdm(range(num_parts), desc='loading data part'):
    ### loading only 1 portion of images
    if id_part>0: break
    f_redir = io.BytesIO()
    with stdout_redirector(f_redir):
        ims, pols, errs = load_root_col_images(col_ids[id_part*part:(id_part+1)*part], arun, dr=dr, n_col=n_col, return_pols=True)
    with open(log_dir+'out_log_p'+str(id_part)+'.log','w') as fout:
        fout.write(f_redir.getvalue().decode('utf-8'))
    with h5py.File(out_data_dir+'data_raw_root_ims_col_'+str(dr*2)+'.h5','a') as dfile:
        dfile.create_dataset(scan_name+'/part'+str(id_part)+'/images', data=ims)
        dfile.create_dataset(scan_name+'/part'+str(id_part)+'/pol_ids', data=pols)
    #print('\n\nloaded in ',datetime.now()-fold,'\n')
    del ims, pols, f_redir
    err_hid |= errs
    gc.collect();

del arun
f.Close()
gc.collect();
print('total loading time:',datetime.now()-start)
print('fake header ids:', len(err_hid),'\n\n',err_hid,'\n')
print('\n'*10)
print('FINISHED')
