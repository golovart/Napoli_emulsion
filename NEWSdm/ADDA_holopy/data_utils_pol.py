import ROOT
import os, re, gc, h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


def short_name(name, thread=None):
    name_tmp = ''
    if 'compare' in name: name_tmp = (name.split('/Scan')[1])[0]
    if 'gamma' in name: name = name.split('/')[0]
    for k in ['30keV','60keV','100keV']: 
        if k in name: name = name.split('/')[0]+'/'+k
    if 'reference' in name: return 'reference'+'/thr'+str(thread)
    if name_tmp: name += '/scan'+name_tmp
    if thread is None: return name
    else: return name+'/thr'+str(thread)
    
    

def load_root_images(yand, root_file, dr=16, n_pol=8, return_pols=False):
    """
    Loads images into numpy array of shape (N_im,2*dr,2*dr,n_pol+1)
    Empty pol images are filled with ZEROS
    
    Arguments:
    yand -- DataFrame with samples details: HeaderID, ViewID, GrainID, cluser_ids, micro-track_flag, num_polarizations
    root_file -- path to the ROOT file with data
    dr -- radius of the image to load
    n_pols -- max number of polarizations of each image
    
    Returns:
    images -- numpy array with N_im cluster images
    """
    
    f = ROOT.TFile.Open(root_file,'read')
    ROOT.gSystem.Load('libDMRoot')
    arun = ROOT.DMRRun(root_file)
    arun.GetHeader().Print()
    arun.SetFixEncoderFaults(1)
    v = arun.GetView()
    ROOT.gStyle.SetOptStat("n")
    
    imgs,pols = [],[]
    for ind, row in tqdm(yand.iterrows(), total=yand.shape[0], desc='loading event data'):
        v = arun.GetEntry(int(row['HeaderID']),1,1,1,1,1,1)
        i_pol = 0
        pixX = arun.GetHeader().pixX
        pixY = arun.GetHeader().pixY
        nx = arun.GetHeader().npixX
        ny = arun.GetHeader().npixY
        hd = v.GetHD()

        if hd.flag or row['n_pol']<2: continue # or row['tr_flag']>=0
        while(row['pol'+str(i_pol)]==-1 and i_pol<8): i_pol += 1
        cl = v.GetCL(int(row['pol'+str(i_pol)]))
        frcl = v.GetFR(cl.ifr)
        fr = v.GetFR(frcl.iz, frcl.ipol)
        #print('cluster', hd.aid,' ', hd.flag, ' ', cl.igr,' ', cl.ID(),' ', cl.ipol)
        x0 = (cl.x+hd.x-fr.x)/pixX + nx/2
        y0 = (cl.y+hd.y-fr.y)/pixY + ny/2

        clust_im = np.zeros((dr*2,dr*2,n_pol+1), dtype=np.uint8)
        skip = False
        while(i_pol<8):
            if row['pol'+str(i_pol)]==-1: i_pol += 1; continue
            im = arun.GetCLIMBFC(int(row['HeaderID']), int(row['pol'+str(i_pol)]), i_pol, dr, int(x0), int(y0) )
            h = im.GetHist2()
            for i,y_ind in enumerate(range(h.GetNbinsY(),1,-1)):
                for j,x_ind in enumerate(range(2,h.GetNbinsX()+1)):
                    clust_im[i,j,i_pol] = h.GetBinContent(x_ind,y_ind)
            del h,im
            gc.collect()
            if not clust_im[...,i_pol].all(): skip = True
            i_pol += 1
        if skip: continue
        clust_im[...,-1] = clust_im[...,0]
        imgs.append(clust_im)
        pols.append(row.values)
    imgs = np.array(imgs, dtype=np.uint8)
    pols = np.array(pols)
    gc.collect()
    if return_pols: return imgs, pols
    return imgs


def load_pol_images(pol_frame, path, class_name='C30keV', n_pol=8):
    """
    Loads images into numpy array of shape (N,h,w,n_pol)
    Empty pol images are filled with ZEROS
    
    Arguments:
    pol_frame -- DataFrame with samples details: HeaderID, ViewID, GrainID, cluser_ids, micro-track_flag, num_polarizations
    path -- path to the directory with all the samples
    class_name -- name of the particular sample we load
    n_pols -- max number of polarizations (channels) of each image
    
    Returns:
    images -- numpy array with N images of shape (h,w) with n_pols channels
    """

    im_array = []
    for hdr, gr, *cl_ids, tr_fl, n_pols in pol_frame.drop(['ViewID'], axis=1).values:
        tmp_im = []
        for i, cl_i in enumerate(cl_ids):
            if cl_i!=-1:
                img_n = str(hdr)+'_gr_'+str(gr)+'_pol_'+str(i)+'_cl_'+str(cl_i)+'_tr_'+str(tr_fl)+'_npol_'+str(n_pols)+'.csv'
                tmp_im.append(pd.read_csv(path+class_name+'/csvs/'+img_n, header=None).drop(32, axis=1).values)
            else:
                tmp_im.append(np.zeros((32,32),dtype=np.uint8))
        if n_pol==9: tmp_im.append(tmp_im[0])
        im_array.append(np.array(tmp_im, dtype=np.uint8).T)
        gc.collect()
        
    return np.array(im_array)


def get_pol_feat(id_frame, n_pol, path_dir, class_name, feat_names):
    """
    Load features from ROOT file using the previously loaded cluster indices.
    
    Arguments:
    id_frame -- DataFrame with samples details: HeaderID, ViewID, GrainID, cluser_ids, micro-track_flag, num_polarizations
    n_pol -- max number of polarizations (channels) of each image
    path_dir -- path to the directory with samples in the ROOT files
    class_name -- name of the particular sample we load
    feat_names -- names of the features to load from ROOT files (last one must be elipticity, it is calculated on the fly)
    
    Returns:
    all_feat -- DataFrame with features of shape (N, n_pol*len(feat_names)+2)
    """
    
    f = ROOT.TFile.Open(path_dir+class_name+'/dm_tracks_cl.dm.root','read')
    t = f.Get('Vdmr')
    all_feat = np.zeros((0,len(feat_names)*n_pol+2))
    feat_array = []
    for i in range(n_pol):
        for name in feat_names:
            feat_array.append(name+str(i))
    feat_array.append('tr_flag')
    feat_array.append('n_pol')
    
    eps=1e-3
    for hdr, *cl_ids, tr_fl, n_pols in id_frame.drop(['ViewID','GrainID'], axis=1).values:
        pol_feat = []
        t.GetEntry(int(hdr))
        for cl_id in cl_ids:
            for name in feat_names[:-1]:
                if cl_id==-1:
                    pol_feat.append(0)
                else:
                    pol_feat.append(t.GetLeaf('cl.'+name).GetValue(int(cl_id)))
            if cl_id==-1:
                pol_feat.append(0)
            else:
                pol_feat.append( (t.GetLeaf('cl.lx').GetValue(int(cl_id))+eps)/(t.GetLeaf('cl.ly').GetValue(int(cl_id))+eps) )
        if n_pol==9:
            for i,name in enumerate(feat_names):
                pol_feat.append(pol_feat[i])        
        pol_feat.append( tr_fl )
        pol_feat.append( n_pols )
        all_feat = np.vstack((all_feat, pol_feat))
        gc.collect()
    return pd.DataFrame(all_feat, columns=feat_array)




def bad_inds(pol_frame, imgs=None, features=None, f_name=None, isolated=True, quant=0.999, n_pols=8, bad_i=[]):
    inds = set(pol_frame.index)
    bad_i = set(bad_i)
    
    if isolated: bad_i |= (inds-set( pol_frame[pol_frame['tr_flag']<0].index ))
    if features is not None:
        bad_i |= (inds-set(features.dropna().index))
        tmp_feat = features.copy()
        for i in range(n_pols):
            for n in ['lx','vol']:
                tmp_feat = tmp_feat[ tmp_feat[n+str(i)]<tmp_feat[n+str(i)].quantile(quant) ]
        bad_i |= (inds-set(tmp_feat.index))
    if imgs is not None:
        for i, im_pols in enumerate(imgs):
            for j, im in enumerate(im_pols.T):
                if j>=n_pols: continue
                if 'pol'+str(j) not in pol_frame.keys(): continue
                if  pol_frame['pol'+str(j)][i]!=-1 and not im.all(): bad_i |= {i}
    if f_name:
        print(len(bad_i),'\tBad ',f_name)
        np.savetxt('bad_sample_ids/'+f_name+'.txt', list(bad_i), fmt='%d')
    return list(bad_i)