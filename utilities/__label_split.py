import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import numpy as np
import tqdm


pj=lambda *args:os.path.join(*args)

ds='./labeled'
dataset_dir1='/home/cyy/datasets/lidar_ds1'
dataset_dir2='/home/cyy/datasets/lidar_ds2'
positions_dir = "pts"
labels_dir = "seg"
os.makedirs(dataset_dir1)
os.makedirs(pj(dataset_dir1,positions_dir))
os.makedirs(pj(dataset_dir1,labels_dir))
os.makedirs(dataset_dir2)
os.makedirs(pj(dataset_dir2,positions_dir))
os.makedirs(pj(dataset_dir2,labels_dir))

for e in os.scandir(ds):
    filename = e.name.split('.')[0]
    
    # pcd
    # x = np.load(pj('./saved_pcs',filename+'.pcd'))
    # npy
    # x = np.load(filename+'.npy')
    # npy
    x = np.genfromtxt(pj(ds,filename+'.pcd'), skip_header=11)
    x=x[:,:4]  # remove dim 4
    x=x[x[:,3]!=3] # remove label 3

    x1=x.copy()
    label = x1[:,3]
    x1[label==2,3]=1
    positions = x1[:, :3]
    labels = x1[:, 3].astype(int)
    np.savetxt(pj(dataset_dir1,positions_dir,filename+'.pts'), positions, delimiter=' ')
    np.savetxt(pj(dataset_dir1,labels_dir,filename+'.seg'), labels, fmt='%d')

    label = x[:,3]
    x2=x[label!=0]
    label = x2[:,3]
    x2[label==1,3]=0
    x2[label==2,3]=1
    positions = x2[:, :3]
    labels = x2[:, 3].astype(int)
    np.savetxt(pj(dataset_dir2,positions_dir,filename+'.pts'), positions, delimiter=' ')
    np.savetxt(pj(dataset_dir2,labels_dir,filename+'.seg'), labels, fmt='%d')
