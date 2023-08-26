"""
old dataset use .pts & .seg
"""

import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import numpy as np



# 首先使用工具对pcd文件添加标签!
# cd ~/temp/semantic-segmentation-editor-x.x.x
# meteor npm install
# meteor npm start

pj=lambda *args:os.path.join(*args)

ds='./labeled'
dataset_dir='/home/cyy/datasets/lidar_ds'
positions_dir = "pts"
labels_dir = "seg"
os.makedirs(dataset_dir)
os.makedirs(pj(dataset_dir,positions_dir))
os.makedirs(pj(dataset_dir,labels_dir))

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

    # 获取位置和标签信息
    positions = x[:, :3]
    labels = x[:, 3].astype(int)

    # 保存位置信息到文件
    np.savetxt(pj(dataset_dir,positions_dir,filename+'.pts'), positions, delimiter=' ')

    # 保存标签信息到文件
    np.savetxt(pj(dataset_dir,labels_dir,filename+'.seg'), labels, fmt='%d')

