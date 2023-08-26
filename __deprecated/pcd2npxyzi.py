import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import open3d as o3d
import numpy as np
import tqdm

source_dir='/home/cyy/datasets/pc082601pcd/'
# 直接将pcd存到标签工具的文件夹
dest_dir='/home/cyy/datasets/pc082601/'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
fs=list(os.listdir(source_dir))
fs=[x for x in fs if os.path.splitext(x)[1]=='.pcd']
for filename in tqdm.tqdm(fs):
    pure_name,sufix=filename.split('.')
    pc = o3d.io.read_point_cloud(pj(source_dir,filename))
    x=np.array(pc.points)
    x=np.concatenate([x,np.zeros((x.shape[0],1))],axis=1)
    np.save(pj(dest_dir,f'{pure_name}.npy'),x)