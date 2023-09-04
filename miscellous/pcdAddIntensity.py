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

source_dir='/home/cyy/datasets/pcEnhance/'
fs=list(os.listdir(source_dir))
fs=[x for x in fs if os.path.splitext(x)[1]=='.pcd']
for filename in tqdm.tqdm(fs):
    pcd = o3d.t.io.read_point_cloud(pj(source_dir,filename))
    n=pcd.point.positions.numpy().shape[0]
    intensity=np.full((n,1),20,dtype=np.float32)
    intensity+=np.random.randint(0,10,(n,1))
    pcd.point.intensity=o3d.core.Tensor(intensity)
    o3d.t.io.write_point_cloud(pj(source_dir,filename), pcd, write_ascii=False)