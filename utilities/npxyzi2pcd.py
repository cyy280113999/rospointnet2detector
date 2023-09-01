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


source_dir='/home/cyy/datasets/pc071803/'
dest_dir='/home/cyy/datasets/pc071803_pcd/'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
fs=list(os.listdir(source_dir))
fs=[x for x in fs if os.path.splitext(x)[1]=='.npy']
for filename in tqdm.tqdm(fs):
    pure_name,sufix=os.path.splitext(filename)
    x = np.load(pj(source_dir,filename))
    assert x.shape[1]==4
    pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(x[:,:3])) # set position
    pcd.point.intensity=o3d.core.Tensor(x[:,3:4]) # set intensity 
    o3d.t.io.write_point_cloud(pj(dest_dir,f'{pure_name}.pcd'),pcd,write_ascii=False)
