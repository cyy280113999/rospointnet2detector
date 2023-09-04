import os
import sys
import open3d as o3d
import numpy as np
import tqdm
pj=lambda *args:os.path.join(*args)

# x y z [intensity, label] (if exist)
source_dir='/home/cyy/datasets/pc071803l/'
dest_dir='/home/cyy/datasets/pc071803l_pcd/'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
fs=list(os.listdir(source_dir))
fs=[x for x in fs if os.path.splitext(x)[1]=='.npy']
for filename in tqdm.tqdm(fs):
    pure_name,sufix=os.path.splitext(filename)
    x = np.load(pj(source_dir,filename))
    pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(x[:,:3])) # set position
    if x.shape[1]>=4:
        pcd.point.intensity=o3d.core.Tensor(x[:,3:4]) # set intensity
    if x.shape[1]>=5:
        pcd.point.label=o3d.core.Tensor(x[:,4:5].astype(np.int32)) # set label
    o3d.t.io.write_point_cloud(pj(dest_dir,f'{pure_name}.pcd'),pcd,write_ascii=False)
