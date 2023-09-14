import os
import tqdm
pj=lambda *args:os.path.join(*args)
import open3d as o3d
import numpy as np
std_len='06d'
# source_dir='saved_npys/' # test dir
# source_dir='/home/cyy/datasets/pc_filtered'
# source_dir=f'/home/cyy/datasets/pcEnhance_pcd'
source_dir=f'/home/lhk/datasets/pc_log/2023-09-11' # 500

fs=sorted(list(os.listdir(source_dir)))
for i,filename in tqdm.tqdm(enumerate(fs)):
    pcd = o3d.t.io.read_point_cloud(pj(source_dir,filename))
    pcd.point.positions=o3d.core.Tensor(pcd.point.positions.numpy().astype(np.float32))
    pcd.point.intensity=o3d.core.Tensor(pcd.point.intensity.numpy().astype(np.float32))
    o3d.t.io.write_point_cloud(pj(source_dir,filename),pcd,write_ascii=False)
    