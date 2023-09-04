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

# x y z [intensity, label] (if exist, unknown order)
source_dir='/home/cyy/datasets/pcEnhancep/'
dest_dir='/home/cyy/datasets/pcEnhancep_npy/'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
fs=list(os.listdir(source_dir))
fs=[x for x in fs if os.path.splitext(x)[1]=='.pcd']
for filename in tqdm.tqdm(fs):
    pure_name,sufix=os.path.splitext(filename)
    pc = o3d.t.io.read_point_cloud(pj(source_dir,filename))
    data=dict(pc.point.items())
    xs=[data['positions'].numpy()]
    if 'intensity' in data:
        xs.append(data['intensity'].numpy())
    if 'label' in data:
        xs.append(data['label'].numpy().astype(np.float32))
    x=np.concatenate(xs,axis=1)
    np.save(pj(dest_dir,f'{pure_name}.npy'),x)