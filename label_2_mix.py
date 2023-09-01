import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import numpy as np
import tqdm
import open3d as o3d
from transfer_dataset import strSort
# 首先使用工具对pcd文件添加标签!
# cd ~/temp/semantic-segmentation-editor-x.x.x
# meteor npm install
# meteor npm start
# 得到包含标签的pcd文件

# 标签文件没有额外intensity信息，将两者混合
# mix the point (xyz intensity) and (label)
point_dir='/home/cyy/datasets/pcEnhancePreLabel/' # xyz intensity (label)
label_dir='/home/cyy/Downloads/' # xyz label
dest_dir='/home/cyy/datasets/pcEnhancel/' # save mixed npy to ..
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
fs=list(os.listdir(label_dir))  # may not label all, label dir has less
fs=[x for x in fs if os.path.splitext(x)[1]=='.pcd']
for filename in tqdm.tqdm(fs): 
    pt = o3d.t.io.read_point_cloud(pj(point_dir,filename))
    lb = o3d.t.io.read_point_cloud(pj(label_dir,filename))
    pt.point.label=lb.point.label # copy label
    o3d.t.io.write_point_cloud(pj(dest_dir,filename),pt,write_ascii=False)



