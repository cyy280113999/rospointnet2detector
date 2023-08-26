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

# 为了使用semantic segmentation tools贴标签，转换为pcd格式
# 采集的npy点云数据保存在：
source_dir='/home/cyy/datasets/pc071803/'
# 直接将pcd存到标签工具的文件夹
dest_dir='/home/cyy/sse-images/'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
for e in tqdm.tqdm(os.scandir(source_dir)):
    x = np.load(pj(source_dir,e.name))
    # new t.pc
    point_cloud = o3d.t.geometry.PointCloud()
    point_cloud.point["positions"] = o3d.core.Tensor(x[:,:3], dtype, device)
    # point_cloud.point["intensities"] = o3d.core.Tensor(x[:,3,None], dtype, device)# pcd格式不能包含除xyz外的其他数据，因此最终需要将两者的融合
    o3d.t.io.write_point_cloud(pj(dest_dir,f'{e.name.split(".")[0]}.pcd'), point_cloud, write_ascii=True)

    # old 
    # 创建一个PointCloud对象并将NumPy数组赋值给它
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(x[:,:3]) # pcd格式不能包含除xyz外的其他数据，因此最终需要将两者的融合

    # 保存PointCloud对象为PCD文件
    # o3d.io.write_point_cloud(pj(des_dir,f'{e.name.split(".")[0]}.pcd'), point_cloud, write_ascii=True)
    
# 下一步，为点云添加标签，运行
# cd ~/temp/semantic-segmentation-editor-x.x.x
# meteor npm install
# meteor npm start
# 得到包含标签的pcd文件