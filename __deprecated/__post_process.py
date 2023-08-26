import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import open3d as o3d
import numpy as np
import pyransac3d as r3d


# 读取点云数据
pc = o3d.io.read_point_cloud("./bk/pc_2.pcd")

# 转换为numpy数组
x_np = np.asarray(pc.points)

# 创建Open3D点云对象
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(x_np)

# 进行体素滤波
# pc = pc.voxel_down_sample(voxel_size=0.02)

# 去除离群值
# pcd,_ = pcd.remove_radius_outlier(nb_points=5,radius=0.05)

# 进行聚类
labels = pc.cluster_dbscan(eps=0.03, min_points=3)

# # 将聚类结果可视化
max_label = max(labels)
colors = np.random.randint(1, 255, size=(max_label + 1, 3)) / 255.
colors = colors[labels]             # 每个点云根据label确定颜色
colors[np.array(labels) < 0] = 0    # 噪点配置为黑色
pc.colors = o3d.utility.Vector3dVector(colors)  # 格式转换(由于pcd.colors需要设置为Vector3dVector格式)


# -------------直线检测
x_np = np.asarray(pc.points)
# c,a,r,i = r3d.Cylinder().fit(x_np, 0.001)
# mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=r,
#                                                           height=np.linalg.norm(a))
# cylinder = o3d.geometry.Cylinder(radius=r, height=np.linalg.norm(a))
# cylinder.transform = np.eye(4)
# cylinder.transform[:3, 3] = c

xs=[]
for i in range(max_label+1):
    xs.append(x_np[np.array(labels)==i])

arrows=[]
for i in range(max_label+1):
    a,b,i = r3d.Line().fit(xs[i], thresh=0.1, maxIteration=1000)
    # e,i=r3d.Plane().fit(x_np, thresh=0.05, minPoints=100, maxIteration=1000)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.005, cone_radius=0.01, cylinder_height=0.02, 
        cone_height=0.01, resolution=20, cylinder_split=4, cone_split=1)
    a_normalized = a / np.linalg.norm(a)
    standard_direction = np.array([0, 0, 1])
    standard_direction_normalized = standard_direction / np.linalg.norm(standard_direction)
    rotation_axis = np.cross(standard_direction_normalized, a_normalized)
    cos_theta = np.dot(standard_direction_normalized, a_normalized)
    theta = np.arccos(cos_theta)
    from scipy.spatial.transform import Rotation

    rotation_matrix = Rotation.from_rotvec(theta * rotation_axis).as_matrix()
    # arrow.transform(rotation_matrix)
    # import copy
    # arrow2 = copy.deepcopy(arrow)
    arrow.translate(b)
    arrow.rotate(rotation_matrix)
    arrows.append(arrow)
    
    # new_i=np.ones(len(x_np))
    # new_i[i]=0
    # new_i=new_i.astype(bool)
    # x_np=x_np[new_i]


# x_np=x_np[i]


# 检测直线
# lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pc, pc)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# 可视化点云列表
o3d.visualization.draw_geometries([pc,axis,*arrows],
                                        window_name="cluster",
                                        width=800,
                                        height=600)