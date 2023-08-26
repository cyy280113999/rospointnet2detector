import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import pyransac3d as r3d
import tqdm

def main():
    # === 从numpy数组创建Open3D点云对象
    fn='/home/cyy/datasets/pc_filtered/pc_000400.npy' # 400
    x = np.load(fn)[:,:3].astype(np.float32)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(x)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # 网格线
    size = 10.0
    points = []
    for i in np.arange(-size, size + 1, 1):
        points.append([i, -size, 0]) # xy-y
        points.append([i, size, 0])
        points.append([-size, i, 0]) # xy-x
        points.append([size, i, 0])
        # points.append([-size, 0, i]) # xz-x 竖着的不画
        # points.append([size, 0, i])
        # points.append([i, 0, -size]) # xz-z
        # points.append([i, 0, size])
        points.append([6, -size, i]) # yz-y, x=6
        points.append([6, size, i])
        points.append([6, i, -size]) # yz-z
        points.append([6, i, size])
    points = np.array(points)
    # 创建线的索引
    lines = []
    for i in range(0, len(points), 2):
        lines.append([i, i + 1]) # 两个一组，两点间直线
    lines = np.array(lines)
    # 创建自定义的网格对象
    grids = o3d.geometry.LineSet()
    grids.points = o3d.utility.Vector3dVector(points)
    grids.lines = o3d.utility.Vector2iVector(lines)
    # o3d.visualization.draw_geometries([point_cloud, axis, grids],window_name='点云加网格')

    # ===========           执行RANSAC平面分割           =====================
    distance_threshold=0.1 # 每条线内部是连续的，这里看的是线间距，0.1m，10cm比较合适
    ransac_n=1000 # 至少多少个点在平面内，每次采集7000个点，1000个属于平面可以
    num_iterations=1000
    plane_model, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
    # inliers是分割出的平面上的点的索引
    # 可视化分割结果
    # inlier_cloud = point_cloud.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1, 0, 0])  # 分割出的平面上的点显示为红色
    # outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    # outlier_cloud.paint_uniform_color([0, 1, 0])  # 非平面上的点显示为绿色
    # 将两个点云合并，可视化分割结果
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],window_name='颜色区分，分割结果')
    # 可视化平面的法向量 norm vector
    if plane_model[0]<0:
        plane_model = -plane_model
    a=plane_model[:3] # 法向量方向
    a = a / np.linalg.norm(a)
    # 创建一个可视化箭头指示法向量
    # arrow = o3d.geometry.TriangleMesh.create_arrow(
    #         cylinder_radius=0.1, cone_radius=0.2, cylinder_height=2, 
    #         cone_height=1, resolution=20, cylinder_split=4, cone_split=1)
    # standard_direction = np.array([0, 0, 1]) # 创建的箭头方向默认是z轴
    # o3d.visualization.draw_geometries([arrow, axis, grids],window_name='arrow test')
    # rotation_axis = np.cross(standard_direction, a) # 二者垂直方向，将z轴的箭头转至a轴（法向量位置）
    # rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
    # cos_theta = np.dot(standard_direction, a)
    # theta = np.arccos(cos_theta) # 二者夹角，转角
    # rotation_matrix = Rotation.from_rotvec(theta * rotation_axis).as_matrix()
    # arrow.rotate(rotation_matrix) # 旋转箭头到法向量
    # arrow.translate(np.array([5,0,0])) # 旋转是按某个中心旋转的，旋转中心不是0，是0.5左右，旋转平移的顺序无所谓
    # o3d.visualization.draw_geometries([inlier_cloud, arrow, axis, grids],window_name='法向量')# 展示

    # ---------- 下采样测试
    # plane_vox = inlier_cloud.voxel_down_sample(voxel_size=0.05) # 网格下采样
    # o3d.visualization.draw_geometries([plane_vox, arrow, axis],window_name='网格下采样')
    # cl, ind = plane_vox.remove_radius_outlier(nb_points=5, radius=0.2) # 离群下采样
    # o3d.visualization.draw_geometries([plane_vox.select_by_index(ind), arrow, axis, grids],window_name='离群下采样')

    # ransac 确实能准确的分割平面，并给出其法向量
    # ==============             坐标规整，1，正视            =============
    standard_direction = np.array([1.0,0,0]) # 箭头方向，规整方向为x轴前向
    rotation_axis = np.cross(a, standard_direction) # 将a轴的转向x轴
    rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
    cos_theta = np.dot(standard_direction, a)
    theta = np.arccos(cos_theta) # 二者夹角
    rotation_matrix1 = Rotation.from_rotvec(theta * rotation_axis).as_matrix()
    norm1 = point_cloud.rotate(rotation_matrix1) # 将所有点旋转， 这个旋转矩阵非常重要
    norm1.paint_uniform_color([0, 0, 1])
    # point_cloud.paint_uniform_color([1, 0, 0]) # 这里两个索引的更新冲突，所以rotate是inplace操作
    # o3d.visualization.draw_geometries([point_cloud],window_name='旋转是不是inplace') # 是
    # o3d.visualization.draw_geometries([norm1, grids])
    plane_model, inliers = norm1.segment_plane(distance_threshold, ransac_n, num_iterations)  # 重新分割平面
    plane_pc = norm1.select_by_index(inliers)
    plane_pc.paint_uniform_color([1, 0, 0])  # 分割出的平面上的点显示为红色
    # o3d.visualization.draw_geometries([plane_pc, grids],window_name='正视')

    # 法向量少一个旋转自由度。类似上面的做法，下面对路面进行直线估计，使路面旋转至y轴正方向
    # ==============             坐标规整，2，转动            =============
    # 直线拟合需要去噪，吗？
    down_pc = plane_pc.voxel_down_sample(voxel_size=0.1)
    cl, ind = down_pc.remove_radius_outlier(nb_points=5, radius=0.3) # 离群下采样 至少30cm内有5个邻近点
    down_pc = down_pc.select_by_index(ind)
    # o3d.visualization.draw_geometries([down_pc, grids],window_name='直线拟合前下采样')
    x=np.array(down_pc.points)
    A = np.column_stack([x[:, 1], np.ones_like(x[:, 1])]) # y,z，以y为拟合的x
    r, _, _, _ = np.linalg.lstsq(A, x[:, 2], rcond=None) # z 为拟合的y
    # 估计的直线参数
    slope = r[0]
    intercept = r[1]
    # 输出直线方程
    # print(f"估计的直线方程为：x = {slope}*y + {intercept}")
    # 方向向量
    a = np.array([0, 1, slope])
    a = a/np.linalg.norm(a)

    # # ransac 直线法，不稳定，已弃用
    # line_thre=5  # 路面范围，路面全包括，不能太大
    # vs=[]
    # counter=0
    # for i in tqdm.tqdm(range(100)):
    #     a,b,ind = r3d.Line().fit(x, thresh=line_thre, maxIteration=100)
    #     if a[1]<0:
    #         a=-a
    #     a = a / np.linalg.norm(a)
    #     if len(x)==len(ind):
    #         vs.append(a)
    #         counter+=1
    #     if counter>=10:
    #         break
    # assert len(vs)>0
    # vs=np.array(vs)
    # print(vs)
    # a = vs.mean(axis=0)
    # a = a / np.linalg.norm(a)
    # print(a)
    # print(vs.std(axis=0))


    # 看看估计的结果
    arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.5, cone_radius=1, cylinder_height=15, 
            cone_height=6, resolution=20, cylinder_split=4, cone_split=1)
    standard_direction = np.array([0,0,1.0])#转这个箭头
    rotation_axis = np.cross(standard_direction, a) # z到a
    rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
    cos_theta = np.dot(standard_direction, a)
    theta = np.arccos(cos_theta)
    rotation_matrix = Rotation.from_rotvec(theta * rotation_axis).as_matrix()
    arrow.rotate(rotation_matrix)
    arrow.translate(np.array([6,0,-2]))#移动
    o3d.visualization.draw_geometries([down_pc, arrow, axis, grids],window_name='直线方向')# 展示

    standard_direction = np.array([0,1.0,0]) # 希望路面朝向y轴
    rotation_axis = np.cross(a, standard_direction) # 将a轴的转向y轴
    rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
    cos_theta = np.dot(standard_direction, a)
    theta = np.arccos(cos_theta)
    rotation_matrix2 = Rotation.from_rotvec(theta * rotation_axis).as_matrix()
    norm2 = norm1.rotate(rotation_matrix2) # 将所有点旋转， 这个旋转矩阵非常重要
    norm2.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([norm2, grids],window_name='正视+转动')

    # =============             一步法，组合两个矩阵       ===============
    rm=np.matmul(rotation_matrix2,rotation_matrix1)
    # rotate是inplace的，重新加载pointcloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.load(fn)[:,:3].astype(np.float32))
    rm=np.matmul(rotation_matrix2,rotation_matrix1)
    pc_norm = point_cloud.rotate(rm)
    pc_norm.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pc_norm, grids],window_name='规整')
    # 平移
    x=np.array(norm2.points)
    shift = x.mean(axis=0) # 减去这个
    shift[0]-=2 # 使减去的中心在[2,0,0]
    calib_info = np.zeros((4,3))
    calib_info[:3]=rm
    calib_info[3]=shift
    np.save('calib_info.npy',calib_info)



if __name__=='__main__':
    main()