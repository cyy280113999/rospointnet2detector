import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import rospy
import numpy as np
import ros_numpy as rosnp
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import time
from transfer_dataset import np_xyz2pc,np_xyzi2pc, strSort
import shutil

# ======== pc dataset filter
# use recorder to save pcs.
# most saved pc are unused(repeatedly). 
# this filter show each pc in rviz, 
# wait keyboard to manually keep those useful pcs. 

source_dir='/home/cyy/datasets/pc0602' # where is the point cloud?
topic_pc_from='/rslidar_points' # visualization topic
lidar_frame='rslidar'
dest_dir='/home/cyy/datasets/pc_filtered'# where to save
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

def waitForConnected(tp_pub):
    count=0
    while True:
        time.sleep(0.1) 
        cn = tp_pub.get_num_connections()
        # if not established, connect count is 0
        if cn!=0:
            print(f'publisher established. topic connect : {cn}')
            break
        count+=1
        if count>100:
            print('cannot establish topic.')
            sys.exit(1)
    return

def main():
    rospy.init_node('pc_filter', anonymous=False)
    pc_pub = rospy.Publisher(topic_pc_from, PointCloud2, queue_size=1)
    filenames=[e.path for e in os.scandir(source_dir)]
    filenames.sort(key=strSort)
    save_indices=[]
    # wait until publisher established, 1s is enough.
    waitForConnected(pc_pub)
    print(f'[{time.asctime()}] filter start.')
    for i, fp in enumerate(filenames):
        x = np.load(fp)
        x = np_xyzi2pc(x)
        x = rosnp.point_cloud2.array_to_pointcloud2(x, frame_id=lidar_frame)
        pc_pub.publish(x)
        # x=x[:,:3]
        # pointcloud_o3d = o3d.geometry.PointCloud()
        # pointcloud_o3d.points = o3d.utility.Vector3dVector(x)
        # o3d.visualization.draw_geometries([pointcloud_o3d], window_name=fp, width=1600, height=800,left=50,top=50)
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pointcloud_o3d)
        # ctr = vis.get_view_control()
        # ctr.set_zoom(0.8)
        # ctr.set_front([-0.2, -0.3, -1.0])
        # ctr.set_lookat([0.5, 0.5, 0.5])
        # vis.run()

        # rater.sleep()
        # time.sleep(3)
        # rospy.spin()
        print(f'keep {fp} ? (./3)')
        cmd=input()
        if cmd[0]=='.':
            save_indices.append(i)
    print('filter over. saving files...')
    for i,idx in enumerate(save_indices):
        shutil.copy(filenames[idx],pj(dest_dir,f'pc_{i:06d}.npy'))# new index by order
    print('save over.')

if __name__ == '__main__':
    main()