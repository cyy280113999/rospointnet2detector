import os
import rospy
import numpy as np
import ros_numpy as rosnp
import threading
from sensor_msgs.msg import PointCloud2
import time
from utilities import *
from genpy.rostime import Time as RosTime
import open3d as o3d
import global_config as CONF
pj=lambda *args:os.path.join(*args)
"""
pcd dataset's ros topic publisher

"""
OFFLINE_RATE=10
# pointcloud_dir=f'/home/{usr_name}/datasets/saved_npys' # 4000
# pointcloud_dir=f'/home/{usr_name}/datasets/pc0602' # 1100
# pointcloud_dir=f'/home/{CONF.usr_name}/datasets/pc0602f' # 500
# pointcloud_dir=f'/home/{usr_name}/datasets/pc063003' # 382  
# pointcloud_dir=f'/home/{usr_name}/datasets/pc_filtered' # 1100
# pointcloud_dir=f'/home/{usr_name}/datasets/pc0701static' # 382
# pointcloud_dir=f'/home/{CONF.usr_name}/datasets/pc071803'
# pointcloud_dir=f'/mnt/d/wsl_tools/pc082303'
# pointcloud_dir=f'/home/{CONF.usr_name}/datasets/pc082601'
pointcloud_dir=f'/home/{CONF.usr_name}/datasets/pc0718m_pcd/' # 500
pointcloud_dir=f'/home/{CONF.usr_name}/datasets/pc063003_pcd/' # 500
pointcloud_dir=f'/home/{CONF.usr_name}/datasets/pc0720/' # 500
pointcloud_dir=f'/home/{CONF.usr_name}/datasets/pc0906/'


npoints=7000

# where to
CONF.TOPIC_OFFLINE

# label_color_map=(
#     (207,   207,    207),
#     (220,   220,    0),
#     (244,   35,     232),
#     (152,   251,    152),
#     (255,   136,    0),
#     (250,   207,    245),
#     (255,   0,      0),
#     (70,    130,    180)
# )
# stage=None
# label_filter=None
# label_map=None
# if stage is None:
#     npoints=4000
# elif stage==1:
#     npoints=4000
#     label_map=[(0,0),(1,1),(2,1)]
# elif stage==2:
#     npoints=2000
#     label_filter=[0]
#     label_map=[(1,0),(2,1)]

            # xyz=x_np[:,:3]
            # xyz = pc_calib(xyz,rm,hei)
            # x_np[:,:3]=xyz

def main():
    rospy.init_node('ros_offline_publisher', anonymous=False)
    pc_puber = PC_Offline_Publisher(source_dir=pointcloud_dir)
    pc_puber.start()
    pc_puber.command_loop()

class PC_Offline_Publisher:
    def __init__(self, source_dir):
        self.rater=rospy.Rate(OFFLINE_RATE)
        self.pc_pub = rospy.Publisher(CONF.TOPIC_OFFLINE, PointCloud2, queue_size=1)
        self.counter=0
        self.source_dir=source_dir
        fs=list(os.listdir(source_dir))
        fs=[x for x in fs if os.path.splitext(x)[1]=='.pcd']
        self.fs=sorted(fs,key=strSort)
        self.stop_flag=False
        self.round_read_ptr=None
    def start(self):
        self.stop_flag=False
        if self.round_read_ptr is None:
            self.round_read_ptr = threading.Thread(target=self.auto_pub)
            self.round_read_ptr.start()
    def command_loop(self):
        while True:
            print('0:auto pub, 2:pause, .:manually pub, 3:exit')
            cmd = input().split()
            if len(cmd)==0:
                continue
            if cmd[0]=='0':
                if self.round_read_ptr is None:
                    self.round_read_ptr = threading.Thread(target=self.auto_pub)
                    self.round_read_ptr.start()
            elif cmd[0]=='2':
                self.stop_flag=True
            elif cmd[0]=='.':
                if self.round_read_ptr is None:
                    self.read_and_publish()
            elif cmd[0]=='3':
                self.stop_flag=True
                print('exit')
                break
            else:
                print('continue')
    def auto_pub(self):
        print(f'{time.asctime()} start auto publish.')
        while not self.stop_flag:
            self.read_and_publish()
            self.rater.sleep()
        print(f'{time.asctime()} stop auto publish.')
        self.round_read_ptr=None
        self.stop_flag=False
    def read_and_publish(self):
        pc = o3d.t.io.read_point_cloud(pj(self.source_dir,self.fs[self.counter]))
        data=dict(pc.point.items())
        if 'intensity' in data:
            x=np.concatenate([data['positions'].numpy(),data['intensity'].numpy()],axis=1)
            x=np_xyzi2pc(x)
        else:
            x=data['positions'].numpy()
            x=np_xyz2pc(x)
        t = RosTime.from_sec(time.time())
        pc = rosnp.point_cloud2.array_to_pointcloud2(x, stamp=t,frame_id=CONF.FRAME_LIDAR)
        self.pc_pub.publish(pc)
        self.counter = (self.counter+1)%len(self.fs)



if __name__ == "__main__":
    main()
