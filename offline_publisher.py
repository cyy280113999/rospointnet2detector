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
import threading
from sensor_msgs.msg import PointCloud2
import time
from transfer_dataset import TransferDataset,np_xyz2pc,np_xyzi2pc, pc_calib
from genpy.rostime import Time as RosTime

import global_config as CONF

rate=10
DSChooseFirst=True
if DSChooseFirst:
    # pointcloud_dir=f'/home/{usr_name}/datasets/saved_npys' # 4000
    # pointcloud_dir=f'/home/{usr_name}/datasets/pc0602' # 1100
    # pointcloud_dir=f'/home/{usr_name}/datasets/pc063003' # 382  
    # pointcloud_dir=f'/home/{usr_name}/datasets/pc_filtered' # 1100
    # pointcloud_dir=f'/home/{usr_name}/datasets/pc0701static' # 382
    # pointcloud_dir=f'/home/{usr_name}/datasets/pc071803'
    pointcloud_dir=f'/mnt/d/wsl_tools/pc082303'
    has_label=False # raw pc has no label
else:
    # -- here are labeled pointclouds
    # pointcloud_dir=f'/home/{usr_name}/datasets/pc_labeled' # 382  
    pointcloud_dir=f'/home/{CONF.usr_name}/datasets/pc071803mix'
    # pointcloud_dir='./test'
    has_label=True # labeled pc [x,y,z,intensity,label]

npoints=7000
normalization=False
augmentation=False
rotation=False
keepChannel=4

# where to
topic_pc_from='/off_line'

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
    dataset = TransferDataset(
        root=pointcloud_dir,
        label_filter=None,
        label_map=None,
        npoints=npoints,
        normalization=normalization,
        augmentation=augmentation,
        rotation=rotation,
        keepChannel=keepChannel,
        has_label=has_label,
        )
    dataset.get_normalization(CONF.norm_file)
    rospy.init_node('ros_offline_publisher', anonymous=False)
    pc_puber = PC_Offline_Publisher(dataset,rate)
    pc_puber.start()
    pc_puber.command_loop()


class PC_Offline_Publisher:
    def __init__(self, dataset,rate=1):
        self.rater=rospy.Rate(rate)
        self.pc_pub = rospy.Publisher(topic_pc_from, PointCloud2, queue_size=1)
        self.counter=0
        self.dataset = dataset
        self.next=True
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
        data = self.dataset[self.counter]
        x_np=data[0].numpy()
        if keepChannel>3:
            x_pc = np_xyzi2pc(x_np)
        else:
            x_pc = np_xyz2pc(x_np)
        t = RosTime.from_sec(time.time())
        pc = rosnp.point_cloud2.array_to_pointcloud2(x_pc, stamp=t,frame_id=CONF.lidar_frame)
        self.pc_pub.publish(pc)
        self.counter = (self.counter+1)%len(self.dataset)



if __name__ == "__main__":
    main()
