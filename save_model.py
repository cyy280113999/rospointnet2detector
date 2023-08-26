import os
import sys

import rospy
import numpy as np
import ros_numpy as rosnp
import torch
from sensor_msgs.msg import PointCloud2
from transfer_pn2 import TransferPn2
from transfer_dataset import *
import open3d as o3d

# where is the point cloud?
# topic_pc_from='/rslidar_points'
topic_pc_from='/off_line'
lidar_frame='rslidar'

topic_seg='/segmentation'
# nn model
# model_path='./model/seg_model.pth'
model_path='./model/final2.pth'
inc=3
outc=8
device ='cuda'
# 4060 4fps

def main():
    torch.set_grad_enabled(False)
    net=torch.load('final3.pth')
    print(net)
    print(net.conv1.weight.device)
       
if __name__ == "__main__":
    main()
