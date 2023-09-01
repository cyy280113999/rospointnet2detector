import rospy
import numpy as np
import ros_numpy as rosnp
import torch
from sensor_msgs.msg import PointCloud2
from transfer_pn2 import TransferPn2
from transfer_dataset import *
import global_config as CONF

# module debug: where is the point cloud?
# topic_pc_from='/rslidar_points'
TOPIC_TEST_FROM=CONF.TOPIC_OFFLINE
topic_seg='/segmentation'
# nn model
# model_path='./model/seg_model.pth'
model_path='./model/final.pth'
inc=3
outc=8
device ='cuda'
# 4060 4fps

def main():
    torch.set_grad_enabled(False)
    rospy.init_node('ros_pointnet2', anonymous=False)
    pc_processer = PC_Processer()
    pc_processer.start()
    rospy.spin() 
    pc_processer.stop()

class PC_Processer:
    def __init__(self):
        self.pc_sub=None
        self.pub_seg = rospy.Publisher(topic_seg, PointCloud2, queue_size=1)
        self.segmentator=Segmentator()

    def start(self):
        self.pc_sub = rospy.Subscriber(TOPIC_TEST_FROM, PointCloud2, self.process, queue_size=1, buff_size=2 ** 24)
    def stop(self):
        self.pc_sub.unregister()
    def send(self,pc):
        x = rosnp.point_cloud2.pointcloud2_to_xyz_array(pc).astype(np.float32)
        x = pc_downsample(x,7000) # down sample as raw pc
        xi = self.segmentator(x)
        color_controller=np.array([[0,0,0,0],[0,0,0,7]])
        xi=np.concatenate([xi,color_controller],axis=0)
        self.pub_seg.publish(rosnp.point_cloud2.array_to_pointcloud2(np_xyzi2pc(xi), frame_id=CONF.lidar_frame))

class Segmentator:
    def __init__(self):
        self.mean,self.std=np.load('./config/ds_norm.npy')
        self.mean=self.mean.astype(np.float32)[:3]
        self.std=self.std.astype(np.float32)[:3]
        # nets
        self.net=TransferPn2(inc=inc,outc=outc)
        self.net.load_state_dict(torch.load(model_path))
        self.net=self.net.to(device).eval()
        x=torch.zeros((1,3,1000),device=device) # run empty once, to escape from first-run-delay
        self.net(x)
    def process(self, x):
        x = pc_downsample(x,7000) # point count consist with training
        xt=normalize(x,self.mean,self.std) # point scale consist..
        xt=torch.from_numpy(xt.T).float().to(device).unsqueeze(0)
        seg = self.net(xt)
        seg = seg.squeeze(0).argmax(axis=-1).cpu().numpy()
        xi=np.concatenate([x,seg.reshape(-1,1).astype(np.float32)],axis=1) # concate result
        return xi

       
if __name__ == "__main__":
    main()
