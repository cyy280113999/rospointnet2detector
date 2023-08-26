"""
this file is no use
"""
import ros_numpy as rosnp
import torch
from pointnet.pointnet.model import PointNetCls, PointNetDenseCls
from sensor_msgs.msg import PointCloud2
shapenet_dir='/home/cyy/datasets/shapenetcore_partanno_segmentation_benchmark_v0/'
class _old_ROSPointNet:
    Net = PointNetCls
    def __init__(self):
        with open(shapenet_dir+'synsetoffset2category.txt', 'r') as f:
            lines = f.readlines()
            lines = [l.strip(' \n') for l in lines]
            lines = [l.split()[0] for l in lines]
            self.cat=lines

        self.model = self.Net(k=num_classes)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)
        self.model.eval()
    
    def forward(self,x):
        return self.model.forward(x)


    def _deprcated(self, pc2:PointCloud2):

        x_np = rosnp.point_cloud2.pointcloud2_to_xyz_array(pc2) # xyz type

        x = torch.from_numpy(x_np.T).to(device).float()
        x = torch.unsqueeze(x, 0)
        out = self.model(x)
        pred:torch.tensor=out[0]
        pred = torch.argmax(pred).item()
        pred = self.cat[pred]
        rospy.loginfo(f'prediction:{pred}')
        global processed
        processed=True

    def pr_send(self, data):
        raise NotImplementedError()
