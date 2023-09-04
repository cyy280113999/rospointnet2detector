import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
"""
copy, modified from pointnet2

transfer pointnet2
easy to change input & output channels


"""
class TransferPn2(nn.Module):
    def __init__(self,inc=3,outc=50, pretrained_state_dict=None):
        super(TransferPn2, self).__init__()
        self.inc = inc
        self.ouc=outc
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], inc, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(n_groups=None, radius=None, group_size=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+3+inc, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, outc, 1)
        if pretrained_state_dict: # load state dict from pn2.
            self.load_state_dict(pretrained_state_dict)
        self.remove_cls_label() # after transfer, model should be loaded manually

    def remove_cls_label(self):
        with torch.no_grad():
            old=self.fp1.mlp_convs[0]
            new_conv=torch.nn.Conv1d(128+3+3,old.out_channels,1,device='cuda')
            new_conv.weight.data=old.weight[:,16:]
            new_conv.bias.data=old.bias
            self.fp1.mlp_convs[0]=new_conv

    def forward(self, xyz):
        # this module must be called remove_cls_label first.
        # Set Abstraction layers
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = x.permute(0, 2, 1).contiguous()
        return x
    
    def in2k(self,k=3): # change input channel
        assert k>=3
        with torch.no_grad():
            for convs in self.sa1.conv_blocks:
                old=convs[0]
                new_conv=torch.nn.Conv2d(k,old.out_channels,1,device='cuda')
                torch.nn.init.xavier_uniform_(new_conv.weight)
                new_conv.weight.data[:,:min(k,old.in_channels)]=old.weight[:,:min(k,old.in_channels)]
                new_conv.bias.data=old.bias
                convs[0]=new_conv

    def out2k(self,k=2): # change classes num
        self.fix_all(True)
        self.conv2 = torch.nn.Conv1d(128, k, 1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)

    def fix_all(self,fix=True): # fix para when training
        for para in self.parameters():
            para.requires_grad=not fix
    def fix_stage1(self,fix=True):
        for para in self.sa1.parameters():
            para.requires_grad=not fix
        for para in self.sa2.parameters():
            para.requires_grad=not fix
        for para in self.sa3.parameters():
            para.requires_grad=not fix
    def fix_stage2(self,fix=True):
        for para in self.fp3.parameters():
            para.requires_grad=not fix
        for para in self.fp2.parameters():
            para.requires_grad=not fix
        for para in self.fp1.parameters():
            para.requires_grad=not fix
    def fix_stage3(self,fix=True):
        for para in self.conv1.parameters():
            para.requires_grad=not fix
        for para in self.bn1.parameters():
            para.requires_grad=not fix
        for para in self.conv2.parameters():
            para.requires_grad=not fix
