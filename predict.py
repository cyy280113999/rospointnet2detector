import os
import torch
import torch.nn.functional as nf
import torch.optim as optim
import torch.utils.data
import tqdm
import open3d as o3d
import numpy as np
from utilities import *
from transfer_pn2 import TransferPn2
import global_config as CONF
pj=lambda *args:os.path.join(*args)
"""
save label for each point cloud

"""
source_dir='/home/cyy/datasets/pcEnhance'
dest_dir='/home/cyy/datasets/pcEnhancePreLabel'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
model_path='./model/final.pth'

npoints=7000
inc=3
outc=8

def main():
    mean,std=np.load(CONF.FILE_NORM)
    mean=mean.astype(np.float32)[:3]
    std=std.astype(np.float32)[:3]
    fs=list(os.listdir(source_dir))
    fs=[x for x in fs if os.path.splitext(x)[1]=='.pcd']

    classifier=TransferPn2(inc=inc,outc=outc)
    classifier.load_state_dict(torch.load(model_path))
    classifier.cuda()
    classifier.eval()

    for filename in tqdm.tqdm(fs):
        pcd=o3d.t.io.read_point_cloud(pj(source_dir,filename))
        x=pcd.point.positions.numpy()
        # x = pc_downsample(x,7000)  # label all
        xt=normalize(x,mean,std) # point scale consist..
        xt=torch.from_numpy(xt.T).float().cuda().unsqueeze(0)
        seg = classifier(xt)
        seg = seg.squeeze(0).argmax(axis=-1).cpu().numpy()
        seg = seg.reshape(-1,1).astype(np.int32) # 4 bytes
        pcd.point.label=o3d.core.Tensor(seg)
        o3d.t.io.write_point_cloud(pj(dest_dir,filename),pcd,write_ascii=False)


if __name__ == '__main__':
    main()