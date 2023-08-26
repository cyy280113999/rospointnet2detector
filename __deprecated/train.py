from __future__ import print_function
import argparse
import json
import os
def change_wd():
    # this use for calling from another directory
    p = lambda f: os.path.abspath(f)
    print(f'pwd: {p(os.path.curdir)}')
    print(f'this file is in: {p(__file__)}')
    print(f'its dir: {os.path.dirname(p(__file__))}')
    os.chdir(os.path.dirname(p(__file__)))
    print(f'pwd: {p(os.path.curdir)}')
change_wd()
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import sys
from pointnet.pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from transfer_train import TransferDataset, TransferPointNet, FocalLoss


# opt.manualSeed = random.randint(1, 10000)  # fix seed
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)
pj=lambda *args:os.path.join(*args)



def main():
    dataset_path='/home/cyy/datasets/lidar_ds1'
    npoints=4000
    tg_cls=2
    
    nepoch=100
    model_path='./model/seg_stage_2.pth'
    feature_transform=True
    batchSize=32
    workers=0
    outf='transfer'

    dataset = TransferDataset(
        root=dataset_path,
        npoints=npoints)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=False,
        num_workers=int(workers),
        pin_memory=True)
    
    classifier = TransferPointNet(k=tg_cls,feature_transform=feature_transform)
    classifier.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()
    try:
        os.makedirs(outf)
    except OSError:
        pass


    num_batch = len(dataset) / batchSize
    loss_ = FocalLoss()
    for epoch in range(1,nepoch+1):
        scheduler.step()
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            pred = pred.softmax(-1)
            loss=loss_(pred.view(-1,tg_cls),target.view(-1))
            # losses=[]
            # for i in range(tg_cls):
            #     index_mask=(target==i)
            #     p_=pred[index_mask]
            #     t_=target[index_mask]
            #     losses.append(loss_(p_.view(-1, tg_cls),t_.view(-1)))
            # loss=sum(losses)
            if feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.max(axis=-1)[1]
            correct = pred_choice.eq(target).flatten()
            acc=correct.sum().item()/len(correct)
            print(f'[{epoch}: {i}/{num_batch}] train loss: {loss} accuracy: {acc}')

            # print(f'[{epoch}: {i}/{num_batch}] train loss: {[l.item()for l in losses]} accuracy: {acc}')

        if epoch % 10 == 0:
            torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (outf, epoch))

if __name__ == '__main__':
    main()