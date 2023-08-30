import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
ROOT_DIR = os.path.dirname(p(__file__))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(pj(ROOT_DIR, 'Pointnet2/models'))
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from transfer_dataset import TransferDataset,  Balence, FocalLoss, Class_acc
from transfer_pn2 import TransferPn2


dataset_path='/home/cyy/datasets/pc071803mix'
transfer=False
lr_rate=0.001
if transfer:
    # pointnet2 checkpoint training with shapenet. class_num=50
    model_path='./model/pretrain_pn2_77.pth' 
else:
    model_path='./model/final2.pth'
    # lr_rate=0.0001
label_filter=None
label_map=[(6,3)] # target to material
# label_map=None
npoints=7000
keepChannel=3
inc=3
tg_cls=8
ori_cls=50


#criterion = torch.nn.CrossEntropyLoss()
criterion = FocalLoss()
nepoch=100
batchSize=8 # 16 will oom
workers=8
# lr_rate=0.001
weight_decay=1e-4
lr_step=10
lr_decr=0.3

outf='transfer'
if not os.path.exists(outf):
    os.makedirs(outf)
model_save_step=10

def main():
    dataset = TransferDataset(
        root=dataset_path,
        label_filter=label_filter, 
        label_map=label_map,
        npoints=npoints,
        normalization=True,
        augmentation=True,
        rotation=True,
        keepChannel=keepChannel,
        has_label=True,
        )
    dataset.get_normalization('ds_norm.npy')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=False,
        num_workers=int(workers),
        pin_memory=True, 
        persistent_workers=True if workers>0 else False)
    
    assert model_path is not None
    if transfer:
        checkpoint = torch.load(model_path)
        classifier = TransferPn2(inc=inc,outc=ori_cls,pretrained_state_dict=checkpoint['model_state_dict'])
        classifier.out2k(tg_cls)
    else:
        classifier=TransferPn2(inc=inc,outc=tg_cls)
        classifier.load_state_dict(torch.load(model_path))
    classifier.cuda()
    classifier.train()
    optimizer = optim.Adam(classifier.parameters(), lr=lr_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD(classifier.parameters(), lr=lr_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decr)

    num_batch = len(dataset) / batchSize
    for epoch in range(1,nepoch+1):
        scheduler.step()
        if epoch==20:
            classifier.fix_stage3(False)
        if epoch==40:
            classifier.fix_stage2(False)
        if epoch==40:
            classifier.fix_stage1(False)
        for i, (points, target) in enumerate(dataloader):
            optimizer.zero_grad()

            points, target = points.cuda(), target.cuda()
            points = points.transpose(2, 1) # b3n

            seg_pred = classifier(points)

            loss=Balence(criterion,seg_pred,target, tg_cls)

            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.data.argmax(-1)
            correct = pred_choice.eq(target.data).flatten()
            acc=correct.sum().item()/len(correct)
            print(f'[{epoch}: {i}/{num_batch}] train loss: {loss:6f} accuracy: {acc:4f}')
            print(Class_acc(pred_choice, target, tg_cls))
            # print(f'[{epoch}: {i}/{num_batch}] train loss: {[l.item()for l in losses]} accuracy: {acc}')

        if epoch % model_save_step == 0:
            torch.save(classifier.state_dict(), f'{outf}/seg_{epoch}.pth')

if __name__ == '__main__':
    main()