import os
import torch
import torch.nn.functional as nf
import torch.optim as optim
import torch.utils.data
import open3d as o3d
from utilities import *
from transfer_pn2 import TransferPn2
import global_config as CONF


def main():
    transfer=False
    dataset_path='/home/cyy/datasets/pc0718m'
    model_path='./model/final.pth'
    # lr_rate=0.0001
    lr_rate=0.001
    if transfer:
        # pointnet2 checkpoint training with shapenet. class_num=50
        model_path='./model/pretrain_pn2_77.pth' 
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
    dataset = TransferDataset(
        root=dataset_path,
        label_filter=label_filter, 
        label_map=label_map,
        npoints=npoints,
        normalization=True,
        augmentation=True,
        rotation=True,
        keepIntensity=False,
        )
    dataset.get_normalization(CONF.FILE_NORM)
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


class TransferDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 label_filter=None,
                 label_map=None,
                 npoints=5000, # downsample
                 normalization=False,
                 augmentation=False,
                 rotation=False,
                 keepIntensity=False,
                 ):
        self.root = root
        self.label_filter=label_filter # label to remove
        self.label_map=label_map # label to remap
        self.npoints = npoints
        self.normalization=normalization
        self.mean=None
        self.std=None
        self.augmentation=augmentation
        self.rotation=rotation
        self.keepIntensity=keepIntensity # keep additional channel
        self.datapath = [e.path for e in os.scandir(root)]
        self.datapath.sort(key=strSort)
    
    def get_normalization(self, fn='ds_norm.npy'):
        if not os.path.exists(fn):
            raise NotImplementedError # wait for pcd version
            # all channel except label, float64 for acc
            datas=[np.load(self.datapath[i]).astype(np.float64)[:,:-1] for i in range(len(self.datapath))]
            x=np.concatenate(datas,axis=0)
            self.mean=np.mean(x,axis=0)
            x-=self.mean
            stdxyz=np.sqrt(np.square(x[:,:3]).sum(axis=1)).mean().repeat(3) # xyz : mean distance
            stdchannel=np.std(x[:,3:],axis=0)
            self.std=np.concatenate([stdxyz,stdchannel])
            self.mean = self.mean.astype(np.float32)
            self.std = self.std.astype(np.float32)
            np.save(fn,(self.mean,self.std))
        else:
            self.mean,self.std=np.load(fn)
            if not self.keepIntensity:
                self.mean = self.mean[:3].astype(np.float32)
                self.std = self.std[:3].astype(np.float32)

    def __getitem__(self, index):
        pcd = o3d.t.io.read_point_cloud(self.datapath[index]) # load from file
        x = pcd.point.positions.numpy()
        if self.keepIntensity:
            x = np.concatenate([x,pcd.point.intensity.numpy()],axis=1)
        x = np.concatenate([x,pcd.point.label.numpy().astype(np.float32)],axis=1)
        if self.label_filter is not None: # in default, label is in last dim
            for i in self.label_filter:
                x=x[x[:,-1]!=i]
        if self.npoints is not None:
            x = pc_downsample(x, self.npoints)
        seg=x[:,-1].astype(np.int64)
        if self.label_map is not None:
            seg=label_remap(seg,self.label_map)
        seg = torch.from_numpy(seg)
        x = x[:,:-1] # remove label
        if not self.keepIntensity:
            x=x[:,:3]
        if self.normalization:
            x = normalize(x, self.mean, self.std)
        pc = x[:,:3]  # pc is in the first three dims
        if self.rotation:
            pc = pc_rotate_z(pc,np.pi/16)
            pc = pc_rotate_x(pc,np.pi/16)
            pc = pc_rotate_y(pc,np.pi/16)
        if self.augmentation:
            pc = pc_reflect(pc,axis=2) # switch left & right(z axis)
            pc = pc_jitter(pc,bound=0.01) 
            pc = pc_scale(pc, scale_low=0.8, scale_high=1.25)
            pc = pc_shift(pc,shift_range=0.3)
        x[:,:3]=pc
        # remove extra feature in addtional channel
        x = torch.from_numpy(x)
        return x, seg

    def __len__(self):
        return len(self.datapath)


class FocalLoss(torch.nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
 
    def forward(self, logits, targets):
        class_mask = torch.zeros_like(logits)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        probs = (logits.softmax(-1)*class_mask).sum(1).view(-1,1)
        log_p=torch.log_softmax(logits,-1)
 
        batch_loss = -(torch.pow((1-probs), self.gamma))*log_p 
        loss = batch_loss.mean()
        return loss

def Balence(criterion,pred,target,cls_num):
    losses=[]
    for j in range(cls_num):
        index_mask=(target==j)
        p_=pred[index_mask]
        t_=target[index_mask]
        if len(p_)>0: # it must have points
            losses.append(criterion(p_.view(-1, cls_num),t_.view(-1)))
    loss=sum(losses) # at least one class is selected, len(losses)>0
    return loss

def Class_acc(pred, gt, cls_num):
    accs=[]
    for i in range(cls_num):
        index_mask=(gt==i)
        p_=pred[index_mask]
        t_=gt[index_mask]
        if len(t_)>0:
            correct = p_.eq(t_).flatten()
            acc=correct.sum().item()/len(correct)
            acc=str(f'{acc:4f}')
        else:
            acc='     nan'
        accs.append(acc)
    return accs

if __name__ == '__main__':
    main()