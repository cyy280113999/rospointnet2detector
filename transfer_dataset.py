import os
import time
import numpy as np
import torch
pj=lambda *args:os.path.join(*args)

def pc_downsample(x, npoints=4000):
    if len(x)>npoints:
        return x[np.random.choice(len(x), npoints, replace=False)]
    else:
        np.random.shuffle(x) # inplace
        return np.concatenate([x,x[np.random.choice(len(x), npoints-len(x), replace=False)]],axis=0)
def centerize(pc, mean=None):
    if mean is None:
        mean = np.mean(pc, axis=0)
    return pc - mean
def standardize(pc, std=None):
    if std is None:
        std=np.sqrt(np.square(pc).sum(axis=1)).mean().repeat(3)
    return pc / std
def normalize(pc,mean=None,std=None):
    return standardize(centerize(pc,mean),std)
def pc_to_sphere(pc):
    radius=np.sqrt(np.square(pc).sum(axis=1)).max()
    return pc / radius
def pc_standardize(x):
    x_std=x.std(0).max()
    return x/x_std
def pc_reflect(x, axis=0, prob=0.5):
    if np.random.rand()>prob:
        x[:,axis]=-x[:,axis]
    return x
def pc_scale(x, scale_low=0.8, scale_high=1.25):
    return x*np.random.uniform(scale_low, scale_high)
def pc_shift(x, shift_range=0.1):
    return x + np.random.uniform(-shift_range, shift_range, (3,)).astype(np.float32)
def pc_jitter(x,bound=0.001):
    return x + np.random.uniform(-bound, bound, size=x.shape).astype(np.float32)
def pc_rotate_z(x, theta=np.pi):
    t = np.random.uniform(-theta,theta)
    rotation_matrix = np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])
    x[:,[0,1]] = x[:,[0,1]].dot(rotation_matrix) # random rotation xy
    return x
def pc_rotate_y(x, theta=np.pi):
    t = np.random.uniform(-theta,theta)
    rotation_matrix = np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])
    x[:,[0,2]] = x[:,[0,2]].dot(rotation_matrix) # random rotation xz
    return x
def pc_rotate_x(x, theta=np.pi):
    t = np.random.uniform(-theta,theta)
    rotation_matrix = np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])
    x[:,[1,2]] = x[:,[1,2]].dot(rotation_matrix) # random rotation yz
    return x
def label_remap(x,map):
    if isinstance(map, dict):
        map=list(map.items())
    ids=[]
    for i in range(len(map)):
        ids.append(x==map[i][0])
    for i in range(len(map)):
        x[ids[i]]=map[i][1]
    return x
# there are 3 types
# pc2 : sensor_msgs.msg.PointCloud2
# np_xyz : numpy.array with shape of (n,2)
# np_pc : numpy.array with named dtype field ['x','y','z',|'intensity']
def np_xyz2pc(x):
    # from numpy xyz type array x:(n,3) to field type array pc:[(x,4),(y,4),(z,4)]
    pc_dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4')])
    # pc_field=rosnp.point_cloud2.dtype_to_fields(pc_dtype)
    pc = np.zeros((1,x.shape[0]), dtype=pc_dtype)
    pc['x'] = x[...,0]
    pc['y'] = x[...,1]
    pc['z'] = x[...,2]
    return pc
def np_xyzi2pc(x):
    pc_dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('intensity','<f4')])
    pc = np.zeros((1,x.shape[0]), dtype=pc_dtype)
    pc['x'] = x[...,0]
    pc['y'] = x[...,1]
    pc['z'] = x[...,2]
    pc['intensity'] = x[...,3]
    return pc
def np_pc2xyz(pc, remove_nans=True):
    if remove_nans:
        mask = np.isfinite(pc['x']) & np.isfinite(pc['y']) & np.isfinite(pc['z'])
        pc = pc[mask]
    x = np.zeros(pc.shape + (3,), dtype=np.float32)
    x[...,0] = pc['x']
    x[...,1] = pc['y']
    x[...,2] = pc['z']
    return x
def np_pc2xyzi(pc, remove_nans=True):
    if remove_nans:
        mask = np.isfinite(pc['x']) & np.isfinite(pc['y']) & np.isfinite(pc['z'])
        pc = pc[mask]
    x = np.zeros(pc.shape + (4,), dtype=np.float32) # float is float64
    x[...,0] = pc['x']
    x[...,1] = pc['y']
    x[...,2] = pc['z']
    x[...,3] = pc['intensity']
    return x
def strSort(s):
    s=s.split('.')[-2]
    s=s.split('_')[-1]
    return int(s)
def pc_bound(x,xb=None,yb=None,zb=None):
    if xb is not None:
        x = x[x[:,0]>=xb[0]]
        x = x[x[:,0]<=xb[1]]
    if yb is not None:
        x = x[x[:,1]>=yb[0]]
        x = x[x[:,1]<=yb[1]]
    if zb is not None:
        x = x[x[:,2]>=zb[0]]
        x = x[x[:,2]<=zb[1]]
    return x
def pc_rotate_m(x, rm):
    return np.matmul(rm,x.T).T
def pc_calib(x,data):
    x = pc_rotate_m(x,data[:3])
    return x - data[3]
#  notice coord. l2a: -y, z, -x. a2l: -z, -x, y
# SHIFT_DATA=np.array([0,0,0])
# arm[-4919,2790,1787]
# lidar[-2278,0121,2987]
# shift in arm[-2641,2669,-1200] first
# SHIFT_DATA=np.array([-2.641,2.669,-1.200])
# second fix
SHIFT_DATA=np.array([-2.641,2.669,-1.180])
def lidar2arm(data):
    x,y,z=data[:,0,None],data[:,1,None],data[:,2,None]
    data=np.concatenate([-y, z, -x],axis=1)
    return data + SHIFT_DATA
def arm2lidar(data):
    data-=SHIFT_DATA
    x,y,z=data[:,0,None],data[:,1,None],data[:,2,None]
    return np.concatenate([-z, -x, y],axis=1)
def insert_with_limit(ls:list,value, maxlen=100):
    ls.insert(0, value)
    if len(ls)>maxlen:
        ls=ls[:maxlen]
    return ls
def find_nearst(array, value):
    array = np.asarray(array)
    ind = (np.abs(array-value)).argmin()
    return ind
def time_str():
    return time.strftime("%Y-%m-%d_%H-%M-%S").split('_') #len=2

class TransferDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 label_filter=None,
                 label_map=None,
                 npoints=5000, # downsample
                 normalization=False,
                 augmentation=False,
                 rotation=False,
                 keepChannel=3,
                 has_label=True,
                 ):
        self.root = root
        self.label_filter=label_filter # label to remove
        self.label_map=label_map # label to remap
        self.npoints = npoints
        self.normalization=normalization
        self.mean=None
        self.augmentation=augmentation
        self.rotation=rotation
        assert keepChannel>=3
        self.keepChannel=keepChannel # keep additional channel
        self.has_label=has_label
        self.datapath = [e.path for e in os.scandir(root)]
        self.datapath.sort(key=strSort)
        self.cache = {}
        self.cache_size = 20000
    
    def get_normalization(self, fn='ds_norm.npy'):
        if not os.path.exists(fn):
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
            self.mean = self.mean[:self.keepChannel].astype(np.float32)
            self.std = self.std[:self.keepChannel].astype(np.float32)

    def __getitem__(self, index):
        if index in self.cache: # load from cache
            return self.cache[index]
        x = np.load(self.datapath[index]).astype(np.float32) # load from file
        if self.has_label and self.label_filter is not None: # in default, label is in last dim
            for i in self.label_filter:
                x=x[x[:,-1]!=i]
        if self.npoints is not None:
            x = pc_downsample(x, self.npoints)
        seg=None
        if self.has_label:
            seg=x[:,-1].astype(np.int64)
            if self.label_map is not None:
                seg=label_remap(seg,self.label_map)
            seg = torch.from_numpy(seg)
            x = x[:,:-1] # remove label
        if self.keepChannel<x.shape[1]:
            x=x[:,:self.keepChannel]
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
        if len(self.cache) < self.cache_size: # save into cache
            self.cache[index] = (x,  seg)
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