import os
import time
import numpy as np
import torch
pj=lambda *args:os.path.join(*args)
"""
some functions

"""


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
    return time.strftime("%Y-%m-%d_%H-%M-%S")
def day_second():
    return time_str().split('_') #len=2
