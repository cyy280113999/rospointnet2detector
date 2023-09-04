import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import numpy as np
import tqdm
from utilities import pc_calib

# run calib on ds

replace=True # replace or save into other dir
source_dir='/home/cyy/datasets/pc_filtered'
other_dir = './test'

CALIB_FILES=('rm.npy','hei.npy')


try:
    rm = np.load(pj('config',CALIB_FILES[0]))
    height = np.load(pj('config',CALIB_FILES[1]))
except:
    print('Error: cannot load calib')
    sys.exit(-1)

for e in tqdm.tqdm(os.scandir(source_dir)):
    x = np.load(e.path)
    xyz=x[:,:3]
    xyz=pc_calib(xyz,rm,height)
    x[:,:3]=xyz
    if replace:
        np.save(e.path,x)
    else:
        np.save(pj(other_dir,e.name),x)