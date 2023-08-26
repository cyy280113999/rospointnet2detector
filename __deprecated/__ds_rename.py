import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import numpy as np
import tqdm

# rename pc_1.npy to pc_00001.npy
std_len='06d'
# source_dir='saved_npys/' # test dir
source_dir='/home/cyy/datasets/pc_filtered'
for e in tqdm.tqdm(os.scandir(source_dir)):
    filename=e.name
    tmp,sufix=filename.split('.')
    counter=tmp.split('_')[-1]
    counter=int(counter)
    os.rename(e.path,pj(os.path.dirname(e.path),f'pc_{counter:06d}.{sufix}'))
    
def strSort(s):
    s=s.split('.')[-2]
    s=s.split('_')[-1]
    return int(s)