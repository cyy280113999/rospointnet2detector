import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import numpy as np
import tqdm
# from transfer_dataset import *
import global_config as CONF

std_len='06d'
# source_dir='saved_npys/' # test dir
# source_dir='/home/cyy/datasets/pc_filtered'
source_dir=pointcloud_dir=f'/home/{CONF.usr_name}/datasets/pc082601'
fs=sorted(list(os.listdir(source_dir)))
for i,filename in tqdm.tqdm(enumerate(fs)):
    pure_name,sufix=filename.split('.')
    os.rename(pj(source_dir,filename),pj(source_dir,f'pc_{i:06d}.{sufix}'))
    