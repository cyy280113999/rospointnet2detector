import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import numpy as np
import tqdm
import shutil

# mix two ds

def strSort(s):
    s=s.split('.')[-2]
    s=s.split('_')[-1]
    return int(s)

dirs=['/home/cyy/datasets/pc_labeled',
      '/home/cyy/datasets/pc071803lb',
      ]
des='/home/cyy/datasets/pc071803mix'
if not os.path.exists(des):
    os.makedirs(des)
counter=0
for d in dirs:
    fs = [e.name for e in os.scandir(d)]
    fs.sort(key=strSort)
    for e in tqdm.tqdm(fs):
        shutil.copy(pj(d,e),pj(des,f'pc_{counter:06d}.npy'))
        counter+=1

    
