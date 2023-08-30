import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import numpy as np
import tqdm

# source_dir='/home/cyy/datasets/pc063003'
files=[e.path for e in os.scandir(source_dir)]
for p in tqdm.tqdm(files):
    x = np.load(pj(source_dir,p))
    x[:,1]=-x[:,1]
    np.save(p,x)