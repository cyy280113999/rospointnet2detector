import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import numpy as np
import tqdm
from transfer_dataset import strSort
# 首先使用工具对pcd文件添加标签!
# cd ~/temp/semantic-segmentation-editor-x.x.x
# meteor npm install
# meteor npm start
# 得到包含标签的pcd文件

# 标签文件没有额外intensity信息，将两者混合
# mix the point (xyz intensity) and (label) 
point_dir='/home/cyy/datasets/pc071803/' # xyz npy data
label_dir='/home/cyy/Downloads/' # label pcd data
dest_dir='/home/cyy/datasets/pc071803lb/' # save mixed npy to ..
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)


fps = [e.name for e in os.scandir(label_dir)]
fps.sort(key=strSort)
for e in tqdm.tqdm(fps): # may not label all, label dir has less
    filename = e.split('.')[0]
    tqdm.tqdm.write(f'save {filename}')
    x = np.load(pj(point_dir,filename+'.npy'))
    l = np.genfromtxt(pj(label_dir,filename+'.pcd'), skip_header=10) # notice this header count
    assert len(x)==len(l)
    l=l[:,3] # pick label
    x=np.concatenate([x,l.reshape(-1,1)],axis=1).astype(np.float32)
    np.save(pj(dest_dir,filename+'.npy'),x)


