import os
import tqdm
pj=lambda *args:os.path.join(*args)

std_len='06d'
# source_dir='saved_npys/' # test dir
# source_dir='/home/cyy/datasets/pc_filtered'
source_dir=pointcloud_dir=f'/home/cyy/datasets/pcEnhance_pcd'
fs=sorted(list(os.listdir(source_dir)))
for i,filename in tqdm.tqdm(enumerate(fs)):
    pure_name,sufix=filename.split('.')
    os.rename(pj(source_dir,filename),pj(source_dir,f'pc_{i:06d}.{sufix}'))
    