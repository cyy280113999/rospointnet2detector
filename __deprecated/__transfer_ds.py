class _old_TransferDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 npoints=5000):
        self.npoints = npoints
        self.root = root
        pc_dir = pj(self.root, 'pts')
        self.datapath = []
        for line in os.scandir(pc_dir):
            _, pcf=line.path.split('/')[-2:]
            pch=pcf.split('.')[0]
            segf = pj(self.root,'seg',pch+'.seg')
            self.datapath.append([line.path, segf])
        if len(self.datapath)==1:
            self.datapath = self.datapath*64

    def __getitem__(self, index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[0]).astype(np.float32)
        seg = np.loadtxt(fn[1]).astype(np.int64)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]
        # if np.random.rand()>0.5:
        point_set = point_set - np.mean(point_set, axis = 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale
        # point_set += np.random.normal(0, 0.002, size=point_set.shape) # random jitter
        theta = np.random.uniform(-np.pi/1,np.pi/1)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        point_set[:,[0,1]] = point_set[:,[0,1]].dot(rotation_matrix) # random rotation xy
        theta = np.random.uniform(-np.pi/1,np.pi/1)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation xz
        theta = np.random.uniform(-np.pi/1,np.pi/1)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        point_set[:,[1,2]] = point_set[:,[1,2]].dot(rotation_matrix) # random rotation yz

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        return point_set, seg

    def __len__(self):
        return len(self.datapath)
