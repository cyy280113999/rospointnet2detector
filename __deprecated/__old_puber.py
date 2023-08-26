class _old_PC_Offline_Publisher:
    def __init__(self, pc_dir=pointcloud_dir,rate=1):
        self.rater=rospy.Rate(rate)
        self.pc_pub = rospy.Publisher(topic_pc_from, PointCloud2, queue_size=1)
        self.counter=0
        self.file_dir = pc_dir
        self.filenames = [e.name for e in os.scandir(pc_dir)]

    def start(self):
        print(f'{time.asctime()} start offline pointcloud publishing thread.')
        self.stop_flag=False
        round_read_ptr = threading.Thread(target=self.read_and_publish)
        round_read_ptr.start()
        
    def stop(self):
        self.stop_flag=True
        print(f'{time.asctime()} ready to stop offline pointcloud publishing.')

    def read_and_publish(self):
        while not self.stop_flag:
            pc_type=self.filenames[self.counter].split('.')[1]
            if pc_type=='npy':
                x_np = np.load(pj(self.file_dir,self.filenames[self.counter])).astype(np.float32)
                # x_np[:,3]=x_np[:,4] # label to intensity
            elif pc_type=='pcd':
                x_np = np.genfromtxt(pj(self.file_dir,self.filenames[self.counter]), skip_header=10)
                # x_np=x_np[x_np[:,3]!=3] # remove label 3
                x_np=x_np[:,:4] # remove extra

            # x_np = pc_downsample(x_np, 4000)

            # x_np[:,:3] = pc_normalize(x_np[:,:3])
            # x_np[:,:3] = pc_scale(x_np[:,:3])
            # x_np[:,:3] = pc_shift(x_np[:,:3])
            # x_np[:,:3] = pc_jitter(x_np[:,:3],bound=0.001)
            # x_np[:,:3] = pc_rotate_z(x_np[:,:3])
            # x_np[:,:3] = pc_rotate_x(x_np[:,:3],np.pi/8)
            # x_np[:,:3] = pc_rotate_y(x_np[:,:3],np.pi/8)

            x_pc = np_xyzi2pc(x_np)
            pc = rosnp.point_cloud2.array_to_pointcloud2(x_pc, frame_id=lidar_frame)
            self.pc_pub.publish(pc)
            self.counter = (self.counter+1)%len(self.filenames)
            self.rater.sleep()
        self.stop_flag = False
        print(f'{time.asctime()} stop offline pointcloud publishing.')
        
