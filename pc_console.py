import os
import sys
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import time
import threading
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import rospy
import ros_numpy as rosnp
from std_msgs.msg import String, Int32
from sensor_msgs.msg import PointCloud2
from transfer_dataset import *
from pc_recorder import PC_Recorder
from mserver import MServer
from mclient import MClient
from move_detector import MoveDetector
from rospn2 import Segmentator
from point_selector import PointSelector
import global_config as CONF

# -- where is the pc?
if CONF.ON_LINE:
    TOPIC_FROM='/rslidar_points' # from online lidar
    PRE_CALIB=True # calib ensure standard rotation at height of 2m, easy to train network
else:
    TOPIC_FROM='/off_line' # from off line files. off line pc saved after calibration, used for debug
    PRE_CALIB=False # off line donot calib
BOUNDING=True # bounding limit calibrated dataset in proper boundary 


def main():
    rospy.init_node('pc_console')
    pc_processer = PC_Console()
    if PRE_CALIB:
        pc_processer.calib(CONF.FILE_CALIB) # pre-calibration
    pc_processer.start()


class PC_Console:
    topic_raw=TOPIC_FROM # where is the raw point cloud?
    def __init__(self):
        self.raw_pc_suber =None
        self.calib_info=None
        self.pub_calib = rospy.Publisher(CONF.TOPIC_CALIB, PointCloud2, queue_size=1)
        # record processed pc
        self.record_flag=False
        self.recorder=PC_Recorder(CONF.RECORD_DIR)

        self.last10=[]
        # move
        self.move_detect=CONF.MOVE_DETECT
        self.motion_dector=MoveDetector()
        self.require_need=CONF.REQUIRE_NEED
        self.detect=CONF.DETECT
        self.segmentator=Segmentator()
        self.pub_seg = rospy.Publisher('/debug_segmentation', PointCloud2, queue_size=1)
        self.pub_require = rospy.Publisher('/debug_require', Int32, queue_size=1)

        self.point_selector=PointSelector()

        # to arm 
        self.shift_puber = rospy.Publisher(CONF.TOPIC_ARM, PointCloud2, queue_size=1)

        # tele
        if CONF.Start_MServer:
            self.mserver = MServer()
            self.mserver.start()
        if CONF.Start_MClient:
            self.mclient=MClient()
            self.mclient.start()
        
    def start(self):
        self.raw_pc_suber= rospy.Subscriber(self.topic_raw, PointCloud2, self.process, queue_size=1, buff_size=2 ** 24)
        while True:
            print('pc console. input command:')
            cmd=input().split()
            if not cmd:
                continue
            elif cmd[0]=='quit' or cmd[0]=='q':
                print('system exit')
                break
            elif cmd[0]=='debug':
                if len(cmd)>=2 and cmd[1]=='0':
                    CONF.DEBUG=False
                    print('debug off')
                else:
                    CONF.DEBUG=True
                    print('debug on')
            elif cmd[0]=='calib':
                print('auto calibration')
                # calib calib.npy
                save_file=load_file=None
                if len(cmd)>=3:
                    save_file=cmd[2]
                if len(cmd)>=2:
                    load_file=cmd[1]
                self.calib(load_file,save_file)
            elif cmd[0]=='clear_calib':
                print('clear calibration')
                self.rm=self.height=None
            elif cmd[0]=='show_calib':
                self.show_calib=True
                if len(cmd)>=2 and cmd[1]=='0':
                    self.show_calib=False
            elif cmd[0]=='process':
                print('start process')
                self.raw_pc_suber = rospy.Subscriber(self.topic_raw, PointCloud2, self.process, queue_size=1, buff_size=2 ** 24)
            elif cmd[0]=='stop':
                print('stop process')
                self.raw_pc_suber.unregister()
                self.raw_pc_suber=None
            elif cmd[0]=='record':
                self.manual_record()
            elif cmd[0]=='auto_record':
                rate=1
                seconds=20
                try:
                    if len(cmd)>=2:
                        rate=float(cmd[1])
                    if len(cmd)>=3:
                        seconds = int(cmd[2])
                except:
                    pass
                self.auto_record_ptr = threading.Thread(target=self.auto_record,kwargs={'rate':rate,'seconds':seconds})
                self.auto_record_ptr.start()
            elif cmd[0]=='move_detect':
                self.move_detect=True
                if len(cmd)>=2 and cmd[1]=='0':
                    self.move_detect=False
            elif cmd[0]=='require':
                self.require_need=True
                if len(cmd)>=2 and cmd[1]=='0':
                    self.require_need=False
            elif cmd[0]=='detect':
                self.detect=True
                if len(cmd)>=2 and cmd[1]=='0':
                    self.detect=False
            else:
                print('not a cmd. continue.')
        # clear other instances
        self.raw_pc_suber.unregister()
        if CONF.Start_MClient:
            self.mclient.stop()
        if CONF.Start_MServer:
            self.mserver.stop()

    def calib(self, load_file=None, save_file=None):
        if load_file:
            try:
                self.calib_info = np.load(load_file)
            except:
                print('Error: cannot load calib')
        else:
            # 尝试读取一次数据
            delay=5
            self.calib_pc=None
            def save_calib_pc(pc2):
                self.calib_pc=pc2
            pc_sub_once = rospy.Subscriber(self.topic_raw, PointCloud2, save_calib_pc, queue_size=1, buff_size=2 ** 24)
            t_start = time.time()
            while True:
                if self.calib_pc is not None:
                    break
                if time.time()-t_start>delay:
                    break
                time.sleep(0.1)
            if self.calib_pc is None:
                print('Error: calibration over time limit.')
                return
            pc_sub_once.unregister()
            x = rosnp.point_cloud2.pointcloud2_to_array(self.calib_pc)
            x = np_pc2xyz(x)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(x)
            # ===========           执行RANSAC平面分割           =====================
            distance_threshold=0.1 # 每条线内部是连续的，这里看的是线间距，0.1m，10cm比较合适
            ransac_n=1000 # 至少多少个点在平面内，每次采集7000个点，1000个属于平面可以
            num_iterations=1000
            plane_model, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
            if plane_model[0]<0:
                plane_model = -plane_model
            a=plane_model[:3] # 法向量方向
            a = a / np.linalg.norm(a)
            # ==============             坐标规整，1，正视            =============
            standard_direction = np.array([1.0,0,0]) # 箭头方向，规整方向为x轴前向
            rotation_axis = np.cross(a, standard_direction) # 将a轴的转向x轴
            rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
            cos_theta = np.dot(standard_direction, a)
            theta = np.arccos(cos_theta) # 二者夹角
            rotation_matrix1 = Rotation.from_rotvec(theta * rotation_axis).as_matrix()
            point_cloud.rotate(rotation_matrix1) # 将所有点旋转， 这个旋转矩阵非常重要
            plane_model, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)  # 重新分割平面
            point_cloud = point_cloud.select_by_index(inliers)
            # ==============             坐标规整，2，转动            =============
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.1)
            cl, ind = point_cloud.remove_radius_outlier(nb_points=5, radius=0.3) # 离群下采样 至少30cm内有5个邻近点
            point_cloud = point_cloud.select_by_index(ind)
            x=np.array(point_cloud.points)
            A = np.column_stack([x[:, 1], np.ones_like(x[:, 1])]) # y,z，以y为拟合的x
            r, _, _, _ = np.linalg.lstsq(A, x[:, 2], rcond=None) # z 为拟合的y
            slope = r[0]
            a = np.array([0, 1, slope])
            a = a/np.linalg.norm(a)
            standard_direction = np.array([0,1.0,0]) # 希望路面朝向y轴
            rotation_axis = np.cross(a, standard_direction) # 将a轴的转向y轴
            rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
            cos_theta = np.dot(standard_direction, a)
            theta = np.arccos(cos_theta)
            rotation_matrix2 = Rotation.from_rotvec(theta * rotation_axis).as_matrix()
            point_cloud = point_cloud.rotate(rotation_matrix2) # 将所有点旋转， 这个旋转矩阵非常重要
            # =============             组合两个矩阵       ===============
            rm=np.matmul(rotation_matrix2,rotation_matrix1)
            # 平移
            x=np.array(point_cloud.points)
            shift = x.mean(axis=0) # 减去这个均值
            shift[0]-=2 # 使减去的中心在[2,0,0]
            calib_info = np.zeros((4,3),dtype=np.float32)
            calib_info[:3]=rm
            calib_info[3]=shift
            # np.save('calib_info.npy',calib_info)
            self.calib_info=calib_info
        if save_file:
            try:
                np.save(pj(self.config_dir,save_file),self.calib_info)
            except:
                print('Error: cannot save calib')
    def manual_record(self):
        while True:
            print(f'{time.asctime()} Press . to record, 3 to exit.')
            cmd=input()
            if not cmd:
                continue
            if cmd[0]=='.':
                self.record_flag=True
            elif cmd[0]=='3':
                print(f'{time.asctime()} Shutting Down.')
                break
    def auto_record(self,rate=1, seconds=20):
        rater=rospy.Rate(rate)
        print(f'{time.asctime()} Auto Recording Start.')
        for _ in range(int(rate*seconds)):
            self.record_flag=True
            rater.sleep()
        print(f'{time.asctime()} Auto Recording Finish.')

    def process(self, pc:PointCloud2):
        x = rosnp.point_cloud2.pointcloud2_to_array(pc)
        stamp = pc.header.stamp
        fi=pc.header.frame_id
        x = np_pc2xyzi(x)
        # calib move pc to be vertical, centerized, at height of 2m
        if self.calib_info is not None:
            xyz=x[:,:3]
            xyz=pc_calib(xyz,self.calib_info)
            x[:,:3]=xyz
            if BOUNDING:
                x = pc_bound(x,zb=(-1.65, 2))
            if CONF.DEBUG:
                self.pub_calib.publish(rosnp.point_cloud2.array_to_pointcloud2(np_xyzi2pc(x), stamp=stamp,frame_id=fi))
        if self.record_flag: # record after calibration
            self.recorder.save_once(x)
            self.record_flag=False
        self.last10=insert_with_limit(self.last10,x,10)
        # net
        t = stamp.to_time()
        is_detect=True
        if self.move_detect:
            # ,xb=(-1.5,9), yb=(0.3, 10) . notice! this zb must change after arm motion box changed
            state=self.motion_dector.process(pc_bound(x, zb=(0.1,10)),t) # fix for arm motion noise
            if CONF.Start_MClient:
                self.mclient.set_moving(state==CONF.MOVE_STATE_STOP) # 1=Stop
            if state!=CONF.MOVE_STATE_STOP:
                is_detect=False
                # self.mclient.set_require(4) # error
        if CONF.Start_MClient and self.require_need:
            rq = self.mclient.get_require()
            if rq[0]!=1: # start detect
                is_detect=False
                if CONF.DEBUG:
                    self.pub_require.publish(Int32(0))
            else:
                if CONF.DEBUG:
                    self.pub_require.publish(Int32(1))
                    s1,s2=time_str()
                    print(f'require: {s1}-{s2}')
                    if not os.path.exists(pj(CONF.DIR_PC_LOG10,s1)):
                        os.makedirs(pj(CONF.DIR_PC_LOG10,s1))
                    for i,xi in enumerate(self.last10):
                        pc = o3d.geometry.PointCloud()
                        pc.points = o3d.utility.Vector3dVector(xi[:,:3]) # remove intensity
                        o3d.io.write_point_cloud(pj(CONF.DIR_PC_LOG10,s1,f'{s2}_{i}.pcd'),pc)
        if is_detect and self.detect:
            # if Start_MClient:
            #     self.mclient.set_require(2) # busy
            xi = self.segmentator.process(x[:,:3])
            if CONF.DEBUG:
                color_controller=np.array([[0,0,0,0],[0,0,0,7]])
                xi_=np.concatenate([xi,color_controller],axis=0)
                self.pub_seg.publish(rosnp.point_cloud2.array_to_pointcloud2(np_xyzi2pc(xi_), frame_id=CONF.FRAME_LIDAR))
            ps=self.point_selector.get_points(xi)
            if CONF.Start_MClient:
                if ps is not None:
                    ps=lidar2arm(ps)
                    ps = (ps*1000).astype(np.int32)
                    for i,p in enumerate(ps):
                        self.mclient.setpoint(i,p)
                    self.mclient.set_require(CONF.REQUIRE_SUCCESS) # over
                    print(f'task success')
                else:
                    self.mclient.set_require(CONF.REQUIRE_FAIL) # error
                    print(f'task fail')

        # shift move pc to arm coord
        # if DEBUG:
        #     x[:,:3] = lidar2arm(x[:,:3])
        #     self.shift_puber.publish(rosnp.point_cloud2.array_to_pointcloud2(np_xyzi2pc(x), stamp=stamp,frame_id=fi))

if __name__ == "__main__":
    main()
