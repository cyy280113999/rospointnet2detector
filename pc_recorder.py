import os
import time
import rospy
import numpy as np
import ros_numpy as rosnp
from sensor_msgs.msg import PointCloud2
import global_config as CONF
pj=lambda *args:os.path.join(*args)

# record pc from topic after calibration

def main():
    rospy.init_node('ros_pc_recorder', anonymous=False)
    pc_processer = PC_Processer()
    pc_processer.start()
    rospy.spin() 
    pc_processer.stop()
    print("Shutting Down")

class PC_Processer:
    def __init__(self):
        self.pc_sub=None
        self.processer=PC_Recorder(CONF.RECORD_DIR)
    def start(self):
        self.pc_sub = rospy.Subscriber(CONF.TOPIC_CALIB, PointCloud2, self.send, queue_size=1, buff_size=2 ** 24)
    def stop(self):
        self.pc_sub.unregister()
    def send(self,pc):
        x = rosnp.point_cloud2.pointcloud2_to_xyz_array(pc)
        self.processer.save_once(x)


class PC_Recorder:
    def __init__(self,record_dir):
        self.record_dir=record_dir
        self.record_flag=False
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
        data_files=[e.name for e in os.scandir(self.record_dir)]
        if data_files:
            nums=[int(f.split('/')[-1].split('.')[0].split('_')[1]) for f in data_files]
            self.counter=max(nums) + 1
        else:
            self.counter=0
    def save_once(self, x):
        np.save(pj(self.record_dir,f'pc_{self.counter:06d}'),x)
        print(f'{time.asctime()} Record once at {self.counter}.')
        self.counter+=1

if __name__ == "__main__":
    main()
