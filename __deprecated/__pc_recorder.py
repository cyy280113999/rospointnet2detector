import os
import sys
# print(sys.executable)
p = lambda f: os.path.abspath(f)
pj=lambda *args:os.path.join(*args)
BASE_DIR = os.path.dirname(p(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import argparse
import rospy
import numpy as np
import ros_numpy as rosnp
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import time
from transfer_dataset import np_pc2xyzi
# import keyboard

# save point cloud
# type : np.array
# xyz [0:3] : any
# intensity [4] : float 0-255

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--topic_from', type=str, default='/rslidar_points', help='where is the raw point cloud?')
    parser.add_argument(
        '--record_dir', type=str, default='/home/cyy/datasets/saved_npys', help='where to record/save pc into disk')
    parser.add_argument(
        '--rate', type=int, default=1, help='record rate. 0 to manually, else to auto record.')
    parser.add_argument(
        '--times', type=int, default=3600*10, help='record times for auto recording by rate.')
    args = parser.parse_args()
    return args 
args=get_params()

def main():
    rospy.init_node('pc_recorder')
    pc_processer = PC_Recorder()
    # pc_processer.start()
    pc_processer.recording(rate=args.rate, times=args.times)  # call to record point cloud
    # try:
    #     rospy.spin()  # python rospy.spin() is only to wait for exit, instead of dispatching messages.
    # except rospy.ROSInterruptException:
    #     pc_processer.stop()
    #     print("Shut Down")


class PC_Recorder:
    def __init__(self):
        self.pc_sub=None
        
        # record
        self.record_flag=False
        self.record_dir = args.record_dir
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
        data_files=[e.name for e in os.scandir(self.record_dir)]
        nums=[int(f.split('/')[-1].split('.')[0].split('_')[1]) for f in data_files]
        if nums:
            self.counter=max(nums) + 1
        else:
            self.counter=0
    
    def recording(self,rate=0, times=20):
        print(f'{time.asctime()} Point Cloud Recorder Start.')
        self.pc_sub = rospy.Subscriber(args.topic_from, PointCloud2, self.process, queue_size=1, buff_size=2 ** 24)
        if rate==0: # manually
            while True:
                print(f'{time.asctime()} Press . to record, 3 to exit.')
                cmd=input()
                if cmd[0]=='.':
                    self.record_flag=True
                elif cmd[0]=='3':
                    print(f'{time.asctime()} Shutting Down.')
                    break
        else:
            rater=rospy.Rate(rate)
            print(f'{time.asctime()} Auto Recording Start.')
            for _ in range(times):
                self.record_flag=True
                rater.sleep()
            print(f'{time.asctime()} Auto Recording Finish.')
        self.pc_sub.unregister()
    
        #--keyboard need root!
        # print(f'{time.asctime()} space: record, q: exit.')
        # keyboard.on_press(self.on_key_press)
        # keyboard.wait('esc')
        # self.kb_sub = rospy.Subscriber('keyboard_input', String, self.on_key_press, queue_size=1)
        # rospy.spin()
        # print("Shutting Down")
    # def on_key_press(self,data):
    #     print(f'press {data.data}')
    #     if data.data=='space':
    #         self.record_flag=True
    #         print(f'{time.asctime()} space: record, q: exit.')
    #     elif data.data=='q':
    #         # keyboard.unhook_all()
    #         pass
    # def on_key_press(self,event):
        # if event.name=='space':
        #     self.record_flag=True
        #     print(f'{time.asctime()} space: record, q: exit.')
        # elif event.name=='q':
        #     # keyboard.unhook_all()
        #     pass
        
    def process(self, pc:PointCloud2):
        if self.record_flag:
            x = rosnp.point_cloud2.pointcloud2_to_array(pc)
            x = np_pc2xyzi(x)
            np.save(pj(self.record_dir,f'pc_{self.counter:06d}'),x)
            print(f'{time.asctime()} Record once at {self.counter}.')
            self.counter+=1
            self.record_flag=False

if __name__ == "__main__":
    main()
