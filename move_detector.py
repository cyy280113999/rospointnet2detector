import time
import rospy
import numpy as np
import ros_numpy as rosnp
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray,Float32, Int32
from utilities import *
from collections import deque
from threading import Timer
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import global_config as CONF

"""
move detect state:
0 no object
1 unrecogized object
2 at one side
3 mass center at given height
4 mass center stop move
5 truck velocity stop
6 truck position stop # wait


"""


PLOT_MAX=3
debug_c_amp=PLOT_MAX/CONF.MOVE_TOLERANCE_CENTER
debug_v_amp=PLOT_MAX/CONF.MOVE_TOLERANCE_VELOCITY # the maximun of tolerance is shown as PLOT_MAX
debug_p_amp=PLOT_MAX/CONF.MOVE_TOLERANCE_POSITION
# where is the point cloud?
TOPIC_TEST='/pointclouds'



def main():
    rospy.init_node('ros_motion_detect', anonymous=False)
    pc_processer = PC_Processer()
    pc_processer.start()
    rospy.spin() 
    pc_processer.stop()
    print("Shutting Down")

class PC_Processer:
    def __init__(self):
        self.pc_sub=None
        self.detector=MoveDetector()
    def start(self):
        self.pc_sub = rospy.Subscriber(TOPIC_TEST, PointCloud2, self.send, queue_size=1, buff_size=2 ** 24)
    def stop(self):
        self.pc_sub.unregister()
    def send(self,pc):
        t = pc.header.stamp.to_time()
        x = rosnp.point_cloud2.pointcloud2_to_xyz_array(pc)
        state=self.detector.process(x,t)


class MoveDetector:
    def __init__(self):
        self.pub_center = rospy.Publisher('/debug_mass_center', Float32, queue_size=1) # center position
        self.pub_move_c = rospy.Publisher('/debug_move_c', Float32, queue_size=1) # mass center move
        self.pub_move_v = rospy.Publisher('/debug_move_v', Float32, queue_size=1) # velocity
        self.pub_move_p = rospy.Publisher('/debug_move_p', Float32, queue_size=1) # position
        self.pub_state = rospy.Publisher('/debug_move_state', Int32, queue_size=1) # move state

        self.time_list = []
        self.pc2d_list = []
        self.center_list = []
        self.v_list=[]
        self.stop_position=None

    def process(self, x:np.ndarray,t):
        state=0 # initial state
        x=x.copy()
        # x = x[x[:,0]<=2-0.1] # remove the road at bottom. this is unused
        x = pc_bound(x,zb=[-0.25,0.25]) # remove the edge from both side. keep z in -0.25 ~ 0.25
        x = x[:,:2] # to 2d points. remove z axis 
        x = x[:,::-1] # to standard viewpoint, come from -x to +x. 1. swap xy axis, new x is old y
        x[:,1] = -x[:,1] # 2. flip new y(old x, depth)
        if len(x)<=0: # no points
            print(f'move-detect: frame {t} has no enough points')
            return state
        self.time_list=insert_with_limit(self.time_list,t, 100)
        self.pc2d_list=insert_with_limit(self.pc2d_list, x, 100) # notice pc2d is at new coordinate.
        if len(self.time_list)<=1: # first estimation has no enough frames, skip
            return state
        center_y=np.mean(x,axis=0)[1].item() # height of mass center
        if CONF.DEBUG:
            self.pub_center.publish(Float32(center_y))
        if -2.3<=center_y<-1.7: # origin is 2m above road. so -2 is the road
            state=0
        elif -1.7<=center_y<-1.0: # small object occur
            state=1
        elif -1.0<=center_y<0: # possibly truck at one side
            state=2
        elif center_y>=0: # truck is at proper position(center), run moving estimation
            state=3
        if state!=3:
            self.center_list.clear()
        else:
            ind_c = find_nearst(self.time_list,t-CONF.MOVE_FRAME_CURRENT).item()
            ind_l = find_nearst(self.time_list,t-CONF.MOVE_FRAME_BEFORE).item()
            if ind_c==ind_l: # if no point comes in 1s
                print('move-detect: time interval is too long')
                return state
            # if ind_c==ind_l: # can not find two different frames
            #     ind_l+=1
            #     if ind_l>=len(self.time_history):
            #         moving=True
            center_c = self.pc2d_list[ind_c].mean(axis=0)
            center_l = self.pc2d_list[ind_l].mean(axis=0)
            move_dist = np.linalg.norm(center_c - center_l)
            self.center_list=insert_with_limit(self.center_list,move_dist,5)
            move_dist=np.mean(self.center_list)
            if CONF.DEBUG:
                self.pub_move_c.publish(Float32(clipMM(debug_c_amp*move_dist)))
            if move_dist <= CONF.MOVE_TOLERANCE_CENTER:
                state = 4
        # must notice not all truck body can observed by lidar,
        # when viewpoint filled, mass center moves not obviously. icp is taken to enhance judgement
        if state!=4: # center detection is not concise, continue estimate
            self.v_list.clear()
        else:
            # 2d icp
            target = self.pc2d_list[ind_c] # target position is current position
            source = self.pc2d_list[ind_l] # source position is at a gap time before
            dist, _ = icp_simplified(source, target)
            if abs(dist)>0.5:
                print(f'move-detect: dist overflow: {dist}')
                dist=max(0.5,min(-0.5,dist))
            v=dist/(self.time_list[ind_c]-self.time_list[ind_l]) # take velocity
            self.v_list=insert_with_limit(self.v_list,v,3) # aveage v by time
            v=np.mean(self.v_list)
            if abs(v)<=CONF.MOVE_TOLERANCE_VELOCITY: # over threshold
                state=5
                if CONF.MOVE_POSITION and self.stop_position is None:
                    self.stop_position = self.pc2d_list[ind_c] # record stop position
            else:
                self.stop_position = None
        if state==5 and self.stop_position is not None:
            source = self.stop_position
            p, _ = icp_simplified(source, target) # which dist + source == target ?
            if abs(p)<=CONF.MOVE_TOLERANCE_POSITION:
                state=6
        if CONF.DEBUG:
            self.pub_state.publish(Int32(state))
            if state>=4:
                self.pub_move_v.publish(Float32(clipMM(v*debug_v_amp)))
            if CONF.MOVE_POSITION and self.stop_position is not None:
                self.pub_move_p.publish(Float32(clipMM(p*debug_p_amp)))
        return state

def clipMM(x):
    return min(2*PLOT_MAX,max(-2*PLOT_MAX,x))

def icp_simplified(source:np.ndarray, target, max_iterations=50, tolerance=1e-3): # detect x-shift from two curve
    # what dist if source + dist == target?
    x_shift = 0.05
    iterations = 0
    prev_shift = 100
    tree = KDTree(target)
    while iterations < max_iterations:
        # 对齐源点云到目标点云
        source_aligned = source.copy()
        source_aligned[:,0] += x_shift
        _, nearest_indices = tree.query(source_aligned)
        nearest_points = target[nearest_indices]
        x_shift = np.mean(nearest_points - source_aligned, axis=0)[0].item()
        if abs(prev_shift - x_shift) < tolerance:
            break
        iterations += 1
        prev_shift = x_shift
    return x_shift, iterations

if __name__ == "__main__":
    main()
