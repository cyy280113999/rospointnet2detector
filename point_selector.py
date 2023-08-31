import rospy
import numpy as np
import ros_numpy as rosnp
from sensor_msgs.msg import PointCloud2
from transfer_dataset import *
import open3d as o3d
import global_config as CONF

BoundShrink=0.20
DIST_ROD=0.20
detect_box=np.array([[-1000, 1492,-1300],
                     [-6900, 2531,  465]],dtype=np.float32)
def bbox2lidar(x):
    x = arm2lidar(x/1000)
    x = x.T
    for i,iv in enumerate(x):
        if iv[0]>iv[1]: # ascending
            x[i]=iv[::-1]
    return x
detect_box=bbox2lidar(detect_box)
class_num=8
patch_num=15
np.random.seed(1)
intens2rgb=np.random.rand(class_num+patch_num,3)
np.random.seed(int(time.time()))
# bbox2 = np.array([[],
#                   []])
# bbox2 = bbox2lidar(bbox2)

class PointSelector:
    def __init__(self):
        self.pub_mater=rospy.Publisher('/debug_material',PointCloud2,queue_size=1)
        self.pub_patch=rospy.Publisher('/debug_patch',PointCloud2,queue_size=1)
        self.pub_max=rospy.Publisher('/debug_max_point',PointCloud2,queue_size=1)
        self.pub_log=rospy.Publisher('/debug_log_points',PointCloud2,queue_size=1)

    def get_points(self,x,n=13):
        # return: list->[points]. None->error 
        if CONF.DEBUG:
            pc_log=x
        avoid_y = x[(x[:,3]==4).__or__(x[:,3]==5),1] # exclude rod and rain
        if len(avoid_y)>0:
            y_intervals=find_clusters(avoid_y)
            y_intervals = interval_expand(y_intervals, radius=DIST_ROD)
            x = inverse_select(x,y_intervals)
        x = pc_bound(x,xb=detect_box[0],yb=detect_box[1],zb=detect_box[2])
        # x = pc_bound(x,xb=bbox2[0],yb=bbox2[1],zb=bbox2[2])
        material = x[(x[:,3]==3).__or__(x[:,3]==6)] # material and target
        if len(material)==0:
            points=None
        else:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(material[:,:3])
            pc,ind = pc.remove_radius_outlier(nb_points=3,radius=0.20)
            material = np.concatenate([np.array(pc.points,dtype=np.float32),material[ind,3,None].astype(np.float32)],axis=1)
            # o3d.visualization.draw_geometries([pc,],window_name="pc down",width=800,height=600)
            if len(material)==0:
                points=None
            else:
                bound = np.max(material,axis=0)
                yM=bound[1]-BoundShrink
                zM=bound[2]#-BoundShrink
                bound = material.min(axis=0)
                ym=bound[1]+BoundShrink
                zm=bound[2]+BoundShrink
                # if CONF.DEBUG: # not record material. 
                #     pc_plot=material+np.array([[-0.2,0,0,0]])
                #     pc_log=np.vstack([pc_log,pc_plot])
                #     self.pub_mater.publish(rosnp.point_cloud2.array_to_pointcloud2(np_xyzi2pc(pc_plot),frame_id=CONF.lidar_frame))
                ysteps=5
                zsteps=3
                patches=[]
                ystep=(yM-ym)/ysteps
                zstep=(zM-zm)/zsteps
                if CONF.DEBUG:
                    pc_plot=[]# plot patches
                # split area into 15 patches
                for i in range(ysteps):
                    indices=(material[:,1]>=(ym+i*ystep)).__and__(material[:,1]<=(ym+(i+1)*ystep))
                    row=material[indices]
                    material=material[~indices]
                    for j in range(zsteps):
                        indices=(row[:,2]>=(zm+j*zstep)).__and__(row[:,2]<=(zm+(j+1)*zstep))
                        col=row[indices]
                        row=row[~indices]
                        if len(col)>0:
                            col[:,3]=class_num+i*zsteps+j # over write intensity
                            patches.append(col)
                if CONF.DEBUG:
                    pc_plot=np.vstack(patches)+np.array([[-0.4,0,0,0]])
                    pc_log=np.vstack([pc_log,pc_plot])
                    self.pub_patch.publish(rosnp.point_cloud2.array_to_pointcloud2(np_xyzi2pc(pc_plot),frame_id=CONF.FRAME_LIDAR))
                if len(patches)<n//2: # less than half
                    points=None
                else:
                    # get max point of each patch
                    points=[]
                    for i in range(len(patches)):
                        patch=patches[i]
                        # ind=patch[:,0].argmin(axis=0) # hightest. min in x axis 
                        h=patch[:,0]
                        ind=find_nearst(h,h.mean())
                        patch[ind,3]=0 # point with zero intensity
                        points.append(patch[ind])
                    points=np.vstack(points)
                    if len(points)<n//2: # fill the list, ensure enough to 15
                        count_lack=n//2-len(points)
                        ind_add=np.random.choice(len(points),count_lack)
                        points=np.vstack([points,points[ind_add]])
                    points=points[np.argsort(points[:,1])[::-1]] # sort by y
                    if n==2:
                        points = points[[0,-1]]
                    if CONF.DEBUG:
                        pc_plot=points+np.array([[-0.8,0,0,0]])
                        pc_log=np.vstack([pc_log,pc_plot])
                        self.pub_max.publish(rosnp.point_cloud2.array_to_pointcloud2(np_xyzi2pc(pc_plot),frame_id=CONF.FRAME_LIDAR))
                        self.pub_log.publish(rosnp.point_cloud2.array_to_pointcloud2(np_xyzi2pc(pc_log),frame_id=CONF.FRAME_LIDAR))
        if CONF.DEBUG:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pc_log[:,:3]) # remove intensity
            pc.colors=o3d.utility.Vector3dVector([intens2rgb[int(i)]for i in pc_log[:,3]])# keep intensity
            s1,s2=day_second()
            if not os.path.exists(pj(CONF.DIR_PC_LOG,s1)):
                os.makedirs(pj(CONF.DIR_PC_LOG,s1))
            o3d.io.write_point_cloud(pj(CONF.DIR_PC_LOG,s1,f'{s2}.pcd'),pc)
        if points is not None and points.shape[1]==4:
            points[:,:3] # remove intensity
        return points

        
def find_clusters(data, max_dist=0.2): # get intervals which has points in distance MAX
    data.sort()  # 对数据进行排序
    clusters = []  # 存储聚集区间
    start = data[0]  # 起始点
    end = data[0]  # 结束点
    for i in range(1, len(data)):
        if data[i] - end <= max_dist:
            end = data[i]
        else:
            clusters.append((start, end))
            start = data[i]
            end = data[i]
    if start!=end:
        clusters.append((start, end))  # 添加最后一个聚集区间（如果存在）
    return clusters

def interval_expand(itvs,radius=0.1): # expand each intervals left and right
    for i,itv in enumerate(itvs):
        itvs[i]=(itv[0]-radius,itv[1]+radius)
    return itvs

def inverse_select(points,intervals): # choose points that are not in these intervals
    for (ym,yM) in intervals:
        points=points[(points[:,1]<=ym).__or__(points[:,1]>=yM)]
    return points
