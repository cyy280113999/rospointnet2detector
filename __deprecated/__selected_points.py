import rospy
 
from sensor_msgs.msg import PointCloud2
 
from visualization_msgs.msg import InteractiveMarkerFeedback
 
import sensor_msgs.point_cloud2 as pc2
 
import rospy
from geometry_msgs.msg import PointStamped

def clicked_point_callback(msg):
    # 提取点击事件中的点坐标
    x = msg.point.x
    y = msg.point.y
    z = msg.point.z

    # 在这里进行进一步的处理，根据需要使用选择的点

def main():
    rospy.init_node('point_selection_listener')

    # 订阅rviz的点击事件主题
    rospy.Subscriber('/clicked_point', PointStamped, clicked_point_callback)

    # 等待接收到点击事件
    rospy.spin()

if __name__ == '__main__':
    main()

# def marker_feedback_callback(msg):
#     print('reseived')
 
#     if msg.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
 
#         # 获取点击的点的位置信息
 
#         x = msg.pose.position.x
 
#         y = msg.pose.position.y
 
#         z = msg.pose.position.z
 
 
#         # 创建点云数据
 
#         fields = [pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
 
#                   pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
 
#                   pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1)]
 
#         header = rospy.Header(frame_id='base_link')  # 假设点云相对于base_link坐标系
 
#         cloud = pc2.create_cloud_xyz32(header, [[x, y, z]])
 
 
#         # 发布点云数据到另一个话题
 
#         pub.publish(cloud)
 
 
# if __name__ == '__main__':
 
#     # 初始化ROS节点
 
#     rospy.init_node('point_cloud_selector', anonymous=True)
 
 
#     # 创建一个发布器，用于发布所选点云数据
 
#     pub = rospy.Publisher('selected_point_cloud_topic', PointCloud2, queue_size=10)
 
 
#     # 创建一个订阅器，用于接收InteractiveMarkerFeedback消息
 
#     rospy.Subscriber('/interactive_marker_feedback', InteractiveMarkerFeedback, marker_feedback_callback)
 
 
#     # 循环等待回调
 
#     rospy.spin()