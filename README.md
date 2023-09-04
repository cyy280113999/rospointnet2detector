开发文档（简明）

### 快速开始 

1. 运行ros核心，bash roscore

2. 运行rslidar sdk。bash ~/rs16/run.sh

sdk是速腾设计的rs-16与ros话题之间数据转发的程序。

经过简单配置，pointcloud2点云被发送至/rslidar_points话题。

3. 运行主程序。python ~/rpc/pc_console.py

主程序创建了命令循环，可以通过输入命令开启或关闭功能。

除开控制命令循环的其他功能，主程序的入口是start

首先创建的话题订阅者，订阅了rslidar sdk的话题，（或离线文件发布的话题）

rospy.Subscriber(self.topic_raw, PointCloud2, self.process, queue_size=1, buff_size=2 ** 24)

同时绑定了回调函数。

当数据传入，激活回调函数，处理点云数据。

1. 点云的坐标转换

2. 区域限制

3. 根据历史判断点云的移动情况

4. 根据网络判断合作机器的请求符号

5. 计算点云分割

6. 根据要求，计算取样点位，2个或13个

7. 发送取样点位

### 点云数据

在线数据 Online Data 从sdk中读取，通过将在线数据写入硬盘，得到离线数据。pc_recorder设计了离线数据的存储流程。

此模块可以独立运行
1. 订阅了某个点云话题
2. 设计文件写入函数
3. 设计回调函数，send(self,pc: sensor_msgs.msg.PointCloud2)，保存所有接受的数据

此模块可以嵌入主程序
1. 通过record命令执行手动记录
2. 通过auto_record 命令执行自动计时记录

### 坐标系

记录的坐标系不是相机的原始坐标系。经过仅平移旋转，将点云方向与空间垂直、与路面平行。点云的原点被移动到路面中心高2m处，
这个过程称为calibration。当前的变换矩阵被记录在文件中，每次执行固定的变换。当相机位置发生偏移，重新执行此过程，可以生成新变换矩阵。

同时，与合作方坐标系也需要变换。

### 标签

使用semantic segmentation editor进行标签 
[site](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor)

meteor npm install # 初次使用

meteor npm start

### 神经网络分割模型

网络是pointnet2（pointnet++）[site](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

使用在shapenet上预训练的模型进行迁移。

transfer_pn2可以导入预训练的模型，实现输入输出通道变换

transfer_train可以训练模型

### 训练流

1. 获得预训练模型
2. 获得数据集
3. 使用标签工具，标注一定数量的数据集
4. 迁移训练模型
5. 使用predict预测未标注的数据集
6. 使用标签工具修正预测结果
7. 融合数据集，再次训练
8. 随时间进行5-7步骤

### 运动估计

move_detector 通过点云历史估计当前的运动状态

### 网络通信

程序与合作方通过modbus交换数据

mserver创建网络存储空间

mclient写入网络存储空间

require表示对方传输的取样请求信号

point表示我方传输的取样点位信息

move表示运动状态信号

### 取样方法

point_selector 设计了取样方法

传入模型分割结果





