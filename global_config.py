# usage: import global_config as CONF
import os
pj=lambda *args:os.path.join(*args)
# ====== config
DEBUG=True  # debug print some messages
usr_name = 'cyy'
# -pc console
ON_LINE=False  # where is pc?
Start_MServer=False
Start_MClient=False
BOUNDING=True # bounding limit calibrated dataset in proper boundary 
# run net
MOVE_DETECT=False
REQUIRE_NEED=False
DETECT=True

# pc
lidar_frame='rslidar'
raw_topic='/rslidar_points'
config_dir='./config'
calib_file=pj(config_dir,'calib_info.npy')
calib_topic='/pointclouds'

offline_topic='/off_line'
norm_file=pj(config_dir,'ds_norm.npy')

# move detect
POSITION_DETECT=False
if not POSITION_DETECT:
    stop_state=5
else:
    stop_state=6
# net
mip='192.168.1.120'
mport=12340

# === auto parameters
# where to save pc after calibration. control in console
RECORD_DIR = f'/home/{usr_name}/datasets/saved_npys'
pc_log_dir=f'/home/{usr_name}/datasets/pc_log' # make sure exist






