# usage: import global_config as CONF
import os
pj=lambda *args:os.path.join(*args)
# ====== config
DEBUG=True  # debug print some messages
AtRemote=False # is server at remote ?
# -pc console
ON_LINE=False  # where is pc?
Start_MServer=False
Start_MClient=False
BOUNDING=True # bounding limit calibrated dataset in proper boundary 
# run net
MOVE_DETECT=True
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

stop_state=None # write in move detector
# net
mip='192.168.1.120'
mport=12340

# === auto parameters
usr_name = 'cyy' if not AtRemote else 'lhk'
# where to save pc after calibration. control in console
RECORD_DIR = f'/home/{usr_name}/datasets/saved_npys'
pc_log_dir=f'/home/{usr_name}/datasets/pc_log' # make sure exist






