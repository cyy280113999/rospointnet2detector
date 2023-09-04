# rename: global_config_default.py --> global_config.py
# usage: import global_config as CONF
import os
pj=lambda *args:os.path.join(*args)
# ============= customized params
DEBUG=True  # debug print some messages
AtRemote=True # is server at remote ?
# usr_name = 'cyy'
usr_name = 'cyy' if not AtRemote else 'lhk'
# ==main
ON_LINE=True  # where is pc?
Start_MServer=True
Start_MClient=True
MOVE_DETECT=True
REQUIRE_NEED=True
DETECT=True
# ==lidar
FRAME_LIDAR='rslidar'
TOPIC_RAW='/rslidar_points'
# ==data
DIR_CONFIG='./config'
FILE_CALIB=pj(DIR_CONFIG,'calib_info.npy')
TOPIC_CALIB='/pointclouds' # show what pc to train
TOPIC_OFFLINE='/off_line'
FILE_NORM=pj(DIR_CONFIG,'ds_norm.npy')
TOPIC_ARM='/arm_pc' # show what pc in arm coord

# ==move
# detect tolerance
MOVE_TOLERANCE_CENTER = 0.40 # not much over 0.2
MOVE_TOLERANCE_VELOCITY = 0.05 # (m/s)
MOVE_TOLERANCE_POSITION = 0.10 # (m)
# when are frames sampled
MOVE_FRAME_CURRENT=0.01
MOVE_FRAME_BEFORE=1.0
MOVE_STATE_STOP=3
MOVE_MASS_CENTER=True
MOVE_VELOCITY=True
MOVE_POSITION=False
if MOVE_MASS_CENTER:
    MOVE_STATE_STOP=4
if MOVE_VELOCITY:
    MOVE_STATE_STOP=5
if MOVE_POSITION:
    MOVE_STATE_STOP=6

# ==net
MODBUS_IP='192.168.1.120'
MODBUS_PORT=12340
MODBUS_WORD_REQUIRE=10 # PLC start at 1. PYMODBUS start at 0
REQUIRE_201=101
REQUIRE_202=102
REQUIRE_203=104
REQUIRE_13=103
REQUIRE_SUCCESS=3
REQUIRE_FAIL=4
MODBUS_WORD_POINTS=11
# point=2*3*13=78
MODBUS_WORD_MOVE=89
MODBUS_MOVE_STOP=0

# ==log
# where to save pc after calibration. control in console
RECORD_DIR = f'/home/{usr_name}/datasets/saved_pcds'
DIR_PC_LOG=f'/home/{usr_name}/datasets/pc_log'
DIR_PC_LOG10=f'/home/{usr_name}/datasets/pc_log10'





