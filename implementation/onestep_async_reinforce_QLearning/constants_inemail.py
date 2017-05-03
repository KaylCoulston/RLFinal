# -*- coding: utf-8 -*-
from ale_python_interface import ALEInterface

LOCAL_T_MAX = 20 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints_seaquest'
LOG_FILE = 'logs_seaquest/'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 16 # parallel thread size
ROM = "roms/seaquest.bin"     # action size = 3
_ale = ALEInterface()
_ale.loadROM(ROM)
ACTION_SIZE = len(_ale.getMinimalActionSet())

FRAME_SKIPS = 30

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA1 = 0.01 # entropy regurarlization constant
ENTROPY_BETA2 = 0.00
MAX_TIME_STEP = 10 * 10**7
LIMIT = MAX_TIME_STEP
GRAD_NORM_CLIP = 40.0 # gradient norm clipping

USE_GPU = False # To use GPU, set True
USE_LSTM = True # True for A3C LSTM, False for A3C FF
CHECK_POINT_INTERVAL = 1000000
SAMPLE_FROM_POLICY = 0.1
DISPLAY = False
RESET_ON_LOST_LIFE = False
SCORES_FILE = LOG_FILE + "scores.txt"
